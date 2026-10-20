import contextlib
import http.server
import io
import json
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from helpers import (
    EXAMPLE,
    FakeHTTP,
    cli,
    example,
    finished_run,
    github_api,
    message,
    repo,
)

import symbolic_acceptance as sa
from symbolic_acceptance import record
from symbolic_acceptance.agents import GRACE, CommandAgent, GitError, diff, git
from symbolic_acceptance.claude import Claude
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.draft import DraftError, draft

# For agents that run git themselves: like the test's own git, theirs reads
# no one's settings, which could run hooks or refuse to commit.
NO_GIT_CONFIG = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


class CommandAgentTests(unittest.TestCase):
    def task(self, number=1, previous=None):
        return sa.Task(
            issue="make it v2", checklist=example(), number=number, previous=previous
        )

    def test_uncommitted_changes_become_the_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            script = "import sys; prompt = sys.stdin.read(); open('app.py', 'w').write(\"print('v2')\\n\"); print('done:', 'make it v2' in prompt)"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            self.assertEqual(agent.base, git(tmp, "rev-parse", "HEAD").strip())
            attempt = agent(self.task())
            assert attempt.diff is not None
            self.assertEqual(attempt.claims, "done: True")
            self.assertIn("+print('v2')", attempt.diff)
            self.assertEqual(len(attempt.ref), 40)
            log = git(tmp, "log", "-1", "--format=%s")
            self.assertEqual(log.strip(), "Attempt 1")
            self.assertEqual(diff(tmp, agent.base, attempt.ref), attempt.diff)

    def test_leftovers_are_committed_in_the_users_name(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as user,
        ):
            local = repo(tmp)
            local("config", "--unset", "user.name")
            local("config", "--unset", "user.email")
            gitconfig = Path(user, "gitconfig")
            gitconfig.write_text("[user]\n\tname = Ada\n\temail = ada@example.com\n")
            # The user's configuration comes from this process's environment;
            # a process of its own has it.
            script = (
                "import sys, symbolic_acceptance as sa\n"
                "from symbolic_acceptance.agents import CommandAgent\n"
                "write = \"open('app.py', 'w').write('v2')\"\n"
                "agent = CommandAgent([sys.executable, '-c', write], cwd=sys.argv[1])\n"
                "agent(sa.Task('', sa.load(sys.argv[2]), 1))\n"
            )
            subprocess.run(
                [sys.executable, "-c", script, tmp, str(EXAMPLE)],
                env=dict(
                    os.environ,
                    GIT_CONFIG_GLOBAL=str(gitconfig),
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                check=True,
            )
            author = local("log", "-1", "--format=%an <%ae>").strip()
        self.assertEqual(author, "Ada <ada@example.com>")

    def test_a_checkout_another_owns_works_once_the_user_trusts_it(self):
        # As in a container job on GitHub Actions: git trusts a checkout
        # someone else owns only if the user's own configuration says so.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as user,
        ):
            repo(tmp)
            gitconfig = Path(user, "gitconfig")
            gitconfig.write_text("[safe]\n\tdirectory = *\n")
            script = (
                "import sys, symbolic_acceptance as sa\n"
                "from symbolic_acceptance.agents import CommandAgent\n"
                "write = \"open('app.py', 'w').write('v2')\"\n"
                "agent = CommandAgent([sys.executable, '-c', write], cwd=sys.argv[1])\n"
                "print(agent(sa.Task('', sa.load(sys.argv[2]), 1)).diff)\n"
            )
            done = subprocess.run(
                [sys.executable, "-c", script, tmp, str(EXAMPLE)],
                env=dict(
                    os.environ,
                    GIT_CONFIG_GLOBAL=str(gitconfig),
                    # Git's own switch to take every repository for another's.
                    GIT_TEST_ASSUME_DIFFERENT_OWNER="1",
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("+v2", done.stdout)

    def test_the_whole_change_counts_from_a_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            git("config", "diff.relative", "true")
            Path(tmp, "sub").mkdir()
            script = (
                "open('../app.py', 'w').write('v2\\n'); open('x.py', 'w').write('')"
            )
            agent = CommandAgent([sys.executable, "-c", script], cwd=Path(tmp, "sub"))
            attempt = agent(self.task())
        assert attempt.diff is not None
        self.assertIn("diff --git a/app.py b/app.py", attempt.diff)

    def test_a_submodule_update_alone_is_an_attempt_whatever_the_checkout_says(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as library,
        ):
            other = repo(library)
            first = other("rev-parse", "HEAD").strip()
            other("commit", "-q", "--allow-empty", "-m", "newer")
            newer = other("rev-parse", "HEAD").strip()
            git = repo(tmp)
            allow = ("-c", "protocol.file.allow=always")
            git(*allow, "submodule", "add", "-q", library, "vendored")
            git("-C", "vendored", "checkout", "-q", first)
            git("commit", "-qam", "vendor")
            git("config", "diff.ignoreSubmodules", "all")
            move = (
                "import subprocess; subprocess.run(['git', '-C', 'vendored',"
                f" 'checkout', '-q', '{newer}'], check=True)"
            )
            agent = CommandAgent(
                [sys.executable, "-c", move], cwd=tmp, env=NO_GIT_CONFIG
            )
            attempt = agent(self.task())
        assert attempt.diff is not None
        self.assertIn(f"+Subproject commit {newer}", attempt.diff)

    def test_a_new_file_is_an_attempt_whatever_the_checkout_says(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            git("config", "status.showUntrackedFiles", "no")
            script = "open('new.py', 'w').write('x = 1\\n')"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = agent(self.task())
        assert attempt.diff is not None
        self.assertIn("+++ b/new.py", attempt.diff)

    def test_the_checkout_cannot_rig_what_the_runner_sees(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            git = repo(tmp)
            Path(tmp, "tests").mkdir()
            Path(tmp, "tests", "test_app.py").write_text("a = 1\nassert True\nb = 2\n")
            git("add", ".")
            git("commit", "-qm", "tests")
            changed = "a = 1\nassert False or True\nb = 2\n"
            # A replace ref would show the old test in place of the change.
            Path(outside, "changed").write_text(changed)
            new = git("hash-object", "-w", str(Path(outside, "changed"))).strip()
            git("replace", new, git("rev-parse", "HEAD:tests/test_app.py").strip())
            # Hooks, a file monitor, diff drivers and settings that would hide
            # or reshape the change.
            hooks, ran = Path(outside, "hooks"), Path(outside, "monitor-ran")
            hooks.mkdir()
            for name, body in (("pre-commit", "exit 1"), ("monitor", f"touch {ran}")):
                Path(hooks, name).write_text(f"#!/bin/sh\n{body}\n")
                Path(hooks, name).chmod(0o755)
            for key, value in {
                "core.hooksPath": str(hooks),
                "core.fsmonitor": str(Path(hooks, "monitor")),
                "core.quotePath": "true",
                "color.ui": "always",
                "diff.noprefix": "true",
                "diff.context": "0",
                "diff.ignoreSubmodules": "all",
                "diff.submodule": "log",
                "diff.renames": "copies",
                "diff.external": "echo HIDDEN",
                "diff.hide.textconv": "echo HIDDEN",
            }.items():
                git("config", key, value)
            Path(tmp, ".git", "info", "attributes").write_text(
                "*.py diff=hide\ntests/test_app.py -diff\n"
            )
            # A gitlink, which ignoreSubmodules or the submodule log would hide.
            head = git("rev-parse", "HEAD").strip()
            Path(tmp, "tests", "vendored").mkdir()
            git("update-index", "--add", "--cacheinfo", f"160000,{head},tests/vendored")
            script = (
                f"open('tests/test_app.py', 'w').write({changed!r});"
                "open('app.py', 'w').write('v2\\n');"
                "open('naïve.txt', 'w').write('x\\n');"
                # A copy of a changed file, which copy detection would show
                # as changed rather than added.
                "open('tests/test_copy.py', 'w').write('a = 1\\nassert True\\nb = 2\\n')"
            )
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = agent(self.task())
            self.assertFalse(ran.exists(), "the file monitor never ran")
        assert attempt.diff is not None
        self.assertIn(
            "diff --git a/tests/test_app.py b/tests/test_app.py", attempt.diff
        )
        self.assertIn(
            " a = 1\n-assert True\n+assert False or True\n b = 2", attempt.diff
        )
        self.assertIn("-print('v1')\n+v2", attempt.diff)
        self.assertEqual(
            [(f.path, f.status) for f in parse(attempt.diff)[0]],
            [
                ("app.py", "modified"),
                ("naïve.txt", "added"),
                ("tests/test_app.py", "modified"),
                ("tests/test_copy.py", "added"),
                ("tests/vendored", "added"),
            ],
        )
        self.assertIn("+++ b/naïve.txt", attempt.diff)
        self.assertNotIn("HIDDEN", attempt.diff)
        self.assertNotIn("\x1b[", attempt.diff)

    def test_the_checkouts_diff_settings_change_nothing_in_the_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            lines = [f"line_{n} = {n}" for n in range(40)]
            Path(tmp, "app.py").write_text("\n".join(["def login():", *lines, ""]))
            Path(tmp, "b.py").write_text("def a():\n\n    return 1\n")
            for name in ("r1", "r2"):
                Path(tmp, f"{name}.txt").write_text(
                    "".join(f"{name} {n}\n" for n in range(30))
                )
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD").strip()
            settings = {
                # One hunk for a whole file, under a header naming any line.
                "diff.interHunkContext": "1000",
                "diff.pick.xfuncname": "^(line_3 .*)$",
                # Other alignments of the same change, and another file order.
                "diff.algorithm": "patience",
                "diff.pick.algorithm": "histogram",
                "diff.indentHeuristic": "false",
                "diff.orderFile": str(Path(tmp, ".git", "order")),
                # Renames read as deletions and additions, other ids, and
                # blank context lines written without their space.
                "diff.renameLimit": "1",
                "core.abbrev": "40",
                "diff.suppressBlankEmpty": "true",
            }
            for key, value in settings.items():
                git("config", key, value)
            Path(tmp, ".git", "order").write_text("b.py\napp.py\n")
            Path(tmp, ".git", "info", "attributes").write_text("*.py diff=pick\n")
            lines[5], lines[35] = "line_5 = 'five'", "line_35 = 'thirty-five'"
            text = "\n".join(["def login():", *lines, ""])
            write = (
                "import os; "
                f"open('app.py', 'w').write({text!r}); "
                "open('b.py', 'w').write('def a():\\n\\n    return 2\\n'); "
                "[os.rename(f'{n}.txt', f'moved_{n}.txt') for n in ('r1', 'r2')]; "
                "[open(f'moved_{n}.txt', 'a').write('more\\n') for n in ('r1', 'r2')]"
            )
            attempt = CommandAgent([sys.executable, "-c", write], cwd=tmp)(self.task())
            for key in settings:
                git("config", "--unset", key)
            Path(tmp, ".git", "info", "attributes").unlink()
            plain = diff(tmp, base, attempt.ref)
        assert attempt.diff is not None
        self.assertEqual(attempt.diff, plain)
        headers = [line for line in plain.splitlines() if line[:2] == "@@"]
        self.assertEqual(
            headers[:3], ["@@ -4,7 +4,7 @@", "@@ -34,7 +34,7 @@", "@@ -1,3 +1,3 @@"]
        )
        self.assertIn("rename from r1.txt", plain)
        self.assertIn("rename from r2.txt", plain)

    def test_attempts_are_pushed_even_when_the_agent_rewrites_them(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            repo(tmp)("remote", "add", "origin", remote)
            script = "open('app.py', 'w').write(\"print('v2')\\n\")"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp, push="agent")
            first = agent(self.task())
            amend = (
                "import subprocess; open('app.py', 'w').write('v3\\n');"
                "subprocess.run(['git', 'commit', '-qa', '--amend', '-m', 'v3'])"
            )
            rewriter = CommandAgent(
                [sys.executable, "-c", amend], cwd=tmp, push="agent", base=agent.base
            )
            second = rewriter(self.task(2, previous=first))
            pushed = git(remote, "rev-parse", "refs/heads/agent").strip()
        self.assertEqual(pushed, second.ref)

    def test_an_attempt_that_changes_workflows_is_not_pushed(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            repo(tmp)("remote", "add", "origin", remote)
            script = (
                "import os; os.makedirs('.github/workflows');"
                "open('.github/workflows/ci.yml', 'w').write('on: push\\n')"
            )
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp, push="agent")
            with self.assertRaisesRegex(
                sa.Escalate, r"changes \.github/workflows/ci\.yml"
            ):
                agent(self.task())
            self.assertEqual(git(remote, "branch", "--list").strip(), "")

    def test_a_push_follows_one_that_a_stopped_run_did_not_record(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            local = repo(tmp)
            local("remote", "add", "origin", remote)
            write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
            base = local("rev-parse", "HEAD").strip()
            first = CommandAgent(write, cwd=tmp, push="agent")(self.task())
            # Attempt 2 was committed and pushed, and the run stopped then.
            Path(tmp, "app.py").write_text("v3\n")
            local("commit", "-qam", "Attempt 2")
            local("push", "-q", "origin", "HEAD:refs/heads/agent")
            # Asked again, the agent rewrites it.
            amend = ["git", "commit", "-q", "--amend", "-m", "Attempt 2, again"]
            agent = CommandAgent(
                amend, cwd=tmp, push="agent", base=base, env=NO_GIT_CONFIG
            )
            second = agent(self.task(2, previous=first))
            self.assertEqual(
                git(remote, "rev-parse", "refs/heads/agent").strip(), second.ref
            )

    def test_a_push_makes_the_branch_again_if_it_is_gone(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            local = repo(tmp)
            local("remote", "add", "origin", remote)
            write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
            base = local("rev-parse", "HEAD").strip()
            first = CommandAgent(write, cwd=tmp, push="agent")(self.task())
            # Deleted, as when its pull request was closed.
            local("push", "-q", "origin", ":refs/heads/agent")
            again = [sys.executable, "-c", "open('app.py', 'w').write('v3\\n')"]
            agent = CommandAgent(again, cwd=tmp, push="agent", base=base)
            second = agent(self.task(2, previous=first))
            self.assertEqual(
                git(remote, "rev-parse", "refs/heads/agent").strip(), second.ref
            )

    def test_a_push_never_overwrites_someone_elses_commit(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            local = repo(tmp)
            local("remote", "add", "origin", remote)
            write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
            base = local("rev-parse", "HEAD").strip()
            first = CommandAgent(write, cwd=tmp, push="agent")(self.task())
            # Someone else pushes to the branch, on top of the first attempt.
            local("commit", "-q", "--allow-empty", "-m", "theirs")
            local("push", "-q", "origin", "HEAD:refs/heads/agent")
            theirs = local("rev-parse", "HEAD").strip()
            local("reset", "-q", "--hard", first.ref)
            again = [sys.executable, "-c", "open('app.py', 'w').write('v3\\n')"]
            agent = CommandAgent(again, cwd=tmp, push="agent", base=base)
            with self.assertRaisesRegex(GitError, "push"):
                agent(self.task(2, previous=first))
            self.assertEqual(
                git(remote, "rev-parse", "refs/heads/agent").strip(), theirs
            )

    def test_an_attempt_follows_the_last_one_recorded(self):
        # A commit of the runner's that no record kept, as when its push
        # failed, counts with the next attempt; a checkout elsewhere does not.
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            base = git("rev-parse", "HEAD").strip()
            noop = [sys.executable, "-c", "pass"]
            agent = CommandAgent(noop, cwd=tmp, base=base)
            Path(tmp, "app.py").write_text("print('v2')\n")
            git("commit", "-qam", "Attempt 1")
            first = agent(self.task())
            self.assertIn("+print('v2')", first.diff)
            Path(tmp, "app.py").write_text("print('v3')\n")
            git("commit", "-qam", "Attempt 2")
            self.assertIn("+print('v3')", agent(self.task(2, previous=first)).diff)
            git("checkout", "-q", base)
            # Elsewhere, or not in this clone at all.
            for previous in (first, sa.Attempt("1" * 40, 1)):
                with self.assertRaisesRegex(sa.Escalate, "does not follow attempt 1"):
                    agent(self.task(2, previous=previous))
            with self.assertRaisesRegex(sa.Escalate, "does not follow the run's base"):
                CommandAgent(noop, cwd=tmp, base=first.ref)(self.task())

    def test_the_agent_never_gets_the_runners_github_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            script = (
                "import os; open('app.py', 'w').write('v2');"
                "print(sorted(k for k in os.environ if k.endswith('TOKEN')))"
            )
            env = {"PATH": os.environ["PATH"], "GITHUB_TOKEN": "t", "GH_TOKEN": "t"}
            agent = CommandAgent(
                [sys.executable, "-c", script],
                cwd=tmp,
                env={**env, "ANTHROPIC_AUTH_TOKEN": "its own"},
            )
            attempt = agent(self.task())
        self.assertEqual(attempt.claims, "['ANTHROPIC_AUTH_TOKEN']")

    def test_the_runners_git_keeps_its_token_from_the_checkouts_programs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            local = repo(tmp)
            seen = Path(outside, "seen")
            spy = f"sh -c 'echo ${{GITHUB_TOKEN:-none}} > {seen}; cat'"
            local("config", "filter.spy.clean", spy)
            Path(tmp, ".gitattributes").write_text("*.py filter=spy\n")
            Path(tmp, "app.py").write_text("print('v2')\n")
            # The runner's git reads this process's environment; a process of
            # its own holds the token.
            script = (
                f"from symbolic_acceptance.agents import git; git({tmp!r}, 'add', '-A')"
            )
            subprocess.run(
                [sys.executable, "-c", script],
                env=dict(
                    os.environ,
                    GITHUB_TOKEN="secret",
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                check=True,
            )
            self.assertEqual(seen.read_text().strip(), "none")

    def test_the_push_carries_the_runners_token(self):
        # A credential helper may read it to authenticate the push.
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            git(remote, "init", "-q", "--bare")
            local = repo(tmp)
            local("remote", "add", "origin", remote)
            seen = Path(outside, "seen")
            spy = f"sh -c 'echo ${{GITHUB_TOKEN:-none}} >> {seen}; exec git-receive-pack \"$1\"' -"
            local("config", "remote.origin.receivepack", spy)
            script = (
                "import sys, symbolic_acceptance as sa\n"
                "from symbolic_acceptance.agents import CommandAgent\n"
                "write = \"open('app.py', 'w').write('v2')\"\n"
                "agent = CommandAgent([sys.executable, '-c', write], cwd=sys.argv[1],"
                " push='agent')\n"
                "agent(sa.Task('', sa.load(sys.argv[2]), 1))\n"
            )
            subprocess.run(
                [sys.executable, "-c", script, tmp, str(EXAMPLE)],
                env=dict(
                    os.environ,
                    GITHUB_TOKEN="secret",
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                check=True,
            )
            self.assertEqual(seen.read_text(), "secret\n")

    def test_a_git_failure_says_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            with self.assertRaisesRegex(GitError, "nothing to commit"):
                git(tmp, "commit", "-m", "empty")

    def test_an_agent_past_its_timeout_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            sleeper = [sys.executable, "-c", "import time; time.sleep(5)"]
            agent = CommandAgent(sleeper, cwd=tmp, timeout=0.5)
            with self.assertRaisesRegex(sa.Escalate, "ran past"):
                agent(self.task())

    def test_no_change_is_an_error(self):
        empty = "git commit -q --allow-empty -m nothing"
        undone = "git commit -qam v2 && git revert --no-edit HEAD >/dev/null"
        # Line endings git normalizes away, and an edit inside a submodule,
        # which only the submodule could commit.
        endings = "printf \"print('v1')\\r\\n\" > app.py"
        inside = "echo x > vendored/new.txt"
        scripts = ("true", empty, f"echo v2 > app.py && {undone}", endings, inside)
        for script in scripts:
            with (
                self.subTest(script),
                tempfile.TemporaryDirectory() as tmp,
                tempfile.TemporaryDirectory() as library,
            ):
                git = repo(tmp)
                repo(library)
                allow = ("-c", "protocol.file.allow=always")
                git(*allow, "submodule", "add", "-q", library, "vendored")
                git("commit", "-qm", "vendor")
                git("config", "core.autocrlf", "input")
                agent = CommandAgent(["sh", "-c", script], cwd=tmp, env=NO_GIT_CONFIG)
                with self.assertRaisesRegex(sa.Escalate, "changed nothing"):
                    agent(self.task())

    def test_what_the_agent_leaves_running_does_not_hold_the_runner_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            script = "echo v2 > app.py; sleep 30 & echo done"
            agent = CommandAgent(["sh", "-c", script], cwd=tmp, timeout=20)
            started = time.monotonic()
            attempt = agent(self.task())
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(attempt.claims, "done")

    def test_failing_agent_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent(
                [sys.executable, "-c", "raise SystemExit('boom')"], cwd=tmp
            )
            with self.assertRaisesRegex(sa.Escalate, "exited with 1: boom"):
                agent(self.task())


class DraftTests(unittest.TestCase):
    def test_bad_drafts_are_sent_back_with_their_problems(self):
        good = EXAMPLE.read_text()
        fake = FakeHTTP(
            message({"checklist": "[q.a\n", "questions_for_reporter": []}),
            message(
                {
                    "checklist": good.replace('by = "llm"', 'by = "robot"'),
                    "questions_for_reporter": [],
                }
            ),
            message(
                {"checklist": good, "questions_for_reporter": ["Which lockout limit?"]}
            ),
        )
        result = draft(
            "Logins are not throttled", claude=Claude(api_key="k", transport=fake)
        )
        self.assertEqual(result.checklist.digest, example().digest)
        self.assertEqual(result.questions, ("Which lockout limit?",))
        retry = fake.body(1)["messages"][0]["content"]
        self.assertIn("It has these problems", retry)
        self.assertIn("Logins are not throttled", retry)
        self.assertIn("unknown-oracle", fake.body(2)["messages"][0]["content"])

    def test_gives_up_after_tries(self):
        bad = message({"checklist": "nope = 1", "questions_for_reporter": []})
        fake = FakeHTTP(bad, bad)
        with self.assertRaisesRegex(DraftError, "no valid draft after 2 tries"):
            draft("x", claude=Claude(api_key="k", transport=fake), tries=2)
        fake = FakeHTTP(bad)
        with self.assertRaisesRegex(DraftError, "the last draft:\nnope = 1$"):
            draft("x", claude=Claude(api_key="k", transport=fake), tries=1)


class CLITests(unittest.TestCase):
    def test_lint_and_show(self):
        code, out = cli("lint", str(EXAMPLE), "--oracles", "ci, llm, human")
        self.assertEqual(code, 0)
        self.assertIn("ok (6 requirements)", out)
        code, out = cli("show", str(EXAMPLE))
        self.assertTrue(out.startswith("Throttle brute-force logins (#342)\n"))
        self.assertIn("protected: any(all(locks_account, can_unlock), backs_off)", out)
        self.assertEqual(cli("digest", str(EXAMPLE))[1].strip(), example().digest)

    def test_lint_fails_on_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "bad.toml")
            path.write_text(
                '[q.a]\nask = "A?"\nby = "robot"\n[[require]]\nthat = "a"\nfail = "x"\n'
            )
            code, out = cli("lint", str(path))
            self.assertEqual(code, 1)
            self.assertIn("unknown-oracle", out)
            with self.assertRaisesRegex(SystemExit, "lint errors"):
                cli("run", str(path), "--agent", "true", "--cwd", tmp)

    def test_strict_lint_fails_on_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "unused.toml")
            path.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[q.b]\nask = "B?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            code, out = cli("lint", str(path))
            self.assertEqual(code, 0)
            self.assertIn("unused-question", out)
            self.assertEqual(cli("lint", "--strict", str(path))[0], 1)

    def test_explain_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            answers = Path(tmp, "answers.json")
            truth: dict[str, object] = {
                "repro_fixed": True,
                "new_test": True,
                "suite_green": True,
                "frozen_untouched": True,
                "locks_account": False,
                "can_unlock": False,
                "backs_off": True,
                "sso_untouched": True,
            }
            answers.write_text(json.dumps(truth))
            self.assertEqual(cli("explain", str(EXAMPLE), str(answers))[0], 0)
            truth["backs_off"] = {"is_met": False, "reasoning": "no delay"}
            answers.write_text(json.dumps(truth))
            code, out = cli("explain", str(EXAMPLE), str(answers))
            self.assertEqual(code, 1)
            self.assertIn("Why: no delay", out)
            del truth["suite_green"]
            answers.write_text(json.dumps(dict(truth, backs_off=True)))
            code, out = cli("explain", str(EXAMPLE), str(answers))
            self.assertEqual(code, 3)
            self.assertIn("To decide it, answer: suite_green", out)
            for data, refusal in (
                ({"backs_off": {"is_met": "no"}}, "backs_off: is_met must be a bool"),
                (["backs_off"], "answers must be a JSON object"),
                ({"backs_off": "yes"}, "backs_off: expected true, false or an answer"),
                ({"backs_of": True}, "answers for unknown questions: backs_of"),
            ):
                with self.subTest(refusal):
                    answers.write_text(json.dumps(data))
                    with self.assertRaisesRegex(SystemExit, refusal):
                        cli("explain", str(EXAMPLE), str(answers))

    def test_run_refuses_setups_that_cannot_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The example's ci questions need GitHub check runs.
            bare = Path(tmp, "bare.toml")
            bare.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\n[[require]]\nthat = "a"\nfail = "x"\n'
            )
            with self.assertRaisesRegex(SystemExit, "a need check, run or frozen"):
                cli("run", str(bare), "--agent", "true", "--cwd", tmp)
            for only in ((), ("--push", "agent"), ("--github", "octo/repo")):
                with (
                    self.subTest(only),
                    self.assertRaisesRegex(SystemExit, "--github and --push"),
                ):
                    cli("run", str(EXAMPLE), "--agent", "true", "--cwd", tmp, *only)
            with self.assertRaisesRegex(SystemExit, "--pr needs --github"):
                cli("run", str(EXAMPLE), "--agent", "true", "--cwd", tmp, "--pr", "5")
            local = Path(tmp, "local.toml")
            local.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            with self.assertRaisesRegex(SystemExit, "no folder to save"):
                cli(
                    "run",
                    str(local),
                    "--agent",
                    "true",
                    "--record",
                    str(Path(tmp, "runs", "run.json")),
                )
            kept = Path(tmp, "kept.json")
            kept.write_text("{}")
            with self.assertRaisesRegex(SystemExit, "pass --resume"):
                cli("run", str(local), "--agent", "true", "--record", str(kept))
            with self.assertRaisesRegex(SystemExit, "is a folder"):
                cli("run", str(local), "--agent", "true", "--record", tmp)
            # Inside the repository, though --cwd names only part of it.
            work = Path(tmp, "work")
            work.mkdir()
            repo(str(work))
            Path(work, "src").mkdir()
            inside = str(Path(work, "run.json"))
            with self.assertRaisesRegex(SystemExit, "outside the agent's checkout"):
                cli(
                    "run",
                    str(local),
                    *("--agent", "true", "--cwd", str(Path(work, "src"))),
                    *("--record", inside),
                )
            # A link there to a file outside, which saving would replace.
            link = Path(work, "linked.json")
            link.symlink_to(Path(tmp, "elsewhere.json"))
            with self.assertRaisesRegex(SystemExit, "outside the agent's checkout"):
                cli(
                    "run",
                    str(local),
                    *("--agent", "true", "--cwd", str(work), "--record", str(link)),
                )
            link.unlink()
            # What would fail only once the agent has worked fails first.
            missing = "no-such-program-xyz"
            linted = Path(tmp, "linted.toml")
            linted.write_text(local.read_text().replace("true", f"{missing} --check"))
            gone = f"/{missing}/bin/check"
            placed = Path(tmp, "placed.toml")
            placed.write_text(local.read_text().replace('"true"', f'"{gone}"'))
            for setup, refusal in (
                ((str(local), "--agent", missing), f"--agent: {missing} is not"),
                ((str(linted), "--agent", "true"), f"q.a.run: {missing} is not"),
                ((str(placed), "--agent", "true"), f"q.a.run: {gone} is not"),
                ((str(local), "--agent", "true", "--github", "octo"), "owner/name"),
                ((str(local), "--agent", "true", "--push", "a"), "remote 'origin'"),
                ((str(local), "--agent", "true", "--push", "a..b"), "branch name"),
            ):
                with self.subTest(refusal), self.assertRaisesRegex(SystemExit, refusal):
                    cli("run", *setup, "--cwd", str(work))
            nowhere = str(Path(tmp, "nowhere"))
            with self.assertRaisesRegex(SystemExit, "--cwd .*nowhere is not a folder"):
                cli("run", str(local), "--agent", "true", "--cwd", nowhere)
            for agent, message in (("", "names no command"), ("x 'y", "--agent: ")):
                with self.assertRaisesRegex(SystemExit, message):
                    cli("run", str(local), "--agent", agent, "--cwd", str(work))
            judged = Path(tmp, "judged.toml")
            judged.write_text(
                '[q.a]\nask = "A?"\nby = "llm"\n[[require]]\nthat = "a"\nfail = "x"\n'
            )
            # Credentials are checked before the agent runs, in a process
            # whose environment has only the ones each case gives it.
            secrets = {
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "GITHUB_TOKEN",
                "GH_TOKEN",
            }
            bare = {k: v for k, v in os.environ.items() if k not in secrets}
            bare["PYTHONPATH"] = str(Path(sa.__file__).parents[1])

            def run(checklist, *more, **given):
                return subprocess.run(
                    [sys.executable, "-m", "symbolic_acceptance", "run", checklist]
                    + ["--agent", "true", "--cwd", str(work), *more],
                    capture_output=True,
                    text=True,
                    env=bare | given,
                    check=False,
                )

            done = run(str(judged))
            self.assertIn("the judge needs ANTHROPIC_API_KEY", done.stderr)
            checked = Path(tmp, "checked.toml")
            checked.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\ncheck = "tests"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            for more in (
                ("--github", "octo/repo", "--pr", "5"),
                ("--github", "octo/repo", "--push", "agent"),
            ):
                done = run(str(checked if "--push" in more else local), *more)
                self.assertIn("--github needs a token in GITHUB_TOKEN", done.stderr)
            # No name to commit in: the agent's leftovers could not be kept.
            nameless = Path(tmp, "nameless")
            nameless.mkdir()
            repo(str(nameless))
            for key in ("user.name", "user.email"):
                git(nameless, "config", "--unset", key)
            git(nameless, "config", "user.useConfigOnly", "true")
            isolated = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
            done = run(str(local), "--cwd", str(nameless), **isolated)
            self.assertIn("identity unknown", done.stderr)
            self.assertEqual(git(work, "log", "--format=%s").split(), ["base"])
            # Either credential will do; with one, the agent runs.
            done = run(str(judged), ANTHROPIC_AUTH_TOKEN="t")
            self.assertIn("attempt 1 changed nothing", done.stdout)

    def test_usage_errors_exit_2(self):
        run = ("run", str(EXAMPLE), "--agent", "x")
        for args in (
            (),
            run[:2],
            (*run, "--samples", "0"),
            (*run, "--max-attempts", "x"),
            (*run, "--agent-timeout", "0"),
        ):
            with (
                self.subTest(args),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                cli(*args)
            self.assertEqual(caught.exception.code, 2)

    def test_run_gives_the_agent_the_time_it_is_given(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            code, out = cli(
                "run",
                str(checklist),
                *("--agent", "sleep 30", "--cwd", tmp, "--agent-timeout", "0.5"),
            )
        self.assertEqual(code, 1)
        self.assertIn("escalated: agent: the agent ran past 0.5s", out)

    def test_run_asks_a_person_on_the_terminal_if_there_is_one(self):
        def run(stdin):
            with (
                tempfile.TemporaryDirectory() as tmp,
                tempfile.TemporaryDirectory() as outside,
            ):
                repo(tmp)
                checklist = Path(outside, "acceptance.toml")
                checklist.write_text(
                    '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                    '[[require]]\nthat = "ok"\nfail = "not ok"\n'
                )
                write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
                saved = Path(outside, "run.json")
                # A process of its own, so a terminal running the tests is not it.
                done = subprocess.run(
                    [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                    + ["--cwd", tmp, "--agent", shlex.join(write)]
                    + ["--record", str(saved)],
                    stdin=stdin,
                    capture_output=True,
                    text=True,
                    # The same package as this process, wherever that came from.
                    env=dict(os.environ, PYTHONPATH=str(Path(sa.__file__).parents[1])),
                    check=False,
                    timeout=60,
                )
                return done, json.loads(saved.read_text())

        def typed(keys):
            keyboard, terminal = os.openpty()
            try:
                os.write(keyboard, keys)
                return run(terminal)
            finally:
                os.close(keyboard)
                os.close(terminal)

        done, _ = run(subprocess.DEVNULL)
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertIn("no terminal to ask a person", done.stdout)
        self.assertRegex(done.stdout, "Run it again with --resume .*run.json to go on")
        # A reply opens with y, yes, n or no, as a word; anything else is
        # asked again.
        done, saved = typed(b"maybe\nnothing to add\nn/a\nYes: looks fine\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stderr.count("ok: Is it ok? [y/n, then why]"), 4)
        self.assertNotIn("Is it ok?", done.stdout, "the result alone")
        self.assertEqual(
            saved["evaluations"][-1]["answers"]["ok"],
            {"is_met": True, "reasoning": "looks fine", "by": "human:terminal"},
        )
        # Ending the input (Ctrl-D) leaves the run waiting, to resume later;
        # after a reply, it ends the reply.
        done, saved = typed(b"\x04")
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertEqual(saved["status"], "waiting")
        done, saved = typed(b"y\x04")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_run_stops_on_a_regression(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                "".join(
                    f'[q.{name}]\nask = "Is there {name}?"\nby = "ci"\nrun = '
                    + json.dumps([sys.executable, "-c", f"open({name!r})"])
                    + f'\n[[require]]\nthat = "{name}"\nfail = "no {name}"\n'
                    for name in ("a", "b")
                )
            )
            # The first attempt adds a; the second swaps it for b.
            swap = (
                "import os\n"
                "if os.path.exists('a'):\n    os.remove('a'); open('b', 'w')\n"
                "else:\n    open('a', 'w')"
            )
            agent = shlex.join([sys.executable, "-c", swap])
            code, out = cli("run", str(checklist), "--cwd", tmp, "--agent", agent)
        self.assertEqual(code, 1)
        self.assertIn("regression: a held in attempt 1 and fails in attempt 2", out)

    def test_run_saves_a_run_and_resumes_it(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            runs = json.dumps([sys.executable, "-c", "import app"])
            checklist.write_text(
                f'[q.runs]\nask = "Does app.py run?"\nby = "ci"\nrun = {runs}\n'
                '[[require]]\nthat = "runs"\nfail = "app.py fails"\n'
            )
            record_path = str(Path(outside, "run.json"))
            write = shlex.join(
                [sys.executable, "-c", "open('app.py', 'w').write(\"print('v2')\\n\")"]
            )
            args = ("run", str(checklist), "--cwd", tmp, "--agent")
            issue = Path(outside, "issue.md")
            issue.write_text("app.py should print v2\n")
            code, out = cli(
                *args, write, "--record", record_path, "--issue", str(issue)
            )
            self.assertEqual((code, out), (0, "approved: every requirement holds\n"))
            saved = record.load(record_path, sa.load(checklist))
            self.assertEqual(saved.status, sa.APPROVED)
            # Resuming a finished run runs no agent: `false` would fail. The
            # record may be named twice, any way that names the same file.
            same = os.path.join(outside, ".", "run.json")
            code, out = cli(*args, "false", "--resume", record_path, "--record", same)
            self.assertEqual((code, out), (0, "approved: every requirement holds\n"))
            other = Path(outside, "other.md")
            other.write_text("Another issue\n")
            with self.assertRaisesRegex(SystemExit, "--issue differs"):
                cli(*args, "false", "--resume", record_path, "--issue", str(other))

    def test_run_starts_from_a_checkout_whose_changes_are_all_committed(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
            tempfile.TemporaryDirectory() as library,
        ):
            git = repo(tmp)
            repo(library)("commit", "-q", "--allow-empty", "-m", "newer")
            allow = ("-c", "protocol.file.allow=always")
            git(*allow, "submodule", "add", "-q", library, "vendored")
            git("config", "-f", ".gitmodules", "submodule.vendored.ignore", "all")
            git("add", ".gitmodules")
            git("commit", "-qm", "vendor")
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            run = ("run", str(checklist), "--agent", "true", "--cwd", tmp)
            Path(tmp, "notes.txt").write_text("mine, not the agent's\n")
            with self.assertRaisesRegex(SystemExit, "changes no commit holds"):
                cli(*run)
            Path(tmp, "notes.txt").unlink()
            # A submodule moved to another commit, which the checkout ignores.
            git("-C", "vendored", "checkout", "-q", "HEAD~1")
            with self.assertRaisesRegex(SystemExit, "changes no commit holds"):
                cli(*run)
            git("-C", "vendored", "checkout", "-q", "-")
            # An edit inside it, which no commit here can hold, is not one.
            Path(tmp, "vendored", "scratch.txt").write_text("x\n")
            _, out = cli(*run)
            self.assertIn("attempt 1 changed nothing", out)

    def test_a_resumed_run_keeps_diffing_from_where_it_started(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            check = json.dumps([sys.executable, "-c", "import app; assert app.V == 3"])
            checklist.write_text(
                f'[q.v3]\nask = "Is V 3?"\nby = "ci"\nrun = {check}\n'
                '[[require]]\nthat = "v3"\nfail = "V is not 3"\n'
            )
            saved = str(Path(outside, "run.json"))

            def agent(value):
                script = f"open('app.py', 'w').write('V = {value}\\n')"
                return shlex.join([sys.executable, "-c", script])

            args = ("run", str(checklist), "--cwd", tmp, "--agent")
            code, _ = cli(*args, agent(2), "--record", saved, "--max-attempts", "1")
            self.assertEqual(code, 1, "escalated at the attempt cap")
            code, _ = cli(*args, agent(3), "--resume", saved, "--max-attempts", "2")
            self.assertEqual(code, 0)
            resumed = record.load(saved, sa.load(checklist))
            base = git(tmp, "rev-parse", "HEAD~2").strip()
        self.assertEqual(resumed.base, base)
        second = resumed.evaluations[1]
        assert second.attempt.diff is not None
        self.assertIn("-print('v1')\n+V = 3", second.attempt.diff)

    def test_report_renders_a_recorded_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "run.json")
            record.save(finished_run(example()), example(), path)
            code, out = cli("report", str(EXAMPLE), str(path))
        self.assertEqual(code, 0)
        self.assertIn("**approved**: `every requirement holds`", out)

    def test_a_failure_outside_the_program_is_an_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            checklist = Path(tmp, "acceptance.toml")
            checklist.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            work = Path(tmp, "work")
            work.mkdir()
            args = ("run", str(checklist), "--agent", "true", "--cwd", str(work))
            with self.assertRaisesRegex(
                SystemExit, "^error: git .*not a git repository"
            ):
                cli(*args)

    def test_files_it_cannot_read_are_errors_not_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing, broken, other, binary = (
                str(Path(tmp, name))
                for name in ("missing.json", "broken.json", "other.json", "issue.md")
            )
            Path(broken).write_text("[1")
            Path(other).write_text('{"format": "something else"}')
            Path(binary).write_bytes(b"\xff")
            checklist = str(EXAMPLE)
            for args, message in (
                (("report", checklist, missing), "No such file"),
                (("report", checklist, broken), "broken.json: Expecting"),
                (("report", checklist, other), "not a symbolic-acceptance/run v1"),
                (("explain", checklist, broken), "broken.json: Expecting"),
                (("run", checklist, "--agent", "true", "--issue", binary), "not UTF-8"),
            ):
                with (
                    self.subTest(args),
                    self.assertRaisesRegex(SystemExit, f"^error: .*{message}"),
                ):
                    cli(*args)

    def test_draft_will_not_overwrite_before_asking_claude(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No issue file either: were the check late, reading it would fail
            # before any call to Claude.
            issue, out = Path(tmp, "issue.md"), Path(tmp, "acceptance.toml")
            out.write_text("approved already")
            with self.assertRaisesRegex(SystemExit, "exists; pass --force"):
                cli("draft", str(issue), "--out", str(out))
            self.assertEqual(out.read_text(), "approved already")
            nowhere = str(Path(tmp, "nowhere", "acceptance.toml"))
            with self.assertRaisesRegex(SystemExit, "no folder to write"):
                cli("draft", str(issue), "--out", nowhere)
            with self.assertRaisesRegex(SystemExit, "is a folder"):
                cli("draft", str(issue), "--out", tmp, "--force")

    def test_draft_asks_claude_over_http_and_writes_the_checklist(self):
        checklist = '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n[[require]]\nthat = "a"\nfail = "x"\n'
        reply = message({"checklist": checklist, "questions_for_reporter": ["Why?"]})
        # First a proxy's page in its place, not even UTF-8, then the API's reply.
        replies = ["<html>Accès refusé</html>".encode("latin-1"), reply.body]
        asked = []

        class MessagesAPI(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["content-length"])
                asked.append(
                    (self.path, self.headers["x-api-key"], self.rfile.read(length))
                )
                body = replies.pop(0)
                self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        # A server on this machine stands in for the Messages API.
        with (
            http.server.ThreadingHTTPServer(("127.0.0.1", 0), MessagesAPI) as server,
            tempfile.TemporaryDirectory() as tmp,
        ):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            issue, out = Path(tmp, "issue.md"), Path(tmp, "acceptance.toml")
            issue.write_text("Logins are not throttled\n")

            def drafting():
                return subprocess.run(
                    [sys.executable, "-m", "symbolic_acceptance", "draft", str(issue)]
                    + ["--out", str(out)],
                    env=dict(
                        os.environ,
                        ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
                        ANTHROPIC_API_KEY="key",
                        NO_PROXY="127.0.0.1",
                        no_proxy="127.0.0.1",
                        PYTHONPATH=str(Path(sa.__file__).parents[1]),
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )

            proxied, done = drafting(), drafting()
            server.shutdown()
            written = out.read_text()
        self.assertEqual(proxied.returncode, 1)
        self.assertEqual(
            proxied.stderr,
            "error: a reply was not JSON: '<html>Acc\ufffds refus\ufffd</html>'\n",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(written, checklist)
        self.assertIn("Questions for the reporter:\n  - Why?", done.stdout)
        for path, key, body in asked:
            self.assertEqual((path, key), ("/v1/messages", "key"))
            self.assertIn("Logins are not throttled", body.decode())

    def run_on_github(self, checklist, tmp, url, *more):
        """``run`` in a process of its own, with GitHub at ``url``."""
        write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        return subprocess.run(
            [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
            + ["--agent", shlex.join(write), "--cwd", tmp, "--github", "owner/repo"]
            + list(more),
            env=env
            | {
                "GITHUB_API_URL": url,
                "GH_TOKEN": "token",
                "NO_PROXY": "127.0.0.1",
                "no_proxy": "127.0.0.1",
                "PYTHONPATH": str(Path(sa.__file__).parents[1]),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_run_asks_on_a_pull_request_and_resumes_from_the_reply(self):
        with (
            github_api() as (url, comments, requests),
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            )
            saved = str(Path(outside, "run.json"))
            first = self.run_on_github(
                checklist, tmp, url, "--pr", "5", "--record", saved
            )
            comments.append(
                {
                    "id": 3,
                    "body": "/yes ok looks right",
                    "user": {"login": "alice", "type": "User"},
                    "author_association": "MEMBER",
                    "created_at": "2026-10-19T10:30:00Z",
                    "updated_at": "2026-10-19T10:30:00Z",
                    "html_url": "https://example.test/3",
                }
            )
            second = self.run_on_github(
                checklist, tmp, url, "--pr", "5", "--resume", saved
            )
        self.assertEqual(first.returncode, 3, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        # One report, kept up to date; the question, asked once; the reply.
        report, asked, _ = comments
        self.assertIn("**approved**", report["body"])
        self.assertIn("`ok`: Is it ok?", asked["body"])
        # The token, from GH_TOKEN, went with every request.
        self.assertEqual({token for _, token in requests}, {"Bearer token"})

    def test_run_without_a_pull_request_neither_asks_nor_reports_on_github(self):
        with (
            github_api() as (url, _, requests),
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            )
            done = self.run_on_github(checklist, tmp, url)
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertIn("no terminal to ask a person", done.stdout)
        self.assertEqual(requests, [])

    def test_an_error_stops_the_checks_still_running(self):
        # The judge cannot reach Claude while a slow check runs beside it.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.slow]\nask = "Slow?"\nby = "ci"\nrun = ["sleep", "60"]\n'
                '[q.judged]\nask = "Judged?"\nby = "llm"\n'
                '[[require]]\nthat = { all = ["slow", "judged"] }\nid = "r"\n'
                'fail = "x"\n'
            )
            with socket.socket() as closed:
                closed.bind(("127.0.0.1", 0))
                port = closed.getsockname()[1]
            write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
            started = time.monotonic()
            done = subprocess.run(
                [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                + ["--agent", shlex.join(write), "--cwd", tmp],
                env=NO_GIT_CONFIG
                | {
                    "PYTHONPATH": str(Path(sa.__file__).parents[1]),
                    "ANTHROPIC_API_KEY": "key",
                    "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
                    "NO_PROXY": "127.0.0.1",
                    "no_proxy": "127.0.0.1",
                },
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=90,
            )
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("connection failed", done.stderr)
        self.assertLess(time.monotonic() - started, 30, "the check was stopped")

    def test_ctrl_c_sigterm_or_a_hang_up_stops_the_agent_and_keeps_the_run(self):
        # The agent is in its own git commit, held up by a hook that takes a
        # second to clean up once asked to stop, when the runner is told to
        # stop; with nobody reading its errors once, as a closed terminal.
        for stop, reading in (
            (signal.SIGINT, True),
            (signal.SIGTERM, True),
            (signal.SIGINT, False),
            (signal.SIGHUP, False),
        ):
            with (
                self.subTest(stop.name, reading=reading),
                tempfile.TemporaryDirectory() as tmp,
                tempfile.TemporaryDirectory() as outside,
            ):
                repo(tmp)
                ready, cleaned = Path(outside, "ready"), Path(outside, "cleaned")
                hook = Path(tmp, ".git", "hooks", "pre-commit")
                hook.write_text(
                    "#!/bin/sh\n"
                    f"trap 'sleep 1; touch {shlex.quote(str(cleaned))}; exit 1' TERM\n"
                    f"touch {shlex.quote(str(ready))}\n"
                    "sleep 60 & wait\n"
                )
                hook.chmod(0o755)
                checklist = Path(outside, "acceptance.toml")
                checklist.write_text(
                    '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                    '[[require]]\nthat = "ok"\nfail = "not ok"\n'
                )
                pid = Path(outside, "pid")
                # Told to stop, it lets git end and ends cleanly, its work
                # half done.
                commits = (
                    "import os, signal, subprocess;"
                    " signal.signal(signal.SIGTERM, lambda *_: None);"
                    " open('app.py', 'w').write('v2');"
                    f" open({str(pid)!r}, 'w').write(str(os.getpid()));"
                    " subprocess.run(['git', 'commit', '-qam', 'v2'])"
                )
                head = git(tmp, "rev-parse", "HEAD")
                saved = Path(outside, "run.json")
                # Its output buffered, as it is unless one asks otherwise.
                env = {
                    k: v for k, v in NO_GIT_CONFIG.items() if k != "PYTHONUNBUFFERED"
                }
                runner = subprocess.Popen(
                    [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                    + ["--agent", shlex.join([sys.executable, "-c", commits])]
                    + ["--cwd", tmp, "--record", str(saved)],
                    env=env | {"PYTHONPATH": str(Path(sa.__file__).parents[1])},
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                lock = Path(tmp, ".git", "index.lock")
                deadline = time.monotonic() + 30
                while not ready.exists():
                    self.assertLess(time.monotonic(), deadline, "no commit began")
                    time.sleep(0.05)
                self.assertTrue(lock.exists(), "git holds its lock")
                assert runner.stderr is not None
                if not reading:
                    runner.stderr.close()
                runner.send_signal(stop)
                runner.wait(timeout=20)
                if reading:
                    self.assertNotIn("Traceback", runner.stderr.read())
                    runner.stderr.close()
                self.assertEqual(runner.returncode, 128 + stop)
                with self.assertRaises(ProcessLookupError, msg="the agent was stopped"):
                    os.kill(int(pid.read_text()), 0)
                self.assertFalse(lock.exists(), "git was let remove its lock")
                self.assertTrue(cleaned.exists(), "the hook was let clean up")
                self.assertEqual(
                    git(tmp, "rev-parse", "HEAD"), head, "nothing was kept"
                )
                self.assertEqual(json.loads(saved.read_text())["status"], "running")

    def test_ctrl_c_stops_a_question_waiting_at_the_terminal(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            )
            write = [sys.executable, "-c", "open('app.py', 'w').write('v2\\n')"]
            saved = Path(outside, "run.json")
            keyboard, terminal = os.openpty()
            # Its output buffered, so that a prompt it does not flush never shows.
            env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
            with subprocess.Popen(
                [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                + ["--cwd", tmp, "--agent", shlex.join(write), "--record", str(saved)],
                env=env | {"PYTHONPATH": str(Path(sa.__file__).parents[1])},
                stdin=terminal,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as runner:
                assert runner.stderr is not None
                try:
                    asked = b""
                    deadline = time.monotonic() + 30
                    while b"[y/n, then why]" not in asked:
                        left = max(0.0, deadline - time.monotonic())
                        readable = select.select([runner.stderr], [], [], left)[0]
                        read = (
                            os.read(runner.stderr.fileno(), 1024) if readable else b""
                        )
                        self.assertTrue(read, "the run never asked")
                        asked += read
                    runner.send_signal(signal.SIGINT)
                    _, err = runner.communicate(timeout=20)
                finally:
                    runner.kill()  # a run that never asked, or never ended
            os.close(keyboard)
            os.close(terminal)
            self.assertEqual(runner.returncode, 128 + signal.SIGINT, err.decode())
            self.assertEqual(json.loads(saved.read_text())["status"], "running")

    def test_ctrl_c_ends_a_draft_quietly(self):
        asked = threading.Event()

        class Silent(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                asked.set()
                time.sleep(30)

        with (
            http.server.ThreadingHTTPServer(("127.0.0.1", 0), Silent) as server,
            tempfile.TemporaryDirectory() as tmp,
        ):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            issue = Path(tmp, "issue.md")
            issue.write_text("Logins are not throttled\n")
            with subprocess.Popen(
                [sys.executable, "-m", "symbolic_acceptance", "draft", str(issue)]
                + ["--out", str(Path(tmp, "acceptance.toml"))],
                env=dict(
                    os.environ,
                    ANTHROPIC_BASE_URL=f"http://127.0.0.1:{server.server_port}",
                    ANTHROPIC_API_KEY="key",
                    NO_PROXY="127.0.0.1",
                    no_proxy="127.0.0.1",
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            ) as drafting:
                self.assertTrue(asked.wait(30), "Claude was never asked")
                drafting.send_signal(signal.SIGINT)
                _, err = drafting.communicate(timeout=20)
            server.shutdown()
        self.assertEqual((drafting.returncode, err), (128 + signal.SIGINT, ""))

    def test_a_second_ctrl_c_quits_at_once(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            )
            # An agent that will not stop when asked to.
            group = Path(outside, "group")
            agent = f"trap '' TERM; echo $$ > {shlex.quote(str(group))}; sleep 60"
            with subprocess.Popen(
                [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                + ["--agent", shlex.join(["sh", "-c", agent]), "--cwd", tmp],
                env=dict(os.environ, PYTHONPATH=str(Path(sa.__file__).parents[1])),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            ) as runner:
                deadline = time.monotonic() + 30
                while not group.exists() or not group.read_text():
                    self.assertLess(
                        time.monotonic(), deadline, "the agent never started"
                    )
                    time.sleep(0.05)
                assert runner.stderr is not None
                runner.send_signal(signal.SIGINT)
                self.assertIn("send it again", runner.stderr.readline())
                runner.send_signal(signal.SIGINT)
                # Long before the agent's grace would be up.
                runner.wait(timeout=GRACE - 1)
            self.assertEqual(runner.returncode, 128 + signal.SIGINT)
            while True:  # its programs killed, once the system has reaped them
                try:
                    os.killpg(int(group.read_text()), 0)
                except ProcessLookupError:
                    break
                self.assertLess(
                    time.monotonic(), deadline, "the agent is still running"
                )
                time.sleep(0.05)

    def test_a_run_goes_on_through_a_ctrl_c_it_was_told_to_ignore(self):
        # As a shell runs a job in the background: SIGINT ignored from the start.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.ok]\nask = "Is it ok?"\nby = "human"\n'
                '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            )
            started = Path(outside, "started")
            agent = (
                f"import time; open({str(started)!r}, 'w').close(); time.sleep(1);"
                " open('app.py', 'w').write('v2')"
            )
            runner = subprocess.Popen(
                ["sh", "-c", 'trap "" INT; exec "$@"', "sh"]
                + [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                + ["--agent", shlex.join([sys.executable, "-c", agent]), "--cwd", tmp],
                env=dict(os.environ, PYTHONPATH=str(Path(sa.__file__).parents[1])),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 30
            while not started.exists():
                self.assertLess(time.monotonic(), deadline, "the agent never started")
                time.sleep(0.05)
            runner.send_signal(signal.SIGINT)
            out, err = runner.communicate(timeout=30)
        self.assertEqual(runner.returncode, 3, err)
        self.assertIn("no terminal to ask a person", out)

    def test_a_run_from_another_thread_than_the_main_one(self):
        # Only the main thread can handle signals; the run goes on without.
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            results = []
            thread = threading.Thread(
                target=lambda: results.append(
                    cli("run", str(checklist), "--agent", "true", "--cwd", tmp)
                )
            )
            thread.start()
            thread.join(timeout=60)
        ((code, out),) = results
        self.assertEqual(code, 1)
        self.assertIn("attempt 1 changed nothing", out)

    def test_stopping_waits_for_the_programs_and_ends_any_started_after(self):
        # A process of its own, as stopping is for good.
        script = (
            "import os, sys, threading, time\n"
            "from symbolic_acceptance.agents import run_program, stop_programs\n"
            "started = sys.argv[1]\n"
            "program = [sys.executable, '-c',"
            " f'import time; open({started!r}, \"w\"); time.sleep(60)']\n"
            "def run():\n"
            "    return run_program(program, cwd='.', env=os.environ, timeout=60)[0]\n"
            "statuses = []\n"
            "thread = threading.Thread(target=lambda: statuses.append(run()))\n"
            "thread.start()\n"
            "while not os.path.exists(started):\n"
            "    time.sleep(0.05)\n"
            "start = time.monotonic()\n"
            "stop_programs(grace=60)\n"
            "thread.join()\n"
            "print(*statuses, run(), time.monotonic() - start < 30)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run(
                [sys.executable, "-c", script, str(Path(tmp, "started"))],
                env=dict(os.environ, PYTHONPATH=str(Path(sa.__file__).parents[1])),
                capture_output=True,
                text=True,
                check=True,
            )
        # Asked to stop, it ended, and the stop with it, long before the
        # grace was up; the program started after it was killed at once.
        self.assertEqual(done.stdout, f"{-signal.SIGTERM} {-signal.SIGKILL} True\n")

    def test_templates_load(self):
        for kind in ("bug", "feature"):
            code, out = cli("template", kind)
            self.assertEqual(code, 0)
            self.assertTrue(sa.loads(out).requirements)


if __name__ == "__main__":
    unittest.main()
