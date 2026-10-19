import asyncio
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import EXAMPLE, FakeHTTP, cli, example, finished_run, message, repo

import symbolic_acceptance as sa
from symbolic_acceptance import record
from symbolic_acceptance.agents import CommandAgent, GitError, diff, git
from symbolic_acceptance.claude import Claude
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.draft import DraftError, draft


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
            attempt = asyncio.run(agent(self.task()))
            assert attempt.diff is not None
            self.assertEqual(attempt.claims, "done: True")
            self.assertIn("+print('v2')", attempt.diff)
            self.assertEqual(len(attempt.ref), 40)
            log = git(tmp, "log", "-1", "--format=%s")
            self.assertEqual(log.strip(), "Attempt 1")
            self.assertEqual(diff(tmp, agent.base, attempt.ref), attempt.diff)

    def test_the_whole_change_counts_from_a_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            git("config", "diff.relative", "true")
            Path(tmp, "sub").mkdir()
            script = (
                "open('../app.py', 'w').write('v2\\n'); open('x.py', 'w').write('')"
            )
            agent = CommandAgent([sys.executable, "-c", script], cwd=Path(tmp, "sub"))
            attempt = asyncio.run(agent(self.task()))
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
            agent = CommandAgent([sys.executable, "-c", move], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
        assert attempt.diff is not None
        self.assertIn(f"+Subproject commit {newer}", attempt.diff)

    def test_a_new_file_is_an_attempt_whatever_the_checkout_says(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            git("config", "status.showUntrackedFiles", "no")
            script = "open('new.py', 'w').write('x = 1\\n')"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
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
            attempt = asyncio.run(agent(self.task()))
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

    def test_attempts_are_pushed_even_when_the_agent_rewrites_them(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            git(remote, "init", "-q", "--bare")
            repo(tmp)("remote", "add", "origin", remote)
            script = "open('app.py', 'w').write(\"print('v2')\\n\")"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp, push="agent")
            first = asyncio.run(agent(self.task()))
            amend = (
                "import subprocess; open('app.py', 'w').write('v3\\n');"
                "subprocess.run(['git', 'commit', '-qa', '--amend', '-m', 'v3'])"
            )
            rewriter = CommandAgent(
                [sys.executable, "-c", amend], cwd=tmp, push="agent", base=agent.base
            )
            second = asyncio.run(rewriter(self.task(2, previous=first)))
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
                asyncio.run(agent(self.task()))
            self.assertEqual(git(remote, "branch", "--list").strip(), "")

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
            first = asyncio.run(CommandAgent(write, cwd=tmp, push="agent")(self.task()))
            # Someone else pushes to the branch, on top of the first attempt.
            local("commit", "-q", "--allow-empty", "-m", "theirs")
            local("push", "-q", "origin", "HEAD:refs/heads/agent")
            theirs = local("rev-parse", "HEAD").strip()
            local("reset", "-q", "--hard", first.ref)
            again = [sys.executable, "-c", "open('app.py', 'w').write('v3\\n')"]
            agent = CommandAgent(again, cwd=tmp, push="agent", base=base)
            with self.assertRaisesRegex(GitError, "push"):
                asyncio.run(agent(self.task(2, previous=first)))
            self.assertEqual(
                git(remote, "rev-parse", "refs/heads/agent").strip(), theirs
            )

    def test_a_later_attempt_starts_from_the_previous_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent([sys.executable, "-c", "pass"], cwd=tmp)
            elsewhere = sa.Attempt("0" * 40, 1)
            with self.assertRaisesRegex(sa.Escalate, "not at attempt 1"):
                asyncio.run(agent(self.task(2, previous=elsewhere)))

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
            attempt = asyncio.run(agent(self.task()))
        self.assertEqual(attempt.claims, "['ANTHROPIC_AUTH_TOKEN']")

    def test_an_agent_past_its_timeout_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            sleeper = [sys.executable, "-c", "import time; time.sleep(5)"]
            agent = CommandAgent(sleeper, cwd=tmp, timeout=0.5)
            with self.assertRaisesRegex(sa.Escalate, "ran past"):
                asyncio.run(agent(self.task()))

    def test_no_change_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent([sys.executable, "-c", "pass"], cwd=tmp)
            with self.assertRaisesRegex(sa.Escalate, "changed nothing"):
                asyncio.run(agent(self.task()))

    def test_failing_agent_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent(
                [sys.executable, "-c", "raise SystemExit('boom')"], cwd=tmp
            )
            with self.assertRaisesRegex(sa.Escalate, "exited with 1: boom"):
                asyncio.run(agent(self.task()))


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


class CLITests(unittest.TestCase):
    def test_lint_and_show(self):
        code, out = cli("lint", str(EXAMPLE))
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
            self.assertEqual(cli("explain", str(EXAMPLE), str(answers))[0], 3)
            answers.write_text(json.dumps({"backs_off": {"is_met": "no"}}))
            with self.assertRaisesRegex(SystemExit, "backs_off: is_met must be a bool"):
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
            record = str(Path(tmp, "run.json"))
            with self.assertRaisesRegex(SystemExit, "outside the agent's checkout"):
                cli(
                    "run",
                    str(local),
                    "--agent",
                    "true",
                    "--cwd",
                    tmp,
                    "--record",
                    record,
                )

    def test_usage_errors_exit_2(self):
        run = ("run", str(EXAMPLE), "--agent", "x")
        for args in (
            (),
            run[:2],
            (*run, "--samples", "0"),
            (*run, "--max-attempts", "x"),
        ):
            with (
                self.subTest(args),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                cli(*args)
            self.assertEqual(caught.exception.code, 2)

    def test_run_waits_for_a_person_it_has_no_terminal_to_ask(self):
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
            # A process of its own, so a terminal running the tests is not it.
            done = subprocess.run(
                [sys.executable, "-m", "symbolic_acceptance", "run", str(checklist)]
                + ["--cwd", tmp, "--agent", shlex.join(write)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                # The same package as this process, wherever that came from.
                env=dict(os.environ, PYTHONPATH=str(Path(sa.__file__).parents[1])),
                check=False,
            )
        self.assertEqual(done.returncode, 3, done.stderr)
        self.assertIn("no terminal to ask a person", done.stdout)

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
            code, out = cli(*args, write, "--record", record_path)
            self.assertEqual((code, out), (0, "approved: every requirement holds\n"))
            saved = record.load(record_path, sa.load(checklist))
            self.assertEqual(saved.status, sa.APPROVED)
            # Resuming a finished run runs no agent: `false` would fail. The
            # record may be named twice, any way that names the same file.
            same = os.path.join(outside, ".", "run.json")
            code, out = cli(*args, "false", "--resume", record_path, "--record", same)
            self.assertEqual((code, out), (0, "approved: every requirement holds\n"))

    def test_run_starts_from_a_checkout_whose_changes_are_all_committed(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            repo(tmp)
            Path(tmp, "notes.txt").write_text("mine, not the agent's\n")
            checklist = Path(outside, "acceptance.toml")
            checklist.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
            with self.assertRaisesRegex(SystemExit, "changes no commit holds"):
                cli("run", str(checklist), "--agent", "true", "--cwd", tmp)

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
                (("run", checklist, "--agent", "x", "--issue", binary), "not UTF-8"),
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

    def test_templates_load(self):
        for kind in ("bug", "feature"):
            code, out = cli("template", kind)
            self.assertEqual(code, 0)
            self.assertTrue(sa.loads(out).requirements)


if __name__ == "__main__":
    unittest.main()
