import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import EXAMPLE, FakeHTTP, cli, example, finished_run, message, repo

import symbolic_acceptance as sa
from symbolic_acceptance import record
from symbolic_acceptance.agents import CommandAgent, local_diff
from symbolic_acceptance.claude import Claude
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
            attempt = asyncio.run(agent(self.task()))
            assert attempt.diff is not None
            self.assertEqual(attempt.claims, "done: True")
            self.assertIn("+print('v2')", attempt.diff)
            self.assertEqual(len(attempt.ref), 40)
            self.assertNotEqual(attempt.base, attempt.ref)
            log = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                cwd=tmp,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(log.strip(), "Attempt 1")
            self.assertEqual(local_diff(tmp)(attempt), attempt.diff)

    def test_a_new_file_is_an_attempt_whatever_the_checkout_says(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            git("config", "status.showUntrackedFiles", "no")
            script = "open('new.py', 'w').write('x = 1\\n')"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
        assert attempt.diff is not None
        self.assertIn("+++ b/new.py", attempt.diff)

    def test_the_checkout_cannot_rig_the_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            Path(tmp, "tests").mkdir()
            Path(tmp, "tests", "test_app.py").write_text("assert True\n")
            git("add", ".")
            git("commit", "-qm", "tests")
            # What an agent could do to hide a change to a frozen test file.
            git("config", "diff.noprefix", "true")
            git("config", "diff.relative", "true")
            git("config", "diff.ignoreSubmodules", "all")
            git("config", "diff.context", "0")
            Path(tmp, ".gitattributes").write_text("*.py -diff\n")
            # ... and a gitlink, which ignoreSubmodules would hide.
            head = git("rev-parse", "HEAD").strip()
            Path(tmp, "tests", "vendored").mkdir()
            git("update-index", "--add", "--cacheinfo", f"160000,{head},tests/vendored")
            script = "open('tests/test_app.py', 'w').write('assert False or True\\n')"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
        assert attempt.diff is not None
        self.assertIn(
            "diff --git a/tests/test_app.py b/tests/test_app.py", attempt.diff
        )
        self.assertIn("+assert False or True", attempt.diff)
        self.assertIn("new file mode 160000", attempt.diff, "the gitlink shows")

    def test_attempts_are_pushed_even_when_the_agent_rewrites_them(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            subprocess.run(["git", "init", "-q", "--bare", remote], check=True)
            git = repo(tmp)
            git("remote", "add", "origin", remote)
            script = "open('app.py', 'w').write(\"print('v2')\\n\")"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp, push="agent")
            first = asyncio.run(agent(self.task()))
            amend = (
                "import subprocess; open('app.py', 'w').write('v3\\n');"
                "subprocess.run(['git', 'commit', '-qa', '--amend', '-m', 'v3'])"
            )
            rewriter = CommandAgent(
                [sys.executable, "-c", amend], cwd=tmp, push="agent", base=first.base
            )
            second = asyncio.run(rewriter(self.task(2, previous=first)))
            pushed = subprocess.run(
                ["git", "rev-parse", "refs/heads/agent"],
                cwd=remote,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        self.assertEqual(pushed, second.ref)

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

    def test_gives_up_after_tries(self):
        bad = message({"checklist": "nope = 1", "questions_for_reporter": []})
        fake = FakeHTTP(bad, bad)
        with self.assertRaises(DraftError):
            draft("x", claude=Claude(api_key="k", transport=fake), tries=2)


class CLITests(unittest.TestCase):
    def test_lint_and_show(self):
        code, out = cli("lint", str(EXAMPLE))
        self.assertEqual(code, 0)
        self.assertIn("ok (6 requirements)", out)
        code, out = cli("show", str(EXAMPLE))
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
                "touches_sso": False,
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
            self.assertEqual(cli("explain", str(EXAMPLE), str(answers))[0], 2)
            answers.write_text(json.dumps({"backs_off": {"is_met": "no"}}))
            with self.assertRaisesRegex(SystemExit, "backs_off: is_met must be a bool"):
                cli("explain", str(EXAMPLE), str(answers))

    def test_run_refuses_setups_that_cannot_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The example's ci questions need GitHub check runs.
            with self.assertRaisesRegex(SystemExit, "--github and --push"):
                cli("run", str(EXAMPLE), "--agent", "true", "--cwd", tmp)
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

    def test_report_renders_a_recorded_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "run.json")
            record.save(finished_run(example()), example(), path)
            code, out = cli("report", str(EXAMPLE), str(path))
        self.assertEqual(code, 0)
        self.assertIn("**approved**: every requirement holds", out)

    def test_draft_will_not_overwrite_before_asking_claude(self):
        with tempfile.TemporaryDirectory() as tmp:
            issue, out = Path(tmp, "issue.md"), Path(tmp, "acceptance.toml")
            issue.write_text("Logins are not throttled")
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
