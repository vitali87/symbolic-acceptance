import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import EXAMPLE, GIT_ENV, FakeHTTP, example, message, repo

import symbolic_acceptance as sa
from symbolic_acceptance.agents import AgentError, CommandAgent, local_diff
from symbolic_acceptance.claude import Claude
from symbolic_acceptance.cli import main
from symbolic_acceptance.draft import DraftError, draft


class CommandAgentTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in GIT_ENV}
        os.environ.update(GIT_ENV)

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def task(self, number=1):
        return sa.Task(issue="make it v2", checklist=example(), number=number)

    def test_uncommitted_changes_become_the_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            script = "import sys; prompt = sys.stdin.read(); open('app.py', 'w').write(\"print('v2')\\n\"); print('done:', 'make it v2' in prompt)"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
            self.assertEqual(attempt.claims, "done: True")
            self.assertIn("+print('v2')", attempt.diff)
            self.assertEqual(len(attempt.ref), 40)
            self.assertNotEqual(attempt.base, attempt.ref)
            log = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                cwd=tmp,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(log.strip(), "Attempt 1")
            self.assertEqual(local_diff(tmp)(attempt), attempt.diff)

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
            Path(tmp, ".gitattributes").write_text("*.py -diff\n")
            script = "open('tests/test_app.py', 'w').write('assert False or True\\n')"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp)
            attempt = asyncio.run(agent(self.task()))
        self.assertIn(
            "diff --git a/tests/test_app.py b/tests/test_app.py", attempt.diff
        )
        self.assertIn("+assert False or True", attempt.diff)

    def test_attempts_are_pushed(self):
        with (
            tempfile.TemporaryDirectory() as remote,
            tempfile.TemporaryDirectory() as tmp,
        ):
            subprocess.run(["git", "init", "-q", "--bare", remote], check=True)
            git = repo(tmp)
            git("remote", "add", "origin", remote)
            script = "open('app.py', 'w').write(\"print('v2')\\n\")"
            agent = CommandAgent([sys.executable, "-c", script], cwd=tmp, push="agent")
            attempt = asyncio.run(agent(self.task()))
            pushed = subprocess.run(
                ["git", "rev-parse", "refs/heads/agent"],
                cwd=remote,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        self.assertEqual(pushed, attempt.ref)

    def test_no_change_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent([sys.executable, "-c", "pass"], cwd=tmp)
            with self.assertRaisesRegex(AgentError, "changed nothing"):
                asyncio.run(agent(self.task()))

    def test_failing_agent_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            agent = CommandAgent(
                [sys.executable, "-c", "raise SystemExit('boom')"], cwd=tmp
            )
            with self.assertRaisesRegex(AgentError, "exited with 1: boom"):
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


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(args))
    return code, out.getvalue()


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
            truth = {
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

    def test_run_refuses_setups_that_cannot_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The example's ci questions need GitHub check runs.
            with self.assertRaisesRegex(SystemExit, "--github and --push"):
                cli("run", str(EXAMPLE), "--agent", "true", "--cwd", tmp)
            local = Path(tmp, "local.toml")
            local.write_text(
                '[q.a]\nask = "A?"\nby = "ci"\nrun = "true"\n'
                '[[require]]\nthat = "a"\nfail = "x"\n'
            )
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

    def test_templates_load(self):
        for kind in ("bug", "feature"):
            code, out = cli("template", kind)
            self.assertEqual(code, 0)
            self.assertTrue(sa.loads(out).requirements)


if __name__ == "__main__":
    unittest.main()
