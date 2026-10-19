import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import DIFF, repo

import symbolic_acceptance as sa
from symbolic_acceptance.checks import Checks, Commands, frozen_files
from symbolic_acceptance.diffs import locate, parse


class ParseTests(unittest.TestCase):
    def test_files(self):
        files, _ = parse(DIFF)
        self.assertEqual(
            [(f.path, f.status, f.old_path) for f in files],
            [
                ("src/auth.py", "modified", None),
                ("tests/test_auth.py", "added", None),
                ("new.py", "renamed", "old.py"),
                ("tests/test_old.py", "deleted", None),
            ],
        )

    def test_lines_and_header_lookalike(self):
        _, lines = parse(DIFF)
        added = [(d.path, d.line, d.text) for d in lines if d.side == "+"]
        self.assertIn(("src/auth.py", 13, "            lock(user)"), added)
        self.assertIn(("src/auth.py", 14, "++ not a header, an added line"), added)
        removed = [d for d in lines if d.side == "-"]
        self.assertEqual(removed[0].where, "src/auth.py:-11")

    def test_bad_hunk_counts_do_not_swallow_the_next_file(self):
        broken = DIFF.replace("@@ -10,3 +10,6 @@", "@@ -10,9 +10,9 @@")
        files, _ = parse(broken)
        self.assertEqual(
            [f.path for f in files][:2], ["src/auth.py", "tests/test_auth.py"]
        )

    def test_quoted_paths_from_real_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            Path(tmp, "docs").mkdir()
            Path(tmp, "docs", "naïve café.md").write_text("hi\n")
            Path(tmp, "app.py").write_text("print('v2')\n")
            git("add", ".")
            git("commit", "-qm", "change")
            diff = git("diff", "HEAD~1", "HEAD")
        self.assertIn("\\303", diff, "git quoted the path with octal escapes")
        files, _ = parse(diff)
        self.assertEqual(
            [(f.path, f.status) for f in files],
            [("app.py", "modified"), ("docs/naïve café.md", "added")],
        )
        q = sa.Question("frozen", "Frozen?", "ci", options={"paths": ["docs/*"]})
        self.assertFalse(frozen_files(q, context(diff)).is_met)

    def test_locate(self):
        _, lines = parse(DIFF)
        found = locate("if failures(user) > MAX_FAILURES:\n    lock(user)", lines)
        self.assertEqual([d.where for d in found], ["src/auth.py:12", "src/auth.py:13"])
        self.assertIsNotNone(locate("+        record_failure(user)", lines))
        self.assertIsNone(locate("rate_limit(user)", lines))
        self.assertIsNone(locate("lock", lines), "too short to prove anything")


def context(diff="", ref="abc"):
    checklist = sa.loads(
        '[q.x]\nask = "X?"\nby = "ci"\n[[require]]\nthat = "x"\nfail = "x"\n'
    )
    return sa.Context(checklist, sa.Attempt(ref, 1, diff=diff), {})


class FrozenTests(unittest.TestCase):
    def question(self, **options):
        return sa.Question("frozen_untouched", "Untouched?", "ci", options=options)

    def test_new_regression_test_is_allowed(self):
        q = self.question(
            check="frozen", paths=["tests/*.py"], allow_new=["tests/test_*.py"]
        )
        diff = DIFF.split("diff --git a/old.py")[0]
        self.assertTrue(frozen_files(q, context(diff)).is_met)

    def test_deleting_or_renaming_frozen_files_fails(self):
        q = self.question(paths=["tests/*", "old.py"], allow_new=["tests/test_*.py"])
        answer = frozen_files(q, context(DIFF))
        self.assertFalse(answer.is_met)
        self.assertEqual(
            answer.evidence,
            ("old.py -> new.py (renamed)", "tests/test_old.py (deleted)"),
        )

    def test_needs_paths(self):
        with self.assertRaises(sa.OracleError):
            frozen_files(self.question(), context(DIFF))
        with self.assertRaisesRegex(ValueError, "a string or a list of strings"):
            frozen_files(self.question(paths=3), context(DIFF))


def run_question(name, code, **options):
    return sa.Question(
        name, f"{name}?", "ci", options={"run": [sys.executable, "-c", code], **options}
    )


class CommandsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.git = repo(self.tmp.name)
        self.head = self.git("rev-parse", "HEAD").strip()
        self.commands = Commands(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, question):
        return self.commands(question, context(ref=self.head))

    def test_exit_status_decides(self):
        ok = self.check(run_question("ok", "print('fine')"))
        self.assertTrue(ok.is_met)
        self.assertEqual(ok.evidence, ("fine",))
        bad = self.check(run_question("bad", "raise SystemExit(3)"))
        self.assertFalse(bad.is_met)
        self.assertIn("exit 3", bad.reasoning)
        inverted = run_question("inv", "raise SystemExit(1)", expect="fail")
        self.assertTrue(self.check(inverted).is_met)

    def test_commands_see_only_the_commit(self):
        # Files outside the commit, such as ignored ones, are not there.
        Path(self.tmp.name, ".gitignore").write_text("planted.py\n")
        Path(self.tmp.name, "planted.py").write_text("x = 1\n")
        Path(self.tmp.name, "app.py").write_text("print('edited, not committed')\n")
        sees = run_question(
            "sees",
            "import os, sys; print(open('app.py').read().strip());"
            "sys.exit(os.path.exists('planted.py'))",
        )
        answer = self.check(sees)
        self.assertTrue(answer.is_met)
        self.assertEqual(answer.evidence, ("print('v1')",))
        listed = self.git("worktree", "list")
        self.assertEqual(len(listed.strip().splitlines()), 1, "the worktree is removed")

    def test_invalid_utf8_output_is_kept(self):
        noisy = run_question(
            "noisy", "import sys; sys.stdout.buffer.write(b'caf\\xe9\\n')"
        )
        self.assertEqual(self.check(noisy).evidence, ("caf\ufffd",))

    def test_a_command_that_cannot_start_is_for_a_person(self):
        missing = sa.Question("m", "M?", "ci", options={"run": "no-such-command-xyz"})
        with self.assertRaisesRegex(sa.OracleError, "could not start"):
            self.check(missing)

    def test_router_merges_local_answers_into_pending(self):
        async def github(questions, ctx):
            raise sa.Pending("CI running")

        router = Checks(repo=self.tmp.name, fallback=github)
        local = run_question("local", "pass")
        remote = sa.Question("remote", "R?", "ci", options={"check": "tests"})
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(router([local, remote], context(ref=self.head)))
        self.assertEqual(list(caught.exception.answers), ["local"])

    def test_router_without_fallback(self):
        with self.assertRaisesRegex(sa.OracleError, "no local check"):
            asyncio.run(
                Checks()(
                    [sa.Question("r", "R?", "ci", options={"check": "x"})], context()
                )
            )


if __name__ == "__main__":
    unittest.main()
