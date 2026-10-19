import asyncio
import sys
import tempfile
import unittest

import symbolic_acceptance as sa
from symbolic_acceptance.checks import Checks, Commands, frozen_files
from symbolic_acceptance.diffs import locate, parse

DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index 1111111..2222222 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,6 @@ def login(user, password):
     if not check(user, password):
-        return False
+        record_failure(user)
+        if failures(user) > MAX_FAILURES:
+            lock(user)
+++ not a header, an added line
     return True
diff --git a/tests/test_auth.py b/tests/test_auth.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/tests/test_auth.py
@@ -0,0 +1,2 @@
+def test_lockout():
+    assert locks_after(5)
diff --git a/old.py b/new.py
similarity index 100%
rename from old.py
rename to new.py
diff --git a/tests/test_old.py b/tests/test_old.py
deleted file mode 100644
index 4444444..0000000
--- a/tests/test_old.py
+++ /dev/null
@@ -1 +0,0 @@
-def test_old(): pass
"""


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
        with self.assertRaisesRegex(sa.OracleError, "list of glob patterns"):
            frozen_files(self.question(paths=3), context(DIFF))


class CommandsTests(unittest.TestCase):
    def test_exit_status_decides(self):
        with tempfile.TemporaryDirectory() as tmp:
            commands = Commands(tmp)
            ok = sa.Question(
                "ok",
                "OK?",
                "ci",
                options={"run": [sys.executable, "-c", "print('fine')"]},
            )
            bad = sa.Question(
                "bad",
                "Bad?",
                "ci",
                options={"run": [sys.executable, "-c", "raise SystemExit(3)"]},
            )
            inverted = sa.Question(
                "inv",
                "Inv?",
                "ci",
                options={
                    "run": [sys.executable, "-c", "raise SystemExit(1)"],
                    "expect": "fail",
                },
            )
            self.assertTrue(commands(ok, context()).is_met)
            self.assertEqual(commands(ok, context()).evidence, ("fine",))
            answer = commands(bad, context())
            self.assertFalse(answer.is_met)
            self.assertIn("exit 3", answer.reasoning)
            self.assertTrue(commands(inverted, context()).is_met)

    def test_a_command_that_cannot_start_is_for_a_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = sa.Question(
                "m", "M?", "ci", options={"run": "no-such-command-xyz --flag"}
            )
            with self.assertRaisesRegex(sa.OracleError, "could not start"):
                Commands(tmp)(missing, context())

    def test_router_merges_local_answers_into_pending(self):
        async def github(questions, ctx):
            raise sa.Pending("CI running")

        with tempfile.TemporaryDirectory() as tmp:
            router = Checks(cwd=tmp, fallback=github)
            local = sa.Question(
                "local", "L?", "ci", options={"run": [sys.executable, "-c", "pass"]}
            )
            remote = sa.Question("remote", "R?", "ci", options={"check": "tests"})
            with self.assertRaises(sa.Pending) as caught:
                asyncio.run(router([local, remote], context()))
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
