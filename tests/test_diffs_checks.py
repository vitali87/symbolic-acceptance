import asyncio
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path

from helpers import DIFF, repo

import symbolic_acceptance as sa
from symbolic_acceptance.agents import GitError, diff
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
        self.assertEqual(
            [(d.where, d.kind, d.text) for d in lines],
            [
                ("src/auth.py:10", "context", "    if not check(user, password):"),
                ("src/auth.py:-11", "removed", "        return False"),
                ("src/auth.py:11", "added", "        record_failure(user)"),
                (
                    "src/auth.py:12",
                    "added",
                    "        if failures(user) > MAX_FAILURES:",
                ),
                ("src/auth.py:13", "added", "            lock(user)"),
                ("src/auth.py:14", "added", "++ not a header, an added line"),
                ("src/auth.py:15", "context", "    return True"),
                ("tests/test_auth.py:1", "added", "def test_lockout():"),
                ("tests/test_auth.py:2", "added", "    assert locks_after(5)"),
                ("tests/test_old.py:-1", "removed", "def test_old(): pass"),
            ],
        )

    def test_lines_carry_the_files_name_after_the_change(self):
        _, lines = parse(RENAMED)
        self.assertEqual(
            [d.where for d in lines], ["new.py:1", "new.py:-2", "new.py:2", "new.py:3"]
        )

    def test_no_newline_markers_and_escaped_paths(self):
        diff = (
            'diff --git "a/tab\\there.txt" "b/tab\\there.txt"\n'
            "index 1111111..2222222 100644\n"
            '--- "a/tab\\there.txt"\n'
            '+++ "b/tab\\there.txt"\n'
            "@@ -1 +1 @@\n"
            "-old\n"
            "\\ No newline at end of file\n"
            "+new\n"
            "\\ No newline at end of file\n"
            "diff --git a/next.py b/next.py\n"
            "new file mode 100644\n"
        )
        files, lines = parse(diff)
        self.assertEqual(
            [(f.path, f.status) for f in files],
            [("tab\there.txt", "modified"), ("next.py", "added")],
        )
        self.assertEqual(
            [(d.side, d.text) for d in lines], [("-", "old"), ("+", "new")]
        )

    def test_bad_hunk_counts_do_not_swallow_the_next_file(self):
        broken = DIFF.replace("@@ -10,3 +10,6 @@", "@@ -10,9 +10,9 @@")
        files, _ = parse(broken)
        self.assertEqual(
            [f.path for f in files][:2], ["src/auth.py", "tests/test_auth.py"]
        )

    def test_unusual_paths_from_real_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            Path(tmp, "docs").mkdir()
            Path(tmp, "docs", "naïve café.md").write_text("hello there\n")
            Path(tmp, "docs", 'say "hi"\there.md').write_text("hi\n")
            Path(tmp, "docs", "notes ").write_text("ends in a space\n")
            Path(tmp, "app.py").write_text("print('v2')\n")
            git("add", ".")
            git("commit", "-qm", "change")
            changed = diff(tmp, "HEAD~1", "HEAD")
        self.assertIn("+++ b/docs/naïve café.md", changed, "shown as it is")
        self.assertIn('"b/docs/say \\"hi\\"\\there.md"', changed, "quoted by git")
        files, lines = parse(changed)
        self.assertEqual(
            [(f.path, f.status) for f in files],
            [
                ("app.py", "modified"),
                ("docs/naïve café.md", "added"),
                ("docs/notes ", "added"),
                ('docs/say "hi"\there.md', "added"),
            ],
        )
        self.assertIsNotNone(locate("hello there", "b/docs/naïve café.md", lines))
        q = sa.Question("frozen", "Frozen?", "ci", options={"frozen": ["docs/*"]})
        self.assertFalse(frozen_files(q, context(changed)).is_met)

    def test_header_only_changes_from_real_git(self):
        # A mode change or an empty file has no ---/+++ lines, so its path
        # comes from the header alone, where git leaves spaces unquoted.
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            Path(tmp, "tests").mkdir()
            Path(tmp, "tests", "my test.py").write_text("x = 1\n")
            Path(tmp, "tests", "empty one.py").write_text("")
            git("add", ".")
            git("commit", "-qm", "tests")
            Path(tmp, "tests", "my test.py").chmod(0o755)
            git("rm", "-q", "tests/empty one.py")
            git("commit", "-qam", "change")
            changed = diff(tmp, "HEAD~1", "HEAD")
        q = sa.Question("frozen", "Frozen?", "ci", options={"frozen": ["tests/*"]})
        self.assertEqual(
            frozen_files(q, context(changed)).evidence,
            ("tests/empty one.py (deleted)", "tests/my test.py (modified)"),
        )

    def test_a_carriage_return_does_not_end_a_line(self):
        # In Go a // comment runs to the newline, so this is one commented line;
        # read as two, it would show code the change does not add.
        with tempfile.TemporaryDirectory() as tmp:
            git = repo(tmp)
            Path(tmp, "app.py").write_bytes(b"x = 1\n// TODO\r+lock(user)\ny = 2\n")
            git("commit", "-qam", "change")
            changed = diff(tmp, "HEAD~1", "HEAD")
        _, lines = parse(changed)
        self.assertEqual(
            [(d.side, d.text) for d in lines],
            [
                ("-", "print('v1')"),
                ("+", "x = 1"),
                ("+", "// TODO\r+lock(user)"),
                ("+", "y = 2"),
            ],
        )
        self.assertIsNone(locate("lock(user)", "app.py", lines))
        self.assertIsNotNone(
            locate("// TODO\r+lock(user)", "app.py", lines), "the line itself"
        )

    def test_locate(self):
        _, lines = parse(DIFF)

        def where(quote, path="src/auth.py"):
            found = locate(quote, path, lines)
            return found and [d.where for d in found]

        self.assertEqual(
            where("if failures(user) > MAX_FAILURES:\n    lock(user)"),
            ["src/auth.py:12", "src/auth.py:13"],
        )
        self.assertEqual(where("+        record_failure(user)"), ["src/auth.py:11"])
        self.assertIsNone(where("rate_limit(user)"))
        self.assertIsNone(where("+ rate_limit(user)"))
        self.assertIsNone(where("MAX_FAILURES"), "part of a line is not a line")
        self.assertIsNone(
            where("record_failure(user)\n    lock(user)"),
            "lines quoted together must be consecutive",
        )
        self.assertIsNone(where("def test_lockout():"), "the line is in another file")

    def test_locate_reads_lines_as_the_file_has_them(self):
        _, lines = parse(SPREAD)

        def where(quote, path="src/auth.py"):
            found = locate(quote, path, lines)
            return found and [d.where for d in found]

        self.assertEqual(
            where("def lock_account(user):\n\n    user.locked = True"),
            ["src/auth.py:1", "src/auth.py:3"],
            "blank lines don't count",
        )
        self.assertIsNone(where("x = 1\ny = 2"), "the lines are far apart")
        self.assertIsNone(where("+z = 3"), "too short to prove anything")
        self.assertEqual(where("lock(user)"), ["src/auth.py:83"])
        self.assertIsNone(
            where("user.locked = True", "docs/notes.txt"),
            "a copy in another file is not this file's line",
        )

    def test_a_line_that_starts_with_a_plus_is_found_whole(self):
        # "+retries" also reads as "retries" with the diff's "+", which is
        # too short to prove anything; the line itself is long enough.
        _, lines = parse(
            "diff --git a/fix.patch b/fix.patch\n"
            "--- a/fix.patch\n"
            "+++ b/fix.patch\n"
            "@@ -1 +1,2 @@\n"
            " retries\n"
            "++retries\n"
        )
        found = locate("+retries", "fix.patch", lines)
        assert found is not None
        self.assertEqual([d.where for d in found], ["fix.patch:2"])

    def test_locate_prefers_where_the_line_changes(self):
        _, lines = parse(RENAMED)
        found = locate("value = compute()", "new.py", lines)
        assert found is not None
        self.assertEqual([d.where for d in found], ["new.py:3"])


SPREAD = """\
diff --git a/src/auth.py b/src/auth.py
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,4 +1,6 @@
 def lock_account(user):
+
+    user.locked = True
     return user
 
 x = 1
@@ -80,2 +82,3 @@
 y = 2
+lock(user)
 z = 3
diff --git a/docs/notes.txt b/docs/notes.txt
new file mode 100644
--- /dev/null
+++ b/docs/notes.txt
@@ -0,0 +1 @@
+lock(user)
"""


RENAMED = """\
diff --git a/old.py b/new.py
similarity index 50%
rename from old.py
rename to new.py
index 1111111..2222222 100644
--- a/old.py
+++ b/new.py
@@ -1,2 +1,3 @@
 value = compute()
-print(value)
+log(value)
+value = compute()
"""


def context(diff: str | None = "", ref="abc"):
    checklist = sa.loads(
        '[q.x]\nask = "X?"\nby = "ci"\n[[require]]\nthat = "x"\nfail = "x"\n'
    )
    return sa.Context(checklist, sa.Attempt(ref, 1, diff=diff), {})


class FrozenTests(unittest.TestCase):
    def question(self, **options):
        return sa.Question("frozen_untouched", "Untouched?", "ci", options=options)

    def test_new_regression_test_is_allowed(self):
        q = self.question(frozen=["tests/*.py"], allow_new=["tests/test_*.py"])
        diff = DIFF.split("diff --git a/old.py")[0]
        self.assertTrue(frozen_files(q, context(diff)).is_met)

    def test_a_new_file_is_allowed_only_as_deep_as_its_pattern(self):
        q = self.question(frozen=["tests/*"], allow_new=["tests/test_*.py"])
        diff = (
            "diff --git a/tests/test_x/conftest.py b/tests/test_x/conftest.py\n"
            "new file mode 100644\n"
        )
        self.assertEqual(
            frozen_files(q, context(diff)).evidence,
            ("tests/test_x/conftest.py (added)",),
        )

    def test_frozen_paths_ignore_case(self):
        # On a case-insensitive file system TESTS/ is tests/.
        q = self.question(frozen=["tests/*"], allow_new=["tests/test_*.py"])
        diff = (
            "diff --git a/TESTS/conftest.py b/TESTS/conftest.py\nnew file mode 100644\n"
        )
        self.assertEqual(
            frozen_files(q, context(diff)).evidence, ("TESTS/conftest.py (added)",)
        )

    def test_deleting_or_renaming_frozen_files_fails(self):
        q = self.question(frozen=["tests/*", "old.py"], allow_new=["tests/test_*.py"])
        answer = frozen_files(q, context(DIFF))
        self.assertFalse(answer.is_met)
        self.assertEqual(
            answer.evidence,
            ("old.py -> new.py (renamed)", "tests/test_old.py (deleted)"),
        )

    def test_an_attempt_without_a_diff_is_an_error(self):
        with self.assertRaisesRegex(sa.OracleError, "has none"):
            frozen_files(self.question(frozen=["tests/*"]), context(diff=None))

    def test_needs_paths(self):
        with self.assertRaisesRegex(ValueError, "frozen lists no paths"):
            frozen_files(self.question(frozen=[]), context(DIFF))
        with self.assertRaisesRegex(ValueError, "a string or a list of strings"):
            frozen_files(self.question(frozen=3), context(DIFF))


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

    def test_a_command_runs_from_the_top_of_the_repository(self):
        Path(self.tmp.name, "src").mkdir()
        commands = Commands(Path(self.tmp.name, "src"))
        sees = run_question("sees", "print(open('app.py').read().strip())")
        self.assertEqual(
            commands(sees, context(ref=self.head)).evidence, ("print('v1')",)
        )

    def test_the_checkout_cannot_set_up_a_filter(self):
        # A filter defined in the agent's checkout would rewrite files as
        # they are checked out; the command must see the commit as it is.
        Path(self.tmp.name, ".git", "info", "attributes").write_text("*.py filter=f\n")
        self.git("config", "filter.f.smudge", "sed s/v1/v2/")
        sees = run_question("sees", "print(open('app.py').read().strip())")
        self.assertEqual(self.check(sees).evidence, ("print('v1')",))

    def test_the_users_git_configuration_cannot_reach_the_checkout(self):
        # The agent can write the user's configuration as well: a template
        # that sets up a filter in every new repository, a filter for every
        # repository, and a rewrite that sends the fetch elsewhere.
        with tempfile.TemporaryDirectory() as user:
            template = Path(user, "template")
            Path(template, "info").mkdir(parents=True)
            Path(template, "info", "attributes").write_text("*.py filter=f\n")
            smudge = '[filter "f"]\n\tsmudge = sed s/v1/v2/\n'
            Path(template, "config").write_text(smudge)
            Path(user, "attributes").write_text("*.py filter=f\n")
            Path(user, "gitconfig").write_text(
                f"[init]\n\ttemplateDir = {template}\n"
                f"[core]\n\tattributesFile = {Path(user, 'attributes')}\n"
                f'{smudge}[url "file:///nowhere/"]\n\tinsteadOf = file:///\n'
            )
            script = (
                "import sys, symbolic_acceptance as sa\n"
                "from symbolic_acceptance.checks import Commands\n"
                "checklist = sa.loads(sys.stdin.read())\n"
                "context = sa.Context(checklist, sa.Attempt(sys.argv[2]), {})\n"
                "answer = Commands(sys.argv[1])(checklist.questions['sees'], context)\n"
                "print(*answer.evidence)\n"
            )
            done = subprocess.run(
                [sys.executable, "-c", script, self.tmp.name, self.head],
                input='[q.sees]\nask = "What is in app.py?"\nby = "ci"\n'
                'run = "cat app.py"\n[[require]]\nthat = "sees"\nfail = "x"\n',
                env=dict(
                    os.environ,
                    GIT_CONFIG_GLOBAL=str(Path(user, "gitconfig")),
                    PYTHONPATH=str(Path(sa.__file__).parents[1]),
                ),
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(done.stdout, "print('v1')\n")

    def test_commands_never_get_the_runners_github_token(self):
        env = {"PATH": os.environ["PATH"], "GITHUB_TOKEN": "t", "GH_TOKEN": "t"}
        commands = Commands(self.tmp.name, env={**env, "OTHER_TOKEN": "kept"})
        sees = run_question(
            "sees", "import os; print(sorted(k for k in os.environ if 'TOKEN' in k))"
        )
        answer = commands(sees, context(ref=self.head))
        self.assertEqual(answer.evidence, ("['OTHER_TOKEN']",))

    def test_a_command_past_its_timeout_fails(self):
        slow = run_question("slow", "import time; time.sleep(5)")
        answer = Commands(self.tmp.name, timeout=0.5)(slow, context(ref=self.head))
        self.assertFalse(answer.is_met)
        self.assertIn("timed out after 0.5s", answer.reasoning)

    def test_what_a_command_leaves_running_neither_holds_it_up_nor_outlives_it(self):
        marker = Path(self.tmp.name).parent / f"{Path(self.tmp.name).name}-late"
        late = f"import time; time.sleep(0.5); open({str(marker)!r}, 'w')"
        script = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {late!r}])"
        started = time.monotonic()
        answer = Commands(self.tmp.name, timeout=30)(
            run_question("leaves", script), context(ref=self.head)
        )
        self.assertTrue(answer.is_met)
        self.assertLess(time.monotonic() - started, 10)
        time.sleep(1)
        self.assertFalse(marker.exists(), "the child it left was stopped")

    def test_a_command_gets_no_input_and_its_output_in_order(self):
        script = (
            "import sys; print('one', flush=True); print('two', file=sys.stderr,"
            " flush=True); print(repr(sys.stdin.read()))"
        )
        self.assertEqual(
            self.check(run_question("io", script)).evidence, ("one", "two", "''")
        )

    def test_a_program_of_the_attempt_that_cannot_run_fails_the_check(self):
        for expect in ("pass", "fail"):
            with self.subTest(expect=expect):
                gone = sa.Question(
                    "g", "G?", "ci", options={"run": "./repro.sh", "expect": expect}
                )
                answer = self.check(gone)
                self.assertFalse(answer.is_met)
                self.assertIn("could not start", answer.reasoning)

    def test_bytes_rewritten_in_the_object_store_are_refused(self):
        blob = self.git("rev-parse", "HEAD:app.py").strip()
        loose = Path(self.tmp.name, ".git", "objects", blob[:2], blob[2:])
        loose.chmod(0o644)
        forged = b"print('forged')\n"
        loose.write_bytes(zlib.compress(b"blob %d\0" % len(forged) + forged))
        with self.assertRaisesRegex(GitError, "fetch"):
            self.check(run_question("sees", "print(open('app.py').read())"))

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

        router = Checks(self.commands, fallback=github)
        local = run_question("local", "pass")
        remote = sa.Question("remote", "R?", "ci", options={"check": "tests"})
        with self.assertRaises(sa.Pending) as caught:
            asyncio.run(router([local, remote], context(ref=self.head)))
        self.assertEqual(list(caught.exception.answers), ["local"])

    def test_router_combines_local_and_fallback_answers(self):
        async def github(questions, ctx):
            return {q.id: sa.Answer(True, by="ci:tests") for q in questions}

        router = Checks(self.commands, fallback=github)
        local = run_question("local", "raise SystemExit(1)")
        remote = sa.Question("remote", "R?", "ci", options={"check": "tests"})
        answers = asyncio.run(router([local, remote], context(ref=self.head)))
        self.assertEqual(
            {qid: (a.is_met, a.by) for qid, a in answers.items()},
            {"local": (False, "ci:run"), "remote": (True, "ci:tests")},
        )

    def test_router_without_fallback(self):
        with self.assertRaisesRegex(sa.OracleError, "no local check"):
            asyncio.run(
                Checks()(
                    [sa.Question("r", "R?", "ci", options={"check": "x"})], context()
                )
            )


if __name__ == "__main__":
    unittest.main()
