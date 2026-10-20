import tempfile
import tomllib
import unittest
from pathlib import Path

from helpers import EXAMPLE, example

import symbolic_acceptance as sa
from symbolic_acceptance.checklist import AllOf, AnyOf, Not, Ref, render

MINIMAL = """
[q.a]
ask = "Is A done?"
by = "ci"

[[require]]
that = "a"
fail = "A is not done"
"""


class LoadTests(unittest.TestCase):
    def test_example_loads(self):
        checklist = example()
        self.assertEqual(len(checklist.questions), 9)
        self.assertEqual(
            [r.id for r in checklist.requirements],
            [
                "repro_fixed",
                "new_test",
                "suite_green",
                "frozen_untouched",
                "protected",
                "sso",
            ],
        )
        protected = checklist.requirement("protected")
        self.assertEqual(
            protected.that,
            AnyOf((AllOf((Ref("locks_account"), Ref("can_unlock"))), Ref("backs_off"))),
        )
        self.assertEqual(checklist.requirement("sso").when, Not(Ref("sso_untouched")))

    def test_options_are_kept_and_frozen(self):
        question = example().questions["suite_green"]
        self.assertEqual(question.options["check"], "tests")
        paths = ["a/*"]
        options: dict[str, object] = {"paths": paths}
        frozen = sa.Question("f", "F?", "ci", options=options)
        paths.append("b/*")
        options["check"] = "frozen"
        self.assertEqual(frozen.strings("paths"), ("a/*",))
        self.assertNotIn("check", frozen.options)

    def test_requirement_id_defaults_to_its_question(self):
        self.assertEqual(sa.loads(MINIMAL).requirements[0].id, "a")

    def test_render(self):
        expr = AnyOf((AllOf((Ref("a"), Ref("b"))), Not(Ref("c"))))
        self.assertEqual(render(expr), "any(all(a, b), not(c))")

    def test_load_reads_files_and_rejects_bad_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.toml"
            path.write_text(MINIMAL, encoding="utf-8")
            self.assertEqual(sa.load(path).source, MINIMAL)
            path.write_bytes(b'title = "\xff"\n' + MINIMAL.encode())
            with self.assertRaisesRegex(sa.ChecklistError, "UTF-8"):
                sa.load(path)


class RejectTests(unittest.TestCase):
    def assertRejected(self, text, pattern):
        with self.assertRaisesRegex(sa.ChecklistError, pattern):
            sa.loads(text)

    def test_toml_syntax_error(self):
        self.assertRejected("[q.a\n", "^<checklist>: .*line 1, column 5")

    def test_unknown_top_level_key(self):
        self.assertRejected("requires = 1\n" + MINIMAL, "unknown top-level")

    def test_needs_a_requirement(self):
        self.assertRejected('[q.a]\nask = "x"\nby = "ci"\n', "at least one")
        self.assertRejected(
            'require = []\n[q.a]\nask = "x"\nby = "ci"\n', "at least one"
        )

    def test_question_needs_ask_and_by(self):
        self.assertRejected(
            '[q.a]\nby = "ci"\n[[require]]\nthat = "a"\nfail = "x"\n', "q.a.ask"
        )
        self.assertRejected(
            '[q.a]\nask = "x"\n[[require]]\nthat = "a"\nfail = "x"\n', "q.a.by"
        )

    def test_unknown_question_reference(self):
        self.assertRejected(
            MINIMAL.replace('that = "a"', 'that = "b"'), "'b' is not a question"
        )

    def test_tables_where_tables_belong(self):
        require = '[[require]]\nthat = "a"\nfail = "x"\n'
        for text, pattern in [
            ("q = 1\n" + require, "q must be a table of questions"),
            ('q.a = "Done?"\n' + require, "q.a must be a table with ask and by"),
            ('require = ["a"]\n[q.a]\nask = "x"\nby = "ci"\n', "#1 must be a table"),
            (MINIMAL.replace('that = "a"', ""), "#1 needs `that`"),
        ]:
            with self.subTest(text):
                self.assertRejected(text, pattern)

    def test_requirement_needs_fail(self):
        self.assertRejected(MINIMAL.replace('fail = "A is not done"', ""), "fail")

    def test_unknown_requirement_key(self):
        self.assertRejected(MINIMAL + 'unless = "a"\n', "unknown key")

    def test_compound_requirement_needs_an_id(self):
        text = MINIMAL.replace('that = "a"', 'that = { all = ["a"] }')
        self.assertRejected(text, "needs an id")

    def test_duplicate_requirement(self):
        self.assertRejected(
            MINIMAL + '\n[[require]]\nthat = "a"\nfail = "again"\n', "twice"
        )

    def test_bad_expressions(self):
        for that, pattern in [
            ('{ some = ["a"] }', "unknown operator"),
            ("{ all = [] }", "non-empty list"),
            ('{ all = ["a"], any = ["a"] }', "one of"),
            ("3", "question id or a table"),
        ]:
            with self.subTest(that=that):
                text = MINIMAL.replace('that = "a"', f'id = "r"\nthat = {that}')
                self.assertRejected(text, pattern)

    def test_identifiers(self):
        self.assertRejected(
            MINIMAL.replace("[q.a]", '[q."a-b"]').replace('"a"', '"a-b"'), "identifier"
        )


class DigestTests(unittest.TestCase):
    def test_comments_and_layout_do_not_change_the_checklist(self):
        reformatted = "# a comment\n" + MINIMAL.replace(
            'ask = "Is A done?"', 'ask   =   "Is A done?"   # same'
        )
        self.assertEqual(sa.loads(MINIMAL).digest, sa.loads(reformatted).digest)
        self.assertEqual(sa.loads(MINIMAL), sa.loads(reformatted))
        self.assertNotIn("a comment", repr(sa.loads(reformatted)))
        options = MINIMAL.replace('by = "ci"', 'by = "ci"\ncheck = "t"\nrun = "x"')
        reordered = MINIMAL.replace('by = "ci"', 'by = "ci"\nrun = "x"\ncheck = "t"')
        self.assertEqual(sa.loads(options).digest, sa.loads(reordered).digest)

    def test_the_digest_stays_the_same_from_one_version_to_the_next(self):
        # Approved checklists are pinned, and runs recorded, by their digest.
        text = MINIMAL.replace("Is A done?", "Is Ä done — on time?")
        self.assertEqual(
            sa.loads(text).digest,
            "9a0b967bfb5ec341184cd8d653b1704f134e451c7478907fbaf73247e5628c53",
        )

    def test_meaning_changes_the_digest(self):
        changed = MINIMAL.replace('by = "ci"', 'by = "llm"')
        self.assertNotEqual(sa.loads(MINIMAL).digest, sa.loads(changed).digest)

    def test_options_may_be_tables_and_lists_that_stay_as_loaded(self):
        def written(env):
            return MINIMAL.replace(
                'by = "ci"', f'by = "ci"\nlabels = ["x", "y"]\nenv = {env}'
            )

        checklist = sa.loads(written('{ B = "2", A = "1" }'))
        reordered = sa.loads(written('{ A = "1", B = "2" }'))
        self.assertEqual(checklist.digest, reordered.digest)
        source = checklist.to_dict()["q"]["a"]
        self.assertEqual(
            (source["env"], source["labels"]), ({"A": "1", "B": "2"}, ["x", "y"])
        )
        with self.assertRaises(TypeError, msg="the digest stays true"):
            checklist.questions["a"].options["env"]["A"] = "3"

    def test_dates_in_options_have_a_digest(self):
        dated = MINIMAL.replace('by = "ci"', 'by = "ci"\nsince = 2026-10-01')
        self.assertNotEqual(sa.loads(dated).digest, sa.loads(MINIMAL).digest)

    def test_to_dict_matches_the_source(self):
        source = tomllib.loads(EXAMPLE.read_text())
        for requirement in source["require"]:
            requirement.setdefault("id", requirement["that"])
        self.assertEqual(example().to_dict(), source)


if __name__ == "__main__":
    unittest.main()
