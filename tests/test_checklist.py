import tempfile
import unittest
from pathlib import Path

from helpers import example

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
        self.assertEqual(checklist.requirement("sso").when, Ref("touches_sso"))

    def test_options_are_kept_and_read_only(self):
        question = example().questions["suite_green"]
        self.assertEqual(question.options["check"], "tests")
        with self.assertRaises(TypeError):
            question.options["check"] = "other"  # type: ignore[index]

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
        self.assertRejected("[q.a\n", "<checklist>")

    def test_unknown_top_level_key(self):
        self.assertRejected("requires = 1\n" + MINIMAL, "unknown top-level")

    def test_needs_a_requirement(self):
        self.assertRejected('[q.a]\nask = "x"\nby = "ci"\n', "at least one")

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
    def test_comments_and_layout_do_not_change_the_digest(self):
        reformatted = "# a comment\n" + MINIMAL.replace(
            'ask = "Is A done?"', 'ask   =   "Is A done?"   # same'
        )
        self.assertEqual(sa.loads(MINIMAL).digest, sa.loads(reformatted).digest)

    def test_meaning_changes_the_digest(self):
        changed = MINIMAL.replace('by = "ci"', 'by = "llm"')
        self.assertNotEqual(sa.loads(MINIMAL).digest, sa.loads(changed).digest)

    def test_dates_in_options_have_a_digest(self):
        dated = MINIMAL.replace('by = "ci"', 'by = "ci"\nsince = 2026-10-01')
        self.assertNotEqual(sa.loads(dated).digest, sa.loads(MINIMAL).digest)

    def test_to_dict_matches_the_source_shape(self):
        data = example().to_dict()
        self.assertEqual(
            data["q"]["suite_green"],
            {"ask": "Does the existing test suite pass?", "by": "ci", "check": "tests"},
        )
        self.assertEqual(data["require"][-1]["when"], "touches_sso")


if __name__ == "__main__":
    unittest.main()
