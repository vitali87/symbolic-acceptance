import json
import unittest

from helpers import PASSING, example

import symbolic_acceptance as sa


def answers(truth, **reasons):
    return {k: sa.Answer(v, reasons.get(k, ""), by="test") for k, v in truth.items()}


CHECKLIST = sa.loads(
    """
    [q.a]
    ask = "A?"
    by = "ci"
    [q.b]
    ask = "B?"
    by = "ci"
    [q.c]
    ask = "C?"
    by = "ci"

    [[require]]
    id = "not_c"
    that = { not = "c" }
    fail = "C must not happen"

    [[require]]
    id = "a_and_not_b"
    that = { all = ["a", { not = "b" }] }
    fail = "Need A without B"
    """
)


class ExplainTests(unittest.TestCase):
    def test_all_lists_only_its_failing_items(self):
        checklist = sa.loads(
            """
            [q.a]
            ask = "A?"
            by = "ci"
            [q.b]
            ask = "B?"
            by = "ci"
            [[require]]
            id = "r"
            that = { all = ["a", "b"] }
            fail = "x"
            """
        )
        reason = sa.explain(
            checklist.requirement("r").that, checklist, answers({"a": True, "b": False})
        )
        assert isinstance(reason, sa.Group)
        self.assertEqual(reason.fix, "each")
        self.assertEqual([fact.question.id for fact in reason.facts()], ["b"])

    def test_any_lists_every_alternative(self):
        checklist = example()
        truth = dict(PASSING, locks_account=True, can_unlock=False, backs_off=False)
        reason = sa.explain(
            checklist.requirement("protected").that, checklist, answers(truth)
        )
        assert isinstance(reason, sa.Group)
        self.assertEqual(reason.fix, "one")
        self.assertEqual(
            [fact.question.id for fact in reason.facts()], ["can_unlock", "backs_off"]
        )

    def test_not_reports_the_yes_that_caused_it(self):
        assessment = sa.assess(CHECKLIST, answers({"a": True, "b": True, "c": True}))
        self.assertEqual(assessment.status, "changes")
        text = sa.render_failures(assessment.failures)
        self.assertIn("c: needs NO, got YES", text)
        self.assertIn("b: needs NO, got YES", text)
        self.assertNotIn("a: needs", text)

    def test_undecided_expression_cannot_be_explained(self):
        with self.assertRaisesRegex(ValueError, "not decided"):
            sa.explain(CHECKLIST.requirement("not_c").that, CHECKLIST, {})


class AnswerTests(unittest.TestCase):
    def test_types_are_checked(self):
        # Answers come from JSON, models and people, which no type checker sees.
        data = json.loads('{"is_met": "yes", "evidence": "a quote"}')
        with self.assertRaisesRegex(TypeError, "is_met must be a bool"):
            sa.Answer(data["is_met"])
        with self.assertRaisesRegex(TypeError, "not one string"):
            sa.Answer(True, evidence=data["evidence"])
        self.assertEqual(sa.Answer(True, evidence=["a", "b"]).evidence, ("a", "b"))


class AssessTests(unittest.TestCase):
    def test_statuses(self):
        checklist = example()
        self.assertEqual(sa.assess(checklist, answers(PASSING)).status, "approved")
        partial = {k: v for k, v in PASSING.items() if k != "suite_green"}
        self.assertEqual(sa.assess(checklist, answers(partial)).status, "open")
        failing = sa.assess(checklist, answers(dict(PASSING, suite_green=False)))
        self.assertEqual(failing.status, "changes")
        self.assertEqual([f.requirement.id for f in failing.failures], ["suite_green"])

    def test_every_answer_set_decides(self):
        # The format guarantees it: with every question answered, an attempt
        # is approved or has named failures, never both and never neither.
        import itertools

        checklist = example()
        ids = list(checklist.questions)
        for values in itertools.product((True, False), repeat=len(ids)):
            assessment = sa.assess(
                checklist, answers(dict(zip(ids, values, strict=True)))
            )
            self.assertIn(assessment.status, ("approved", "changes"))
            self.assertEqual(assessment.status == "changes", bool(assessment.failures))

    def test_render_mentions_the_gate(self):
        checklist = example()
        truth = dict(PASSING, sso_untouched=False, sso_exempt=False)
        text = sa.render_failures(
            sa.assess(
                checklist, answers(truth, sso_exempt="needs security review")
            ).failures
        )
        self.assertIn("1. sso: Single sign-on changed without approval", text)
        self.assertIn(
            "Applies because not(sso_untouched) is YES (sso_untouched is NO).", text
        )
        self.assertIn("Why: needs security review", text)

    def test_render_names_each_part_of_a_compound_gate(self):
        checklist = sa.loads(
            """
            [q.a]
            ask = "A?"
            by = "ci"
            [q.b]
            ask = "B?"
            by = "ci"
            [q.c]
            ask = "C?"
            by = "ci"
            [[require]]
            when = { all = ["a", { not = "b" }] }
            that = "c"
            fail = "x"
            """
        )
        failures = sa.assess(checklist, answers({"a": True, "b": False, "c": False}))
        self.assertIn(
            "Applies because all(a, not(b)) is YES (a is YES, b is NO).",
            sa.render_failures(failures.failures),
        )

    def test_render_without_failures(self):
        self.assertEqual(sa.render_failures([]), "No requirement failed.\n")


if __name__ == "__main__":
    unittest.main()
