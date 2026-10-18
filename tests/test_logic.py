import unittest

from helpers import PASSING, example

import symbolic_acceptance as sa
from symbolic_acceptance.checklist import AllOf, AnyOf, Not, Ref
from symbolic_acceptance.logic import FAIL, NA, OPEN, PASS, decided, plan, status, value

T, F, U = True, False, None


def facts(**values):
    return {k: v for k, v in values.items() if v is not None}


class ValueTests(unittest.TestCase):
    def test_kleene_all(self):
        expr = AllOf((Ref("a"), Ref("b")))
        cases = {(T, T): T, (T, F): F, (F, U): F, (T, U): U, (U, U): U}
        for (a, b), expected in cases.items():
            with self.subTest(a=a, b=b):
                self.assertIs(value(expr, facts(a=a, b=b)), expected)

    def test_kleene_any(self):
        expr = AnyOf((Ref("a"), Ref("b")))
        cases = {(F, F): F, (T, U): T, (F, U): U, (U, U): U}
        for (a, b), expected in cases.items():
            with self.subTest(a=a, b=b):
                self.assertIs(value(expr, facts(a=a, b=b)), expected)

    def test_not(self):
        self.assertIs(value(Not(Ref("a")), {"a": True}), False)
        self.assertIs(value(Not(Ref("a")), {}), None)


class StatusTests(unittest.TestCase):
    def requirement(self, when=None):
        return sa.Requirement("r", Ref("t"), "fails", when)

    def test_unconditional(self):
        r = self.requirement()
        self.assertEqual(status(r, {"t": True}), PASS)
        self.assertEqual(status(r, {"t": False}), FAIL)
        self.assertEqual(status(r, {}), OPEN)

    def test_conditional(self):
        r = self.requirement(Ref("w"))
        self.assertEqual(status(r, {"w": False}), NA)
        self.assertEqual(status(r, {"w": False, "t": False}), NA)
        self.assertEqual(status(r, {"w": True, "t": False}), FAIL)
        self.assertEqual(status(r, {"t": False}), OPEN)
        # Holding is enough even while `when` is open.
        self.assertEqual(status(r, {"t": True}), PASS)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.checklist = example()

    def plan(self, known, last=("human",)):
        return plan(self.checklist, known, last=lambda q: q.by in last)

    def test_first_round_asks_every_automated_question_but_gated_ones(self):
        asked = self.plan({})
        self.assertEqual(
            asked,
            ["repro_fixed", "new_test", "suite_green", "frozen_untouched",
             "locks_account", "can_unlock", "backs_off", "touches_sso"],
        )

    def test_gated_question_waits_for_its_condition(self):
        known = {k: v for k, v in PASSING.items() if k != "sso_exempt"}
        known["touches_sso"] = True
        self.assertEqual(self.plan(known), ["sso_exempt"])

    def test_moot_person_is_never_asked(self):
        known = {k: v for k, v in PASSING.items() if k != "sso_exempt"}
        self.assertEqual(self.plan(known), [])
        self.assertTrue(decided(self.checklist, known))

    def test_failure_is_not_held_up_by_people(self):
        known = {k: v for k, v in PASSING.items() if k != "sso_exempt"}
        known.update(touches_sso=True, new_test=False)
        self.assertEqual(self.plan(known), [])

    def test_satisfied_alternative_skips_the_rest(self):
        known = dict(PASSING)
        for qid in ("locks_account", "can_unlock", "backs_off", "sso_exempt"):
            del known[qid]
        known["backs_off"] = True
        self.assertEqual(self.plan(known), [])

    def test_failing_all_still_reports_every_item(self):
        checklist = sa.loads(
            """
            [q.a]
            ask = "A?"
            by = "ci"
            [q.b]
            ask = "B?"
            by = "llm"
            [[require]]
            id = "both"
            that = { all = ["a", "b"] }
            fail = "not both"
            """
        )
        asked = plan(checklist, {"a": False}, last=lambda q: False)
        self.assertEqual(asked, ["b"])

    def test_people_asked_minimally_once_alone(self):
        checklist = sa.loads(
            """
            [q.h1]
            ask = "H1?"
            by = "human"
            [q.h2]
            ask = "H2?"
            by = "human"
            [[require]]
            id = "either"
            that = { any = ["h1", "h2"] }
            fail = "neither"
            """
        )
        self.assertEqual(plan(checklist, {}, last=lambda q: True), ["h1", "h2"])
        self.assertEqual(plan(checklist, {"h1": True}, last=lambda q: True), [])


if __name__ == "__main__":
    unittest.main()
