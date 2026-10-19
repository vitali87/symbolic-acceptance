import asyncio
import unittest

from helpers import PASSING, Scripted, example

import symbolic_acceptance as sa


def evaluate(checklist, oracles, **kwargs):
    return asyncio.run(sa.evaluate(checklist, sa.Attempt("abc", 1), oracles, **kwargs))


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.checklist = example()

    def test_approves_without_asking_a_moot_person(self):
        script = Scripted(PASSING)
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.APPROVED)
        self.assertNotIn("sso_exempt", script.asked_ids())
        self.assertEqual(result.rounds[0][0], "repro_fixed")

    def test_reports_every_failure_at_once(self):
        script = Scripted(
            dict(
                PASSING,
                repro_fixed=False,
                suite_green=False,
                backs_off=False,
                can_unlock=False,
            )
        )
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.CHANGES)
        self.assertEqual(result.failing, {"repro_fixed", "suite_green", "protected"})
        self.assertIn("fix any one of", result.feedback)

    def test_asks_people_last(self):
        script = Scripted(dict(PASSING, touches_sso=True))
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual(script.asked[-1], ("human", ("sso_exempt",)))
        self.assertEqual(result.answers["sso_exempt"].by, "human")

    def test_waits_for_people_and_resumes(self):
        script = Scripted(dict(PASSING, touches_sso=True), silent={"human"})
        waiting = evaluate(self.checklist, script.oracles())
        self.assertEqual(waiting.status, sa.WAITING)
        self.assertEqual(waiting.waiting_on, ("sso_exempt",))
        script.silent.clear()
        resumed = evaluate(
            self.checklist,
            script.oracles(),
            answers=waiting.answers,
            rounds=waiting.rounds,
        )
        self.assertEqual(resumed.status, sa.APPROVED)
        self.assertEqual(script.asked_ids("ci").count("repro_fixed"), 1)

    def test_failure_does_not_wait_for_people(self):
        script = Scripted(
            dict(PASSING, touches_sso=True, new_test=False), silent={"human"}
        )
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.CHANGES)
        self.assertNotIn("human", [who for who, _ in script.asked])

    def test_pending_keeps_partial_answers(self):
        async def ci(questions, context):
            raise sa.Pending("still running", answers={"repro_fixed": sa.Answer(True)})

        script = Scripted(PASSING)
        oracles = dict(script.oracles(), ci=ci)
        result = evaluate(self.checklist, oracles)
        self.assertEqual(result.status, sa.WAITING)
        self.assertIn("repro_fixed", result.answers)
        self.assertNotIn("repro_fixed", result.waiting_on)

    def test_escalation(self):
        async def llm(questions, context):
            raise sa.Escalate("instructions to the reviewer in the diff")

        result = evaluate(self.checklist, dict(Scripted(PASSING).oracles(), llm=llm))
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertIn("instructions to the reviewer", result.reason)

    def test_oracle_contract(self):
        script = Scripted(PASSING)

        async def silent(questions, context):
            return {}

        async def overreach(questions, context):
            return {"sso_exempt": sa.Answer(True)}

        async def wrong_type(questions, context):
            return {q.id: True for q in questions}

        for bad, pattern in [
            (silent, "none of"),
            (overreach, "not asked"),
            (wrong_type, "expected Answer"),
        ]:
            with self.subTest(bad=bad.__name__):
                with self.assertRaisesRegex(sa.OracleError, pattern):
                    evaluate(self.checklist, dict(script.oracles(), ci=bad))

    def test_sync_oracles_run_in_a_thread(self):
        def ci(questions, context):
            return {q.id: sa.Answer(True) for q in questions}

        result = evaluate(self.checklist, dict(Scripted(PASSING).oracles(), ci=ci))
        self.assertEqual(result.status, sa.APPROVED)

    def test_missing_oracle(self):
        with self.assertRaisesRegex(ValueError, "no oracle for: human"):
            evaluate(self.checklist, Scripted(PASSING).oracles(("ci", "llm")))

    def test_context_carries_issue_and_answers(self):
        seen = {}

        async def llm(questions, context):
            seen["issue"] = context.issue
            seen["known"] = sorted(context.answers)
            return {q.id: sa.Answer(PASSING[q.id]) for q in questions}

        evaluate(
            self.checklist,
            dict(Scripted(PASSING).oracles(), llm=llm),
            issue="#342 text",
        )
        self.assertEqual(seen["issue"], "#342 text")
        self.assertEqual(seen["known"], [])


class Agent:
    """Each call takes the next truth table and returns a new ref."""

    def __init__(self, script, truths):
        self.script = script
        self.truths = list(truths)
        self.tasks = []

    async def __call__(self, task):
        self.tasks.append(task)
        self.script.truth = self.truths[len(self.tasks) - 1]
        return f"commit{len(self.tasks)}"


def run(checklist, truths, **kwargs):
    script = Scripted(truths[0])
    agent = Agent(script, truths)
    result = asyncio.run(
        sa.run(checklist, agent=agent, oracles=script.oracles(), **kwargs)
    )
    return result, agent


class RunTests(unittest.TestCase):
    def setUp(self):
        self.checklist = example()

    def test_feedback_reaches_the_agent_until_approval(self):
        result, agent = run(
            self.checklist,
            [dict(PASSING, new_test=False), PASSING],
            issue="Brute force logins",
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual([a.number for a in result.attempts], [1, 2])
        self.assertEqual(agent.tasks[0].feedback, ())
        self.assertEqual(
            [f.requirement.id for f in agent.tasks[1].feedback], ["new_test"]
        )
        prompt = agent.tasks[1].prompt()
        self.assertIn("Brute force logins", prompt)
        self.assertIn("Attempt 1 (commit1) was not accepted", prompt)
        self.assertIn("new_test: needs YES, got NO", prompt)

    def test_prompt_names_the_frozen_files(self):
        prompt = sa.Task(issue="", checklist=self.checklist, number=1).prompt()
        self.assertIn(
            "Do not change files matching `acceptance/*`, `.github/*`, `tests/*` "
            "(new files matching `tests/test_*.py` are allowed).",
            prompt,
        )
        bare = sa.loads(
            '[q.a]\nask = "A?"\nby = "ci"\n[[require]]\nthat = "a"\nfail = "x"\n'
        )
        self.assertNotIn(
            "Do not change", sa.Task(issue="", checklist=bare, number=1).prompt()
        )

    def test_no_progress_escalates(self):
        failing = dict(PASSING, new_test=False)
        result, _ = run(self.checklist, [failing, failing, PASSING])
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertIn("no progress", result.reason)
        self.assertEqual(len(result.evaluations), 2)

    def test_regression_escalates(self):
        result, _ = run(
            self.checklist,
            [dict(PASSING, new_test=False), dict(PASSING, suite_green=False), PASSING],
        )
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertIn("regression: suite_green", result.reason)

    def test_a_requirement_that_newly_applies_is_not_a_regression(self):
        result, agent = run(
            self.checklist,
            [
                dict(PASSING, new_test=False),
                dict(PASSING, touches_sso=True, sso_exempt=False),
                PASSING,
            ],
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual([f.requirement.id for f in agent.tasks[2].feedback], ["sso"])

    def test_regression_can_go_back_to_the_agent(self):
        result, _ = run(
            self.checklist,
            [dict(PASSING, new_test=False), dict(PASSING, suite_green=False), PASSING],
            stop_on_regression=False,
        )
        self.assertEqual(result.status, sa.APPROVED)

    def test_attempt_cap(self):
        truths = [
            dict(PASSING, new_test=False),
            dict(PASSING, suite_green=False, new_test=False),
        ]
        result, _ = run(
            self.checklist, truths, max_attempts=2, stop_on_regression=False
        )
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertIn("attempt cap", result.reason)

    def test_waiting_run_resumes(self):
        truth = dict(PASSING, touches_sso=True)
        script = Scripted(truth, silent={"human"})
        agent = Agent(script, [truth])
        first = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles())
        )
        self.assertEqual(first.status, sa.WAITING)
        script.silent.clear()
        second = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles(), resume=first)
        )
        self.assertEqual(second.status, sa.APPROVED)
        self.assertEqual(len(agent.tasks), 1)

    def test_resume_keeps_the_issue(self):
        truth = dict(PASSING, touches_sso=True)
        script = Scripted(truth, silent={"human"})
        agent = Agent(script, [truth])
        first = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles(), issue="#342")
        )
        self.assertEqual(first.issue, "#342")
        script.silent.clear()
        second = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles(), resume=first)
        )
        self.assertEqual(second.issue, "#342")
        with self.assertRaisesRegex(ValueError, "issue text differs"):
            asyncio.run(
                sa.run(
                    self.checklist,
                    agent=agent,
                    oracles=script.oracles(),
                    resume=first,
                    issue="another issue",
                )
            )

    def test_resume_rejects_a_changed_checklist(self):
        result, agent = run(
            self.checklist, [dict(PASSING, new_test=False), PASSING], max_attempts=1
        )
        changed = sa.loads(self.checklist.source.replace("Throttle", "Limit"))
        with self.assertRaisesRegex(ValueError, "checklist changed"):
            asyncio.run(
                sa.run(
                    changed,
                    agent=agent,
                    oracles=Scripted(PASSING).oracles(),
                    resume=result,
                )
            )

    def test_progress_and_diff_source(self):
        seen = []
        script = Scripted(PASSING)
        agent = Agent(script, [PASSING])

        def diff(attempt):
            return f"diff of {attempt.ref}"

        diffs = []

        async def llm(questions, context):
            diffs.append(context.attempt.diff)
            return {q.id: sa.Answer(PASSING[q.id]) for q in questions}

        result = asyncio.run(
            sa.run(
                self.checklist,
                agent=agent,
                oracles=dict(script.oracles(), llm=llm),
                diff=diff,
                on_progress=lambda r: seen.append((r.status, len(r.evaluations))),
            )
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual(seen, [(sa.RUNNING, 1)])
        self.assertEqual(diffs, ["diff of commit1"])

    def test_agent_must_return_a_ref(self):
        async def agent(task):
            return None

        with self.assertRaises(TypeError):
            asyncio.run(
                sa.run(self.checklist, agent=agent, oracles=Scripted(PASSING).oracles())
            )


if __name__ == "__main__":
    unittest.main()
