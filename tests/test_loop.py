import asyncio
import dataclasses
import time
import unittest

from helpers import EXAMPLE, PASSING, Scripted, example

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
        self.assertIn("repro_fixed", result.rounds[0], "CI is asked in the first round")

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
        script = Scripted(dict(PASSING, sso_untouched=False))
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual(script.asked[-1], ("human", ("sso_exempt",)))
        self.assertEqual(result.answers["sso_exempt"].by, "human")

    def test_waits_for_people_and_resumes(self):
        script = Scripted(dict(PASSING, sso_untouched=False), silent={"human"})
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

    def test_a_persons_no_decides_without_waiting_for_the_rest(self):
        checklist = sa.loads(
            '[q.ux]\nask = "UX ok?"\nby = "human"\n'
            '[q.docs]\nask = "Docs ok?"\nby = "human"\n'
            '[[require]]\nid = "r"\nthat = { all = ["ux", "docs"] }\nfail = "x"\n'
        )

        async def human(questions, context):
            raise sa.Pending("asked", answers={"ux": sa.Answer(False, "confusing")})

        result = evaluate(checklist, {"human": human})
        self.assertEqual(result.status, sa.CHANGES)
        self.assertEqual(result.failing, {"r"})

    def test_a_persons_no_decides_once_what_their_answers_unlocked_is_asked(self):
        checklist = sa.loads(
            '[q.ok]\nask = "OK?"\nby = "human"\n'
            '[q.risky]\nask = "Risky?"\nby = "human"\n'
            '[q.docs]\nask = "Docs ok?"\nby = "human"\n'
            '[q.load]\nask = "Load test written?"\nby = "llm"\n'
            '[[require]]\nthat = "ok"\nfail = "not ok"\n'
            '[[require]]\nid = "gated"\nwhen = "risky"\nthat = "load"\nfail = "x"\n'
            '[[require]]\nthat = "docs"\nfail = "no docs"\n'
        )
        answers = {"ok": sa.Answer(False, "no"), "risky": sa.Answer(True)}

        async def human(questions, context):
            raise sa.Pending("asked", answers=answers)

        script = Scripted({"load": True})
        result = evaluate(checklist, {"human": human, "llm": script.oracle("llm")})
        self.assertEqual(result.status, sa.CHANGES)
        self.assertEqual(result.failing, {"ok"})
        self.assertEqual(script.asked_ids("llm"), ["load"])

    def test_an_escalation_keeps_the_answers_given_with_a_pause(self):
        checklist = sa.loads(
            '[q.ux]\nask = "UX ok?"\nby = "human"\n'
            '[q.docs]\nask = "Docs ok?"\nby = "human"\n'
            '[q.api]\nask = "API kept?"\nby = "llm"\n'
            '[[require]]\nid = "r"\nthat = { all = ["ux", "docs", "api"] }\n'
            'fail = "x"\n'
        )

        async def human(questions, context):
            raise sa.Pending("asked", answers={"ux": sa.Answer(False, "confusing")})

        async def llm(questions, context):
            raise sa.Escalate("the change addresses the reviewer")

        result = evaluate(checklist, {"human": human, "llm": llm}, last=())
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertEqual(result.answers["ux"].reasoning, "confusing")

    def test_an_answer_that_cannot_matter_is_not_waited_for(self):
        checklist = sa.loads(
            '[q.tests]\nask = "Tests pass?"\nby = "ci"\n'
            '[q.risky]\nask = "Risky?"\nby = "github"\n'
            '[[require]]\nthat = "tests"\nfail = "x"\n'
            '[[require]]\nid = "r"\nwhen = "risky"\nthat = "tests"\nfail = "y"\n'
        )
        script = Scripted({"tests": True}, silent={"github"})
        result = evaluate(checklist, script.oracles(("ci", "github")))
        self.assertEqual(result.status, sa.APPROVED)

    def test_an_oracle_cannot_change_the_answers_the_others_read(self):
        async def meddler(questions, context):
            context.answers["sso_exempt"] = sa.Answer(True)

        oracles = {**Scripted(PASSING).oracles(), "llm": meddler}
        with self.assertRaisesRegex(TypeError, "does not support item assignment"):
            evaluate(self.checklist, oracles)

    def test_failure_does_not_wait_for_people(self):
        script = Scripted(
            dict(PASSING, sso_untouched=False, new_test=False), silent={"human"}
        )
        result = evaluate(self.checklist, script.oracles())
        self.assertEqual(result.status, sa.CHANGES)
        self.assertNotIn("human", [who for who, _ in script.asked])

    def test_a_paused_oracle_is_asked_again_so_corrections_count(self):
        checklist = sa.loads(
            '[q.ux]\nask = "UX ok?"\nby = "human"\n'
            '[q.docs]\nask = "Docs ok?"\nby = "human"\n'
            '[[require]]\nid = "r"\nthat = { all = ["ux", "docs"] }\nfail = "x"\n'
        )
        # A person says YES to ux while docs waits, then takes it back.
        replies = [
            {"ux": sa.Answer(True, "fine")},
            {"ux": sa.Answer(False, "I misread it"), "docs": sa.Answer(True)},
        ]

        async def human(questions, context):
            reply = replies.pop(0)
            answers = {q.id: reply[q.id] for q in questions if q.id in reply}
            if len(answers) < len(questions):
                raise sa.Pending("asked", answers=answers)
            return answers

        waiting = evaluate(checklist, {"human": human})
        self.assertEqual(waiting.status, sa.WAITING)
        self.assertEqual(waiting.waiting_on, ("docs",))
        resumed = evaluate(
            checklist, {"human": human}, answers=waiting.answers, rounds=waiting.rounds
        )
        self.assertEqual(resumed.status, sa.CHANGES)
        self.assertEqual(resumed.answers["ux"].reasoning, "I misread it")

    def test_an_escalation_wins_over_a_pause_in_the_same_round(self):
        async def ci(questions, context):
            raise sa.Pending("CI running")

        async def llm(questions, context):
            raise sa.Escalate("the change addresses the reviewer")

        oracles = dict(Scripted(PASSING).oracles(), ci=ci, llm=llm)
        result = evaluate(self.checklist, oracles)
        self.assertEqual(result.status, sa.ESCALATED)

    def test_escalation(self):
        async def llm(questions, context):
            raise sa.Escalate("instructions to the reviewer in the diff")

        result = evaluate(self.checklist, dict(Scripted(PASSING).oracles(), llm=llm))
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertIn("instructions to the reviewer", result.reason)

    def test_a_pause_may_only_carry_answers_to_what_was_asked(self):
        async def ci(questions, context):
            raise sa.Pending("still running", answers={"sso_exempt": sa.Answer(True)})

        with self.assertRaisesRegex(sa.OracleError, "not asked"):
            evaluate(self.checklist, dict(Scripted(PASSING).oracles(), ci=ci))

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
            with (
                self.subTest(bad=bad.__name__),
                self.assertRaisesRegex(sa.OracleError, pattern),
            ):
                evaluate(self.checklist, dict(script.oracles(), ci=bad))

    def test_an_oracle_error_stops_the_others(self):
        stopped = []

        async def slow(questions, context):
            try:
                await asyncio.sleep(30)
            finally:
                stopped.append("ci")

        async def broken(questions, context):
            raise RuntimeError("the judge is down")

        async def main():
            oracles = dict(Scripted(PASSING).oracles(), ci=slow, llm=broken)
            with self.assertRaisesRegex(RuntimeError, "the judge is down"):
                await sa.evaluate(self.checklist, sa.Attempt("abc", 1), oracles)
            await asyncio.sleep(0)
            # Before the end of the loop, which would stop anything left.
            return list(stopped)

        started = time.monotonic()
        self.assertEqual(asyncio.run(main()), ["ci"])
        self.assertLess(time.monotonic() - started, 10)

    def test_sync_oracles_work(self):
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

    def test_what_a_run_hands_out_cannot_be_changed(self):
        # Oracles, agents and listeners get the same objects, so none of them
        # can change what the others see.
        result, agent = run(self.checklist, [dict(PASSING, new_test=False), PASSING])
        evaluation = result.evaluations[0]
        failure = evaluation.failures[0]
        context = sa.Context(self.checklist, evaluation.attempt, evaluation.answers)
        for value, name in (
            (result, "status"),
            (evaluation, "status"),
            (evaluation.attempt, "diff"),
            (context, "answers"),
            (agent.tasks[1], "feedback"),
            (failure, "reason"),
            (failure.reason, "answer"),
            (evaluation.answers["new_test"], "is_met"),
            (self.checklist, "requirements"),
            (self.checklist.questions["new_test"], "ask"),
            (self.checklist.requirement("new_test"), "fail"),
            (self.checklist.requirement("new_test").that, "id"),
            (self.checklist.requirement("protected").that, "items"),
            (self.checklist.requirement("sso").when, "item"),
        ):
            with (
                self.subTest(type(value).__name__),
                self.assertRaises(dataclasses.FrozenInstanceError),
            ):
                setattr(value, name, None)

    def test_feedback_reaches_the_agent_until_approval(self):
        result, agent = run(
            self.checklist,
            [dict(PASSING, new_test=False), PASSING],
            issue="Brute force logins",
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual([e.attempt.number for e in result.evaluations], [1, 2])
        self.assertEqual(agent.tasks[0].feedback, ())
        self.assertEqual(
            [f.requirement.id for f in agent.tasks[1].feedback], ["new_test"]
        )
        prompt = agent.tasks[1].prompt()
        self.assertIn("Brute force logins", prompt)
        self.assertIn("  A test that fails on the base branch for an unrelated", prompt)
        self.assertIn("Attempt 1 (commit1) was not accepted", prompt)
        self.assertIn("new_test: needs YES, got NO", prompt)

    def test_prompt_says_which_files_each_frozen_check_watches(self):
        prompt = sa.Task(issue="", checklist=self.checklist, number=1).prompt()
        self.assertIn(
            "\n  YES only when no file matching `acceptance/*`, `.github/*`, "
            "`tests/*` changes (new files matching `tests/test_*.py` excepted).\n",
            prompt,
        )
        # The SSO check decides whether a person must approve; it forbids
        # nothing, and the prompt does not say it does.
        self.assertIn(
            "- sso_untouched [ci]: Does the change leave the single sign-on code "
            "alone?\n  YES only when no file matching `src/auth/sso/*` changes.\n",
            prompt,
        )
        self.assertNotIn("Do not change", prompt)

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
                dict(PASSING, sso_untouched=False, sso_exempt=False),
                PASSING,
            ],
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual([f.requirement.id for f in agent.tasks[2].feedback], ["sso"])

    def test_a_regression_from_any_earlier_attempt_escalates(self):
        # x holds in attempt 1, does not apply in attempt 2, fails in 3.
        checklist = sa.loads(
            "".join(f'[q.{q}]\nask = "{q}?"\nby = "ci"\n' for q in "gabc")
            + '[[require]]\nid = "x"\nwhen = "g"\nthat = "a"\nfail = "x"\n'
            + '[[require]]\nid = "y"\nthat = "b"\nfail = "y"\n'
            + '[[require]]\nid = "z"\nthat = "c"\nfail = "z"\n'
        )
        result, _ = run(
            checklist,
            [
                {"g": True, "a": True, "b": False, "c": False},
                {"g": False, "a": True, "b": False, "c": True},
                {"g": True, "a": False, "b": True, "c": True},
            ],
        )
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertEqual(
            result.reason, "regression: x held in attempt 1 and fails in attempt 3"
        )

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
        truth = dict(PASSING, sso_untouched=False)
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

    def test_resume_keeps_the_issue_and_base(self):
        truth = dict(PASSING, sso_untouched=False)
        script = Scripted(truth, silent={"human"})
        agent = Agent(script, [truth])
        first = asyncio.run(
            sa.run(
                self.checklist,
                agent=agent,
                oracles=script.oracles(),
                issue="#342",
                base="b0",
            )
        )
        self.assertEqual((first.issue, first.base), ("#342", "b0"))
        script.silent.clear()
        second = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles(), resume=first)
        )
        self.assertEqual((second.issue, second.base), ("#342", "b0"))
        for other, refusal in (
            ({"issue": "another issue"}, "issue text differs"),
            ({"base": "b1"}, "base differs"),
        ):
            with self.subTest(refusal), self.assertRaisesRegex(ValueError, refusal):
                asyncio.run(
                    sa.run(
                        self.checklist,
                        agent=agent,
                        oracles=script.oracles(),
                        resume=first,
                        **other,
                    )
                )

    def test_resume_rejects_a_changed_checklist(self):
        result, agent = run(
            self.checklist, [dict(PASSING, new_test=False), PASSING], max_attempts=1
        )
        changed = sa.loads(EXAMPLE.read_text().replace("Throttle", "Limit"))
        with self.assertRaisesRegex(ValueError, "checklist changed"):
            asyncio.run(
                sa.run(
                    changed,
                    agent=agent,
                    oracles=Scripted(PASSING).oracles(),
                    resume=result,
                )
            )

    def test_progress_sees_every_evaluation_and_the_outcome(self):
        seen = []
        script = Scripted(PASSING)
        agent = Agent(script, [dict(PASSING, new_test=False), PASSING])
        result = asyncio.run(
            sa.run(
                self.checklist,
                agent=agent,
                oracles=script.oracles(),
                on_progress=lambda r: seen.append(
                    (r.status, [e.status for e in r.evaluations])
                ),
            )
        )
        self.assertEqual(result.status, sa.APPROVED)
        self.assertEqual(
            seen,
            [
                (sa.RUNNING, []),
                (sa.RUNNING, [sa.WAITING]),
                (sa.RUNNING, [sa.CHANGES]),
                (sa.RUNNING, [sa.CHANGES, sa.WAITING]),
                (sa.APPROVED, [sa.CHANGES, sa.APPROVED]),
            ],
        )

    def test_a_run_stopped_while_checking_resumes_by_checking_the_attempt(self):
        seen = []
        script = Scripted(PASSING)
        agent = Agent(script, [PASSING])

        async def broken(questions, context):
            raise RuntimeError("the network went away")

        with self.assertRaisesRegex(RuntimeError, "network"):
            asyncio.run(
                sa.run(
                    self.checklist,
                    agent=agent,
                    oracles=dict(script.oracles(), llm=broken),
                    on_progress=seen.append,
                    base="b0",
                )
            )
        saved = seen[-1]
        self.assertEqual(saved.base, "b0")
        self.assertEqual(saved.last.reason if saved.last else "", "being checked")
        resumed = asyncio.run(
            sa.run(self.checklist, agent=agent, oracles=script.oracles(), resume=saved)
        )
        self.assertEqual(resumed.status, sa.APPROVED)
        self.assertEqual(len(agent.tasks), 1, "the agent was not asked again")
        self.assertEqual(resumed.base, "b0")

    def test_an_agent_that_cannot_go_on_escalates(self):
        seen = []

        async def agent(task):
            raise sa.Escalate("attempt 1 changed nothing")

        result = asyncio.run(
            sa.run(
                self.checklist,
                agent=agent,
                oracles=Scripted(PASSING).oracles(),
                on_progress=seen.append,
                base="b0",
            )
        )
        self.assertEqual(result.status, sa.ESCALATED)
        self.assertEqual(result.reason, "agent: attempt 1 changed nothing")
        self.assertEqual(seen[-1], result, "the outcome is reported, so it is saved")
        self.assertEqual(seen[0].base, "b0", "the base is saved before the agent runs")
        script = Scripted(PASSING)
        fixed = Agent(script, [PASSING])
        resumed = asyncio.run(
            sa.run(self.checklist, agent=fixed, oracles=script.oracles(), resume=result)
        )
        self.assertEqual(resumed.status, sa.APPROVED, "a fixed agent can go on")

    def test_agent_must_return_a_ref(self):
        for empty in ("", sa.Attempt("")):
            with (
                self.subTest(empty=empty),
                self.assertRaisesRegex(TypeError, "non-empty ref"),
            ):
                asyncio.run(
                    sa.run(
                        self.checklist,
                        agent=lambda task, empty=empty: empty,
                        oracles=Scripted(PASSING).oracles(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
