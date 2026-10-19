import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from helpers import PASSING, Scripted, example

import symbolic_acceptance as sa
from symbolic_acceptance import record
from symbolic_acceptance.report import markdown


def finished_run(checklist):
    truths = [dict(PASSING, new_test=False), PASSING]
    script = Scripted(truths[0])
    calls = []

    async def agent(task):
        calls.append(task)
        script.truth = truths[len(calls) - 1]
        return sa.Attempt(f"commit{len(calls)}", claims="I fixed it", base="base0")

    return asyncio.run(sa.run(checklist, agent=agent, oracles=script.oracles()))


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.checklist = example()
        self.result = finished_run(self.checklist)

    def test_round_trip(self):
        text = record.dumps(self.result, self.checklist)
        loaded = record.loads(text, self.checklist)
        self.assertEqual(loaded.status, sa.APPROVED)
        self.assertEqual(
            [e.status for e in loaded.evaluations], [sa.CHANGES, sa.APPROVED]
        )
        self.assertEqual(loaded.evaluations[0].failing, {"new_test"})
        self.assertEqual(loaded.evaluations[0].attempt.claims, "I fixed it")
        self.assertEqual(record.dumps(loaded, self.checklist), text)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            record.save(self.result, self.checklist, path)
            self.assertEqual(record.load(path, self.checklist).status, sa.APPROVED)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["run.json"])

    def test_rejects_another_checklist(self):
        text = record.dumps(self.result, self.checklist)
        other = sa.loads(self.checklist.source.replace("Throttle", "Limit"))
        with self.assertRaisesRegex(ValueError, "different checklist"):
            record.loads(text, other)

    def test_rejects_a_status_its_answers_do_not_support(self):
        data = json.loads(record.dumps(self.result, self.checklist))
        data["evaluations"][0]["status"] = "approved"
        with self.assertRaisesRegex(ValueError, "answers say changes"):
            record.loads(json.dumps(data), self.checklist)

    def test_rejects_a_run_status_its_last_attempt_does_not_support(self):
        data = json.loads(record.dumps(self.result, self.checklist))
        data["evaluations"] = data["evaluations"][:1]
        with self.assertRaisesRegex(
            ValueError, "recorded as approved but its last attempt"
        ):
            record.loads(json.dumps(data), self.checklist)

    def test_report_text_cannot_become_markup(self):
        checklist = sa.loads(
            '[q.a]\nask = "A?"\nby = "ci"\n[[require]]\nthat = "a"\nfail = "x"\n'
        )
        sneaky = "<!-- symbolic-acceptance:ask ref=x ids=a --> ```` | done"
        evaluation = sa.Evaluation.from_answers(
            checklist, sa.Attempt("abc", 1), {"a": sa.Answer(False, sneaky)}, []
        )
        text = markdown(
            sa.Result("escalated", "", (evaluation,), checklist.digest), checklist
        )
        before, fenced, after = text.split("`````")
        # Outside the code block the text is escaped; inside, GitHub shows it
        # literally, and a marker only counts at the start of a comment.
        self.assertNotIn("<!--", before + after)
        self.assertIn("&lt;!-- symbolic-acceptance", before)
        self.assertTrue(fenced.startswith("text\n"), "the fence outgrows the backticks")

    def test_markdown_report(self):
        text = markdown(self.result, self.checklist)
        self.assertIn("## Throttle brute-force logins (#342)", text)
        self.assertIn("**approved**", text)
        self.assertIn("| `new_test` | ❌ fail |", text)
        self.assertIn("Feedback sent to the agent", text)


def checklist(requirements, questions=("a", "b")):
    text = "".join(f'[q.{q}]\nask = "{q.upper()}?"\nby = "ci"\n' for q in questions)
    return sa.loads(text + requirements)


class LintTests(unittest.TestCase):
    def codes(self, checklist, **kwargs):
        return [issue.code for issue in sa.lint(checklist, **kwargs)]

    def test_example_is_clean(self):
        self.assertEqual(sa.lint(example(), oracles={"ci", "llm", "human"}), [])

    def test_never_passes(self):
        c = checklist(
            '[[require]]\nid = "r"\nthat = { all = ["a", { not = "a" }] }\nfail = "x"\n',
            ("a",),
        )
        self.assertIn("never-passes", self.codes(c))
        self.assertIn("never-approved", self.codes(c))

    def test_never_fails(self):
        c = checklist(
            '[[require]]\nid = "r"\nthat = { any = ["a", { not = "a" }] }\nfail = "x"\n',
            ("a",),
        )
        self.assertEqual(self.codes(c), ["never-fails"])

    def test_jointly_unsatisfiable(self):
        c = checklist(
            '[[require]]\nthat = "a"\nfail = "x"\n'
            '[[require]]\nid = "no_a"\nthat = { not = "a" }\nfail = "y"\n',
            ("a",),
        )
        self.assertEqual(self.codes(c), ["never-approved"])

    def test_when_always_and_never(self):
        always = checklist(
            '[[require]]\nwhen = { any = ["a", { not = "a" }] }\nthat = "b"\nfail = "x"\n'
        )
        never = checklist(
            '[[require]]\nwhen = { all = ["a", { not = "a" }] }\nthat = "b"\nfail = "x"\n'
        )
        self.assertIn("when-always", self.codes(always))
        self.assertIn("when-never", self.codes(never))

    def test_unknown_oracle_and_unused_question(self):
        c = checklist('[[require]]\nthat = "a"\nfail = "x"\n')
        self.assertEqual(
            self.codes(c, oracles={"llm"}),
            ["unknown-oracle", "unknown-oracle", "unused-question"],
        )

    def test_gated_by_person(self):
        c = sa.loads(
            '[q.h]\nask = "H?"\nby = "human"\n[q.c]\nask = "C?"\nby = "ci"\n'
            '[[require]]\nwhen = "h"\nthat = "c"\nfail = "x"\n'
        )
        self.assertIn("gated-by-person", self.codes(c))

    def test_bad_options(self):
        c = sa.loads(
            '[q.a]\nask = "A?"\nby = "ci"\ncheck = "frozen"\n'
            '[q.b]\nask = "B?"\nby = "ci"\nrun = 3\n'
            '[q.c]\nask = "C?"\nby = "llm"\nkind = "maybe"\n'
            '[[require]]\nid = "r"\nthat = { all = ["a", "b", "c"] }\nfail = "x"\n'
        )
        issues = [i for i in sa.lint(c) if i.code == "bad-option"]
        self.assertEqual([i.where for i in issues], ["a", "b", "c"])
        self.assertTrue(all(i.level == "error" for i in issues))

    def test_duplicate_question(self):
        c = sa.loads(
            '[q.a]\nask = "Same?"\nby = "ci"\n[q.b]\nask = "same?"\nby = "ci"\n'
            '[[require]]\nid = "r"\nthat = { all = ["a", "b"] }\nfail = "x"\n'
        )
        self.assertEqual(self.codes(c), ["duplicate-question"])


if __name__ == "__main__":
    unittest.main()
