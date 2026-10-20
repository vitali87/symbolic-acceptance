import json
import tempfile
import unittest
from pathlib import Path

from helpers import EXAMPLE, example, finished_run

import symbolic_acceptance as sa
from symbolic_acceptance import record, templates
from symbolic_acceptance.lint import MAX_EXHAUSTIVE
from symbolic_acceptance.report import markdown


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

    def test_rejects_what_is_not_a_record_of_this_checklist(self):
        with self.assertRaisesRegex(ValueError, "not a symbolic-acceptance/run v1"):
            record.loads('{"format": "something else"}', self.checklist)
        data = json.loads(record.dumps(self.result, self.checklist))
        with self.assertRaisesRegex(ValueError, "not a symbolic-acceptance/run v1"):
            record.loads(json.dumps(dict(data, version=2)), self.checklist)
        data["evaluations"][0]["answers"]["planted"] = {"is_met": True}
        with self.assertRaisesRegex(ValueError, r"unknown questions: \['planted'\]"):
            record.loads(json.dumps(data), self.checklist)

    def test_a_failed_save_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            taken = Path(tmp, "run.json")
            taken.mkdir()  # a directory where the record should go
            with self.assertRaises(OSError):
                record.save(self.result, self.checklist, taken)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["run.json"])

    def test_rejects_another_checklist(self):
        text = record.dumps(self.result, self.checklist)
        other = sa.loads(EXAMPLE.read_text().replace("Throttle", "Limit"))
        with self.assertRaisesRegex(ValueError, "different .* as checklist.source$"):
            record.loads(text, other)

    def test_rejects_a_status_its_answers_do_not_support(self):
        data = json.loads(record.dumps(self.result, self.checklist))
        data["evaluations"][0]["status"] = "approved"
        with self.assertRaisesRegex(ValueError, "answers say changes"):
            record.loads(json.dumps(data), self.checklist)
        data["evaluations"][0]["answers"] = {}
        with self.assertRaisesRegex(ValueError, "not decided by its answers"):
            record.loads(json.dumps(data), self.checklist)

    def test_saves_a_run_only_with_its_own_checklist(self):
        other = sa.loads(EXAMPLE.read_text().replace("Throttle", "Limit"))
        with self.assertRaisesRegex(ValueError, "made with a different checklist"):
            record.dumps(self.result, other)

    def test_rejects_a_run_status_its_attempts_do_not_support(self):
        data = json.loads(record.dumps(self.result, self.checklist))
        approved = data["evaluations"][1]
        first = dict(approved, attempt=dict(approved["attempt"], number=1))
        for status, evaluations in (
            ("approved", data["evaluations"][:1]),
            ("escalated", data["evaluations"]),
            ("running", data["evaluations"]),
            ("approved", [first, approved]),
        ):
            with self.subTest(status, attempts=len(evaluations)):
                changed = dict(data, status=status, evaluations=evaluations)
                with self.assertRaisesRegex(ValueError, f"recorded as {status} but"):
                    record.loads(json.dumps(changed), self.checklist)

    def test_rejects_attempts_not_numbered_in_turn(self):
        data = json.loads(record.dumps(self.result, self.checklist))
        data["evaluations"][1]["attempt"]["number"] = 7
        with self.assertRaisesRegex(ValueError, "attempt 2 is numbered 7"):
            record.loads(json.dumps(data), self.checklist)

    def test_rejects_what_is_not_shaped_like_a_record(self):
        def broken(path, value):
            top = json.loads(record.dumps(self.result, self.checklist))
            *parents, key = path
            data = top
            for part in parents:
                data = data[part]
            if value is None:
                del data[key]
            else:
                data[key] = value
            return path, json.dumps(top)

        cases = [
            broken(("evaluations", 0, "rounds"), None),
            broken(("evaluations", 0, "attempt", "ref"), 7),
            broken(("evaluations", 0, "answers", "repro_fixed", "by"), 1),
            broken(("evaluations", 0, "answers", "repro_fixed", "said"), "x"),
            broken(("reason",), ["x"]),
            broken(("status",), "unheard of"),
        ]
        for path, text in [((), "[]"), *cases]:
            with (
                self.subTest(path),
                self.assertRaisesRegex(
                    ValueError, "not a symbolic-acceptance/run v1 record"
                ),
            ):
                record.loads(text, self.checklist)

    def test_any_text_an_oracle_gives_can_be_saved(self):
        reasoning = json.loads('"half a pair: \\ud800"')
        evaluation = sa.Evaluation.from_answers(
            self.checklist,
            sa.Attempt("abc", 1),
            {"repro_fixed": sa.Answer(False, reasoning)},
            [],
            "waiting",
        )
        result = sa.Result("waiting", "", (evaluation,), self.checklist.digest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "run.json")
            record.save(result, self.checklist, path)
            loaded = record.load(path, self.checklist)
        self.assertEqual(
            loaded.evaluations[0].answers["repro_fixed"].reasoning, reasoning
        )

    def test_report_text_cannot_become_markup(self):
        checklist = sa.loads(
            'title = "Fix it\\n<!-- hidden"\n'
            '[q.a]\nask = "A?"\nby = "ci"\n[[require]]\nthat = "a"\nfail = "x"\n'
        )
        sneaky = (
            "<!-- symbolic-acceptance:ask ref=x ids=a --> ```` | @team "
            "![ok](https://example.com/ok.png)"
        )
        evaluation = sa.Evaluation.from_answers(
            checklist,
            sa.Attempt("abc`\n<!--", 1),
            {"a": sa.Answer(False, sneaky, ["quoted | line"])},
            [],
        )
        text = markdown(
            sa.Result("escalated", sneaky, (evaluation,), checklist.digest), checklist
        )
        # What the run produced is inline code, which GitHub shows as it is:
        # no comment, link, image or mention, and nothing that ends a cell.
        shown = sneaky.replace("|", "\\|")
        self.assertIn(f"**escalated**: `````{shown}`````", text)
        self.assertIn(f"| `````{shown} — quoted \\| line````` |", text)
        self.assertIn("## Fix it &lt;!-- hidden\n", text)
        self.assertIn("### Attempt 1 · ``abc` <!--`` · changes", text)
        self.assertFalse([line for line in text.splitlines() if "<!--" in line[:4]])

    def test_markdown_report(self):
        text = markdown(self.result, self.checklist)
        self.assertIn("## Throttle brute-force logins (#342)", text)
        self.assertIn("**approved**", text)
        self.assertIn("| `new_test` | ❌ fail |", text)
        self.assertLess(text.index("### Attempt 2"), text.index("### Attempt 1"))
        # Only the newest attempt shows its answers; the record has the rest.
        self.assertEqual(text.count("<summary>Answers</summary>"), 1)
        self.assertNotIn("Feedback sent to the agent", text)
        self.assertEqual(text.count("What the agent says it did"), 1)
        self.assertIn("I fixed it", text)
        waiting = sa.Evaluation.from_answers(
            self.checklist, sa.Attempt("abc", 1), {}, [], "waiting", "CI: tests queued"
        )
        waiting_run = sa.Result("waiting", "", (waiting,), self.result.digest)
        self.assertIn("> `CI: tests queued`", markdown(waiting_run, self.checklist))
        sent_back = sa.Result(
            sa.ESCALATED, "", self.result.evaluations[:1], self.result.digest
        )
        self.assertIn("Feedback sent to the agent", markdown(sent_back, self.checklist))

    def test_long_evidence_is_cut_in_the_report(self):
        evaluation = sa.Evaluation.from_answers(
            self.checklist,
            sa.Attempt("abc", 1),
            {"repro_fixed": sa.Answer(False, "x" * 100_000)},
            [],
            "waiting",
        )
        result = sa.Result("waiting", "", (evaluation,), self.checklist.digest)
        self.assertLess(len(markdown(result, self.checklist)), 12_000)


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

    def test_never_passing_where_it_applies_is_a_warning(self):
        c = checklist(
            '[[require]]\nid = "r"\nwhen = "a"\nthat = { not = "a" }\nfail = "x"\n',
            ("a",),
        )
        self.assertEqual(
            [(f.level, f.code) for f in sa.lint(c)], [("warning", "never-passes")]
        )

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
        people_only = sa.loads(
            '[q.h]\nask = "H?"\nby = "human"\n[q.g]\nask = "G?"\nby = "human"\n'
            '[[require]]\nwhen = "h"\nthat = "g"\nfail = "x"\n'
        )
        self.assertNotIn("gated-by-person", self.codes(people_only))

    def test_bad_options(self):
        c = sa.loads(
            '[q.a]\nask = "A?"\nby = "ci"\nfrozen = []\n'
            '[q.b]\nask = "B?"\nby = "ci"\nrun = 3\n'
            '[q.c]\nask = "C?"\nby = "llm"\nkind = "maybe"\n'
            '[q.d]\nask = "D?"\nby = "ci"\nrun = "pytest -k \'x"\n'
            '[q.e]\nask = "E?"\nby = "ci"\nrun = []\nexpect = "maybe"\n'
            '[q.f]\nask = "F?"\nby = "ci"\ncheck = 3\n'
            '[q.g]\nask = "G?"\nby = "ci"\ncheck = ""\n'
            '[q.h]\nask = "H?"\nby = "ci"\nfrozen = ["x"]\nrun = "true"\n'
            '[q.i]\nask = "I?"\nby = "llm"\nfrozen = ["x"]\n'
            '[q.j]\nask = "J?"\nby = "human"\nkind = "absence"\n'
            '[q.k]\nask = "K?"\nby = "scanner"\nfrozen = ["x"]\n'
            '[q.l]\nask = "L?"\nby = "ci"\nrun = "./repro.sh | grep -q ok"\n'
            '[q.m]\nask = "M?"\nby = "ci"\nrun = "make test 2>&1 > log"\n'
            '[q.n]\nask = "N?"\nby = "ci"\nrun = ["sh", "-c", "a | b"]\n'
            '[q.o]\nask = "O?"\nby = "ci"\nrun = ["tr", "|", ","]\n'
            '[q.p]\nask = "P?"\nby = "ci"\nrun = "grep -q \'<b>x</b>\' f"\n'
            '[q.q]\nask = "Q?"\nby = "ci"\nrun = "python3 v.py \'>=3.11\'"\n'
            '[[require]]\nid = "r"\nthat = { all = ["a", "b", "c", "d", "e", '
            '"f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q"] }\n'
            'fail = "x"\n'
        )
        findings = [f for f in sa.lint(c) if f.code == "bad-option"]
        shell = (
            ' but runs without a shell; for one, write run = ["sh", "-c", "<script>"]'
        )
        self.assertEqual(
            [(f.where, f.message) for f in findings],
            [
                ("a", "q.a.frozen lists no paths"),
                ("b", "q.b.run must be a string or a list of strings"),
                ("c", 'q.c.kind must be "presence" or "absence"'),
                ("d", "q.d.run: No closing quotation"),
                ("e", 'q.e.expect must be "pass" or "fail"'),
                ("e", "q.e.run is empty"),
                ("f", "q.f.check must name a check run"),
                ("g", "q.g.check must name a check run"),
                ("h", "q.h has both frozen and run; split it in two"),
                ("i", 'q.i.frozen works only with by = "ci"'),
                ("j", 'q.j.kind works only with by = "llm"'),
                ("l", "q.l.run has |" + shell),
                ("m", "q.m.run has >& >" + shell),
            ],
        )
        self.assertTrue(all(f.level == "error" for f in findings))

    def test_an_option_no_oracle_reads_is_taken_for_a_typo(self):
        c = sa.loads(
            '[q.a]\nask = "A?"\nby = "ci"\nfrozen = ["tests/*"]\n'
            'alow_new = ["tests/test_*.py"]\n'
            '[q.b]\nask = "B?"\nby = "llm"\ncontxt = "Only src/ counts."\n'
            '[q.c]\nask = "C?"\nby = "human"\nowner = "security"\n'
            '[q.d]\nask = "D?"\nby = "scanner"\nseverity = "high"\n'
            '[[require]]\nid = "r"\nthat = { all = ["a", "b", "c", "d"] }\n'
            'fail = "x"\n'
        )
        findings = [f for f in sa.lint(c) if f.code == "unknown-option"]
        unread = "is read by no built-in oracle"
        self.assertEqual(
            [(f.level, f.where, f.message) for f in findings],
            [
                ("warning", "a", f"q.a.alow_new {unread}; did you mean allow_new?"),
                ("warning", "b", f"q.b.contxt {unread}; did you mean context?"),
                ("warning", "c", f"q.c.owner {unread}"),
            ],
        )

    def test_blanks_a_template_leaves_are_reported_until_filled_in(self):
        findings = [
            f for f in sa.lint(sa.loads(templates.FEATURE)) if f.code == "placeholder"
        ]
        self.assertIn(
            "q.criterion_1.check still holds the template's "
            "<test or job that decides it>",
            [f.message for f in findings],
        )
        self.assertTrue(all(f.level == "warning" for f in findings))
        # A criterion copied to make another, and blanks of the format guide.
        blank = "<acceptance criterion 2, as a yes/no question>"
        copied = sa.loads(
            f'[q.criterion_2]\nask = "{blank}"\nby = "llm"\n'
            '[q.sso]\nask = "SSO alone?"\nby = "ci"\nfrozen = ["<their paths>"]\n'
            '[q.repro]\nask = "Fixed?"\nby = "ci"\nrun = ["sh", "-c", "<script>"]\n'
            '[[require]]\nthat = "criterion_2"\nfail = "x"\n'
            '[[require]]\nthat = "sso"\nfail = "y"\n'
            '[[require]]\nthat = "repro"\nfail = "z"\n'
        )
        self.assertEqual(
            [f.message for f in sa.lint(copied) if f.code == "placeholder"],
            [
                f"q.criterion_2.ask still holds the template's {blank}",
                "q.sso.frozen[0] still holds the template's <their paths>",
                "q.repro.run[2] still holds the template's <script>",
            ],
        )
        # Other text in angle brackets is just text.
        for ask in (
            "Does the page escape <script> tags?",
            "Is lo returned when x < lo, and hi when x > hi?",
            "Does cache stay a Map<String, Long>?",
        ):
            with self.subTest(ask):
                text = sa.loads(
                    f'[q.a]\nask = "{ask}"\nby = "llm"\n'
                    '[[require]]\nthat = "a"\nfail = "x"\n'
                )
                self.assertNotIn("placeholder", [f.code for f in sa.lint(text)])

    def test_check_runs_need_frozen_workflows(self):
        unfrozen = sa.loads(
            '[q.t]\nask = "Tests pass?"\nby = "ci"\ncheck = "tests"\n'
            '[[require]]\nthat = "t"\nfail = "x"\n'
        )
        self.assertEqual(self.codes(unfrozen), ["unfrozen-workflows"])
        self.assertNotIn("unfrozen-workflows", self.codes(example()))
        for by, frozen in (
            ("ci", 'frozen = ["src/*"]'),
            ("ci", 'frozen = [".github/*"]\nallow_new = [".github/workflows/*"]'),
            ("ci", "frozen = []"),
            ("ci", 'frozen = [".github/workflows/*.yml"]'),
            ("llm", 'frozen = [".github/*"]'),
        ):
            with self.subTest(frozen, by=by):
                elsewhere = sa.loads(
                    '[q.t]\nask = "Tests pass?"\nby = "ci"\ncheck = "tests"\n'
                    f'[q.f]\nask = "Frozen?"\nby = "{by}"\n{frozen}\n'
                    '[[require]]\nid = "r"\nthat = { all = ["t", "f"] }\nfail = "x"\n'
                )
                self.assertIn("unfrozen-workflows", self.codes(elsewhere))
        # A pattern in allow_new lets in only files as deep as itself.
        shallow = sa.loads(
            '[q.t]\nask = "Tests pass?"\nby = "ci"\ncheck = "tests"\n'
            '[q.f]\nask = "Frozen?"\nby = "ci"\nfrozen = [".github/*"]\n'
            'allow_new = ["*.yml"]\n'
            '[[require]]\nid = "r"\nthat = { all = ["t", "f"] }\nfail = "x"\n'
        )
        self.assertNotIn("unfrozen-workflows", self.codes(shallow))

    def test_large_checklists_skip_exhaustive_checks(self):
        ids = [f"q{n}" for n in range(MAX_EXHAUSTIVE + 1)]
        text = "".join(f'[q.{q}]\nask = "{q}?"\nby = "ci"\n' for q in ids)
        text += (
            f'[[require]]\nid = "r"\nthat = {{ any = {json.dumps(ids)} }}\nfail = "x"\n'
        )
        findings = [(f.code, f.where) for f in sa.lint(sa.loads(text))]
        self.assertEqual(findings, [("not-exhaustive", "r"), ("not-exhaustive", None)])

    def test_duplicate_question(self):
        c = sa.loads(
            '[q.a]\nask = "Same?"\nby = "ci"\n[q.b]\nask = "same?"\nby = "ci"\n'
            '[[require]]\nid = "r"\nthat = { all = ["a", "b"] }\nfail = "x"\n'
        )
        self.assertEqual(self.codes(c), ["duplicate-question"])


if __name__ == "__main__":
    unittest.main()
