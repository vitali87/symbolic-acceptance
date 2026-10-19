"""A whole run with real git, a real agent command and real CI commands."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import repo

import symbolic_acceptance as sa
from symbolic_acceptance.agents import CommandAgent
from symbolic_acceptance.checks import Checks, Commands

AGENT = """\
import re, sys
prompt = sys.stdin.read()
number = int(re.search(r"Make attempt (\\d+)", prompt).group(1))
# The first attempt is wrong; the second follows the feedback.
feedback = prompt.partition("was not accepted")[2]
fixed = "prints_v2: needs YES, got NO" in feedback
open("app.py", "w").write("print('v2')\\n" if fixed else "print('v1.5')\\n")
print(f"attempt {number}: I am confident this is correct")
"""


def checklist_text(python):
    check = (
        "import subprocess, sys; "
        "out = subprocess.run([sys.executable, 'app.py'], capture_output=True, text=True).stdout; "
        "raise SystemExit(out.strip() != 'v2')"
    )
    return f"""
title = "Print v2"

[q.prints_v2]
ask = "Does app.py print v2?"
by = "ci"
run = {json.dumps([python, "-c", check])}

[q.frozen_untouched]
ask = "Is the checklist left alone?"
by = "ci"
check = "frozen"
paths = ["acceptance.toml"]

[q.small_change]
ask = "Does the change only touch app.py?"
by = "llm"

[[require]]
that = "prints_v2"
fail = "app.py does not print v2"

[[require]]
that = "frozen_untouched"
fail = "The checklist was changed"

[[require]]
that = "small_change"
fail = "The change touches more than app.py"
"""


class EndToEndTests(unittest.TestCase):
    def test_failed_attempt_is_fixed_from_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo(tmp)
            checklist = sa.loads(checklist_text(sys.executable))
            seen_claims = []

            async def judge(questions, context):
                # Stands in for Judge: reads only the runner-computed diff.
                seen_claims.append(context.attempt.claims)
                files = [
                    line
                    for line in context.attempt.diff.splitlines()
                    if line.startswith("diff --git")
                ]
                return {
                    q.id: sa.Answer(
                        files == ["diff --git a/app.py b/app.py"], f"files: {files}"
                    )
                    for q in questions
                }

            saved = Path(tmp, "..", f"{Path(tmp).name}-run.json").resolve()
            try:
                result = asyncio.run(
                    sa.run(
                        checklist,
                        agent=CommandAgent([sys.executable, "-c", AGENT], cwd=tmp),
                        oracles={
                            "ci": Checks(Commands(tmp)),
                            "llm": judge,
                            "human": lambda q, c: {},
                        },
                        max_attempts=3,
                        on_progress=lambda run: sa.record.save(run, checklist, saved),
                    )
                )
                self.assertEqual(result.status, sa.APPROVED, result.reason)
                first, second = result.evaluations
                self.assertEqual(first.failing, {"prints_v2"})
                self.assertIn("exit 1", first.answers["prints_v2"].reasoning)
                self.assertTrue(second.answers["prints_v2"].is_met)
                self.assertTrue(second.answers["frozen_untouched"].is_met)
                self.assertEqual(
                    seen_claims[0], "attempt 1: I am confident this is correct"
                )
                self.assertEqual(
                    sa.record.load(saved, checklist).evaluations[-1].status, sa.APPROVED
                )
            finally:
                saved.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
