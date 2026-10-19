"""Markdown reports of a run, for a pull request comment or an audit."""

from __future__ import annotations

import re

from symbolic_acceptance.checklist import Checklist
from symbolic_acceptance.loop import Evaluation, Result

__all__ = ["markdown"]

_ICONS = {"pass": "✅", "fail": "❌", "n/a": "➖", "open": "⏳"}
# The most of a table cell, and of the feedback, that a report shows; the
# run record keeps all of it.
CELL = 500
FEEDBACK = 10_000


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(text: str) -> str:
    """One table cell of the checklist's text: a single line, with no HTML
    and nothing that ends the cell; long text is cut."""
    return _cut(" ".join(text.split()), CELL).replace("<", "&lt;").replace("|", "\\|")


def _inert(text: str) -> str:
    """Text a run produced, which the agent can shape, as inline code: one
    line, cut, and shown as it is, with no links, images, mentions or HTML,
    and nothing that ends a table cell."""
    line = _cut(" ".join(text.split()), CELL)
    if not line:
        return ""
    longest = max((len(run) for run in re.findall(r"`+", line)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if "`" in (line[0], line[-1]) else ""
    return f"{fence}{pad}{line}{pad}{fence}".replace("|", "\\|")


def _fenced(text: str) -> list[str]:
    """A code block whose fence no run of backticks in ``text`` can close."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", text, fence]


def _evaluation(
    evaluation: Evaluation, checklist: Checklist, detailed: bool
) -> list[str]:
    attempt = evaluation.attempt
    ref = _inert(attempt.ref[:12])
    lines = [f"### Attempt {attempt.number} · {ref} · {evaluation.status}"]
    if evaluation.reason:
        lines += ["", f"> {_inert(evaluation.reason)}"]
    lines += ["", "| Requirement | Status | If it fails |", "|---|---|---|"]
    for requirement in checklist.requirements:
        state = evaluation.requirements[requirement.id]
        lines.append(
            f"| `{requirement.id}` | {_ICONS[state]} {state} | "
            f"{_cell(requirement.fail)} |"
        )
    if detailed and evaluation.answers:
        lines += [
            "",
            "<details><summary>Answers</summary>",
            "",
            "| Question | Answer | By | Why |",
            "|---|---|---|---|",
        ]
        for qid, answer in evaluation.answers.items():
            why = answer.reasoning
            if answer.evidence:
                why += " — " + "; ".join(answer.evidence)
            lines.append(
                f"| `{qid}` | {answer.word} | {_inert(answer.by)} | {_inert(why)} |"
            )
        lines += ["", "</details>"]
    if detailed and attempt.claims:
        lines += ["", "<details><summary>What the agent says it did</summary>", ""]
        lines += [*_fenced(_cut(attempt.claims, FEEDBACK)), "", "</details>"]
    if detailed and evaluation.failures:
        lines += ["", "<details><summary>Feedback sent to the agent</summary>", ""]
        feedback = _cut(evaluation.feedback.rstrip(), FEEDBACK)
        lines += [*_fenced(feedback), "", "</details>"]
    return lines


def markdown(result: Result, checklist: Checklist) -> str:
    """The run: its outcome, then each attempt, newest first; only the newest
    shows its answers and feedback."""
    title = checklist.title or "Acceptance run"
    lines = [
        f"## {title}",
        "",
        f"**{result.status}**"
        + (f": {_inert(result.reason)}" if result.reason else ""),
        "",
        f"Checklist `{checklist.digest[:12]}` · {len(result.evaluations)} attempt(s)",
    ]
    newest = result.last
    for evaluation in reversed(result.evaluations):
        lines += ["", *_evaluation(evaluation, checklist, evaluation is newest)]
    return "\n".join(lines) + "\n"
