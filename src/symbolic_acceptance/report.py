"""Markdown reports of a run, for a pull request comment or an audit."""

from __future__ import annotations

from symbolic_acceptance.checklist import Checklist
from symbolic_acceptance.loop import Evaluation, Result

__all__ = ["markdown"]

_ICONS = {"pass": "✅", "fail": "❌", "n/a": "➖", "open": "⏳"}


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _evaluation(evaluation: Evaluation, checklist: Checklist) -> list[str]:
    attempt = evaluation.attempt
    lines = [f"### Attempt {attempt.number} · `{attempt.ref[:12]}` · {evaluation.status}"]
    if evaluation.reason:
        lines += ["", f"> {_cell(evaluation.reason)}"]
    lines += ["", "| Requirement | Status | If it fails |", "|---|---|---|"]
    for requirement in checklist.requirements:
        state = evaluation.requirements.get(requirement.id, "open")
        lines.append(
            f"| `{requirement.id}` | {_ICONS.get(state, '')} {state} | "
            f"{_cell(requirement.fail)} |"
        )
    if evaluation.answers:
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
                f"| `{qid}` | {answer.word} | {_cell(answer.by)} | {_cell(why)} |"
            )
        lines += ["", "</details>"]
    if evaluation.failures:
        lines += ["", "<details><summary>Feedback sent to the agent</summary>", ""]
        lines += ["```text", evaluation.feedback.rstrip(), "```", "", "</details>"]
    return lines


def markdown(result: Result, checklist: Checklist) -> str:
    """The whole run: outcome first, then each attempt with its answers."""
    title = checklist.title or "Acceptance run"
    lines = [
        f"## {title}",
        "",
        f"**{result.status}**" + (f": {result.reason}" if result.reason else ""),
        "",
        f"Checklist `{checklist.digest[:12]}` · {len(result.evaluations)} attempt(s)",
    ]
    for evaluation in result.evaluations:
        lines += ["", *_evaluation(evaluation, checklist)]
    return "\n".join(lines) + "\n"
