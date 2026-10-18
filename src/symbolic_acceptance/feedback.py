"""What failed and why, in the form the agent and reviewers read.

A failure is explained along its expression, not by listing every NO: an
``all`` that is NO is explained by the items that are NO (fix each), an
``any`` that is NO by all of its alternatives (fix any one), and ``not`` flips
what is wanted. So a NO is only reported where YES was needed, and a YES is
reported where NO was needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    AllOf,
    AnyOf,
    Checklist,
    Expr,
    Not,
    Question,
    Ref,
    Requirement,
    render,
)
from symbolic_acceptance.logic import FAIL, OPEN, statuses, value

__all__ = ["Assessment", "Failure", "Reason", "assess", "explain", "render_failures"]


@dataclass(frozen=True)
class Reason:
    """Why ``expr`` has the value ``actual``.

    ``children`` are the parts that decided it. To change the value, change
    each child when ``fix`` is ``"each"``, or any one when it is ``"one"``.
    """

    expr: Expr
    actual: bool
    question: Question | None = None
    answer: Answer | None = None
    fix: str | None = None
    children: tuple[Reason, ...] = ()

    def leaves(self) -> tuple[Reason, ...]:
        if self.question is not None:
            return (self,)
        return tuple(leaf for child in self.children for leaf in child.leaves())


@dataclass(frozen=True)
class Failure:
    requirement: Requirement
    reason: Reason
    because: Reason | None = None


@dataclass(frozen=True)
class Assessment:
    """``approved``, ``changes`` or ``open``, with each requirement's status."""

    status: str
    requirements: Mapping[str, str]
    failures: tuple[Failure, ...]


def _facts(answers: Mapping[str, Answer]) -> dict[str, bool]:
    return {qid: answer.is_met for qid, answer in answers.items()}


def explain(
    expr: Expr, checklist: Checklist, answers: Mapping[str, Answer]
) -> Reason:
    """Explain a decided expression; raises ``ValueError`` if it is open."""
    facts = _facts(answers)
    actual = value(expr, facts)
    if actual is None:
        raise ValueError(f"{render(expr)} is not decided yet")
    if isinstance(expr, Ref):
        return Reason(
            expr, actual, question=checklist.questions[expr.id], answer=answers[expr.id]
        )
    if isinstance(expr, Not):
        return Reason(expr, actual, children=(explain(expr.item, checklist, answers),))
    # The items with the expression's own value decided it: the NO items of a
    # NO all(), the YES items of a YES any(), and every item otherwise.
    deciding = [item for item in expr.items if value(item, facts) is actual]
    if isinstance(expr, AllOf):
        fix = "one" if actual else "each"
    else:
        assert isinstance(expr, AnyOf)
        fix = "each" if actual else "one"
    return Reason(
        expr,
        actual,
        fix=fix,
        children=tuple(explain(item, checklist, answers) for item in deciding),
    )


def assess(checklist: Checklist, answers: Mapping[str, Answer]) -> Assessment:
    """Decide what the answers in hand say about the attempt."""
    facts = _facts(answers)
    current = statuses(checklist, facts)
    failures = []
    for requirement in checklist.requirements:
        if current[requirement.id] != FAIL:
            continue
        because = None
        if requirement.when is not None:
            because = explain(requirement.when, checklist, answers)
        failures.append(
            Failure(
                requirement,
                reason=explain(requirement.that, checklist, answers),
                because=because,
            )
        )
    if failures:
        result = "changes"
    elif OPEN in current.values():
        result = "open"
    else:
        result = "approved"
    return Assessment(result, current, tuple(failures))


def _word(flag: bool) -> str:
    return "YES" if flag else "NO"


def _leaf_lines(reason: Reason, wanted: bool, indent: str) -> list[str]:
    question, answer = reason.question, reason.answer
    assert question is not None and answer is not None
    who = answer.by or question.by
    lines = [
        f"{indent}- {question.id}: needs {_word(wanted)}, got {answer.word} [{who}]",
        f"{indent}  Q: {question.ask}",
    ]
    if answer.reasoning:
        lines.append(f"{indent}  Why: {_one_line(answer.reasoning)}")
    for item in answer.evidence:
        lines.append(f"{indent}  Evidence: {_one_line(item)}")
    return lines


def _reason_lines(reason: Reason, wanted: bool, indent: str) -> list[str]:
    if reason.question is not None:
        return _leaf_lines(reason, wanted, indent)
    if isinstance(reason.expr, Not):
        return _reason_lines(reason.children[0], not wanted, indent)
    how = "fix each of" if reason.fix == "each" else "fix any one of"
    lines = [f"{indent}- {render(reason.expr)}: needs {_word(wanted)}; {how}:"]
    for child in reason.children:
        lines.extend(_reason_lines(child, wanted, indent + "  "))
    return lines


def _one_line(text: str) -> str:
    return " ".join(text.split())


def render_failures(failures: tuple[Failure, ...] | list[Failure]) -> str:
    """Plain-text feedback: every failed requirement, its message and causes."""
    if not failures:
        return "No requirement failed."
    lines: list[str] = []
    for number, failure in enumerate(failures, start=1):
        requirement = failure.requirement
        lines.append(f"{number}. {requirement.id}: {requirement.fail}")
        if failure.because is not None:
            because = failure.because
            text = f"   Applies because {render(because.expr)} is YES"
            if not isinstance(because.expr, Ref):
                facts = ", ".join(
                    f"{leaf.question.id} is {_word(leaf.actual)}"
                    for leaf in because.leaves()
                    if leaf.question is not None
                )
                text += f" ({facts})"
            lines.append(text + ".")
        lines.extend(_reason_lines(failure.reason, True, "   "))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
