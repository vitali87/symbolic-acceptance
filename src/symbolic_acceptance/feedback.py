"""What failed and why, in the form the agent and reviewers read.

A failure is explained along its expression, not by listing every NO: an
``all`` that is NO is explained by the items that are NO (fix each), an
``any`` that is NO by all of its alternatives (fix any one), and ``not`` flips
what is wanted. So a NO is only reported where YES was needed, and a YES is
reported where NO was needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from symbolic_acceptance.answer import Answer, yes_no
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
from symbolic_acceptance.logic import FAIL, OPEN, Facts, Status, facts, statuses, value

__all__ = [
    "Assessment",
    "Fact",
    "Failure",
    "Group",
    "Reason",
    "assess",
    "explain",
    "render_failures",
]


@dataclass(frozen=True)
class Fact:
    """An answered question that helped decide an expression."""

    question: Question
    answer: Answer

    @property
    def actual(self) -> bool:
        return self.answer.is_met

    def facts(self) -> tuple[Fact, ...]:
        return (self,)


@dataclass(frozen=True)
class Group:
    """Why ``all``, ``any`` or ``not`` has the value ``actual``.

    ``parts`` decided it. To change the value, change each part when ``fix``
    is ``"each"``, or any one of them when it is ``"one"``; ``not`` has a
    single part and no ``fix``.
    """

    expr: AllOf | AnyOf | Not
    actual: bool
    parts: tuple[Reason, ...]
    fix: Literal["each", "one"] | None = None

    def facts(self) -> tuple[Fact, ...]:
        return tuple(fact for part in self.parts for fact in part.facts())


Reason = Fact | Group


@dataclass(frozen=True)
class Failure:
    requirement: Requirement
    reason: Reason
    because: Reason | None = None


@dataclass(frozen=True)
class Assessment:
    """What the answers in hand decide, with each requirement's status."""

    status: Literal["approved", "changes", "open"]
    requirements: Mapping[str, Status]
    failures: tuple[Failure, ...]


def _explain(
    expr: Expr, checklist: Checklist, answers: Mapping[str, Answer], known: Facts
) -> Reason:
    actual = value(expr, known)
    if actual is None:
        raise ValueError(f"{render(expr)} is not decided yet")
    if isinstance(expr, Ref):
        return Fact(checklist.questions[expr.id], answers[expr.id])
    if isinstance(expr, Not):
        return Group(expr, actual, (_explain(expr.item, checklist, answers, known),))
    # The items with the expression's own value decided it: the NO items of a
    # NO all(), the YES items of a YES any(), and every item otherwise.
    parts = tuple(
        _explain(item, checklist, answers, known)
        for item in expr.items
        if value(item, known) is actual
    )
    # Undoing a NO all() or a YES any() takes every part; the others, one.
    each = not actual if isinstance(expr, AllOf) else actual
    return Group(expr, actual, parts, "each" if each else "one")


def explain(expr: Expr, checklist: Checklist, answers: Mapping[str, Answer]) -> Reason:
    """Explain a decided expression; raises ``ValueError`` if it is open."""
    return _explain(expr, checklist, answers, facts(answers))


def assess(checklist: Checklist, answers: Mapping[str, Answer]) -> Assessment:
    """Decide what the answers in hand say about the attempt."""
    known = facts(answers)
    current = statuses(checklist, known)
    failures = tuple(
        Failure(
            requirement,
            reason=_explain(requirement.that, checklist, answers, known),
            because=(
                None
                if requirement.when is None
                else _explain(requirement.when, checklist, answers, known)
            ),
        )
        for requirement in checklist.requirements
        if current[requirement.id] == FAIL
    )
    if failures:
        return Assessment("changes", current, failures)
    return Assessment("open" if OPEN in current.values() else "approved", current, ())


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _lines(reason: Reason, wanted: bool, indent: str) -> list[str]:
    if isinstance(reason, Fact):
        question, answer = reason.question, reason.answer
        lines = [
            f"{indent}- {question.id}: needs {yes_no(wanted)}, got {answer.word} "
            f"[{answer.by or question.by}]",
            f"{indent}  Q: {question.ask}",
        ]
        if answer.reasoning:
            lines.append(f"{indent}  Why: {_one_line(answer.reasoning)}")
        lines += [f"{indent}  Evidence: {_one_line(item)}" for item in answer.evidence]
        return lines
    if isinstance(reason.expr, Not):
        return _lines(reason.parts[0], not wanted, indent)
    how = "fix each of" if reason.fix == "each" else "fix any one of"
    lines = [f"{indent}- {render(reason.expr)}: needs {yes_no(wanted)}; {how}:"]
    for part in reason.parts:
        lines += _lines(part, wanted, indent + "  ")
    return lines


def _because(reason: Reason) -> str:
    if isinstance(reason, Fact):
        return f"   Applies because {reason.question.id} is YES."
    listed = ", ".join(f"{f.question.id} is {yes_no(f.actual)}" for f in reason.facts())
    return f"   Applies because {render(reason.expr)} is YES ({listed})."


def render_failures(failures: Sequence[Failure]) -> str:
    """Plain-text feedback: every failed requirement, its message and causes."""
    if not failures:
        return "No requirement failed."
    lines: list[str] = []
    for number, failure in enumerate(failures, start=1):
        lines.append(f"{number}. {failure.requirement.id}: {failure.requirement.fail}")
        if failure.because is not None:
            lines.append(_because(failure.because))
        lines += [*_lines(failure.reason, True, "   "), ""]
    return "\n".join(lines).rstrip() + "\n"
