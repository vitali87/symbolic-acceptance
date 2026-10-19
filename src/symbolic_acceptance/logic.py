"""Three-valued evaluation of a checklist and the plan for what to ask next.

A fact is a question's answer: ``True`` (YES), ``False`` (NO), or missing.
Expressions follow Kleene logic, so a value is decided as soon as the answers
in hand force it: ``all`` is NO once any item is NO, ``any`` is YES once any
item is YES.

A requirement holds unless it applies and its ``that`` is NO:

=========  ==========  ========
``when``   ``that``    status
=========  ==========  ========
NO         anything    ``n/a``
YES/none   YES         ``pass``
YES/none   NO          ``fail``
other      other       ``open``
=========  ==========  ========

(A requirement whose ``that`` is YES passes even while ``when`` is open.)
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Literal

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    AllOf,
    Checklist,
    Expr,
    Not,
    Question,
    Ref,
    Requirement,
)

__all__ = [
    "DEFAULT_LAST",
    "FAIL",
    "NA",
    "OPEN",
    "PASS",
    "Status",
    "by_oracle",
    "facts",
    "plan",
    "status",
    "statuses",
    "value",
    "wanted",
]

Status = Literal["pass", "fail", "n/a", "open"]
PASS: Status = "pass"
FAIL: Status = "fail"
NA: Status = "n/a"
OPEN: Status = "open"

# The oracles asked last: people, by default.
DEFAULT_LAST = ("human",)

Facts = Mapping[str, bool]


def facts(answers: Mapping[str, Answer]) -> dict[str, bool]:
    """The YES/NO of each answer, which is all the logic needs."""
    return {qid: answer.is_met for qid, answer in answers.items()}


def value(expr: Expr, facts: Facts) -> bool | None:
    """The expression's value, or ``None`` while the answers don't force one."""
    if isinstance(expr, Ref):
        return facts.get(expr.id)
    if isinstance(expr, Not):
        inner = value(expr.item, facts)
        return None if inner is None else not inner
    values = [value(item, facts) for item in expr.items]
    if isinstance(expr, AllOf):
        if False in values:
            return False
        return None if None in values else True
    if True in values:
        return True
    return None if None in values else False


def status(requirement: Requirement, facts: Facts) -> Status:
    """``pass``, ``fail``, ``n/a`` or ``open``; see the module docstring."""
    applies = True if requirement.when is None else value(requirement.when, facts)
    if applies is False:
        return NA
    holds = value(requirement.that, facts)
    if holds is True:
        return PASS
    if holds is False and applies is True:
        return FAIL
    return OPEN


def statuses(checklist: Checklist, facts: Facts) -> dict[str, Status]:
    return {r.id: status(r, facts) for r in checklist.requirements}


def _deciding(expr: Expr, facts: Facts, out: dict[str, None]) -> None:
    """Unanswered questions that could still decide ``expr``, either way."""
    if value(expr, facts) is not None:
        return
    if isinstance(expr, Ref):
        out[expr.id] = None
    elif isinstance(expr, Not):
        _deciding(expr.item, facts, out)
    else:
        for item in expr.items:
            _deciding(item, facts, out)


def _unmet(expr: Expr, wanted: bool, facts: Facts, out: dict[str, None]) -> None:
    """Unanswered questions under every part of ``expr`` not yet as wanted.

    Parts that already have the wanted value are skipped, because nothing
    under them can change it. Parts that already have the other value are
    still entered, so every unmet item is reported, not only the first.
    """
    if value(expr, facts) is wanted:
        return
    if isinstance(expr, Ref):
        if expr.id not in facts:
            out[expr.id] = None
    elif isinstance(expr, Not):
        _unmet(expr.item, not wanted, facts, out)
    else:
        for item in expr.items:
            _unmet(item, wanted, facts, out)


def plan(
    checklist: Checklist,
    facts: Facts,
    *,
    last: Collection[str] = DEFAULT_LAST,
) -> list[str]:
    """The questions to ask now, in checklist order; empty once there are none.

    Questions for oracles not in ``last`` are asked first, with full coverage:
    every question that could show an applicable requirement unmet.
    ``when`` conditions are asked before the ``that`` they gate, so a gated
    check is never asked while it might not apply.

    The ``last`` questions (people, by default) come only when nothing else is
    left to ask, but the questions they gate, and no requirement has failed,
    and then only those that can still change a requirement's status as
    three-valued logic sees it. A failure is therefore never held up by a
    person, and nobody is asked a question whose answer cannot matter, unless
    a requirement reads it twice and the logic cannot see it is moot.
    """
    early: dict[str, None] = {}
    for requirement in checklist.requirements:
        if value(requirement.that, facts) is True:
            continue  # it holds whether or not it applies
        when = requirement.when
        if when is not None and value(when, facts) is not True:
            _deciding(when, facts, early)  # nothing, once `when` is NO
        else:
            _unmet(requirement.that, True, facts, early)
    first = [qid for qid in early if checklist.questions[qid].by not in last]
    if first:
        return first

    current = statuses(checklist, facts)
    if FAIL in current.values():
        return []
    late: dict[str, None] = {}
    for requirement in checklist.requirements:
        if current[requirement.id] == OPEN:
            when = requirement.when
            if when is not None and value(when, facts) is None:
                _deciding(when, facts, late)
            else:
                _deciding(requirement.that, facts, late)
    return list(late)


def wanted(checklist: Checklist) -> dict[str, frozenset[bool]]:
    """For each question, the answers that help the attempt pass.

    A requirement fails only when it applies and its ``that`` is NO, so YES
    helps inside ``that`` and NO helps inside ``when``; ``not`` flips which.
    A question read in both ways is helped by both answers.
    """
    helps: dict[str, set[bool]] = {}

    def walk(expr: Expr, good: bool) -> None:
        if isinstance(expr, Ref):
            helps.setdefault(expr.id, set()).add(good)
        elif isinstance(expr, Not):
            walk(expr.item, not good)
        else:
            for item in expr.items:
                walk(item, good)

    for requirement in checklist.requirements:
        walk(requirement.that, True)
        if requirement.when is not None:
            walk(requirement.when, False)
    return {qid: frozenset(answers) for qid, answers in helps.items()}


def by_oracle(
    checklist: Checklist, question_ids: Iterable[str]
) -> dict[str, list[Question]]:
    """Group questions by the oracle that answers them, keeping order."""
    groups: dict[str, list[Question]] = {}
    for qid in question_ids:
        question = checklist.questions[qid]
        groups.setdefault(question.by, []).append(question)
    return groups
