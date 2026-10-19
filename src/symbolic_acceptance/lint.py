"""Checks a person should see before approving a checklist.

Loading already rejects anything malformed, and the format rules out the
silent cases (an attempt that is neither approved nor told what failed). Lint
looks for checklists that load but cannot mean what their author intended:
requirements that can never fail or never pass, a checklist that can never
be approved, unknown oracles, and checks held up behind a person.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from typing import Literal

from symbolic_acceptance.checklist import Checklist, Expr, Requirement, refs, render
from symbolic_acceptance.logic import DEFAULT_LAST, FAIL, NA, status, value

__all__ = ["MAX_EXHAUSTIVE", "Issue", "lint"]

# Exhaustive checks enumerate 2**n answer sets; past this many questions they
# are skipped and reported as such.
MAX_EXHAUSTIVE = 16


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    where: str | None = None

    def __str__(self) -> str:
        where = f" [{self.where}]" if self.where else ""
        return f"{self.level}: {self.code}{where}: {self.message}"


def _assignments(ids: tuple[str, ...]) -> Iterator[dict[str, bool]]:
    for values in itertools.product((True, False), repeat=len(ids)):
        yield dict(zip(ids, values, strict=True))


def _constant(expr: Expr) -> bool | None:
    """The value ``expr`` has under every answer set, if it has one."""
    ids = refs(expr)
    if len(ids) > MAX_EXHAUSTIVE:
        return None
    seen = {value(expr, facts) for facts in _assignments(ids)}
    return seen.pop() if len(seen) == 1 else None


def _requirement_issues(requirement: Requirement) -> list[Issue]:
    issues: list[Issue] = []
    ids = requirement.questions()
    rid = requirement.id
    if len(ids) > MAX_EXHAUSTIVE:
        return [
            Issue(
                "info",
                "not-exhaustive",
                f"reads {len(ids)} questions; exhaustive checks skipped",
                rid,
            )
        ]
    if requirement.when is not None:
        constant = _constant(requirement.when)
        if constant is True:
            issues.append(
                Issue(
                    "warning",
                    "when-always",
                    f"`when` {render(requirement.when)} is always YES; drop it",
                    rid,
                )
            )
        elif constant is False:
            issues.append(
                Issue(
                    "warning",
                    "when-never",
                    f"`when` {render(requirement.when)} is never YES, so this "
                    "requirement never applies",
                    rid,
                )
            )
    results = [status(requirement, facts) for facts in _assignments(ids)]
    applicable = [result for result in results if result != NA]
    if applicable and all(result == FAIL for result in applicable):
        issues.append(
            Issue(
                "error",
                "never-passes",
                "fails under every answer set where it applies, so it can never be met",
                rid,
            )
        )
    elif FAIL not in results:
        issues.append(
            Issue(
                "warning",
                "never-fails",
                f"holds under every answer set, so it checks nothing: "
                f"{render(requirement.that)}",
                rid,
            )
        )
    return issues


def lint(
    checklist: Checklist,
    *,
    oracles: Collection[str] | None = None,
    last: Collection[str] = DEFAULT_LAST,
) -> list[Issue]:
    """Return issues, errors first. ``oracles`` lists the oracle names known."""
    issues: list[Issue] = []
    used = set(checklist.used())

    for question in checklist.questions.values():
        if question.id not in used:
            issues.append(
                Issue(
                    "warning",
                    "unused-question",
                    "no requirement reads this question",
                    question.id,
                )
            )
        if oracles is not None and question.by not in oracles:
            issues.append(
                Issue(
                    "error",
                    "unknown-oracle",
                    f"by = {question.by!r}; known oracles: "
                    f"{', '.join(sorted(oracles))}",
                    question.id,
                )
            )
    asks: dict[str, str] = {}
    for question in checklist.questions.values():
        key = " ".join(question.ask.lower().split())
        if key in asks:
            issues.append(
                Issue(
                    "warning",
                    "duplicate-question",
                    f"asks the same as q.{asks[key]}",
                    question.id,
                )
            )
        else:
            asks[key] = question.id

    for requirement in checklist.requirements:
        issues.extend(_requirement_issues(requirement))
        if requirement.when is not None:
            gated_by = [
                qid
                for qid in refs(requirement.when)
                if checklist.questions[qid].by in last
            ]
            waiting = [
                qid
                for qid in refs(requirement.that)
                if checklist.questions[qid].by not in last
            ]
            if gated_by and waiting:
                issues.append(
                    Issue(
                        "warning",
                        "gated-by-person",
                        f"automated checks ({', '.join(waiting)}) wait for "
                        f"{', '.join(gated_by)} to be answered by a person",
                        requirement.id,
                    )
                )

    ordered = checklist.used()
    if len(ordered) <= MAX_EXHAUSTIVE:
        approvable = any(
            all(status(r, facts) != FAIL for r in checklist.requirements)
            for facts in _assignments(ordered)
        )
        if not approvable:
            issues.append(
                Issue(
                    "error",
                    "never-approved",
                    "no answer set satisfies every requirement at once",
                )
            )
    else:
        issues.append(
            Issue(
                "info",
                "not-exhaustive",
                f"{len(ordered)} questions; the whole-checklist check was skipped",
            )
        )
    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda issue: rank[issue.level])
