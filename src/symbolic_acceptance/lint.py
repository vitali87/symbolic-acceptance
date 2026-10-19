"""Checks a person should see before approving a checklist.

Loading already rejects anything malformed, and the format rules out the
silent cases (an attempt that is neither approved nor told what failed). Lint
looks for checklists that load but cannot mean what their author intended:

* errors: unknown oracles (``unknown-oracle``), options the built-in oracles
  cannot read (``bad-option``), requirements that can never pass
  (``never-passes``), and checklists that can never be approved
  (``never-approved``);
* warnings: requirements that can never fail (``never-fails``), ``when``
  conditions that are always or never YES (``when-always``, ``when-never``),
  checks held up behind a person (``gated-by-person``), questions nothing
  reads (``unused-question``) or asked twice (``duplicate-question``), and
  CI check runs an attempt could fake (``unfrozen-workflows``);
* info: checklists too large for the exhaustive checks (``not-exhaustive``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Literal

from symbolic_acceptance.checklist import (
    Checklist,
    Expr,
    Question,
    Requirement,
    refs,
    render,
)
from symbolic_acceptance.checks import command, frozen_paths, matches
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
    seen = {value(expr, facts) for facts in _assignments(refs(expr))}
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


def _option_issues(question: Question) -> list[Issue]:
    """Options the built-in oracles read, checked before any run needs them."""
    options = question.options
    reads: list[Callable[[], object]] = [
        lambda: question.strings("allow_new"),
        lambda: question.choice("expect", "pass", "fail"),
        lambda: question.choice("kind", "presence", "absence"),
    ]
    if "frozen" in options:
        reads.append(lambda: frozen_paths(question))
    if "run" in options:
        reads.append(lambda: command(question))
    problems: list[str] = []
    for read in reads:
        try:
            read()
        except ValueError as error:
            problems.append(str(error))
    if not isinstance(options.get("check", ""), str):
        problems.append(f"q.{question.id}.check must be a string")
    return [Issue("error", "bad-option", problem, question.id) for problem in problems]


def _freezes(question: Question, path: str) -> bool:
    """Whether ``question`` is a frozen check that fails a new file at ``path``."""
    if "frozen" not in question.options:
        return False
    try:
        frozen, allowed = frozen_paths(question), question.strings("allow_new")
    except ValueError:
        return False  # reported as a bad option
    return matches(path, frozen) and not matches(path, allowed)


def _check_runs_issues(checklist: Checklist) -> list[Issue]:
    """CI check runs can be faked by a workflow the attempt adds."""
    used = [checklist.questions[qid] for qid in checklist.used()]
    probe = ".github/workflows/added-by-the-attempt.yml"
    if any(_freezes(q, probe) for q in used):
        return []
    return [
        Issue(
            "warning",
            "unfrozen-workflows",
            f"answered by the CI check run {q.options['check']!r}, but no frozen "
            "check covers .github/workflows, so an attempt could add a workflow "
            "that reports a check of that name",
            q.id,
        )
        for q in used
        if q.by == "ci"
        and "run" not in q.options
        and "frozen" not in q.options
        and isinstance(q.options.get("check"), str)
    ]


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
        issues.extend(_option_issues(question))
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

    issues.extend(_check_runs_issues(checklist))

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
