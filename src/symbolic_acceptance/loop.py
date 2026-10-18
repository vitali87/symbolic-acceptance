"""Drive an agent until its attempt is approved or a person has to step in.

``evaluate`` checks one attempt against a checklist; ``run`` repeats
agent-then-evaluate, handing every failure back to the agent, and stops when
the attempt is approved or escalates when progress stalls.

Oracles are ``(questions, context) -> {question_id: Answer}``, sync or async.
An oracle may answer some of its questions and leave the rest for the next
round. It raises :class:`Pending` when it has asked someone and must wait,
which pauses the run so it can be resumed later (for example, from the next
GitHub event), and :class:`Escalate` to hand the attempt to a person.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Checklist, Question, render
from symbolic_acceptance.feedback import Failure, assess, render_failures
from symbolic_acceptance.logic import NA, PASS, by_oracle, plan

__all__ = [
    "APPROVED",
    "CHANGES",
    "DEFAULT_LAST",
    "ESCALATED",
    "RUNNING",
    "WAITING",
    "Attempt",
    "Context",
    "Escalate",
    "Evaluation",
    "OracleError",
    "Pending",
    "Result",
    "Task",
    "evaluate",
    "run",
]

APPROVED = "approved"
CHANGES = "changes"
WAITING = "waiting"
ESCALATED = "escalated"
RUNNING = "running"

DEFAULT_LAST = ("human",)


@dataclass(frozen=True)
class Attempt:
    """One commit to judge.

    Oracles must read facts from ``ref`` (CI results for that commit, the diff
    computed by the runner). ``claims`` is what the agent says it did; it is
    shown to people but is never evidence.
    """

    ref: str
    number: int = 0
    diff: str = ""
    claims: str = ""
    base: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "number": self.number,
            "diff": self.diff,
            "claims": self.claims,
            "base": self.base,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Attempt:
        return cls(
            ref=data["ref"],
            number=data.get("number", 0),
            diff=data.get("diff", ""),
            claims=data.get("claims", ""),
            base=data.get("base", ""),
        )


@dataclass(frozen=True)
class Context:
    """What an oracle gets besides its questions."""

    checklist: Checklist
    attempt: Attempt
    answers: Mapping[str, Answer]
    issue: str = ""


class Pending(Exception):
    """An oracle has asked (a person, a CI run still going) and must wait.

    Answers it already has can be passed along and are kept.
    """

    def __init__(
        self,
        message: str = "waiting for an answer",
        *,
        answers: Mapping[str, Answer] | None = None,
    ) -> None:
        super().__init__(message)
        self.answers = dict(answers or {})


class Escalate(Exception):
    """An oracle hands the attempt to a person, e.g. on suspected tampering."""


class OracleError(RuntimeError):
    """An oracle broke its contract: no answers, or answers it was not asked."""


Oracle = Callable[[Sequence[Question], Context], Any]


@dataclass(frozen=True)
class Evaluation:
    """One attempt, checked: every answer, and what they decided."""

    attempt: Attempt
    status: str
    answers: Mapping[str, Answer]
    rounds: tuple[tuple[str, ...], ...]
    requirements: Mapping[str, str]
    failures: tuple[Failure, ...]
    reason: str = ""
    waiting_on: tuple[str, ...] = ()

    @property
    def failing(self) -> frozenset[str]:
        return frozenset(f.requirement.id for f in self.failures)

    @property
    def feedback(self) -> str:
        return render_failures(self.failures)


@dataclass(frozen=True)
class Task:
    """What the agent is asked to do for its next attempt."""

    issue: str
    checklist: Checklist
    number: int
    feedback: tuple[Failure, ...] = ()
    previous: Attempt | None = None

    def prompt(self) -> str:
        """Instructions for a coding agent: the issue, the bar, the feedback."""
        checklist = self.checklist
        lines = []
        if checklist.title:
            lines += [f"# {checklist.title}", ""]
        if self.issue:
            lines += ["## Issue", "", self.issue.strip(), ""]
        lines += [
            "## Definition of done",
            "",
            "Your change is accepted only when every requirement below holds.",
            "Each question is answered by CI, by an independent reviewer or by a",
            "person, from your commit itself. What you write about the change is",
            "not evidence. Do not edit this checklist, CI configuration or",
            "existing tests the checks rely on; add new tests instead.",
            "",
        ]
        for requirement in checklist.requirements:
            gate = ""
            if requirement.when is not None:
                gate = f" (only when {render(requirement.when)} is YES)"
            lines.append(
                f"- {requirement.id}: {render(requirement.that)} must be YES{gate}."
                f" Otherwise: {requirement.fail}"
            )
        lines += ["", "Questions:", ""]
        for qid in checklist.used():
            question = checklist.questions[qid]
            lines.append(f"- {qid} [{question.by}]: {question.ask}")
        if self.feedback:
            ref = self.previous.ref if self.previous else ""
            lines += [
                "",
                f"## Attempt {self.number - 1} ({ref}) was not accepted",
                "",
                render_failures(self.feedback).rstrip(),
            ]
        lines += [
            "",
            f"Make attempt {self.number}: change the code and commit. "
            "The commit is the attempt.",
        ]
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Result:
    """``approved``, ``escalated`` (a person must decide) or ``waiting``.

    ``running`` is only seen by an ``on_progress`` listener mid-run.
    """

    status: str
    reason: str
    evaluations: tuple[Evaluation, ...]
    digest: str

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        return tuple(e.attempt for e in self.evaluations)

    @property
    def last(self) -> Evaluation | None:
        return self.evaluations[-1] if self.evaluations else None


Agent = Callable[[Task], Any]
DiffSource = Callable[[Attempt], Any]
Listener = Callable[[Result], Any]


async def _call(function: Callable[..., Any], *args: Any) -> Any:
    """Await an async callable; run a sync one in a thread."""
    method = type(function).__call__ if not inspect.isroutine(function) else None
    if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(method):
        return await function(*args)
    result = await asyncio.to_thread(function, *args)
    if inspect.isawaitable(result):
        result = await result
    return result


def _check_oracles(checklist: Checklist, oracles: Mapping[str, Oracle]) -> None:
    missing = sorted(
        {checklist.questions[qid].by for qid in checklist.used()} - set(oracles)
    )
    if missing:
        raise ValueError(f"no oracle for: {', '.join(missing)}")


def _accepted(name: str, asked: Collection[str], result: Any) -> dict[str, Answer]:
    if not isinstance(result, Mapping):
        raise OracleError(
            f"oracle {name!r} returned {type(result).__name__}, expected a mapping "
            "of question id to Answer"
        )
    unasked = sorted(set(result) - set(asked))
    if unasked:
        raise OracleError(
            f"oracle {name!r} answered questions it was not asked: {unasked}"
        )
    answers: dict[str, Answer] = {}
    for qid, answer in result.items():
        if not isinstance(answer, Answer):
            raise OracleError(
                f"oracle {name!r} answered {qid!r} with {type(answer).__name__}, "
                "expected Answer"
            )
        answers[qid] = answer if answer.by else replace(answer, by=name)
    return answers


def _finish(
    checklist: Checklist,
    attempt: Attempt,
    answers: dict[str, Answer],
    rounds: list[tuple[str, ...]],
    status: str | None = None,
    reason: str = "",
    waiting_on: tuple[str, ...] = (),
) -> Evaluation:
    assessment = assess(checklist, answers)
    if status is None:
        # plan() only stops once a requirement fails or none is left open.
        assert assessment.status != "open", "evaluation stopped while undecided"
        status = APPROVED if assessment.status == "approved" else CHANGES
    return Evaluation(
        attempt=attempt,
        status=status,
        answers=MappingProxyType(dict(answers)),
        rounds=tuple(rounds),
        requirements=MappingProxyType(dict(assessment.requirements)),
        failures=assessment.failures,
        reason=reason,
        waiting_on=waiting_on,
    )


async def evaluate(
    checklist: Checklist,
    attempt: Attempt,
    oracles: Mapping[str, Oracle],
    *,
    last: Collection[str] = DEFAULT_LAST,
    answers: Mapping[str, Answer] | None = None,
    rounds: Sequence[tuple[str, ...]] = (),
    issue: str = "",
) -> Evaluation:
    """Ask the oracles until the attempt is decided, or an oracle must wait.

    ``last`` names the oracles asked only once nothing else is left and no
    requirement has failed (see :func:`symbolic_acceptance.logic.plan`).
    Pass the ``answers`` and ``rounds`` of a waiting evaluation to resume it.
    """
    _check_oracles(checklist, oracles)
    deferred = frozenset(last)
    known = dict(answers or {})
    done = list(rounds)
    while True:
        facts = {qid: answer.is_met for qid, answer in known.items()}
        ask = plan(checklist, facts, last=lambda q: q.by in deferred)
        if not ask:
            return _finish(checklist, attempt, known, done)
        groups = by_oracle(checklist, ask)
        context = Context(checklist, attempt, MappingProxyType(dict(known)), issue)
        names = list(groups)
        results = await asyncio.gather(
            *(_call(oracles[name], tuple(groups[name]), context) for name in names),
            return_exceptions=True,
        )
        got: dict[str, Answer] = {}
        waiting: list[str] = []
        notes: list[str] = []
        escalations: list[str] = []
        for name, result in zip(names, results, strict=True):
            asked = [q.id for q in groups[name]]
            if isinstance(result, Pending):
                partial = _accepted(name, asked, result.answers)
                got.update(partial)
                waiting += [qid for qid in asked if qid not in partial]
                notes.append(f"{name}: {result}")
            elif isinstance(result, Escalate):
                escalations.append(f"{name}: {result}")
            elif isinstance(result, BaseException):
                raise result
            else:
                answered = _accepted(name, asked, result)
                if not answered:
                    raise OracleError(f"oracle {name!r} answered none of {asked}")
                got.update(answered)
        if got:
            known.update(got)
            done.append(tuple(got))
        if escalations:
            return _finish(
                checklist, attempt, known, done, ESCALATED, "; ".join(escalations)
            )
        if waiting:
            return _finish(
                checklist,
                attempt,
                known,
                done,
                WAITING,
                "; ".join(notes),
                tuple(waiting),
            )


def _verdict(
    evaluations: Sequence[Evaluation], max_attempts: int, stop_on_regression: bool
) -> tuple[str, str] | None:
    """Why the run stops after the latest evaluation, or ``None`` to go on."""
    latest = evaluations[-1]
    if latest.status == APPROVED:
        return APPROVED, "every requirement holds"
    if latest.status in (WAITING, ESCALATED):
        return latest.status, latest.reason
    number = latest.attempt.number
    failing = latest.failing
    if stop_on_regression:
        for earlier in evaluations[:-1]:
            regressed = sorted(
                rid for rid in failing if earlier.requirements.get(rid) in (PASS, NA)
            )
            if regressed:
                return ESCALATED, (
                    f"regression: {', '.join(regressed)} held in attempt "
                    f"{earlier.attempt.number} and fails in attempt {number}"
                )
    if len(evaluations) >= 2 and failing == evaluations[-2].failing:
        return ESCALATED, (
            f"no progress: attempt {number} fails the same requirements as "
            f"attempt {evaluations[-2].attempt.number} ({', '.join(sorted(failing))})"
        )
    if len(evaluations) >= max_attempts:
        return ESCALATED, f"attempt cap: {max_attempts} attempts without approval"
    return None


async def run(
    checklist: Checklist,
    *,
    agent: Agent,
    oracles: Mapping[str, Oracle],
    issue: str = "",
    max_attempts: int = 5,
    last: Collection[str] = DEFAULT_LAST,
    diff: DiffSource | None = None,
    resume: Result | None = None,
    on_progress: Listener | None = None,
    stop_on_regression: bool = True,
) -> Result:
    """Run the agent until approval or escalation.

    The agent is ``(task) -> Attempt | ref``. ``diff`` computes an attempt's
    diff from its ref, so the judge never reads a diff the agent supplied.
    ``on_progress`` gets the run so far after each evaluation, e.g. to save
    it with :func:`symbolic_acceptance.record.save`. Pass a saved ``Result``
    as ``resume`` to continue it.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    _check_oracles(checklist, oracles)
    evaluations: list[Evaluation] = []

    async def finished(evaluation: Evaluation) -> None:
        evaluations.append(evaluation)
        if on_progress is not None:
            progress = Result(RUNNING, "", tuple(evaluations), checklist.digest)
            await _call(on_progress, progress)

    if resume is not None:
        if resume.digest != checklist.digest:
            raise ValueError(
                "the checklist changed since this run started; start a new run"
            )
        evaluations = list(resume.evaluations)
        if evaluations and evaluations[-1].status == WAITING:
            waiting = evaluations.pop()
            await finished(
                await evaluate(
                    checklist,
                    waiting.attempt,
                    oracles,
                    last=last,
                    answers=waiting.answers,
                    rounds=waiting.rounds,
                    issue=issue,
                )
            )

    while True:
        if evaluations:
            verdict = _verdict(evaluations, max_attempts, stop_on_regression)
            if verdict is not None:
                return Result(verdict[0], verdict[1], tuple(evaluations), checklist.digest)
        previous = evaluations[-1] if evaluations else None
        task = Task(
            issue=issue,
            checklist=checklist,
            number=len(evaluations) + 1,
            feedback=previous.failures if previous else (),
            previous=previous.attempt if previous else None,
        )
        produced = await _call(agent, task)
        if isinstance(produced, Attempt):
            attempt = replace(produced, number=task.number)
        elif isinstance(produced, str) and produced:
            attempt = Attempt(ref=produced, number=task.number)
        else:
            raise TypeError("the agent must return an Attempt or a non-empty ref")
        if diff is not None:
            attempt = replace(attempt, diff=await _call(diff, attempt))
        await finished(
            await evaluate(checklist, attempt, oracles, last=last, issue=issue)
        )

