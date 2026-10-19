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
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, ParamSpec

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Checklist, Question, render
from symbolic_acceptance.feedback import Failure, assess, render_failures
from symbolic_acceptance.logic import DEFAULT_LAST, PASS, Status, by_oracle, facts, plan

__all__ = [
    "APPROVED",
    "CHANGES",
    "ESCALATED",
    "RUNNING",
    "WAITING",
    "Agent",
    "Attempt",
    "Context",
    "Escalate",
    "Evaluation",
    "EvaluationStatus",
    "Oracle",
    "OracleError",
    "Pending",
    "Result",
    "RunStatus",
    "Task",
    "evaluate",
    "run",
]

EvaluationStatus = Literal["approved", "changes", "waiting", "escalated"]
RunStatus = Literal["approved", "escalated", "waiting", "running"]

APPROVED: Literal["approved"] = "approved"
CHANGES: Literal["changes"] = "changes"
WAITING: Literal["waiting"] = "waiting"
ESCALATED: Literal["escalated"] = "escalated"
RUNNING: Literal["running"] = "running"


@dataclass(frozen=True)
class Attempt:
    """One commit to judge.

    Oracles must read facts from ``ref`` (CI results for that commit, the diff
    computed by the runner); ``diff`` is ``None`` when nothing computed one.
    ``claims`` is what the agent says it did; it is shown to people but is
    never evidence.
    """

    ref: str
    number: int = 0
    diff: str | None = None
    claims: str = ""


@dataclass(frozen=True)
class Context:
    """What an oracle gets besides its questions."""

    checklist: Checklist
    attempt: Attempt
    answers: Mapping[str, Answer]
    issue: str = ""


class Pending(Exception):
    """An oracle has asked (a person, a CI run still going) and must wait.

    Answers it already has can be passed along. They decide the attempt if
    they can, as a person's NO may; if not, the oracle is asked all its
    questions again when the run resumes, so a correction made meanwhile (a
    person's later reply) counts.
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
    """An oracle cannot answer: it broke its contract (no answers, or answers
    it was not asked), or the attempt lacks what it needs."""


Answers = Mapping[str, Answer]
Oracle = Callable[[Sequence[Question], Context], Answers | Awaitable[Answers]]


@dataclass(frozen=True)
class Evaluation:
    """One attempt, checked: every answer, and what they decided."""

    attempt: Attempt
    status: EvaluationStatus
    answers: Mapping[str, Answer]
    rounds: tuple[tuple[str, ...], ...]
    requirements: Mapping[str, Status]
    failures: tuple[Failure, ...]
    reason: str = ""
    waiting_on: tuple[str, ...] = ()

    @classmethod
    def from_answers(
        cls,
        checklist: Checklist,
        attempt: Attempt,
        answers: Mapping[str, Answer],
        rounds: Sequence[tuple[str, ...]],
        paused: Literal["waiting", "escalated"] | None = None,
        reason: str = "",
        waiting_on: tuple[str, ...] = (),
    ) -> Evaluation:
        """Decide an attempt from its answers.

        Approval and change requests follow from the answers alone; only a
        pause or an escalation, which they don't show, is given as ``paused``.
        """
        assessment = assess(checklist, answers)
        if paused is not None:
            status: EvaluationStatus = paused
        elif assessment.status == "open":
            raise ValueError(f"attempt {attempt.number} is not decided by its answers")
        else:
            status = assessment.status
        return cls(
            attempt=attempt,
            status=status,
            answers=MappingProxyType(dict(answers)),
            rounds=tuple(rounds),
            requirements=MappingProxyType(dict(assessment.requirements)),
            failures=assessment.failures,
            reason=reason,
            waiting_on=waiting_on,
        )

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
            "not evidence.",
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
            if "frozen" in question.options:
                allowed = _globs(question, "allow_new")
                but = f" (new files matching {allowed} excepted)" if allowed else ""
                frozen = _globs(question, "frozen")
                lines.append(f"  YES only when no file matching {frozen} changes{but}.")
            if question.context:
                lines.append(f"  {' '.join(question.context.split())}")
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
            (
                f"Make attempt {self.number}: change the code and commit. "
                "The commit is the attempt."
            ),
        ]
        return "\n".join(lines) + "\n"


def _globs(question: Question, key: str) -> str:
    return ", ".join(f"`{pattern}`" for pattern in question.strings(key))


@dataclass(frozen=True)
class Result:
    """``approved``, ``escalated`` (a person must decide) or ``waiting``.

    ``running`` is only seen mid-run: by an ``on_progress`` listener, or in a
    record it saved.
    """

    status: RunStatus
    reason: str
    evaluations: tuple[Evaluation, ...]
    digest: str
    issue: str = ""
    base: str = ""

    @property
    def last(self) -> Evaluation | None:
        return self.evaluations[-1] if self.evaluations else None


Agent = Callable[[Task], Attempt | str | Awaitable[Attempt | str]]
Listener = Callable[[Result], object]


P = ParamSpec("P")


async def _call(
    function: Callable[P, object], *args: P.args, **kwargs: P.kwargs
) -> object:
    """Call ``function`` in a thread, so a sync one cannot block the loop, and
    await what it returns when that is awaitable (an async one's coroutine)."""
    result = await asyncio.to_thread(function, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def _put(
    oracle: Oracle, questions: tuple[Question, ...], context: Context
) -> object:
    """Ask ``oracle``. A pause or a hand-over is its result; any other error is
    raised, so the questions put to other oracles at the same time are not
    waited for."""
    try:
        return await _call(oracle, questions, context)
    except (Pending, Escalate) as stop:
        return stop


def _check_oracles(checklist: Checklist, oracles: Mapping[str, Oracle]) -> None:
    missing = sorted(
        {checklist.questions[qid].by for qid in checklist.used()} - set(oracles)
    )
    if missing:
        raise ValueError(f"no oracle for: {', '.join(missing)}")


def _accepted(name: str, asked: Collection[str], result: object) -> dict[str, Answer]:
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
    known = dict(answers or {})
    done = list(rounds)
    # Oracles that paused, the questions they owe and why: they are not asked
    # again until the run resumes.
    paused: set[str] = set()
    waiting: list[str] = []
    notes: list[str] = []
    # Answers passed with a pause. They count if the attempt is decided now;
    # if it must wait, they are asked for again on resume, so a correction
    # made meanwhile (a person's later reply) counts.
    provisional: dict[str, Answer] = {}
    escalations: list[str] = []
    # Provisional answers count when planning: once one fails a requirement
    # (a person's NO), people are asked nothing more, so unless an automated
    # check is still owed the attempt is decided and nobody is waited for.
    while ask := plan(checklist, facts(known | provisional), last=last):
        groups = {
            name: questions
            for name, questions in by_oracle(checklist, ask).items()
            if name not in paused
        }
        if not groups:
            return Evaluation.from_answers(
                checklist,
                attempt,
                known,
                done,
                WAITING,
                "; ".join(notes),
                tuple(waiting),
            )
        context = Context(
            checklist, attempt, MappingProxyType(known | provisional), issue
        )
        names = list(groups)
        results = await asyncio.gather(
            *(_put(oracles[name], tuple(groups[name]), context) for name in names)
        )
        got: dict[str, Answer] = {}
        for name, result in zip(names, results, strict=True):
            asked = [q.id for q in groups[name]]
            if isinstance(result, Pending):
                partial = _accepted(name, asked, result.answers)
                provisional.update(partial)
                paused.add(name)
                waiting += [qid for qid in asked if qid not in partial]
                notes.append(f"{name}: {result}")
            elif isinstance(result, Escalate):
                escalations.append(f"{name}: {result}")
            else:
                answered = _accepted(name, asked, result)
                if not answered:
                    raise OracleError(f"oracle {name!r} answered none of {asked}")
                got.update(answered)
        if got:
            known.update(got)
            done.append(tuple(got))
        if escalations:
            break
    if provisional:
        known.update(provisional)
        done.append(tuple(provisional))
    if escalations:
        return Evaluation.from_answers(
            checklist, attempt, known, done, ESCALATED, "; ".join(escalations)
        )
    return Evaluation.from_answers(checklist, attempt, known, done)


def _verdict(
    evaluations: Sequence[Evaluation], max_attempts: int, stop_on_regression: bool
) -> tuple[RunStatus, str] | None:
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
                rid for rid in failing if earlier.requirements.get(rid) == PASS
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
    resume: Result | None = None,
    on_progress: Listener | None = None,
    stop_on_regression: bool = True,
    base: str = "",
) -> Result:
    """Run the agent until approval or escalation.

    The agent is ``(task) -> Attempt | ref``, and raises :class:`Escalate`
    when it cannot make an attempt. It is the runner's code around the
    coding agent, such as :class:`symbolic_acceptance.agents.CommandAgent`,
    which computes the attempt's diff itself.
    ``on_progress`` gets the run whenever it changes: before each attempt,
    when an attempt is made and is being checked, and when the check ends,
    ``running`` or as the run ended, e.g. to save it with
    :func:`symbolic_acceptance.record.save`. A run saved that way can be
    resumed from any of these points. ``base`` names what the run started
    from, such as :attr:`symbolic_acceptance.agents.CommandAgent.base`, and
    is kept with it. Pass a saved ``Result`` as ``resume`` to continue it;
    its issue text and base carry over.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    _check_oracles(checklist, oracles)
    evaluations: list[Evaluation] = []

    async def report(status: RunStatus, reason: str = "") -> Result:
        result = Result(
            status, reason, tuple(evaluations), checklist.digest, issue, base
        )
        if on_progress is not None:
            await _call(on_progress, result)
        return result

    if resume is not None:
        if resume.digest != checklist.digest:
            raise ValueError(
                "the checklist changed since this run started; start a new run"
            )
        if issue and resume.issue and issue != resume.issue:
            raise ValueError("the issue text differs from the run being resumed")
        if base and resume.base and base != resume.base:
            raise ValueError("the base differs from the run being resumed")
        issue = issue or resume.issue
        base = base or resume.base
        evaluations = list(resume.evaluations)
        if evaluations and evaluations[-1].status == WAITING:
            waiting = evaluations.pop()
            evaluations.append(
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
                return await report(*verdict)
        await report(RUNNING)
        previous = evaluations[-1] if evaluations else None
        task = Task(
            issue=issue,
            checklist=checklist,
            number=len(evaluations) + 1,
            feedback=previous.failures if previous else (),
            previous=previous.attempt if previous else None,
        )
        try:
            produced = await _call(agent, task)
        except Escalate as stop:
            return await report(ESCALATED, f"agent: {stop}")
        if isinstance(produced, str):
            produced = Attempt(ref=produced)
        if not isinstance(produced, Attempt) or not produced.ref:
            raise TypeError("the agent must return an Attempt or a non-empty ref")
        attempt = replace(produced, number=task.number)
        # Kept before it is checked, so a run stopped by an error meanwhile
        # resumes by checking it rather than asking the agent again.
        evaluations.append(
            Evaluation.from_answers(
                checklist, attempt, {}, (), WAITING, "being checked"
            )
        )
        await report(RUNNING)
        evaluations[-1] = await evaluate(
            checklist, attempt, oracles, last=last, issue=issue
        )
