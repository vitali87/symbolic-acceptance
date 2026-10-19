"""Save and load runs as JSON, to audit them or to resume a waiting one.

The record keeps answers, not conclusions: approval, the requirements' states
and the failures are worked out again from the answers on load, so a record
cannot claim an approval its answers don't support; a pause or an escalation
is kept as recorded, and the run's status must fit its attempts. The
checklist's digest is stored and checked, and loading against a different
checklist fails.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict

from symbolic_acceptance.answer import Answer, AnswerData
from symbolic_acceptance.checklist import Checklist
from symbolic_acceptance.loop import (
    APPROVED,
    CHANGES,
    ESCALATED,
    RUNNING,
    WAITING,
    Attempt,
    Evaluation,
    EvaluationStatus,
    Result,
    RunStatus,
)

__all__ = ["dumps", "load", "loads", "save"]

FORMAT = "symbolic-acceptance/run"
VERSION = 1
# What the last attempt of a run with each status can be (a running run's
# may still be being checked); every attempt before it was sent back with
# changes.
_LAST: dict[str, set[str | None]] = {
    APPROVED: {APPROVED},
    WAITING: {WAITING},
    RUNNING: {None, CHANGES, WAITING},
    ESCALATED: {None, CHANGES, ESCALATED},
}


class _Attempt(TypedDict):
    ref: str
    number: int
    diff: str | None
    claims: str


class _Evaluation(TypedDict):
    attempt: _Attempt
    status: EvaluationStatus
    reason: str
    waiting_on: list[str]
    rounds: list[list[str]]
    answers: dict[str, AnswerData]


class _Checklist(TypedDict):
    digest: str
    source: str | None


class _Record(TypedDict):
    format: str
    version: int
    checklist: _Checklist
    issue: str
    base: str
    status: RunStatus
    reason: str
    evaluations: list[_Evaluation]


def _evaluation(evaluation: Evaluation) -> _Evaluation:
    attempt = evaluation.attempt
    return {
        "attempt": {
            "ref": attempt.ref,
            "number": attempt.number,
            "diff": attempt.diff,
            "claims": attempt.claims,
        },
        "status": evaluation.status,
        "reason": evaluation.reason,
        "waiting_on": list(evaluation.waiting_on),
        "rounds": [list(ids) for ids in evaluation.rounds],
        "answers": {qid: a.to_dict() for qid, a in evaluation.answers.items()},
    }


def dumps(result: Result, checklist: Checklist) -> str:
    """Serialise a run; ``checklist`` is embedded so the record is self-contained."""
    if result.digest != checklist.digest:
        raise ValueError("this run was made with a different checklist")
    data: _Record = {
        "format": FORMAT,
        "version": VERSION,
        "checklist": {"digest": checklist.digest, "source": checklist.source},
        "issue": result.issue,
        "base": result.base,
        "status": result.status,
        "reason": result.reason,
        "evaluations": [_evaluation(e) for e in result.evaluations],
    }
    return json.dumps(data, indent=2) + "\n"


def loads(text: str, checklist: Checklist) -> Result:
    """Rebuild a run, re-deciding every evaluation from its answers."""
    try:
        return _loads(text, checklist)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(f"not a {FORMAT} v{VERSION} record: {error!r}") from None


def _loads(text: str, checklist: Checklist) -> Result:
    data: _Record = json.loads(text)
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError(f"not a {FORMAT} v{VERSION} record")
    digest = data["checklist"]["digest"]
    if digest != checklist.digest:
        raise ValueError(
            "the record was made with a different checklist "
            f"(record {digest[:12]}, checklist {checklist.digest[:12]})"
        )
    evaluations = []
    for item in data["evaluations"]:
        answers = {qid: Answer(**body) for qid, body in item["answers"].items()}
        unknown = sorted(set(answers) - set(checklist.questions))
        if unknown:
            raise ValueError(f"record answers unknown questions: {unknown}")
        status = item["status"]
        paused = status if status in (WAITING, ESCALATED) else None
        evaluation = Evaluation.from_answers(
            checklist,
            Attempt(**item["attempt"]),
            answers,
            [tuple(ids) for ids in item["rounds"]],
            paused,
            item["reason"],
            tuple(item["waiting_on"]),
        )
        if evaluation.status != status:
            raise ValueError(
                f"attempt {evaluation.attempt.number} is recorded as {status} but "
                f"its answers say {evaluation.status}"
            )
        evaluations.append(evaluation)
    recorded = data["status"]
    last = evaluations[-1].status if evaluations else None
    if last not in _LAST.get(recorded, set()) or any(
        evaluation.status != CHANGES for evaluation in evaluations[:-1]
    ):
        raise ValueError(
            f"the run is recorded as {recorded} but its attempts end "
            f"{[evaluation.status for evaluation in evaluations]}"
        )
    return Result(
        recorded,
        data["reason"],
        tuple(evaluations),
        digest,
        data["issue"],
        data["base"],
    )


def save(result: Result, checklist: Checklist, path: str | Path) -> None:
    """Write a run atomically, so a crash never leaves half a record."""
    path = Path(path)
    text = dumps(result, checklist)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load(path: str | Path, checklist: Checklist) -> Result:
    return loads(Path(path).read_text(encoding="utf-8"), checklist)
