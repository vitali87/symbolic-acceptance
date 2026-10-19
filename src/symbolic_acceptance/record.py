"""Save and load runs as JSON, to audit them or to resume a waiting one.

The record keeps answers, not conclusions: statuses and failures are worked
out again from the answers on load, so a record cannot claim an outcome its
answers don't support. The checklist's digest is stored and checked, and
loading against a different checklist fails.
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
    ESCALATED,
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


class _Attempt(TypedDict):
    ref: str
    number: int
    diff: str | None
    claims: str
    base: str


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
            "base": attempt.base,
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
        "status": result.status,
        "reason": result.reason,
        "evaluations": [_evaluation(e) for e in result.evaluations],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def loads(text: str, checklist: Checklist) -> Result:
    """Rebuild a run, re-deciding every evaluation from its answers."""
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
    if recorded in (APPROVED, WAITING) and recorded != last:
        raise ValueError(
            f"the run is recorded as {recorded} but its last attempt is {last}"
        )
    return Result(recorded, data["reason"], tuple(evaluations), digest, data["issue"])


def save(result: Result, checklist: Checklist, path: str | Path) -> None:
    """Write a run atomically, so a crash never leaves half a record."""
    path = Path(path)
    text = dumps(result, checklist)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load(path: str | Path, checklist: Checklist) -> Result:
    return loads(Path(path).read_text(encoding="utf-8"), checklist)
