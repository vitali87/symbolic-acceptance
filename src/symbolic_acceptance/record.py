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
from typing import Any

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Checklist
from symbolic_acceptance.loop import APPROVED, WAITING, Attempt, Evaluation, Result

__all__ = ["dumps", "load", "loads", "save"]

FORMAT = "symbolic-acceptance/run"
VERSION = 1


def _evaluation(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "attempt": evaluation.attempt.to_dict(),
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
    data = {
        "format": FORMAT,
        "version": VERSION,
        "checklist": {"digest": checklist.digest, "source": checklist.source},
        "status": result.status,
        "reason": result.reason,
        "evaluations": [_evaluation(e) for e in result.evaluations],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def loads(text: str, checklist: Checklist) -> Result:
    """Rebuild a run, re-deciding every evaluation from its answers."""
    data = json.loads(text)
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
        answers = {qid: Answer.from_dict(body) for qid, body in item["answers"].items()}
        unknown = sorted(set(answers) - set(checklist.questions))
        if unknown:
            raise ValueError(f"record answers unknown questions: {unknown}")
        status = item["status"]
        paused = status if status in ("waiting", "escalated") else None
        evaluation = Evaluation.from_answers(
            checklist,
            Attempt.from_dict(item["attempt"]),
            answers,
            [tuple(ids) for ids in item["rounds"]],
            paused,
            item.get("reason", ""),
            tuple(item.get("waiting_on", ())),
        )
        if evaluation.status != status:
            raise ValueError(
                f"attempt {evaluation.attempt.number} is recorded as {status} but "
                f"its answers say {evaluation.status}"
            )
        evaluations.append(evaluation)
    status = data["status"]
    last = evaluations[-1].status if evaluations else None
    if status in (APPROVED, WAITING) and status != last:
        raise ValueError(
            f"the run is recorded as {status} but its last attempt is {last}"
        )
    return Result(status, data.get("reason", ""), tuple(evaluations), digest)


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
