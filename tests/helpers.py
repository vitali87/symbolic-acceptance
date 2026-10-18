"""Shared test fixtures: the example checklist, scripted oracles, fake HTTP."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import symbolic_acceptance as sa
from symbolic_acceptance.http import Response

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "issue-342.toml"

PASSING = {
    "repro_fixed": True,
    "new_test": True,
    "suite_green": True,
    "frozen_untouched": True,
    "locks_account": True,
    "can_unlock": True,
    "backs_off": True,
    "touches_sso": False,
    "sso_exempt": True,
}


def example() -> sa.Checklist:
    return sa.load(EXAMPLE)


class Scripted:
    """Oracles answering from a truth table, recording what each was asked."""

    def __init__(
        self,
        truth: Mapping[str, bool],
        *,
        silent: Iterable[str] = (),
    ) -> None:
        self.truth = dict(truth)
        self.silent = set(silent)
        self.asked: list[tuple[str, tuple[str, ...]]] = []

    def oracle(self, name: str) -> Callable[..., Any]:
        async def answer(questions, context):
            self.asked.append((name, tuple(q.id for q in questions)))
            if name in self.silent:
                raise sa.Pending(f"{name} is not answering")
            return {
                q.id: sa.Answer(self.truth[q.id], reasoning=f"scripted {q.id}")
                for q in questions
            }

        return answer

    def oracles(self, names: Iterable[str] = ("ci", "llm", "human")) -> dict:
        return {name: self.oracle(name) for name in names}

    def asked_ids(self, name: str | None = None) -> list[str]:
        return [qid for who, ids in self.asked if name in (None, who) for qid in ids]


class FakeHTTP:
    """A transport that replays canned responses and records requests."""

    def __init__(self, *responses: Response | Callable[[urllib.request.Request], Response]):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request {request.get_method()} {request.full_url}")
        reply = self.responses.pop(0)
        return reply(request) if callable(reply) else reply

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].data.decode("utf-8"))

    def header(self, name: str, index: int = -1) -> str | None:
        request = self.requests[index]
        for key, value in request.header_items():
            if key.lower() == name.lower():
                return value
        return None


def json_response(data: Any, status: int = 200, headers: Mapping[str, str] | None = None) -> Response:
    return Response(status, json.dumps(data).encode("utf-8"), dict(headers or {}))
