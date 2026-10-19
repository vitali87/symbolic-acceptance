"""Shared test fixtures: the example checklist, scripted oracles, fake HTTP."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import symbolic_acceptance as sa
from symbolic_acceptance.http import Response
from symbolic_acceptance.loop import Oracle

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

    def oracle(self, name: str) -> Oracle:
        async def answer(questions, context):
            self.asked.append((name, tuple(q.id for q in questions)))
            if name in self.silent:
                raise sa.Pending(f"{name} is not answering")
            return {
                q.id: sa.Answer(self.truth[q.id], reasoning=f"scripted {q.id}")
                for q in questions
            }

        return answer

    def oracles(
        self, names: Iterable[str] = ("ci", "llm", "human")
    ) -> dict[str, Oracle]:
        return {name: self.oracle(name) for name in names}

    def asked_ids(self, name: str | None = None) -> list[str]:
        return [qid for who, ids in self.asked if name in (None, who) for qid in ids]


class FakeHTTP:
    """A transport that replays canned responses and records requests."""

    def __init__(
        self, *responses: Response | Callable[[urllib.request.Request], Response]
    ):
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(
                f"unexpected request {request.get_method()} {request.full_url}"
            )
        reply = self.responses.pop(0)
        return reply if isinstance(reply, Response) else reply(request)

    def body(self, index: int = -1):
        """The JSON body of a request."""
        data = self.requests[index].data
        assert isinstance(data, bytes)
        return json.loads(data)

    def header(self, name: str, index: int = -1) -> str | None:
        request = self.requests[index]
        for key, value in request.header_items():
            if key.lower() == name.lower():
                return value
        return None


def json_response(
    data: object, status: int = 200, headers: Mapping[str, str] | None = None
) -> Response:
    return Response(status, json.dumps(data).encode("utf-8"), dict(headers or {}))


GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def repo(tmp: str) -> Callable[..., str]:
    """A git repository in ``tmp`` with one commit of ``app.py``."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=tmp,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **GIT_ENV},
        ).stdout

    git("init", "-q")
    Path(tmp, "app.py").write_text("print('v1')\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    return git


def message(
    data: object, stop: str = "end_turn", model: str = "claude-opus-5-5"
) -> Response:
    """A Messages API reply carrying ``data`` as its JSON text."""
    return json_response(
        {
            "model": model,
            "stop_reason": stop,
            "stop_details": {"category": "cyber"} if stop == "refusal" else None,
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": json.dumps(data)},
            ],
        }
    )


DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index 1111111..2222222 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,3 +10,6 @@ def login(user, password):
     if not check(user, password):
-        return False
+        record_failure(user)
+        if failures(user) > MAX_FAILURES:
+            lock(user)
+++ not a header, an added line
     return True
diff --git a/tests/test_auth.py b/tests/test_auth.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/tests/test_auth.py
@@ -0,0 +1,2 @@
+def test_lockout():
+    assert locks_after(5)
diff --git a/old.py b/new.py
similarity index 100%
rename from old.py
rename to new.py
diff --git a/tests/test_old.py b/tests/test_old.py
deleted file mode 100644
index 4444444..0000000
--- a/tests/test_old.py
+++ /dev/null
@@ -1 +0,0 @@
-def test_old(): pass
"""
