"""Shared test fixtures: the example checklist, scripted oracles, fake HTTP."""

from __future__ import annotations

import asyncio
import contextlib
import html
import http.server
import io
import json
import os
import signal
import socketserver
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path

import symbolic_acceptance as sa
from symbolic_acceptance.agents import git as run_git
from symbolic_acceptance.cli import main
from symbolic_acceptance.github import Comment
from symbolic_acceptance.http import Response
from symbolic_acceptance.loop import Oracle

# The suite runs the same whoever runs it. Git takes no identity from the
# environment and reads no one's settings, so that a developer's
# commit.gpgsign, say, fails no test, unless the suite is pointed at some
# (GIT_CONFIG_GLOBAL, to try the runner against hostile ones).
_MACHINE = (
    "GIT_AUTHOR_",
    "GIT_COMMITTER_",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)
for _key in [key for key in os.environ if key.startswith(_MACHINE)]:
    del os.environ[_key]
os.environ.setdefault("GIT_CONFIG_GLOBAL", os.devnull)
os.environ.setdefault("GIT_CONFIG_SYSTEM", os.devnull)
# A signal ignored here, as Ctrl-C is in a shell's background job and a
# hang-up under nohup, would be ignored by the processes the tests start too,
# and the tests that stop them would fail. One handled here is not inherited.
for _stop in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    if signal.getsignal(_stop) is signal.SIG_IGN:
        signal.signal(_stop, lambda signum, frame: None)

ROOT = Path(__file__).resolve().parent.parent
# The package under test, for a process of its own to import.
SRC = str(Path(sa.__file__).parents[1])
EXAMPLE = ROOT / "examples" / "issue-342.toml"

PASSING = {
    "repro_fixed": True,
    "new_test": True,
    "suite_green": True,
    "frozen_untouched": True,
    "locks_account": True,
    "can_unlock": True,
    "backs_off": True,
    "sso_untouched": True,
    "sso_exempt": True,
}


def example() -> sa.Checklist:
    return sa.load(EXAMPLE)


def finished_run(checklist: sa.Checklist) -> sa.Result:
    """A run of the example: one attempt sent back, the next approved."""
    truths = [dict(PASSING, new_test=False), PASSING]
    script = Scripted(truths[0])
    calls: list[sa.Task] = []

    async def agent(task: sa.Task) -> sa.Attempt:
        calls.append(task)
        script.truth = truths[len(calls) - 1]
        return sa.Attempt(f"commit{len(calls)}", claims="I fixed it")

    return asyncio.run(
        sa.run(checklist, agent=agent, oracles=script.oracles(), base="base0")
    )


def cli(*args: str) -> tuple[int, str]:
    """Run the command line; its exit code and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(args))
    return code, out.getvalue()


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


def repo(tmp: str) -> Callable[..., str]:
    """A git repository in ``tmp`` with one commit of ``app.py``.

    Its git is the runner's, so neither the developer's git configuration nor
    a hook's environment can reach it.
    """

    def git(*args: str) -> str:
        return run_git(tmp, *args, isolated=True)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
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


def rendered(body: str) -> str:
    """The HTML GitHub renders a comment of plain lines as (body_html)."""
    lines = "<br>\n".join(html.escape(line) for line in body.split("\n"))
    return f'<p dir="auto">{lines}</p>'


def folds_case() -> bool:
    """Whether the temporary folder, where a check checks a commit out, takes
    ``A`` and ``a`` for one name, as macOS does by default."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "A").touch()
        return Path(tmp, "a").exists()


@contextlib.contextmanager
def serve(handler: type[socketserver.BaseRequestHandler]) -> Generator[str]:
    """``handler`` answering on this machine, at the URL this yields.

    It is socketserver's server rather than http.server's, which first looks
    up the name of the address it listens on, a lookup that can take half a
    minute on macOS.
    """
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


@contextlib.contextmanager
def github_api() -> Generator[tuple[str, list[Comment], list[tuple[str, str]]]]:
    """A stand-in for GitHub's API for issue comments, on this machine: its
    URL, the comments it holds and each request's method and token."""
    comments: list[Comment] = []
    requests: list[tuple[str, str]] = []

    class API(http.server.BaseHTTPRequestHandler):
        def reply(self, data: object) -> None:
            requests.append((self.command, self.headers["authorization"]))
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def sent(self) -> str:
            length = int(self.headers["content-length"])
            return str(json.loads(self.rfile.read(length))["body"])

        def do_GET(self) -> None:
            self.reply(comments)

        def do_POST(self) -> None:
            at = f"2026-10-19T10:00:{len(comments):02d}Z"
            body = self.sent()
            comment: Comment = {
                "id": len(comments) + 1,
                "body": body,
                "body_html": rendered(body),
                "user": {"login": "runner[bot]", "type": "Bot"},
                "author_association": "NONE",
                "created_at": at,
                "updated_at": at,
                "html_url": f"https://example.test/{len(comments) + 1}",
            }
            comments.append(comment)
            self.reply(comment)

        def do_PATCH(self) -> None:
            comment = comments[int(self.path.rsplit("/", 1)[1]) - 1]
            body = self.sent()
            comment.update(
                body=body, body_html=rendered(body), updated_at="2026-10-19T11:00:00Z"
            )
            self.reply(comment)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with serve(API) as url:
        yield url, comments, requests
