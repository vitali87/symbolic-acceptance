"""A small JSON-over-HTTPS client on ``urllib``, with retries.

The transport is injectable so integrations can be tested without a network.
Error messages carry the URL and status, never request headers, so tokens do
not end up in logs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from typing import TypeAlias

__all__ = ["JSON", "HTTPError", "Response", "Transport", "request"]

JSON: TypeAlias = "dict[str, JSON] | list[JSON] | str | int | float | bool | None"

IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
# Statuses meaning the server did not act on the request, so even a POST
# can be sent again.
UNPROCESSED = frozenset({408, 429, 529})
MAX_WAIT = 60.0


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str | None:
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


class HTTPError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, detail: str) -> None:
        super().__init__(f"{method} {url} failed with HTTP {status}: {detail}")


Transport = Callable[[urllib.request.Request, float], Response]


def urlopen(request: urllib.request.Request, timeout: float) -> Response:
    """The default transport: ``urllib``, honouring ``HTTPS_PROXY`` and system CAs."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return Response(reply.status, reply.read(), dict(reply.headers.items()))
    except urllib.error.HTTPError as error:
        return Response(error.code, error.read(), dict(error.headers.items()))


def _wait(response: Response | None, attempt: int) -> float:
    """Seconds before the next try: the server's ``Retry-After``, else backoff."""
    value = response.header("retry-after") if response is not None else None
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
            except ValueError:
                seconds = 2.0**attempt
    else:
        seconds = 2.0**attempt
    return min(MAX_WAIT, max(0.0, seconds))


def _detail(response: Response) -> str:
    text = response.body.decode("utf-8", errors="replace").strip()
    try:
        data = json.loads(text)
    except ValueError:
        return text[:500]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:500]
        if data.get("message"):
            return str(data["message"])[:500]
    return text[:500]


def request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    json_body: JSON = None,
    timeout: float = 600.0,
    retries: int = 2,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    repeatable: bool | None = None,
) -> Response:
    """Send a request, retrying failures that are safe to retry.

    Returns the response for any 2xx status and raises :class:`HTTPError`
    otherwise. Idempotent methods are retried on connection failures, rate
    limits and server errors. Other methods, such as POST, may already have
    taken effect when they fail, so they are retried only when the server
    says it did not process the request (408, 429, 529). Pass ``repeatable``
    to say whether sending this request twice is harmless.
    """
    send = transport or urlopen
    data = None
    all_headers = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        all_headers.setdefault("content-type", "application/json")
    idempotent = method in IDEMPOTENT if repeatable is None else repeatable
    attempt = 0
    while True:
        final = attempt == retries
        prepared = urllib.request.Request(
            url, data=data, headers=all_headers, method=method
        )
        try:
            response = send(prepared, timeout)
        # No reply, or one cut off: the connection failed either way.
        except (OSError, HTTPException) as error:
            if final or not idempotent:
                raise HTTPError(method, url, 0, f"connection failed: {error}") from None
            sleep(_wait(None, attempt))
        else:
            if 200 <= response.status < 300:
                return response
            retryable = RETRY_STATUSES if idempotent else UNPROCESSED
            if final or response.status not in retryable:
                raise HTTPError(method, url, response.status, _detail(response))
            sleep(_wait(response, attempt))
        attempt += 1
