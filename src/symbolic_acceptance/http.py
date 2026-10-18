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
from typing import Any

__all__ = ["HTTPError", "Response", "Transport", "request"]

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


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

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class HTTPError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, detail: str) -> None:
        super().__init__(f"{method} {url} failed with HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


Transport = Callable[[urllib.request.Request, float], Response]


def urlopen(request: urllib.request.Request, timeout: float) -> Response:
    """The default transport: ``urllib``, honouring ``HTTPS_PROXY`` and system CAs."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            return Response(reply.status, reply.read(), dict(reply.headers.items()))
    except urllib.error.HTTPError as error:
        return Response(error.code, error.read(), dict(error.headers.items()))


def _retry_after(response: Response | None, attempt: int) -> float:
    if response is not None:
        value = response.header("retry-after")
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
                except (TypeError, ValueError):
                    pass
    return min(60.0, 2.0 ** attempt)


def _detail(response: Response) -> str:
    text = response.text().strip()
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
    json_body: Any = None,
    timeout: float = 600.0,
    retries: int = 2,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Response:
    """Send a request; retry timeouts, rate limits and server errors.

    Returns the response for any 2xx status and raises :class:`HTTPError`
    otherwise, after ``retries`` further tries on retryable failures.
    """
    send = transport or urlopen
    data = None
    all_headers = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        all_headers.setdefault("content-type", "application/json")
    response: Response | None = None
    for attempt in range(retries + 1):
        prepared = urllib.request.Request(
            url, data=data, headers=all_headers, method=method
        )
        try:
            response = send(prepared, timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == retries:
                raise HTTPError(method, url, 0, f"connection failed: {error}") from None
            sleep(_retry_after(None, attempt))
            continue
        if 200 <= response.status < 300:
            return response
        if response.status not in RETRY_STATUSES or attempt == retries:
            break
        sleep(_retry_after(response, attempt))
    assert response is not None
    raise HTTPError(method, url, response.status, _detail(response))
