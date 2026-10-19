"""The Claude Messages API over raw HTTPS, for structured JSON answers.

Raw HTTP rather than the official SDK keeps the package free of dependencies.
Requests use structured outputs (``output_config.format``), so the reply is
JSON that matches the schema, and server-side refusal fallback
(``fallbacks: "default"``), so a classifier false positive is retried on the
model Anthropic recommends instead of failing the check. Sampling parameters
such as ``temperature`` are not sent: current models reject them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import (
    Generic,
    Literal,
    NotRequired,
    TypedDict,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from symbolic_acceptance import http
from symbolic_acceptance.http import JSON

__all__ = ["DEFAULT_MODEL", "Claude", "ClaudeError", "Refusal", "Reply"]

DEFAULT_MODEL = "claude-opus-5-5"
API_VERSION = "2023-06-01"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
OAUTH_BETA = "oauth-2025-04-20"


class ClaudeError(RuntimeError):
    """The API answered, but not with a usable result."""


class Refusal(ClaudeError):
    """Safety classifiers declined the request, after any fallback."""

    def __init__(self, category: str | None) -> None:
        super().__init__(f"the model declined the request (category: {category})")
        self.category = category


T = TypeVar("T")


@dataclass(frozen=True)
class Reply(Generic[T]):
    data: T
    model: str


class _Block(TypedDict):
    type: str
    text: NotRequired[str]


class _StopDetails(TypedDict):
    category: str | None


class _Message(TypedDict):
    model: str
    content: list[_Block]
    stop_reason: str | None
    stop_details: _StopDetails | None  # null unless the reply is a refusal


def _schema(shape: object) -> dict[str, JSON]:
    """The JSON schema for ``shape``, a ``TypedDict`` whose keys are required.

    Values may be strings, booleans, string ``Literal``s and lists of them.
    """
    if shape is str:
        return {"type": "string"}
    if shape is bool:
        return {"type": "boolean"}
    if get_origin(shape) is Literal:
        return {"type": "string", "enum": [str(value) for value in get_args(shape)]}
    if get_origin(shape) is list:
        return {"type": "array", "items": _schema(get_args(shape)[0])}
    fields = get_type_hints(shape)
    return {
        "type": "object",
        "properties": {name: _schema(kind) for name, kind in fields.items()},
        "required": list(fields),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Claude:
    """Settings for calling the Messages API.

    Unless passed in, credentials come from ``ANTHROPIC_API_KEY``, or
    ``ANTHROPIC_AUTH_TOKEN`` for an OAuth token, and the endpoint from
    ``ANTHROPIC_BASE_URL``.
    """

    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 16000
    fallbacks: bool = True
    timeout: float = 600.0
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"), repr=False
    )
    auth_token: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_AUTH_TOKEN"), repr=False
    )
    base_url: str = field(
        default_factory=lambda: (
            os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        )
    )
    transport: http.Transport | None = None

    def _headers(self) -> dict[str, str]:
        betas = [FALLBACK_BETA] if self.fallbacks else []
        headers = {"anthropic-version": API_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        elif self.auth_token:
            headers["authorization"] = f"Bearer {self.auth_token}"
            betas.append(OAUTH_BETA)
        else:
            raise ClaudeError(
                "no credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)"
            )
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def json(
        self, *, system: str, content: str | list[JSON], shape: type[T]
    ) -> Reply[T]:
        """Ask once and return the reply, which has the keys and types of ``shape``."""
        body: dict[str, JSON] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _schema(shape)},
            },
        }
        if self.fallbacks:
            body["fallbacks"] = "default"
        response = http.request(
            "POST",
            self.base_url.rstrip("/") + "/v1/messages",
            headers=self._headers(),
            json_body=body,
            timeout=self.timeout,
            transport=self.transport,
            # A repeated request costs tokens but changes nothing, so server
            # errors are retried, as the official SDKs do.
            repeatable=True,
        )
        message: _Message = json.loads(response.body)
        stop = message["stop_reason"]
        if stop == "refusal":
            details = message["stop_details"]
            raise Refusal(details["category"] if details else None)
        # With no tools and no stop sequences, any other stop cuts the reply
        # short: max_tokens, or model_context_window_exceeded.
        if stop != "end_turn":
            raise ClaudeError(
                f"the reply stopped early ({stop}; max_tokens is {self.max_tokens})"
            )
        # Structured outputs guarantee a reply in the shape the schema gives.
        data: T = json.loads(
            "".join(
                b.get("text", "") for b in message["content"] if b["type"] == "text"
            )
        )
        return Reply(data, message["model"])
