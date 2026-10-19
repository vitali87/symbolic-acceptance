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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from symbolic_acceptance import http

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


@dataclass(frozen=True)
class Reply:
    data: Any
    model: str
    usage: Mapping[str, Any]


@dataclass(frozen=True)
class Claude:
    """Settings for calling the Messages API.

    Credentials come from ``ANTHROPIC_API_KEY``, or ``ANTHROPIC_AUTH_TOKEN``
    for an OAuth token, unless passed in.
    """

    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 16000
    fallbacks: bool = True
    timeout: float = 600.0
    api_key: str | None = None
    auth_token: str | None = None
    base_url: str | None = None
    transport: http.Transport | None = None

    def _headers(self) -> dict[str, str]:
        betas = [FALLBACK_BETA] if self.fallbacks else []
        headers = {"anthropic-version": API_VERSION}
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        token = self.auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if api_key:
            headers["x-api-key"] = api_key
        elif token:
            headers["authorization"] = f"Bearer {token}"
            betas.append(OAUTH_BETA)
        else:
            raise ClaudeError(
                "no credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)"
            )
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def json(
        self,
        *,
        system: str | Sequence[Mapping[str, Any]],
        content: str | Sequence[Mapping[str, Any]],
        schema: Mapping[str, Any],
    ) -> Reply:
        """Ask once and return the JSON reply that matches ``schema``."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self.fallbacks:
            body["fallbacks"] = "default"
        base = (
            self.base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )
        response = http.request(
            "POST",
            base.rstrip("/") + "/v1/messages",
            headers=self._headers(),
            json_body=body,
            timeout=self.timeout,
            transport=self.transport,
        )
        message = response.json()
        stop = message.get("stop_reason")
        if stop == "refusal":
            details = message.get("stop_details") or {}
            raise Refusal(details.get("category"))
        if stop == "max_tokens":
            raise ClaudeError(
                f"the reply hit max_tokens ({self.max_tokens}) and is incomplete"
            )
        text = "".join(
            block.get("text", "")
            for block in message.get("content", ())
            if block.get("type") == "text"
        )
        try:
            data = json.loads(text)
        except ValueError:
            raise ClaudeError("the reply was not valid JSON") from None
        return Reply(data, message.get("model", self.model), message.get("usage", {}))
