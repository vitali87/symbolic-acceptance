"""GitHub over its REST API: CI results, reviewers' answers, diffs, reports.

* :class:`GitHubChecks` answers ``ci`` questions from the check runs of the
  attempt's commit; a question names its run with ``check = "<name>"``.
* :class:`Reviews` answers ``human`` questions from pull request comments. It
  posts the questions once per attempt and reads replies such as
  ``/yes sso_exempt reviewed with security`` from people allowed to answer.
* :func:`compare_diff` and :func:`post_report` fetch a diff and keep one
  report comment up to date.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from symbolic_acceptance import http
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.loop import Attempt, Context, OracleError, Pending

__all__ = [
    "API_VERSION",
    "REPORT_MARKER",
    "GitHub",
    "GitHubChecks",
    "Reviews",
    "compare_diff",
    "diff_source",
    "post_report",
]

API_VERSION = "2026-03-10"
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
_REPLY = re.compile(r"^/(yes|no)\s+([A-Za-z_][A-Za-z0-9_]*)\b[ \t:—-]*(.*)$", re.I | re.M)
_TRUSTED = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_FAILED = frozenset(
    {"failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale"}
)
REPORT_MARKER = "<!-- symbolic-acceptance:report -->"


@dataclass(frozen=True)
class GitHub:
    """A repository, ``owner/name``; the token defaults to ``GITHUB_TOKEN``."""

    repo: str
    token: str | None = None
    api: str = "https://api.github.com"
    transport: http.Transport | None = None

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "accept": accept,
            "x-github-api-version": API_VERSION,
            "user-agent": "symbolic-acceptance",
        }
        token = self.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.api.rstrip('/')}/repos/{self.repo}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return url

    def call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any = None,
        accept: str = "application/vnd.github+json",
    ) -> http.Response:
        return http.request(
            method,
            self._url(path, params),
            headers=self._headers(accept),
            json_body=body,
            timeout=60,
            transport=self.transport,
        )

    def pages(self, path: str, *, key: str | None = None, **params: Any) -> list[Any]:
        """Every item of a paginated list, following ``Link: rel="next"``."""
        items: list[Any] = []
        url: str | None = self._url(path, {"per_page": 100, **params})
        while url:
            response = http.request(
                "GET",
                url,
                headers=self._headers("application/vnd.github+json"),
                timeout=60,
                transport=self.transport,
            )
            data = response.json()
            items.extend(data[key] if key else data)
            link = response.header("link") or ""
            match = _NEXT.search(link)
            url = match.group(1) if match else None
        return items

    def check_runs(self, ref: str) -> list[dict[str, Any]]:
        return self.pages(f"/commits/{urllib.parse.quote(ref)}/check-runs", key="check_runs")

    def comments(self, number: int) -> list[dict[str, Any]]:
        return self.pages(f"/issues/{number}/comments")

    def comment(self, number: int, body: str) -> dict[str, Any]:
        return self.call("POST", f"/issues/{number}/comments", body={"body": body}).json()

    def edit_comment(self, comment_id: int, body: str) -> dict[str, Any]:
        return self.call(
            "PATCH", f"/issues/comments/{comment_id}", body={"body": body}
        ).json()


def compare_diff(github: GitHub, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as GitHub computes it."""
    path = f"/compare/{urllib.parse.quote(base)}...{urllib.parse.quote(head)}"
    return github.call("GET", path, accept="application/vnd.github.diff").text()


def post_report(github: GitHub, number: int, markdown: str) -> dict[str, Any]:
    """Create or update the single report comment on an issue or pull request."""
    body = f"{REPORT_MARKER}\n{markdown}"
    for existing in github.comments(number):
        if REPORT_MARKER in (existing.get("body") or ""):
            return github.edit_comment(existing["id"], body)
    return github.comment(number, body)


class GitHubChecks:
    """A ``ci`` oracle reading check runs for the attempt's commit.

    A completed run named by the question's ``check`` option answers it:
    ``success`` is YES; ``failure``, ``timed_out``, ``cancelled`` and the like
    are NO, and so are ``skipped`` and ``neutral``, because a check that did
    not run proves nothing. Runs still going, or not started, are waited for
    up to ``wait`` seconds and then reported as :class:`Pending`.
    """

    def __init__(self, github: GitHub, *, wait: float = 0.0, poll: float = 15.0) -> None:
        self.github = github
        self.wait = wait
        self.poll = poll

    def _read(self, questions: Sequence[Question], ref: str) -> tuple[dict[str, Answer], list[str]]:
        runs = self.github.check_runs(ref)
        answers: dict[str, Answer] = {}
        waiting: list[str] = []
        for question in questions:
            name = question.options.get("check")
            if not isinstance(name, str):
                raise OracleError(f"q.{question.id} needs check = \"<check run name>\"")
            matching = [run for run in runs if run.get("name") == name]
            if not matching:
                waiting.append(f"{question.id} (no check run named {name!r} yet)")
                continue
            latest = max(matching, key=lambda run: run.get("id", 0))
            if latest.get("status") != "completed":
                waiting.append(f"{question.id} ({name!r} is {latest.get('status')})")
                continue
            conclusion = latest.get("conclusion")
            output = latest.get("output") or {}
            summary = output.get("title") or output.get("summary") or ""
            evidence = tuple(item for item in (latest.get("html_url"), summary) if item)
            if conclusion == "success":
                answers[question.id] = Answer(
                    True, f"{name} passed", evidence, by=f"ci:{name}"
                )
            else:
                why = "failed" if conclusion in _FAILED else f"concluded {conclusion}"
                answers[question.id] = Answer(
                    False, f"{name} {why}", evidence, by=f"ci:{name}"
                )
        return answers, waiting

    async def __call__(self, questions: Sequence[Question], context: Context) -> dict[str, Answer]:
        deadline = time.monotonic() + self.wait
        while True:
            answers, waiting = await asyncio.to_thread(
                self._read, questions, context.attempt.ref
            )
            if not waiting:
                return answers
            if time.monotonic() >= deadline:
                raise Pending("CI: " + "; ".join(waiting), answers=answers)
            await asyncio.sleep(self.poll)


class Reviews:
    """A ``human`` oracle asking on a pull request and reading the replies.

    Answers count only from ``reviewers`` when given, otherwise from the
    repository's owners, members and collaborators, never from bots or from
    ``exclude`` (the agent's own account). Only replies posted after the
    questions for this attempt's commit count, and the latest reply for a
    question wins.
    """

    def __init__(
        self,
        github: GitHub,
        number: int,
        *,
        reviewers: Collection[str] | None = None,
        exclude: Collection[str] = (),
    ) -> None:
        self.github = github
        self.number = number
        self.reviewers = {r.lower() for r in reviewers} if reviewers else None
        self.exclude = {e.lower() for e in exclude}

    def _marker(self, ref: str) -> str:
        return f"<!-- symbolic-acceptance:ask ref={ref} -->"

    def _allowed(self, comment: dict[str, Any]) -> bool:
        user = comment.get("user") or {}
        login = str(user.get("login", "")).lower()
        if not login or login in self.exclude or user.get("type") == "Bot":
            return False
        if login.endswith("[bot]"):
            return False
        if self.reviewers is not None:
            return login in self.reviewers
        return comment.get("author_association") in _TRUSTED

    def _ask_body(self, questions: Sequence[Question], ref: str) -> str:
        lines = [
            self._marker(ref),
            f"**Questions for a person** about commit `{ref[:12]}`",
            "",
            "Reply with one line per question: `/yes <id> <why>` or `/no <id> <why>`.",
            "",
        ]
        for question in questions:
            lines.append(f"- `{question.id}`: {question.ask}")
            if question.context:
                lines.append(f"  - {question.context}")
        return "\n".join(lines)

    def _read(self, questions: Sequence[Question], ref: str) -> dict[str, Answer]:
        comments = self.github.comments(self.number)
        marker = self._marker(ref)
        asked_at = None
        for comment in comments:
            if marker in (comment.get("body") or ""):
                asked_at = comment.get("created_at")
                break
        if asked_at is None:
            self.github.comment(self.number, self._ask_body(questions, ref))
            return {}
        wanted = {q.id for q in questions}
        answers: dict[str, Answer] = {}
        for comment in sorted(comments, key=lambda c: (c.get("created_at", ""), c.get("id", 0))):
            if comment.get("created_at", "") < asked_at or not self._allowed(comment):
                continue
            login = comment["user"]["login"]
            for verdict, qid, why in _REPLY.findall(comment.get("body") or ""):
                if qid in wanted:
                    answers[qid] = Answer(
                        verdict.lower() == "yes",
                        why.strip(),
                        tuple(x for x in (comment.get("html_url"),) if x),
                        by=f"human:@{login}",
                    )
        return answers

    async def __call__(self, questions: Sequence[Question], context: Context) -> dict[str, Answer]:
        answers = await asyncio.to_thread(self._read, questions, context.attempt.ref)
        missing = [q.id for q in questions if q.id not in answers]
        if missing:
            raise Pending(
                f"asked on #{self.number}; waiting for {', '.join(missing)}",
                answers=answers,
            )
        return answers


def diff_source(github: GitHub, base: str) -> Callable[[Attempt], str]:
    """A ``run(diff=...)`` source: GitHub's diff from ``base`` to the attempt."""

    def source(attempt: Attempt) -> str:
        return compare_diff(github, attempt.base or base, attempt.ref)

    return source

