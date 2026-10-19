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
import json
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import TypedDict

from symbolic_acceptance import http
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.loop import Attempt, Context, OracleError, Pending

__all__ = [
    "API_VERSION",
    "REPORT_MARKER",
    "CheckRun",
    "Comment",
    "GitHub",
    "GitHubChecks",
    "Output",
    "Reviews",
    "User",
    "compare_diff",
    "diff_source",
    "post_report",
]

API_VERSION = "2026-03-10"
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
_REPLY = re.compile(
    r"^/(yes|no)\s+([A-Za-z_]\w*)\b[ \t:—-]*(.*)$",
    re.IGNORECASE | re.MULTILINE | re.ASCII,
)
# Markers count only at the very start of a comment, so text quoted inside a
# comment (CI output in a report, say) can never pose as one.
_ASKED = re.compile(
    r"<!-- symbolic-acceptance:ask ref=(?P<ref>\S+) ids=(?P<ids>[\w,]+) -->", re.ASCII
)
_TRUSTED = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
_FAILED = frozenset(
    {"failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale"}
)
REPORT_MARKER = "<!-- symbolic-acceptance:report -->"


class User(TypedDict):
    login: str
    type: str


class Comment(TypedDict):
    """The fields of an issue comment that are read here."""

    id: int
    body: str
    user: User | None
    author_association: str
    created_at: str
    updated_at: str
    html_url: str


class Output(TypedDict):
    title: str | None
    summary: str | None


class CheckRun(TypedDict):
    """The fields of a check run that are read here."""

    id: int
    name: str
    status: str
    conclusion: str | None
    output: Output
    html_url: str | None


class _CheckRuns(TypedDict):
    check_runs: list[CheckRun]


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
        token = (
            self.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        )
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.api.rstrip('/')}/repos/{self.repo}{path}"

    def call(
        self,
        method: str,
        path: str,
        *,
        body: http.JSON = None,
        accept: str = "application/vnd.github+json",
    ) -> http.Response:
        return http.request(
            method,
            self._url(path),
            headers=self._headers(accept),
            json_body=body,
            timeout=60,
            transport=self.transport,
        )

    def _pages(self, path: str) -> Iterator[bytes]:
        """The body of every page of a list, following ``Link: rel="next"``."""
        url: str | None = self._url(f"{path}?per_page=100")
        while url:
            response = http.request(
                "GET",
                url,
                headers=self._headers("application/vnd.github+json"),
                timeout=60,
                transport=self.transport,
            )
            yield response.body
            match = _NEXT.search(response.header("link") or "")
            url = match.group(1) if match else None

    def check_runs(self, ref: str) -> list[CheckRun]:
        path = f"/commits/{urllib.parse.quote(ref)}/check-runs"
        pages: list[_CheckRuns] = [json.loads(page) for page in self._pages(path)]
        return [run for page in pages for run in page["check_runs"]]

    def comments(self, number: int) -> list[Comment]:
        path = f"/issues/{number}/comments"
        pages: list[list[Comment]] = [json.loads(page) for page in self._pages(path)]
        return [comment for page in pages for comment in page]

    def comment(self, number: int, body: str) -> None:
        self.call("POST", f"/issues/{number}/comments", body={"body": body})

    def edit_comment(self, comment_id: int, body: str) -> None:
        self.call("PATCH", f"/issues/comments/{comment_id}", body={"body": body})


def compare_diff(github: GitHub, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as GitHub computes it."""
    path = f"/compare/{urllib.parse.quote(base)}...{urllib.parse.quote(head)}"
    return github.call("GET", path, accept="application/vnd.github.diff").text()


def _author(comment: Comment) -> tuple[str, bool, str]:
    """The comment's login (lowercased), whether a bot wrote it, its association.

    A deleted account has no user, so its login is empty.
    """
    user = comment["user"]
    login = user["login"].lower() if user else ""
    bot = user is not None and (user["type"] == "Bot" or login.endswith("[bot]"))
    return login, bot, comment["author_association"]


def _edited(comment: Comment) -> bool:
    """Whether the comment changed after it was posted.

    Anyone with write access can edit another account's comment, and the
    comment keeps its author, so an edited comment cannot vouch for anyone.
    """
    return comment["updated_at"] != comment["created_at"]


def _by_runner(comment: Comment, exclude: Collection[str]) -> bool:
    """Whether the runner could have written it: a bot, or a trusted account."""
    login, bot, association = _author(comment)
    return login not in exclude and (bot or association in _TRUSTED)


def post_report(
    github: GitHub, number: int, markdown: str, *, exclude: Collection[str] = ()
) -> None:
    """Create or update the single report comment on an issue or pull request.

    ``exclude`` lists accounts (the agent's) whose comments are never taken
    for the report, whatever they contain.
    """
    body = f"{REPORT_MARKER}\n{markdown}"
    excluded = {login.lower() for login in exclude}
    for existing in github.comments(number):
        if existing["body"].startswith(REPORT_MARKER) and _by_runner(
            existing, excluded
        ):
            github.edit_comment(existing["id"], body)
            return
    github.comment(number, body)


class GitHubChecks:
    """A ``ci`` oracle reading check runs for the attempt's commit.

    A completed run named by the question's ``check`` option answers it:
    ``success`` is YES; ``failure``, ``timed_out``, ``cancelled`` and the like
    are NO, and so are ``skipped`` and ``neutral``, because a check that did
    not run proves nothing. Runs still going, or not started, are waited for
    up to ``wait`` seconds and then reported as :class:`Pending`.
    """

    def __init__(
        self, github: GitHub, *, wait: float = 0.0, poll: float = 15.0
    ) -> None:
        self.github = github
        self.wait = wait
        self.poll = poll

    def _read(
        self, questions: Sequence[Question], ref: str
    ) -> tuple[dict[str, Answer], list[str]]:
        runs = self.github.check_runs(ref)
        answers: dict[str, Answer] = {}
        waiting: list[str] = []
        for question in questions:
            name = question.options.get("check")
            if not isinstance(name, str):
                raise OracleError(f'q.{question.id} needs check = "<check run name>"')
            matching = [run for run in runs if run["name"] == name]
            if not matching:
                waiting.append(f"{question.id} (no check run named {name!r} yet)")
                continue
            latest = max(matching, key=lambda run: run["id"])
            if latest["status"] != "completed":
                waiting.append(f"{question.id} ({name!r} is {latest['status']})")
                continue
            conclusion = latest["conclusion"]
            summary = latest["output"]["title"] or latest["output"]["summary"]
            evidence = tuple(item for item in (latest["html_url"], summary) if item)
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

    async def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
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

    Each question is posted once per commit, in a comment that names the
    commit and the question ids; a later round posts only the new ones. A
    reply counts only if it comes after the comment that asked it, from
    ``reviewers`` when given and otherwise from the repository's owners,
    members and collaborators, never from bots or from ``exclude`` (the
    agent's own account). Edited comments never count, since anyone with
    write access can edit another person's comment; correct an answer with
    a new reply. The latest reply to a question wins.
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

    def _allowed(self, comment: Comment) -> bool:
        login, bot, association = _author(comment)
        if not login or bot or login in self.exclude:
            return False
        if self.reviewers is not None:
            return login in self.reviewers
        return association in _TRUSTED

    def _ask_body(self, questions: Sequence[Question], ref: str) -> str:
        ids = ",".join(q.id for q in questions)
        lines = [
            f"<!-- symbolic-acceptance:ask ref={ref} ids={ids} -->",
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
        comments = sorted(
            self.github.comments(self.number), key=lambda c: (c["created_at"], c["id"])
        )
        asked_at: dict[str, str] = {}
        for comment in comments:
            match = _ASKED.match(comment["body"])
            if (
                match
                and match["ref"] == ref
                and _by_runner(comment, self.exclude)
                and not _edited(comment)
            ):
                for qid in match["ids"].split(","):
                    asked_at.setdefault(qid, comment["created_at"])
        new = [q for q in questions if q.id not in asked_at]
        if new:
            self.github.comment(self.number, self._ask_body(new, ref))
        answers: dict[str, Answer] = {}
        for comment in comments:
            if _edited(comment) or not self._allowed(comment):
                continue
            for verdict, qid, why in _REPLY.findall(comment["body"]):
                if qid in asked_at and comment["created_at"] >= asked_at[qid]:
                    answers[qid] = Answer(
                        verdict.lower() == "yes",
                        why.strip(),
                        (comment["html_url"],),
                        by=f"human:@{_author(comment)[0]}",
                    )
        return {q.id: answers[q.id] for q in questions if q.id in answers}

    async def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
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
