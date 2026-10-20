"""GitHub over its REST API: CI results, reviewers' answers, reports.

* :class:`GitHubChecks` answers ``ci`` questions from the check runs of the
  attempt's commit; a question names its run with ``check = "<name>"``.
* :class:`Reviews` answers ``human`` questions from pull request comments. It
  posts the questions once per attempt and reads replies such as
  ``/yes sso_exempt reviewed with security`` from people allowed to answer.
* :func:`post_report` keeps one report comment up to date.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.parse
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TypedDict

from symbolic_acceptance import http
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.loop import Context, Pending

__all__ = [
    "API_VERSION",
    "REPORT_MARKER",
    "App",
    "CheckRun",
    "Comment",
    "GitHub",
    "GitHubChecks",
    "Output",
    "Reviews",
    "User",
    "post_report",
]

API_VERSION = "2026-03-10"
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
_REPLY = re.compile(
    r"^/(yes|no)[ \t]+([A-Za-z_]\w*)\b[ \t:—-]*(.*)$",
    re.IGNORECASE | re.MULTILINE | re.ASCII,
)
# The runner's comments start with a marker. Markers count only at the very
# start of a comment, so text quoted inside a comment (CI output in a report,
# say) can never pose as one.
_MARKER = "<!-- symbolic-acceptance:"
_ASKED = re.compile(
    re.escape(_MARKER) + r"ask ref=(?P<ref>\S+) ids=(?P<ids>[\w,]+) -->", re.ASCII
)
# A code fence as GitHub reads one (CommonMark): a fence of backticks takes
# no backtick after it, and only spaces and tabs may follow a closing one.
_FENCE = re.compile(r" {0,3}(`{3,}(?!.*`)|~{3,})(.*)$")
_TRUSTED = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
REPORT_MARKER = f"{_MARKER}report -->"
# GitHub refuses a comment over 65,536 characters; this leaves room for
# however it counts them.
MAX_COMMENT = 60_000


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


class App(TypedDict):
    slug: str


class CheckRun(TypedDict):
    """The fields of a check run that are read here."""

    id: int
    name: str
    app: App | None
    status: str
    conclusion: str | None
    output: Output
    html_url: str | None


class _CheckRuns(TypedDict):
    check_runs: list[CheckRun]


@dataclass(frozen=True)
class GitHub:
    """A repository, ``owner/name``; the token defaults to ``GITHUB_TOKEN``
    (or ``GH_TOKEN``), and the API to ``GITHUB_API_URL``, which GitHub
    Actions sets to its server's (GitHub Enterprise Server's, say)."""

    repo: str
    token: str | None = field(
        default_factory=lambda: (
            os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        ),
        repr=False,
    )
    api: str = field(
        default_factory=lambda: (
            os.environ.get("GITHUB_API_URL") or "https://api.github.com"
        )
    )
    transport: http.Transport | None = None

    def _url(self, path: str) -> str:
        return f"{self.api.rstrip('/')}/repos/{self.repo}{path}"

    def _request(self, method: str, url: str, body: http.JSON = None) -> http.Response:
        headers = {
            "accept": "application/vnd.github+json",
            "x-github-api-version": API_VERSION,
            "user-agent": "symbolic-acceptance",
        }
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        return http.request(
            method,
            url,
            headers=headers,
            json_body=body,
            timeout=60,
            transport=self.transport,
        )

    def _pages(self, path: str) -> Iterator[str]:
        """The body of every page of a list, following ``Link: rel="next"``;
        decoded leniently, so that a reply that is not JSON is reported as such."""
        url: str | None = self._url(f"{path}?per_page=100")
        while url:
            response = self._request("GET", url)
            yield response.body.decode(errors="replace")
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
        self._request("POST", self._url(f"/issues/{number}/comments"), {"body": body})

    def edit_comment(self, comment_id: int, body: str) -> None:
        self._request(
            "PATCH", self._url(f"/issues/comments/{comment_id}"), {"body": body}
        )


def _author(comment: Comment) -> tuple[str, bool, str]:
    """The comment's login as :func:`_logins` gives it, whether a bot wrote
    it, and its association.

    A deleted account has no user, so its login is empty.
    """
    user = comment["user"]
    login = user["login"].lower() if user else ""
    bot = user is not None and (user["type"] == "Bot" or login.endswith("[bot]"))
    return login.removesuffix("[bot]"), bot, comment["author_association"]


def _edited(comment: Comment) -> bool:
    """Whether the comment changed after it was posted.

    Anyone with write access can edit another account's comment, and the
    comment keeps its author, so an edited comment cannot vouch for anyone.
    """
    return comment["updated_at"] != comment["created_at"]


def _logins(names: Collection[str]) -> set[str]:
    """Logins as GitHub compares them: no spaces or "@", any case, and an
    app's account by its name, with or without the "[bot]" GitHub adds."""
    return {
        name.strip().removeprefix("@").lower().removesuffix("[bot]")
        for name in names
        if name.strip()
    }


def _outside_code(body: str) -> str:
    """The comment's own lines: a fenced code block quotes, it never replies."""
    lines, fence = [], ""
    for line in re.split(r"\r\n|\r|\n", body):
        match = _FENCE.match(line)
        if not fence:
            if match:
                fence = match[1]
            else:
                lines.append(line)
        elif match and match[1].startswith(fence) and not match[2].strip(" \t"):
            fence = ""
    return "\n".join(lines)


def _by_runner(comment: Comment, exclude: Collection[str]) -> bool:
    """Whether the runner could have written it: a bot, or a trusted account."""
    login, bot, association = _author(comment)
    return login not in exclude and (bot or association in _TRUSTED)


def post_report(
    github: GitHub, number: int, markdown: str, *, exclude: Collection[str] = ()
) -> None:
    """Create or update the single report comment on an issue or pull request.

    ``exclude`` lists accounts (the agent's) whose comments are never taken
    for the report, whatever they contain. A report too long for a comment
    is cut to fit.
    """
    body = f"{REPORT_MARKER}\n{markdown}"
    if len(body) > MAX_COMMENT:
        cut = "\n\n…(cut to fit a comment; the run record has the rest)"
        body = body[: MAX_COMMENT - len(cut)] + cut
    excluded = _logins(exclude)
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
    not run proves nothing. Runs still going, or not started, are read again
    every ``poll`` seconds for up to ``wait`` seconds, and then reported as
    :class:`Pending`.

    Only runs reported by ``app`` count, GitHub Actions by default: another
    CI app installed on the repository could run a configuration the attempt
    added and report a check of the same name.
    """

    def __init__(
        self,
        github: GitHub,
        *,
        wait: float = 0.0,
        poll: float = 15.0,
        app: str = "github-actions",
    ) -> None:
        self.github = github
        self.wait = wait
        self.poll = poll
        self.app = app

    def _read(
        self, questions: Sequence[Question], ref: str
    ) -> tuple[dict[str, Answer], list[str]]:
        runs = self.github.check_runs(ref)
        answers: dict[str, Answer] = {}
        waiting: list[str] = []
        for question in questions:
            name = question.options.get("check")
            if not isinstance(name, str) or not name:
                raise ValueError(f'q.{question.id} needs check = "<check run name>"')
            matching = [
                run
                for run in runs
                if run["name"] == name and run["app"] and run["app"]["slug"] == self.app
            ]
            if not matching:
                waiting.append(
                    f"{question.id} (no check run named {name!r} from {self.app} yet)"
                )
                continue
            latest = max(matching, key=lambda run: run["id"])
            if latest["status"] != "completed":
                waiting.append(f"{question.id} ({name!r} is {latest['status']})")
                continue
            summary = latest["output"]["title"] or latest["output"]["summary"]
            answers[question.id] = Answer(
                latest["conclusion"] == "success",
                f"{name} concluded {latest['conclusion']}",
                tuple(item for item in (latest["html_url"], summary) if item),
                by=f"ci:{name}",
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
            now = time.monotonic()
            if now >= deadline:
                raise Pending("; ".join(waiting), answers=answers)
            await asyncio.sleep(min(self.poll, deadline - now))


class Reviews:
    """A ``human`` oracle asking on a pull request and reading the replies.

    Each question is posted once per commit, in a comment that names the
    commit and the question ids; a later round posts only the new ones. A
    reply answers the latest question with its id posted before it, so it
    counts for that question's commit alone. It must come from ``reviewers``
    when given and otherwise from the repository's owners, members and
    collaborators, never from bots or from ``exclude`` (the agent's own
    account). Edited comments never count, since anyone with write access
    can edit another person's comment; correct an answer with a new reply.
    Nor do the runner's own comments, whoever posts them, or lines inside a
    fenced code block: both quote what the agent printed, which can look
    like a reply. The latest reply to a question wins.
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
        self.reviewers = _logins(reviewers) if reviewers else None
        self.exclude = _logins(exclude)

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
            f"{_MARKER}ask ref={ref} ids={ids} -->",
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
        # The commit each question was last asked about, comment by comment.
        latest: dict[str, str] = {}
        answers: dict[str, Answer] = {}
        for comment in comments:
            if _edited(comment):
                continue
            if comment["body"].startswith(_MARKER):
                match = _ASKED.match(comment["body"])
                if match and _by_runner(comment, self.exclude):
                    latest.update(dict.fromkeys(match["ids"].split(","), match["ref"]))
            elif self._allowed(comment):
                for verdict, qid, why in _REPLY.findall(_outside_code(comment["body"])):
                    if latest.get(qid) == ref:
                        answers[qid] = Answer(
                            verdict.lower() == "yes",
                            why.strip(),
                            (comment["html_url"],),
                            by=f"human:@{_author(comment)[0]}",
                        )
        new = [q for q in questions if q.id not in answers and latest.get(q.id) != ref]
        if new:
            self.github.comment(self.number, self._ask_body(new, ref))
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
