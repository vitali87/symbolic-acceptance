"""The ``symbolic-acceptance`` command (also ``python -m symbolic_acceptance``)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from symbolic_acceptance import __version__, record
from symbolic_acceptance.agents import CommandAgent, GitError
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    Checklist,
    ChecklistError,
    Question,
    load,
    render,
)
from symbolic_acceptance.checks import Checks, Commands
from symbolic_acceptance.claude import DEFAULT_MODEL, Claude, ClaudeError
from symbolic_acceptance.draft import DraftError, draft
from symbolic_acceptance.feedback import assess, render_failures
from symbolic_acceptance.github import GitHub, GitHubChecks, Reviews, post_report
from symbolic_acceptance.http import HTTPError
from symbolic_acceptance.judge import Judge
from symbolic_acceptance.lint import lint
from symbolic_acceptance.loop import (
    APPROVED,
    WAITING,
    Context,
    Oracle,
    OracleError,
    Pending,
    Result,
    run,
)
from symbolic_acceptance.report import markdown
from symbolic_acceptance.templates import ORACLES, TEMPLATES

__all__ = ["main"]


def _load(path: str) -> Checklist:
    try:
        return load(path)
    except (OSError, ChecklistError) as error:
        raise SystemExit(f"error: {error}") from None


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"error: {error}") from None
    except UnicodeDecodeError:
        raise SystemExit(f"error: {path} is not UTF-8 text") from None


def _json(path: str) -> object:
    try:
        data: object = json.loads(_read(path))
    except json.JSONDecodeError as error:
        raise SystemExit(f"error: {path}: {error}") from None
    return data


def _record(path: str, checklist: Checklist) -> Result:
    try:
        return record.loads(_read(path), checklist)
    except ValueError as error:
        raise SystemExit(f"error: {path}: {error}") from None


def _cmd_lint(args: argparse.Namespace) -> int:
    oracles = tuple(args.oracles.split(",")) if args.oracles else ORACLES
    worst = 0
    for path in args.files:
        try:
            checklist = load(path)
        except (OSError, ChecklistError) as error:
            print(f"{path}: error: {error}")
            worst = 1
            continue
        findings = lint(checklist, oracles=oracles)
        for finding in findings:
            print(f"{path}: {finding}")
        if any(f.level == "error" for f in findings) or (
            args.strict and any(f.level == "warning" for f in findings)
        ):
            worst = 1
        elif not findings:
            print(f"{path}: ok ({len(checklist.requirements)} requirements)")
    return worst


def _cmd_show(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    print(checklist.title or args.file)
    print(f"digest {checklist.digest}")
    print()
    print("Requirements:")
    for requirement in checklist.requirements:
        gate = f" when {render(requirement.when)}" if requirement.when else ""
        print(f"  {requirement.id}: {render(requirement.that)}{gate}")
        print(f"      fails with: {requirement.fail}")
    print()
    print("Questions:")
    for question in checklist.questions.values():
        print(f"  {question.id} [{question.by}]: {question.ask}")
    return 0


def _cmd_digest(args: argparse.Namespace) -> int:
    print(_load(args.file).digest)
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    sys.stdout.write(TEMPLATES[args.kind])
    return 0


def _answers(path: str) -> dict[str, Answer]:
    data = _json(path)
    if not isinstance(data, dict):
        raise SystemExit(
            "error: answers must be a JSON object of question id to answer"
        )
    answers = {}
    for qid, value in data.items():
        if not isinstance(value, bool | dict):
            raise SystemExit(f"error: {qid}: expected true, false or an answer object")
        try:
            answers[qid] = Answer(value) if isinstance(value, bool) else Answer(**value)
        except TypeError as error:
            raise SystemExit(f"error: {qid}: {error}") from None
    return answers


def _cmd_explain(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    answers = _answers(args.answers)
    unknown = sorted(set(answers) - set(checklist.questions))
    if unknown:
        raise SystemExit(f"error: answers for unknown questions: {', '.join(unknown)}")
    assessment = assess(checklist, answers)
    print(assessment.status)
    for rid, state in assessment.requirements.items():
        print(f"  {rid}: {state}")
    if assessment.failures:
        print()
        sys.stdout.write(render_failures(assessment.failures))
    return {"approved": 0, "changes": 1, "open": 3}[assessment.status]


def _cmd_draft(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"error: {out} exists; pass --force to replace it")
    issue = _read(args.issue)
    result = draft(issue, template=args.template, claude=Claude(model=args.model))
    text = result.text if result.text.endswith("\n") else result.text + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}; review it before approving")
    for finding in result.findings:
        print(f"  {finding}")
    if result.questions:
        print("Questions for the reporter:")
        for question in result.questions:
            print(f"  - {question}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    sys.stdout.write(markdown(_record(args.record, checklist), checklist))
    return 0


class TerminalReviewer:
    """A ``human`` oracle that asks on the terminal."""

    def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
        if not sys.stdin.isatty():
            raise Pending("no terminal to ask a person; pass --pr to ask on GitHub")
        attempt = context.attempt
        answers = {}
        print(f"\nQuestions about attempt {attempt.number} ({attempt.ref[:12]}):")
        for question in questions:
            if question.context:
                print(f"  ({' '.join(question.context.split())})")
            while True:
                reply = input(
                    f"  {question.id}: {question.ask} [y/n, then why] "
                ).strip()
                if reply[:1].lower() in ("y", "n"):
                    break
            answers[question.id] = Answer(
                reply[:1].lower() == "y", reply[1:].strip(" ,:-"), by="human:terminal"
            )
        return answers


def _cmd_run(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    errors = [i for i in lint(checklist, oracles=ORACLES) if i.level == "error"]
    if errors:
        raise SystemExit(
            "error: the checklist has lint errors:\n" + "\n".join(map(str, errors))
        )
    issue = _read(args.issue) if args.issue else ""
    if args.pr and not args.github:
        raise SystemExit("error: --pr needs --github")
    github = GitHub(args.github) if args.github else None
    ci_fallback = GitHubChecks(github, wait=args.ci_wait) if github else None
    exclude = args.exclude.split(",") if args.exclude else []
    human: Oracle = TerminalReviewer()
    if github and args.pr:
        human = Reviews(
            github,
            args.pr,
            reviewers=args.reviewers.split(",") if args.reviewers else None,
            exclude=exclude,
        )
    claude = Claude(model=args.model)
    ci = Checks(Commands(args.cwd), fallback=ci_fallback)
    remote = [
        qid
        for qid in checklist.used()
        if checklist.questions[qid].by == "ci"
        and not ci.handles(checklist.questions[qid])
    ]
    if remote and not (github and args.push):
        raise SystemExit(
            f"error: {', '.join(remote)} are answered by GitHub check runs; pass "
            "--github and --push so that every attempt reaches CI"
        )
    oracles: dict[str, Oracle] = {
        "ci": ci,
        "llm": Judge(claude, samples=args.samples),
        "human": human,
    }
    resume: Result | None = None
    if args.resume:
        resume = _record(args.resume, checklist)
    record_path = args.record or args.resume
    if args.record and args.record != args.resume and Path(args.record).exists():
        raise SystemExit(
            f"error: {args.record} holds a run; pass --resume {args.record} to "
            "continue it"
        )
    if record_path and not Path(record_path).resolve().parent.is_dir():
        raise SystemExit(f"error: no folder to save {record_path} in")
    if record_path and Path(record_path).resolve().is_relative_to(
        Path(args.cwd).resolve()
    ):
        raise SystemExit(
            "error: keep the run record outside the agent's checkout (--cwd); "
            "inside it, the record would be committed with the attempt and the "
            "agent could rewrite it"
        )
    # A resumed run keeps diffing against the commit it started from.
    base = resume.evaluations[0].attempt.base if resume and resume.evaluations else ""

    def progress(result: Result) -> None:
        if record_path:
            record.save(result, checklist, record_path)
        if github and args.pr:
            post_report(github, args.pr, markdown(result, checklist), exclude=exclude)

    result = asyncio.run(
        run(
            checklist,
            agent=CommandAgent(
                shlex.split(args.agent), cwd=args.cwd, base=base or None, push=args.push
            ),
            oracles=oracles,
            issue=issue,
            max_attempts=args.max_attempts,
            resume=resume,
            on_progress=progress,
            stop_on_regression=not args.keep_going_on_regression,
        )
    )
    print(f"{result.status}: {result.reason}")
    if result.last and result.last.failures:
        sys.stdout.write(result.last.feedback)
    return 0 if result.status == APPROVED else 3 if result.status == WAITING else 1


def _count(text: str) -> int:
    """A whole number from 1 up, for an option."""
    number = int(text)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-acceptance",
        description="Acceptance checklists for coding agents.",
        epilog="Exit status: 0 approved (or clean), 1 changes, escalated or "
        "errors, 2 bad usage, 3 waiting (or undecided).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("lint", help="check checklists before approving them")
    sub.add_argument("files", nargs="+")
    sub.add_argument(
        "--oracles", help="comma-separated oracle names (default ci,llm,human)"
    )
    sub.add_argument("--strict", action="store_true", help="fail on warnings too")
    sub.set_defaults(handler=_cmd_lint)

    sub = commands.add_parser("show", help="print a checklist's requirements")
    sub.add_argument("file")
    sub.set_defaults(handler=_cmd_show)

    sub = commands.add_parser("digest", help="print a checklist's SHA-256 digest")
    sub.add_argument("file")
    sub.set_defaults(handler=_cmd_digest)

    sub = commands.add_parser("template", help="print a starting checklist")
    sub.add_argument("kind", choices=sorted(TEMPLATES))
    sub.set_defaults(handler=_cmd_template)

    sub = commands.add_parser("explain", help="decide a checklist from answers in JSON")
    sub.add_argument("file")
    sub.add_argument("answers", help='JSON: {"question_id": true, ...}')
    sub.set_defaults(handler=_cmd_explain)

    sub = commands.add_parser(
        "draft", help="draft a checklist from an issue with Claude"
    )
    sub.add_argument("issue", help="file with the issue text")
    sub.add_argument("--template", choices=sorted(TEMPLATES), default="bug")
    sub.add_argument("--out", default="acceptance.toml")
    sub.add_argument("--model", default=DEFAULT_MODEL)
    sub.add_argument("--force", action="store_true")
    sub.set_defaults(handler=_cmd_draft)

    sub = commands.add_parser("report", help="render a recorded run as Markdown")
    sub.add_argument("file", help="the checklist")
    sub.add_argument("record", help="the run record (JSON)")
    sub.set_defaults(handler=_cmd_report)

    sub = commands.add_parser("run", help="drive an agent until approval")
    sub.add_argument("file", help="the approved checklist")
    sub.add_argument("--agent", required=True, help='agent command, e.g. "claude -p"')
    sub.add_argument("--issue", help="file with the issue text")
    sub.add_argument("--cwd", default=".", help="the agent's checkout")
    sub.add_argument("--max-attempts", type=_count, default=5)
    sub.add_argument("--record", help="save the run here after every attempt")
    sub.add_argument("--resume", help="continue a saved run (and keep saving to it)")
    sub.add_argument("--github", help="owner/name, for CI results and reviews")
    sub.add_argument("--push", help="push each attempt to this branch, for CI")
    sub.add_argument(
        "--pr", type=_count, help="pull request to ask people on and report to"
    )
    sub.add_argument(
        "--ci-wait", type=float, default=1800.0, help="seconds to wait for CI"
    )
    sub.add_argument("--reviewers", help="comma-separated logins allowed to answer")
    sub.add_argument("--exclude", help="comma-separated logins never allowed to answer")
    sub.add_argument(
        "--model", default=DEFAULT_MODEL, help="Claude model for the judge"
    )
    sub.add_argument(
        "--samples", type=_count, default=1, help="judge samples per question"
    )
    sub.add_argument(
        "--keep-going-on-regression",
        action="store_true",
        help="hand a regression back to the agent instead of escalating",
    )
    sub.set_defaults(handler=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ClaudeError, DraftError, GitError, HTTPError, OracleError) as error:
        # What went wrong outside this program, or in how it was set up; the
        # record, if any, keeps the run as it was.
        raise SystemExit(f"error: {error}") from None
    except BrokenPipeError:
        # The reader went away (e.g. `| head`); stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
