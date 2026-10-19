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
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    Checklist,
    ChecklistError,
    Question,
    load,
    render,
)
from symbolic_acceptance.feedback import assess, render_failures
from symbolic_acceptance.lint import lint
from symbolic_acceptance.loop import (
    APPROVED,
    WAITING,
    Context,
    Oracle,
    Pending,
    Result,
    run,
)
from symbolic_acceptance.report import markdown
from symbolic_acceptance.templates import TEMPLATES

__all__ = ["main"]

ORACLES = ("ci", "llm", "human")


def _load(path: str) -> Checklist:
    try:
        return load(path)
    except (OSError, ChecklistError) as error:
        raise SystemExit(f"error: {error}") from None


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
        issues = lint(checklist, oracles=oracles)
        for issue in issues:
            print(f"{path}: {issue}")
        if any(i.level == "error" for i in issues) or (
            args.strict and any(i.level == "warning" for i in issues)
        ):
            worst = 1
        elif not issues:
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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(
            "error: answers must be a JSON object of question id to answer"
        )
    answers = {}
    for qid, value in data.items():
        if isinstance(value, bool):
            answers[qid] = Answer(value)
        elif isinstance(value, dict):
            answers[qid] = Answer.from_dict(value)
        else:
            raise SystemExit(f"error: {qid}: expected true, false or an answer object")
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
    return {"approved": 0, "changes": 1}.get(assessment.status, 2)


def _cmd_draft(args: argparse.Namespace) -> int:
    from symbolic_acceptance.claude import Claude
    from symbolic_acceptance.draft import DraftError, draft

    issue = Path(args.issue).read_text(encoding="utf-8")
    claude = Claude(model=args.model) if args.model else Claude()
    try:
        result = draft(issue, template=args.template, claude=claude)
    except DraftError as error:
        raise SystemExit(f"error: {error}") from None
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"error: {out} exists; pass --force to replace it")
    out.write_text(result.text if result.text.endswith("\n") else result.text + "\n")
    print(f"wrote {out}; review it before approving")
    for issue_found in result.issues:
        print(f"  {issue_found}")
    if result.questions:
        print("Questions for the reporter:")
        for question in result.questions:
            print(f"  - {question}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    sys.stdout.write(markdown(record.load(args.record, checklist), checklist))
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
    from symbolic_acceptance.agents import CommandAgent
    from symbolic_acceptance.checks import Checks
    from symbolic_acceptance.claude import Claude
    from symbolic_acceptance.github import GitHub, GitHubChecks, Reviews, post_report
    from symbolic_acceptance.judge import Judge

    checklist = _load(args.file)
    errors = [i for i in lint(checklist, oracles=ORACLES) if i.level == "error"]
    if errors:
        raise SystemExit(
            "error: the checklist has lint errors:\n" + "\n".join(map(str, errors))
        )
    issue = Path(args.issue).read_text(encoding="utf-8") if args.issue else ""
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
    claude = Claude(model=args.model) if args.model else Claude()
    ci = Checks(cwd=args.cwd, fallback=ci_fallback)
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
    oracles = {"ci": ci, "llm": Judge(claude, samples=args.samples), "human": human}
    resume: Result | None = None
    if args.resume:
        resume = record.load(args.resume, checklist)
    record_path = args.record or args.resume
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
    progress(result)
    print(f"{result.status}: {result.reason}")
    if result.last and result.last.failures:
        sys.stdout.write(result.last.feedback)
    return 0 if result.status == APPROVED else 3 if result.status == WAITING else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symbolic-acceptance",
        description="Acceptance checklists for coding agents.",
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
    sub.add_argument("--model")
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
    sub.add_argument("--max-attempts", type=int, default=5)
    sub.add_argument("--record", help="save the run here after every attempt")
    sub.add_argument("--resume", help="continue a saved run (and keep saving to it)")
    sub.add_argument("--github", help="owner/name, for CI results and reviews")
    sub.add_argument("--push", help="push each attempt to this branch, for CI")
    sub.add_argument(
        "--pr", type=int, help="pull request to ask people on and report to"
    )
    sub.add_argument(
        "--ci-wait", type=float, default=1800.0, help="seconds to wait for CI"
    )
    sub.add_argument("--reviewers", help="comma-separated logins allowed to answer")
    sub.add_argument("--exclude", help="comma-separated logins never allowed to answer")
    sub.add_argument("--model", help="Claude model for the judge")
    sub.add_argument(
        "--samples", type=int, default=1, help="judge samples per question"
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
    except BrokenPipeError:
        # The reader went away (e.g. `| head`); stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
