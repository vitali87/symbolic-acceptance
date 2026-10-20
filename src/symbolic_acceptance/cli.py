"""The ``symbolic-acceptance`` command (also ``python -m symbolic_acceptance``)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import sys
import threading
import urllib.request
from collections.abc import Collection, Sequence
from pathlib import Path

from symbolic_acceptance import __version__, record
from symbolic_acceptance.agents import (
    CommandAgent,
    GitError,
    git,
    stop_programs,
    stopping,
)
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    Checklist,
    ChecklistError,
    Question,
    load,
    render,
)
from symbolic_acceptance.checks import Checks, Commands, command
from symbolic_acceptance.claude import DEFAULT_MODEL, Claude, ClaudeError
from symbolic_acceptance.draft import DraftError, draft
from symbolic_acceptance.feedback import assess, render_failures
from symbolic_acceptance.github import GitHub, GitHubChecks, Reviews, post_report
from symbolic_acceptance.http import HTTPError, Response, urlopen
from symbolic_acceptance.judge import Judge
from symbolic_acceptance.lint import lint
from symbolic_acceptance.logic import facts, plan
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


def _record(path: str, checklist: Checklist) -> Result:
    try:
        return record.loads(_read(path), checklist)
    except ValueError as error:
        raise SystemExit(f"error: {path}: {error}") from None


def _cmd_lint(args: argparse.Namespace) -> int:
    oracles = (
        tuple(name.strip() for name in args.oracles.split(","))
        if args.oracles
        else ORACLES
    )
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
    try:
        data: object = json.loads(_read(path))
    except json.JSONDecodeError as error:
        raise SystemExit(f"error: {path}: {error}") from None
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
    if assessment.status == "open":
        ask = plan(checklist, facts(answers), last=())
        print(f"\nTo decide it, answer: {', '.join(ask)}")
    return {"approved": 0, "changes": 1, "open": 3}[assessment.status]


def _cmd_draft(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.is_dir():
        raise SystemExit(f"error: {out} is a folder; name the file to write")
    if out.exists() and not args.force:
        raise SystemExit(f"error: {out} exists; pass --force to replace it")
    # Checked before Claude is asked, and paid, for a draft.
    if not out.parent.is_dir():
        raise SystemExit(f"error: no folder to write {out} in")
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


# A reply at the terminal: y, yes, n or no, as a word, then why.
_REPLY = re.compile(r"(yes|y|no|n)(?:[\s,:;.!-]+(.*))?", re.IGNORECASE)


async def _ask_at_the_terminal(
    questions: Sequence[Question], context: Context
) -> dict[str, Answer]:
    """A ``human`` oracle that asks on the terminal. It writes to stderr, so
    that the questions show when the run's result is sent elsewhere."""
    if not sys.stdin.isatty():
        raise Pending("no terminal to ask a person; pass --pr to ask on GitHub")
    attempt = context.attempt
    answers: dict[str, Answer] = {}
    asked = f"\nQuestions about attempt {attempt.number} ({attempt.ref[:12]}):"
    print(asked, file=sys.stderr)
    for question in questions:
        if question.context:
            print(f"  ({' '.join(question.context.split())})", file=sys.stderr)
        while True:
            prompt = f"  {question.id}: {question.ask} [y/n, then why] "
            print(prompt, end="", flush=True, file=sys.stderr)
            typed = await _typed()
            if not typed:
                raise Pending(
                    "the person at the terminal stopped answering", answers=answers
                )
            if reply := _REPLY.fullmatch(typed.strip()):
                yes = reply[1][0].lower() == "y"
                why = reply[2] or ""
                answers[question.id] = Answer(yes, why, by="human:terminal")
                break
    return answers


async def _typed() -> str:
    """The next line typed at the terminal, as far as Ctrl-D if that ends it,
    or "" once its input ends.

    It is waited for on the event loop, which a stopping run stops, rather
    than by input() in a thread, which nothing can stop; and read in one go,
    as much as the terminal has, so that a line Ctrl-D ends cannot block.
    """
    loop = asyncio.get_running_loop()
    typed = asyncio.Event()
    loop.add_reader(sys.stdin, typed.set)
    try:
        await typed.wait()
    finally:
        loop.remove_reader(sys.stdin)
    return os.read(sys.stdin.fileno(), 1 << 16).decode(errors="replace")


# Ctrl-C, what a supervisor or a cancelled CI job sends, and a hang-up: the
# terminal closed, which would end the runner and leave the agent working.
_STOPS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def _stop(signum: int, stops: Collection[int]) -> None:
    """End the agent, check or git running now, start and send nothing more,
    and stop as soon as the rest has, the record kept as last saved; a
    second signal, for a request still waiting on the network, quits at
    once."""
    for stop in stops:
        signal.signal(stop, _quit)
    name = signal.Signals(signum).name
    try:
        print(f"stopping on {name}; send it again to quit at once", file=sys.stderr)
    except OSError:
        # Whoever read stderr went with the same Ctrl-C. What is written there
        # from now on goes nowhere, so that flushing it cannot fail the exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stderr.fileno())
    stop_programs()
    raise SystemExit(128 + signum)


def _quit(signum: int, frame: object) -> None:
    stop_programs(grace=0)
    os._exit(128 + signum)


def _stop_on_signals() -> None:
    """Stop the running loop on Ctrl-C, SIGTERM or SIGHUP as :func:`_stop`
    does, unless one was ignored from the start (a shell's background job),
    or this is not the main thread, the only one that can handle signals.

    The loop takes them, as a callback between its steps, rather than a
    handler Python runs wherever the main thread is: the main thread, waiting
    on the loop, is not woken by a signal another thread receives, and a
    handler run in the middle of a write to stderr cannot write to it.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    loop = asyncio.get_running_loop()
    stops = [stop for stop in _STOPS if signal.getsignal(stop) is not signal.SIG_IGN]
    for stop in stops:
        loop.add_signal_handler(stop, _stop, stop, stops)


def _send(request: urllib.request.Request, timeout: float) -> Response:
    """Send nothing once the run is stopping, as git runs nothing then."""
    if stopping():
        raise HTTPError(
            request.get_method(), request.full_url, 0, "not sent: the run is stopping"
        )
    return urlopen(request, timeout)


def _installed(program: str, what: str) -> None:
    """Refuse a program the machine lacks: one named without a folder, looked
    for on PATH, or by its whole path. One named by a relative path is the
    checkout's, which an attempt may add."""
    if ("/" not in program or os.path.isabs(program)) and not shutil.which(program):
        raise SystemExit(f"error: {what}: {program} is not a program on this machine")


def _cmd_run(args: argparse.Namespace) -> int:
    checklist = _load(args.file)
    errors = [i for i in lint(checklist, oracles=ORACLES) if i.level == "error"]
    if errors:
        raise SystemExit(
            "error: the checklist has lint errors:\n" + "\n".join(map(str, errors))
        )
    try:
        argv = shlex.split(args.agent)
    except ValueError as error:
        raise SystemExit(f"error: --agent: {error}") from None
    if not argv:
        raise SystemExit("error: --agent names no command")
    if not Path(args.cwd).is_dir():
        raise SystemExit(f"error: --cwd {args.cwd} is not a folder")
    # What would fail only once the agent has worked is checked now.
    _installed(argv[0], "--agent")
    used = [checklist.questions[qid] for qid in checklist.used()]
    for question in used:
        if "run" in question.options:
            _installed(command(question)[0], f"q.{question.id}.run")
    issue = _read(args.issue) if args.issue else ""
    if args.github and not re.fullmatch(r"[^/\s]+/[^/\s]+", args.github):
        raise SystemExit("error: --github takes a repository as owner/name")
    if args.pr and not args.github:
        raise SystemExit("error: --pr needs --github")
    github = GitHub(args.github, transport=_send) if args.github else None
    ci_fallback = GitHubChecks(github, wait=args.ci_wait) if github else None
    exclude = args.exclude.split(",") if args.exclude else []
    human: Oracle = _ask_at_the_terminal
    if github and args.pr:
        human = Reviews(
            github,
            args.pr,
            reviewers=args.reviewers.split(",") if args.reviewers else None,
            exclude=exclude,
        )
    claude = Claude(model=args.model, transport=_send)
    ci = Checks(Commands(args.cwd), fallback=ci_fallback)
    remote = [q for q in used if q.by == "ci" and not ci.handles(q)]
    unnamed = [q.id for q in remote if "check" not in q.options]
    if unnamed:
        raise SystemExit(f"error: {', '.join(unnamed)} need check, run or frozen")
    if remote and not (github and args.push):
        raise SystemExit(
            f"error: GitHub check runs answer {', '.join(q.id for q in remote)}; "
            "pass --github and --push so that every attempt reaches CI"
        )
    # Without one, GitHub hides a private repository and soon limits the rate.
    if github and (args.pr or remote) and not github.token:
        raise SystemExit("error: --github needs a token in GITHUB_TOKEN or GH_TOKEN")
    oracles: dict[str, Oracle] = {
        "ci": ci,
        "llm": Judge(claude, samples=args.samples),
        "human": human,
    }
    resume: Result | None = None
    if args.resume:
        resume = _record(args.resume, checklist)
        if issue and resume.issue and issue != resume.issue:
            raise SystemExit("error: --issue differs from the run being resumed")
    record_path = args.record or args.resume
    if args.record and Path(args.record).is_dir():
        raise SystemExit(f"error: {args.record} is a folder; name the file to save in")
    if (
        args.record
        and Path(args.record).exists()
        and not (args.resume and Path(args.record).samefile(args.resume))
    ):
        raise SystemExit(
            f"error: {args.record} holds a run; pass --resume {args.record} to "
            "continue it"
        )
    if record_path:
        # Saving replaces the file named in this folder, even a link.
        folder = Path(record_path).parent.resolve()
        if not folder.is_dir():
            raise SystemExit(f"error: no folder to save {record_path} in")
        # The same folder, however its name is spelt: macOS takes a name in
        # other letters, or another Unicode form, for the checkout's.
        top = Path(git(args.cwd, "rev-parse", "--show-toplevel").strip())
        if any(top.samefile(path) for path in (folder, *folder.parents)):
            raise SystemExit(
                "error: keep the run record outside the agent's checkout (--cwd); "
                "inside it, the record would be committed with the attempt and "
                "the agent could rewrite it"
            )
    # What a commit could hold, whatever the checkout's settings would hide.
    status = "status --porcelain --untracked-files=all --ignore-submodules=dirty"
    if resume is None and git(args.cwd, *status.split()):
        raise SystemExit(
            "error: the checkout (--cwd) has changes no commit holds; commit or "
            "stash them, so that the attempts hold only what the agent does"
        )
    # Checked now, before the agent runs, rather than when first needed: a
    # name to commit the agent's leftovers in, and where to push them.
    git(args.cwd, "var", "GIT_COMMITTER_IDENT")
    if args.push:
        git(args.cwd, "check-ref-format", "--branch", args.push)
        git(args.cwd, "remote", "get-url", "origin")
    judged = any(q.by == "llm" for q in used)
    if judged and not (claude.api_key or claude.auth_token):
        raise SystemExit(
            "error: the judge needs ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"
        )
    # A resumed run keeps diffing against the commit it started from.
    agent = CommandAgent(
        argv,
        cwd=args.cwd,
        timeout=args.agent_timeout,
        base=resume.base if resume else None,
        push=args.push,
    )

    report: int | None = None  # the report comment's id, once known

    def progress(result: Result) -> None:
        nonlocal report
        if record_path:
            record.save(result, checklist, record_path)
        if github and args.pr:
            text = markdown(result, checklist)
            report = post_report(github, args.pr, text, exclude=exclude, report=report)

    async def stoppable() -> Result:
        _stop_on_signals()
        try:
            return await run(
                checklist,
                agent=agent,
                oracles=oracles,
                issue=issue,
                max_attempts=args.max_attempts,
                resume=resume,
                on_progress=progress,
                stop_on_regression=not args.keep_going_on_regression,
                base=agent.base,
            )
        except Exception:
            # As Ctrl-C does, so that the error is not held back until a check
            # still running in a thread, whose answer nobody reads now, ends.
            stop_programs()
            raise

    result = asyncio.run(stoppable())
    print(f"{result.status}: {result.reason}")
    if result.status == WAITING and record_path:
        print(f"Run it again with --resume {record_path} to go on.")
    if result.last and result.last.failures:
        sys.stdout.write(result.last.feedback)
    return 0 if result.status == APPROVED else 3 if result.status == WAITING else 1


def _seconds(text: str) -> float:
    """A number of seconds from 0 up, for an option."""
    seconds = float(text)
    if not seconds >= 0:  # nan is not
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {text}")
    return seconds


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
        "errors (options it refuses included), 2 malformed arguments, 3 waiting "
        "(or undecided), 130, 143 or 129 stopped by Ctrl-C, SIGTERM or a hang-up.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("lint", help="check checklists before approving them")
    sub.add_argument("files", nargs="+", help="checklists (TOML)")
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
    sub.add_argument(
        "answers", help='a file of answers in JSON: {"question_id": true, ...}'
    )
    sub.set_defaults(handler=_cmd_explain)

    sub = commands.add_parser(
        "draft", help="draft a checklist from an issue with Claude"
    )
    sub.add_argument("issue", help="file with the issue text")
    sub.add_argument(
        "--template",
        choices=sorted(TEMPLATES),
        default="bug",
        help="the template to start from (default bug)",
    )
    sub.add_argument(
        "--out",
        default="acceptance.toml",
        help="where to write the draft (default acceptance.toml)",
    )
    sub.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})"
    )
    sub.add_argument("--force", action="store_true", help="replace --out if it exists")
    sub.set_defaults(handler=_cmd_draft)

    sub = commands.add_parser("report", help="render a recorded run as Markdown")
    sub.add_argument("file", help="the checklist")
    sub.add_argument("record", help="the run record (JSON)")
    sub.set_defaults(handler=_cmd_report)

    sub = commands.add_parser("run", help="drive an agent until approval")
    sub.add_argument("file", help="the approved checklist")
    sub.add_argument("--agent", required=True, help='agent command, e.g. "claude -p"')
    sub.add_argument("--issue", help="file with the issue text")
    sub.add_argument("--cwd", default=".", help="the agent's checkout (default .)")
    sub.add_argument(
        "--agent-timeout",
        type=_seconds,
        default=3600.0,
        help="seconds the agent has for each attempt (default 3600)",
    )
    sub.add_argument(
        "--max-attempts",
        type=_count,
        default=5,
        help="attempts before a person must decide (default 5)",
    )
    sub.add_argument("--record", help="save the run here after every attempt")
    sub.add_argument("--resume", help="continue a saved run (and keep saving to it)")
    sub.add_argument("--github", help="owner/name, for CI results and reviews")
    sub.add_argument("--push", help="push each attempt to this branch, for CI")
    sub.add_argument(
        "--pr", type=_count, help="pull request to ask people on and report to"
    )
    sub.add_argument(
        "--ci-wait",
        type=_seconds,
        default=1800.0,
        help="seconds to wait for CI before pausing the run (default 1800)",
    )
    sub.add_argument("--reviewers", help="comma-separated logins allowed to answer")
    sub.add_argument("--exclude", help="comma-separated logins never allowed to answer")
    sub.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model for the judge (default {DEFAULT_MODEL})",
    )
    sub.add_argument(
        "--samples",
        type=_count,
        default=1,
        help="times the judge answers each question, all to agree for a pass "
        "(default 1)",
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
        code = int(args.handler(args))
        # Here rather than at exit, where a failure is past handling.
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # The reader went away (e.g. `| head`); stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    except KeyboardInterrupt:
        # Ctrl-C outside run, which stops what it runs first: draft waiting
        # on Claude, say.
        return 128 + signal.SIGINT
    except (
        ClaudeError,
        DraftError,
        GitError,
        HTTPError,
        OracleError,
        OSError,
    ) as error:
        # What went wrong outside this program, or in how it was set up; the
        # record, if any, keeps the run as it was.
        raise SystemExit(f"error: {error}") from None
    except json.JSONDecodeError as error:
        # A service's reply that is not JSON: a proxy's page, say, or another
        # server at ANTHROPIC_BASE_URL or GITHUB_API_URL.
        raise SystemExit(f"error: a reply was not JSON: {error.doc[:300]!r}") from None
