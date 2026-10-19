"""``ci`` questions answered on this machine, and a router for them.

:class:`Checks` answers each question by its options:

* ``check = "frozen"``: the attempt leaves the frozen files alone. ``paths``
  lists the frozen glob patterns, and ``allow_new`` the patterns a new file
  may match anyway (the regression test). Patterns use :mod:`fnmatch`, where
  ``*`` also matches ``/``.
* ``run = "<command>"``: the command passes in a fresh checkout of the
  attempt's commit. ``expect = "fail"`` inverts it. Commands run without a
  shell.
* ``check = "<name>"`` with a handler registered under that name.

Anything else goes to ``fallback``, such as
:class:`symbolic_acceptance.github.GitHubChecks`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import shlex
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.agents import git
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Context, Oracle, OracleError, Pending, _call

__all__ = ["Checks", "Commands", "Handler", "command", "frozen_files"]

Handler = Callable[[Question, Context], Answer | Awaitable[Answer]]


def _matches(path: str | None, patterns: Sequence[str]) -> bool:
    return path is not None and any(fnmatch.fnmatchcase(path, p) for p in patterns)


def command(question: Question) -> list[str]:
    """The arguments of the question's ``run`` command, a string or a list."""
    run = question.options.get("run")
    if isinstance(run, str):
        try:
            argv = shlex.split(run)
        except ValueError as error:  # an unclosed quote
            raise OracleError(f"q.{question.id}.run: {error}") from None
    else:
        argv = list(question.strings("run"))
    if not argv:
        raise OracleError(f"q.{question.id}.run is empty")
    return argv


def frozen_files(question: Question, context: Context) -> Answer:
    """YES when no frozen path is added, changed, renamed or deleted."""
    frozen = question.strings("paths")
    if not frozen:
        raise OracleError(f"q.{question.id} needs paths = [<frozen globs>]")
    allow_new = question.strings("allow_new")
    files, _ = parse(context.attempt.diff)
    violations = []
    for change in files:
        if not (_matches(change.path, frozen) or _matches(change.old_path, frozen)):
            continue
        if change.status == "added" and _matches(change.path, allow_new):
            continue
        violations.append(
            f"{change.old_path} -> {change.path} (renamed)"
            if change.old_path
            else f"{change.path} ({change.status})"
        )
    if violations:
        return Answer(
            False,
            f"{len(violations)} frozen file(s) changed",
            tuple(violations),
            by="ci:frozen",
        )
    return Answer(
        True, f"no frozen file changed ({len(files)} file(s) touched)", by="ci:frozen"
    )


class Commands:
    """Run a question's ``run`` command on the attempt's commit; exit 0 is YES.

    Each command runs in a fresh checkout of exactly that commit, made with
    ``git worktree`` from the repository at ``repo`` and removed afterwards.
    Nothing outside the commit (ignored files, skip-worktree edits, a smudge
    filter) can change the result, and one command cannot affect the next. A
    command that runs past ``timeout`` counts as failing; one that cannot
    start at all is an :class:`OracleError`, for a person to fix.
    """

    def __init__(
        self, repo: str | Path, *, timeout: float = 1800.0, tail: int = 30
    ) -> None:
        self.repo = Path(repo)
        self.timeout = timeout
        self.tail = tail

    def __call__(self, question: Question, context: Context) -> Answer:
        argv = command(question)
        expect = question.options.get("expect", "pass")
        if expect not in ("pass", "fail"):
            raise OracleError(f'q.{question.id}.expect must be "pass" or "fail"')
        with tempfile.TemporaryDirectory(prefix="symbolic-acceptance-") as scratch:
            tree = str(Path(scratch, "tree"))
            git(
                self.repo,
                "worktree",
                "add",
                "--detach",
                "--quiet",
                tree,
                context.attempt.ref,
            )
            try:
                return self._run(argv, tree, expect == "pass")
            finally:
                git(self.repo, "worktree", "remove", "--force", tree)

    def _run(self, argv: list[str], tree: str, wanted: bool) -> Answer:
        shown = shlex.join(argv)
        try:
            done = subprocess.run(
                argv,
                cwd=tree,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Answer(
                False, f"`{shown}` timed out after {self.timeout:.0f}s", by="ci:run"
            )
        except OSError as error:
            raise OracleError(f"`{shown}` could not start: {error}") from None
        output = (done.stdout + done.stderr).strip().splitlines()[-self.tail :]
        passed = done.returncode == 0
        verb = "passed" if passed else f"failed (exit {done.returncode})"
        return Answer(passed == wanted, f"`{shown}` {verb}", tuple(output), by="ci:run")


class Checks:
    """A ``ci`` oracle: local checks first, then ``fallback`` for the rest."""

    def __init__(
        self,
        handlers: Mapping[str, Handler] | None = None,
        *,
        repo: str | Path | None = None,
        fallback: Oracle | None = None,
        timeout: float = 1800.0,
    ) -> None:
        self.handlers: dict[str, Handler] = {"frozen": frozen_files, **(handlers or {})}
        self.commands = Commands(repo, timeout=timeout) if repo is not None else None
        self.fallback = fallback

    def handles(self, question: Question) -> bool:
        """Whether a local check answers ``question``, without ``fallback``."""
        name = question.options.get("check")
        if isinstance(name, str) and name in self.handlers:
            return True
        return "run" in question.options and self.commands is not None

    async def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        rest: list[Question] = []
        for question in questions:
            name = question.options.get("check")
            if not self.handles(question):
                rest.append(question)
            elif isinstance(name, str) and name in self.handlers:
                answer = await _call(self.handlers[name], question, context)
                if not isinstance(answer, Answer):
                    raise OracleError(f"the {name!r} check did not return an Answer")
                answers[question.id] = answer
            else:
                assert self.commands is not None  # handles() checked it
                answers[question.id] = await asyncio.to_thread(
                    self.commands, question, context
                )
        if rest:
            if self.fallback is None:
                ids = ", ".join(q.id for q in rest)
                raise OracleError(
                    f"no local check for {ids}: give it `run`, register a handler "
                    "for its `check`, or pass a fallback such as GitHubChecks"
                )
            try:
                more = await _call(self.fallback, tuple(rest), context)
            except Pending as pending:
                raise Pending(
                    str(pending), answers={**answers, **pending.answers}
                ) from None
            if not isinstance(more, Mapping):
                raise OracleError("the fallback did not return a mapping of answers")
            answers.update(more)
        return answers
