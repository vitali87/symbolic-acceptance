"""``ci`` questions answered on this machine, and a router for them.

:class:`Checks` answers each question by its options:

* ``check = "frozen"``: the attempt leaves the frozen files alone. ``paths``
  lists the frozen glob patterns, and ``allow_new`` the patterns a new file
  may match anyway (the regression test). Patterns use :mod:`fnmatch`, where
  ``*`` also matches ``/``.
* ``run = "<command>"``: the command passes in a checkout of the attempt.
  ``expect = "fail"`` inverts it. Commands run without a shell.
* ``check = "<name>"`` with a handler registered under that name.

Anything else goes to ``fallback``, such as
:class:`symbolic_acceptance.github.GitHubChecks`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Context, OracleError, Pending, _call

__all__ = ["Checks", "Commands", "frozen_files"]

Handler = Callable[[Question, Context], Any]


def _patterns(question: Question, key: str) -> tuple[str, ...]:
    value = question.options.get(key, ())
    if isinstance(value, str):
        value = (value,)
    if not all(isinstance(item, str) for item in value):
        raise OracleError(f"q.{question.id}.{key} must be a list of glob patterns")
    return tuple(value)


def _matches(path: str | None, patterns: Sequence[str]) -> bool:
    return path is not None and any(fnmatch.fnmatchcase(path, p) for p in patterns)


def frozen_files(question: Question, context: Context) -> Answer:
    """YES when no frozen path is added, changed, renamed or deleted."""
    frozen = _patterns(question, "paths")
    if not frozen:
        raise OracleError(f"q.{question.id} needs paths = [<frozen globs>]")
    allow_new = _patterns(question, "allow_new")
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
    return Answer(True, f"no frozen file changed ({len(files)} file(s) touched)", by="ci:frozen")


class Commands:
    """Run a question's ``run`` command in ``cwd``; exit status 0 is YES.

    ``cwd`` must be checked out at the attempt's commit; when it is a git
    checkout at another commit the check refuses to run.
    """

    def __init__(self, cwd: str | Path, *, timeout: float = 1800.0, tail: int = 30) -> None:
        self.cwd = Path(cwd)
        self.timeout = timeout
        self.tail = tail

    def _head(self) -> str | None:
        try:
            done = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def __call__(self, question: Question, context: Context) -> Answer:
        command = question.options.get("run")
        if isinstance(command, str):
            argv = shlex.split(command)
        elif isinstance(command, Sequence) and all(isinstance(a, str) for a in command):
            argv = list(command)
        else:
            raise OracleError(f"q.{question.id}.run must be a command string or list")
        if not argv:
            raise OracleError(f"q.{question.id}.run is empty")
        head = self._head()
        ref = context.attempt.ref
        if head is not None and ref and not head.startswith(ref) and not ref.startswith(head):
            raise OracleError(
                f"{self.cwd} is at {head[:12]}, not at the attempt's commit {ref[:12]}"
            )
        expect = question.options.get("expect", "pass")
        if expect not in ("pass", "fail"):
            raise OracleError(f"q.{question.id}.expect must be \"pass\" or \"fail\"")
        shown = shlex.join(argv)
        try:
            done = subprocess.run(
                argv,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Answer(False, f"`{shown}` timed out after {self.timeout:.0f}s", by="ci:run")
        except OSError as error:
            return Answer(False, f"`{shown}` could not start: {error}", by="ci:run")
        output = (done.stdout + done.stderr).strip().splitlines()[-self.tail :]
        passed = done.returncode == 0
        is_met = passed if expect == "pass" else not passed
        verb = "passed" if passed else f"failed (exit {done.returncode})"
        return Answer(is_met, f"`{shown}` {verb}", tuple(output), by="ci:run")


class Checks:
    """A ``ci`` oracle: local checks first, then ``fallback`` for the rest."""

    def __init__(
        self,
        handlers: Mapping[str, Handler] | None = None,
        *,
        cwd: str | Path | None = None,
        fallback: Callable[..., Any] | None = None,
        timeout: float = 1800.0,
    ) -> None:
        self.handlers: dict[str, Handler] = {"frozen": frozen_files, **(handlers or {})}
        self.commands = Commands(cwd, timeout=timeout) if cwd is not None else None
        self.fallback = fallback

    async def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        rest: list[Question] = []
        for question in questions:
            name = question.options.get("check")
            if isinstance(name, str) and name in self.handlers:
                answers[question.id] = await _call(self.handlers[name], question, context)
            elif "run" in question.options and self.commands is not None:
                answers[question.id] = await asyncio.to_thread(
                    self.commands, question, context
                )
            else:
                rest.append(question)
        if rest:
            if self.fallback is None:
                ids = ", ".join(q.id for q in rest)
                raise OracleError(
                    f"no local check for {ids}: give it `run`, register a handler "
                    "for its `check`, or pass a fallback such as GitHubChecks"
                )
            try:
                answers.update(await _call(self.fallback, tuple(rest), context))
            except Pending as pending:
                raise Pending(str(pending), answers={**answers, **pending.answers}) from None
        return answers
