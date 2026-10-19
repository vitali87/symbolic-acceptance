"""``ci`` questions answered on this machine, and a router for them.

:class:`Checks` answers each question by its options:

* ``frozen = [<globs>]``: the attempt leaves the files matching these
  patterns alone; ``allow_new`` lists the patterns a new file may match
  anyway (the regression test). Patterns use :mod:`fnmatch`, where ``*``
  also matches ``/``, and ignore case.
* ``run = "<command>"``: the command passes in a fresh checkout of the
  attempt's commit (see :class:`Commands`). ``expect = "fail"`` inverts it.
  Commands run without a shell.

Anything else goes to ``fallback``, such as
:class:`symbolic_acceptance.github.GitHubChecks`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.agents import for_agent, git
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Context, Oracle, OracleError, Pending, _call

__all__ = ["Checks", "Commands", "command", "frozen_files", "frozen_paths", "matches"]

# Lines of a command's output kept as evidence.
TAIL = 30


def matches(path: str | None, patterns: Sequence[str]) -> bool:
    """Whether ``path`` matches one of the glob ``patterns``, ignoring case:
    on a case-insensitive file system ``Tests/x.py`` is ``tests/x.py``."""
    return path is not None and any(
        fnmatch.fnmatchcase(path.casefold(), pattern.casefold()) for pattern in patterns
    )


def command(question: Question) -> list[str]:
    """The arguments of the question's ``run`` command, a string or a list."""
    run = question.options.get("run")
    if isinstance(run, str):
        try:
            argv = shlex.split(run)
        except ValueError as error:  # an unclosed quote
            raise ValueError(f"q.{question.id}.run: {error}") from None
    else:
        argv = list(question.strings("run"))
    if not argv:
        raise ValueError(f"q.{question.id}.run is empty")
    return argv


def frozen_paths(question: Question) -> tuple[str, ...]:
    """The glob patterns of the question's ``frozen`` option."""
    paths = question.strings("frozen")
    if not paths:
        raise ValueError(f"q.{question.id}.frozen lists no paths")
    return paths


def frozen_files(question: Question, context: Context) -> Answer:
    """YES when no frozen path is added, changed, renamed or deleted."""
    frozen = frozen_paths(question)
    allow_new = question.strings("allow_new")
    diff = context.attempt.diff
    if diff is None:
        raise OracleError(f"q.{question.id} reads the attempt's diff, and it has none")
    files, _ = parse(diff)
    violations = []
    for change in files:
        if not (matches(change.path, frozen) or matches(change.old_path, frozen)):
            continue
        if change.status == "added" and matches(change.path, allow_new):
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

    Each command runs in a fresh checkout of exactly that commit, in a clone
    of the repository at ``repo`` that shares its objects but none of its
    configuration or attributes, nor the user's, and is removed afterwards.
    Nothing in the checkout outside the commit (ignored files, skip-worktree
    edits, a filter set up to rewrite files on checkout) can change the
    result, and one command cannot affect the next. The command runs the
    attempt's code, so like the agent it gets ``env`` (by default this
    process's environment) without the runner's GitHub token. A command
    that runs past ``timeout`` counts as failing; one that cannot start at
    all is an :class:`OracleError`, for a person to fix.
    """

    def __init__(
        self,
        repo: str | Path,
        *,
        timeout: float = 1800.0,
        env: Mapping[str, str] = os.environ,
    ) -> None:
        self.repo = Path(repo)
        self.timeout = timeout
        self.env = env

    def __call__(self, question: Question, context: Context) -> Answer:
        argv = command(question)
        expect = question.choice("expect", "pass", "fail")
        with tempfile.TemporaryDirectory(prefix="symbolic-acceptance-") as scratch:
            tree = str(Path(scratch, "tree"))
            source = str(self.repo.resolve())
            clone = ("clone", "--quiet", "--shared", "--no-checkout", source, tree)
            git(scratch, *clone, isolated=True)
            git(
                tree,
                "checkout",
                "--quiet",
                "--detach",
                context.attempt.ref,
                isolated=True,
            )
            return self._run(argv, tree, expect == "pass")

    def _run(self, argv: list[str], tree: str, wanted: bool) -> Answer:
        shown = shlex.join(argv)
        try:
            done = subprocess.run(
                argv,
                cwd=tree,
                env=for_agent(self.env),
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
        output = (done.stdout + done.stderr).strip().splitlines()[-TAIL:]
        passed = done.returncode == 0
        verb = "passed" if passed else f"failed (exit {done.returncode})"
        return Answer(passed == wanted, f"`{shown}` {verb}", tuple(output), by="ci:run")


class Checks:
    """A ``ci`` oracle: local checks first, then ``fallback`` for the rest.

    ``run`` questions need ``commands``; without it they go to ``fallback``.
    """

    def __init__(
        self, commands: Commands | None = None, *, fallback: Oracle | None = None
    ) -> None:
        self.commands = commands
        self.fallback = fallback

    def _local(
        self, question: Question
    ) -> Callable[[Question, Context], Answer] | None:
        if "frozen" in question.options:
            return frozen_files
        return self.commands if "run" in question.options else None

    def handles(self, question: Question) -> bool:
        """Whether a local check answers ``question``, without ``fallback``."""
        return self._local(question) is not None

    async def __call__(
        self, questions: Sequence[Question], context: Context
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        rest: list[Question] = []
        for question in questions:
            local = self._local(question)
            if local is None:
                rest.append(question)
            else:
                answers[question.id] = await asyncio.to_thread(local, question, context)
        if rest:
            if self.fallback is None:
                ids = ", ".join(q.id for q in rest)
                raise OracleError(
                    f"no local check for {ids}: give it `run` and Checks a Commands, "
                    "or pass a fallback such as GitHubChecks"
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
