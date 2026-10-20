"""``ci`` questions answered on this machine, and a router for them.

:class:`Checks` answers each question by its options:

* ``frozen = [<globs>]``: the attempt leaves the files matching these
  patterns alone; ``allow_new`` lists the patterns a new file, or one moved
  in, may match anyway (the regression test). Patterns use :mod:`fnmatch`,
  where ``*`` also matches ``/``, and ignore case; in ``allow_new``, which
  lets files in, a pattern matches only paths as deep as itself, so ``*``
  stays within a directory.
* ``run = "<command>"``: the command passes in a fresh checkout of the
  attempt's commit (see :class:`Commands`). ``expect = <status>`` asks for
  that exit status instead, and ``expect = "fail"`` for any but 0. Commands
  run without a shell.

Anything else goes to ``fallback``, such as
:class:`symbolic_acceptance.github.GitHubChecks`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import shlex
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.agents import for_agent, git, run_program
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import Question
from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Context, Oracle, OracleError, Pending, _call

__all__ = [
    "Checks",
    "Commands",
    "admits",
    "command",
    "expected",
    "frozen_files",
    "frozen_paths",
    "matches",
    "patterns",
]

# Lines of a command's output kept as evidence.
TAIL = 30


def matches(path: str | None, patterns: Sequence[str]) -> bool:
    """Whether ``path`` matches one of the glob ``patterns``, ignoring case:
    on a case-insensitive file system ``Tests/x.py`` is ``tests/x.py``."""
    return path is not None and any(
        fnmatch.fnmatchcase(path.casefold(), pattern.casefold()) for pattern in patterns
    )


def admits(path: str, allow_new: Sequence[str]) -> bool:
    """Whether ``allow_new`` lets a new file at ``path`` in: a pattern matches
    only paths as deep as itself, so its ``*`` stays within a directory."""
    depth = path.count("/")
    return matches(path, [p for p in allow_new if p.count("/") == depth])


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


def patterns(question: Question, key: str) -> tuple[str, ...]:
    """The glob patterns of the question's option ``key``, which name files
    from the top of the repository: ``tests/*``, where ``./tests/*``,
    ``/tests/*`` and ``tests/`` would match nothing."""
    globs = question.strings(key)
    for glob in globs:
        if glob.startswith(("/", "./")) or glob.endswith("/"):
            raise ValueError(
                f"q.{question.id}.{key} has {glob!r}, which matches no file: "
                "patterns name files from the top of the repository, as tests/*"
            )
    return globs


def expected(question: Question) -> Callable[[int], bool]:
    """Whether an exit status is what the question's ``expect`` asks for: 0
    (``"pass"``, the default), any other (``"fail"``), or that very status."""
    expect = question.options.get("expect", "pass")
    if isinstance(expect, int) and not isinstance(expect, bool):
        return lambda status: status == expect
    if expect == "pass":
        return lambda status: status == 0
    if expect == "fail":
        return lambda status: status != 0
    raise ValueError(f'q.{question.id}.expect must be "pass", "fail" or an exit status')


def frozen_paths(question: Question) -> tuple[str, ...]:
    """The glob patterns of the question's ``frozen`` option."""
    paths = patterns(question, "frozen")
    if not paths:
        raise ValueError(f"q.{question.id}.frozen lists no paths")
    return paths


def frozen_files(question: Question, context: Context) -> Answer:
    """YES when no frozen path is added, changed, renamed or deleted."""
    frozen = frozen_paths(question)
    allow_new = patterns(question, "allow_new")
    diff = context.attempt.diff
    if diff is None:
        raise OracleError(f"q.{question.id} reads the attempt's diff, and it has none")
    files, _ = parse(diff)
    violations = []
    for change in files:
        was_frozen = matches(change.old_path, frozen)
        if not (matches(change.path, frozen) or was_frozen):
            continue
        # A file moved in from elsewhere is as new to the frozen ones as one
        # added: the new regression test may come from the issue's repro.
        added = change.status == "added" or (
            change.status == "renamed" and not was_frozen
        )
        if added and admits(change.path, allow_new):
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
    """Run a question's ``run`` command on the attempt's commit; exit 0 is YES,
    or the status ``expect`` names.

    Each command runs in a fresh checkout of exactly that commit, fetched
    from the repository at ``repo`` with none of its configuration or
    attributes, nor the user's, and removed afterwards. The fetch checks
    every object against its id, so neither anything in the checkout outside
    the commit (ignored files, skip-worktree edits, a filter set up to
    rewrite files on checkout) nor bytes rewritten in its object store can
    change the result, and no command sees another's checkout. Submodules
    are left empty and Git LFS files as their pointers: their contents are
    not the commit's. The command runs the attempt's code, so like the agent
    it gets ``env`` (by default this process's environment) without the
    runner's GitHub token, and no input; whatever it starts is stopped when
    it ends, but for a daemon, which leaves its session. A command that runs
    past ``timeout`` answers NO, whatever ``expect`` says, as does a program
    of the attempt's own (a relative path, like ``./repro.sh``) that is gone
    or cannot run; a program the machine lacks is an :class:`OracleError`,
    for a person to fix.
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
        expect = expected(question)
        top = git(self.repo, "rev-parse", "--show-toplevel").strip()
        with tempfile.TemporaryDirectory(prefix="symbolic-acceptance-") as scratch:
            tree = str(Path(scratch, "tree"))
            source = Path(top).as_uri()
            git(scratch, "init", "--quiet", tree, isolated=True)
            # Every file as the commit holds it, whatever the commit's own
            # .gitattributes would make of it on checkout: line endings, an
            # encoding, $Id$. These attributes outrank the commit's.
            info = Path(tree, ".git", "info")
            info.mkdir(exist_ok=True)
            Path(info, "attributes").write_text(
                "* -text -ident -working-tree-encoding -filter\n"
            )
            # Git found the checkout trusted just above: the user's, or named
            # by their safe.directory, which the git that reads it for this
            # fetch does not see, so it is told.
            upload = "--upload-pack=git -c 'safe.directory=*' upload-pack"
            fetch = ("fetch", "--quiet", "--depth=1", "--no-tags", upload, source)
            # As long as a command may take: a large repository takes long.
            git(tree, *fetch, context.attempt.ref, isolated=True, timeout=self.timeout)
            checkout = ("checkout", "--quiet", "--detach", "FETCH_HEAD")
            git(tree, *checkout, isolated=True, timeout=self.timeout)
            return self._run(argv, tree, expect)

    def _run(self, argv: list[str], tree: str, expect: Callable[[int], bool]) -> Answer:
        shown = shlex.join(argv)
        env = for_agent(self.env) | {"PWD": tree}
        try:
            status, text = run_program(argv, cwd=tree, env=env, timeout=self.timeout)
        except OSError as error:
            # A program the commit holds (./repro.sh) is the attempt's to keep
            # working; one the machine lacks is not.
            if "/" not in argv[0] or os.path.isabs(argv[0]):
                raise OracleError(f"`{shown}` could not start: {error}") from None
            return Answer(False, f"`{shown}` could not start: {error}", by="ci:run")
        output = tuple(text.strip().splitlines()[-TAIL:])
        if status is None:
            reason = f"`{shown}` timed out after {self.timeout:g}s"
            return Answer(False, reason, output, by="ci:run")
        verb = "passed" if status == 0 else f"failed (exit {status})"
        return Answer(expect(status), f"`{shown}` {verb}", output, by="ci:run")


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
