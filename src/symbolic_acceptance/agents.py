"""Run a coding agent as a command; the commit it leaves is the attempt.

Any agent with a command line works. With Claude Code, for example::

    CommandAgent(["claude", "-p", "--permission-mode", "auto"], cwd=checkout)

The task prompt is sent on standard input. If the agent leaves changes
uncommitted, they are committed for it, so every attempt is one commit; the
diff is computed here with git, from where the run started to that commit,
and the agent's own output is kept only as its claims.

The agent runs without the runner's GitHub token, which would let it report
a check run or comment as the runner; an agent set on getting it, as the
runner's user, still can (docs/design.md says how to sandbox one). It can
write to the checkout's git configuration and attributes, so every git
command here overrides what they could change: hooks, replace refs, the file
monitor, and in the diff textconv and external drivers, binary files, path
prefixes, quoting, renames and how many files rename detection reads,
submodules, context and blank lines, object ids, how lines are matched up
and files ordered, and the function names in hunk headers, which it leaves
out. A filter can still rewrite what the agent's changes are committed as,
which is the agent's to decide anyway: everyone judges the commit.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Attempt, Escalate, Task

__all__ = [
    "GRACE",
    "OUTPUT",
    "RUNNER_SECRETS",
    "CommandAgent",
    "GitError",
    "diff",
    "for_agent",
    "git",
    "run_program",
    "stop_programs",
]

_OVERRIDES = (
    "--no-replace-objects",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    # Names as they are, so the judge can quote them; git still quotes a
    # name with quotes, backslashes or control characters in it.
    "core.quotePath=false",
)
# Git without the user's or the system's configuration, which the agent may
# have written too.
_ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
# Variables that point git at another repository, as git sets them for its
# hooks (`git rev-parse --local-env-vars`); git here works on the one in cwd.
_ELSEWHERE = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_GRAFT_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)
# The variables the runner reads its GitHub token from (see github.GitHub).
RUNNER_SECRETS = frozenset({"GITHUB_TOKEN", "GH_TOKEN"})


def for_agent(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` for running code the agent wrote: without
    :data:`RUNNER_SECRETS`, and without git variables pointing elsewhere."""
    return {k: v for k, v in env.items() if k not in RUNNER_SECRETS | _ELSEWHERE}


# The process groups of the programs run_program is running now, and
# whether stop_programs has been called, after which none may run.
_running: set[int] = set()
_stopped = threading.Event()
# Seconds the programs asked to stop have to end before they are killed.
GRACE = 5.0
# The most of a program's output kept, from its end: what the runner reads
# of it is its last lines.
OUTPUT = 1 << 20


def stop_programs(grace: float = GRACE) -> None:
    """End every program :func:`run_program` is running, and what they
    started, for a runner told to stop, and any it starts from now on.

    They are asked to stop (SIGTERM), so that git, say, removes its locks,
    and killed if they have not all ended within ``grace`` seconds.
    """
    _stopped.set()
    groups = _running.copy()
    for group in groups:
        _signal(group, signal.SIGTERM)
    deadline = time.monotonic() + grace
    # Signal 0 only asks whether a process of the group is left. One that
    # ended is left until it is reaped, which nothing does where the runner
    # is a container's first process: there, stopping takes the whole grace.
    while any(_signal(group, 0) for group in groups) and time.monotonic() < deadline:
        time.sleep(0.05)
    for group in groups:
        _signal(group, signal.SIGKILL)


def _signal(group: int, number: int) -> bool:
    """Send the process group signal ``number``: whether it had a process."""
    try:
        os.killpg(group, number)
    except ProcessLookupError:
        return False
    return True


def run_program(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    timeout: float,
    given: str = "",
) -> tuple[int | None, str]:
    """Run ``argv`` to its end and return its exit status (``None`` past
    ``timeout``) and its output, errors mixed in where they came: the last
    :data:`OUTPUT` bytes of it.

    It reads ``given`` and nothing else. It runs in a session of its own, so
    whatever it starts is stopped when it ends, but for a daemon, which
    leaves that session; and it writes to a file, so a child it leaves
    running with the output open cannot hold the runner up. A program that
    cannot start raises :class:`OSError`.
    """
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as log:
        stdin.write(given.encode("utf-8"))
        stdin.seek(0)
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=stdin,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _running.add(process.pid)
        if _stopped.is_set():  # stop_programs came as it started
            os.killpg(process.pid, signal.SIGKILL)
        try:
            status: int | None = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = None
        finally:
            # What it left running, unless stop_programs is giving it time.
            if not _stopped.is_set():
                _signal(process.pid, signal.SIGKILL)
            process.wait()
            _running.discard(process.pid)
        log.seek(max(0, log.seek(0, os.SEEK_END) - OUTPUT))
        return status, log.read().decode("utf-8", errors="replace")


_DIFF = (
    # A setting no option of git diff turns off.
    "-c",
    "diff.suppressBlankEmpty=false",
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--text",
    "--no-relative",
    "--ignore-submodules=none",
    "--submodule=short",
    "--find-renames",
    "-l1000",
    "--full-index",
    "--unified=3",
    "--inter-hunk-context=0",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "-O/dev/null",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)
# What follows a hunk's line numbers: the name of the function it is in, as
# a line the checkout's attributes can choose (diff.<driver>.xfuncname).
_FUNCTION_NAME = re.compile(r"^(@@ [^@\n]* @@).*$", re.MULTILINE)


class GitError(RuntimeError):
    """A git command the runner ran failed, or was not run because the
    runner is stopping (:func:`stop_programs`)."""


def git(
    cwd: str | Path, *args: str, isolated: bool = False, secrets: bool = False
) -> str:
    """Run git on the repository in ``cwd``, overriding its own settings.

    ``isolated`` also leaves out the user's and the system's configuration.
    Git may run programs the checkout's configuration names (a filter, a
    credential helper), so the runner's GitHub token is withheld unless
    ``secrets``, for a push whose credential helper needs it. Git does not
    ask for a password itself (ssh may, on the terminal), and gives up
    after five minutes.
    """
    # A stopped run commits, pushes or checks nothing a program left.
    if _stopped.is_set():
        raise GitError(f"git {' '.join(args)} not run: the runner is stopping")
    hidden = _ELSEWHERE if secrets else _ELSEWHERE | RUNNER_SECRETS
    env = {k: v for k, v in os.environ.items() if k not in hidden}
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        done = subprocess.run(
            ["git", *_OVERRIDES, *args],
            cwd=cwd,
            env=env | _ISOLATED if isolated else env,
            capture_output=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise GitError(f"git {' '.join(args)} ran past 300s") from None
    if done.returncode != 0:
        error = (done.stderr or done.stdout).decode(errors="replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {error}")
    # Decoded here rather than in text mode, which would take a carriage
    # return inside a line for a line break: file content could then forge
    # lines of a diff. "replace" covers binary and non-UTF-8 files.
    return done.stdout.decode(errors="replace")


def diff(cwd: str | Path, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as plain text, with no
    function names in its hunk headers.

    Its options override every setting that could shape it, so it may read
    the user's configuration like the runner's other git commands, and their
    ``safe.directory`` trusts a checkout someone else owns.
    """
    return _FUNCTION_NAME.sub(r"\1", git(cwd, *_DIFF, base, head))


class CommandAgent:
    """Run ``argv`` in ``cwd`` once per attempt.

    ``base`` is the commit attempts are compared with; it defaults to the
    checkout's HEAD now, before the agent has run, so pass it to
    :func:`symbolic_acceptance.loop.run` to keep it with the run, and pass
    the recorded one back when resuming. Each attempt starts from the
    previous one, or ``base`` for the first, which the checkout must hold:
    a commit the runner made after it that no record kept, as when a push
    failed, counts with the attempt. With ``push``, each attempt is pushed
    to that branch of ``origin``, so CI runs on it; the agent may rewrite
    its earlier attempts, and the branch follows as long as nobody else has
    pushed to it. An attempt that changes ``.github/`` is not pushed but
    handed to a person: pushing would run the workflows it holds, with
    whatever token they ask for, before any check could refuse them. The
    agent gets ``env`` (by default this process's environment) without
    :data:`RUNNER_SECRETS`.

    An agent that fails, runs past ``timeout`` or changes nothing hands the
    run to a person (:class:`Escalate`).
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path = ".",
        timeout: float = 3600.0,
        base: str | None = None,
        push: str | None = None,
        env: Mapping[str, str] = os.environ,
    ) -> None:
        if not argv:
            raise ValueError("argv must name the agent's command")
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.timeout = timeout
        self.push = push
        self.env = env
        self.base = base or self._head()

    def _head(self) -> str:
        return git(self.cwd, "rev-parse", "HEAD").strip()

    def _tree(self, commit: str) -> str:
        return git(self.cwd, "rev-parse", f"{commit}^{{tree}}").strip()

    def __call__(self, task: Task) -> Attempt:
        before = self._head()
        start, what = (
            (task.previous.ref, f"attempt {task.number - 1}")
            if task.previous
            else (self.base, "the run's base")
        )
        # What follows it is the runner's commits that no record kept, as when
        # a push failed, which count with this attempt; another clone may not
        # hold it at all.
        held = git(self.cwd, "rev-list", "-1", "--ignore-missing", start).strip()
        if not held or git(self.cwd, "rev-list", "-1", start, "--not", before).strip():
            raise Escalate(
                f"the checkout is at {before[:12]}, which does not follow {what} "
                f"({start[:12]}); check that out to continue"
            )
        status, output = run_program(
            self.argv,
            cwd=self.cwd,
            env=for_agent(self.env) | {"PWD": str(self.cwd.resolve())},
            timeout=self.timeout,
            given=task.prompt(),
        )
        if status is None:
            raise Escalate(f"the agent ran past {self.timeout:g}s")
        if status != 0:
            raise Escalate(f"the agent exited with {status}: {output.strip()[-2000:]}")
        # Whatever a commit can hold is committed, whatever the checkout's
        # settings would hide; what none can (an edit inside a submodule,
        # line endings git normalizes) is no change.
        git(self.cwd, "add", "--all")
        if git(self.cwd, "write-tree").strip() != self._tree("HEAD"):
            git(self.cwd, "commit", "--quiet", "-m", f"Attempt {task.number}")
        after = self._head()
        # The tree, not the commit: an empty commit, or one and its revert,
        # change nothing either.
        if self._tree(after) == self._tree(start):
            raise Escalate(f"attempt {task.number} changed nothing")
        changed = diff(self.cwd, self.base, after)
        if self.push is not None:
            workflows = sorted(
                {
                    path
                    for change in parse(changed)[0]
                    for path in (change.path, change.old_path)
                    if path and path.casefold().split("/")[0] == ".github"
                }
            )
            if workflows:
                raise Escalate(
                    f"attempt {task.number} changes {', '.join(workflows)}, which "
                    "pushing would run; a person must review it first"
                )
            branch = f"refs/heads/{self.push}"
            # The runner replaces only its own commit on the branch, which the
            # agent may have rewritten: the previous attempt, or the commit
            # this one started from, pushed by a run stopped before it kept it.
            # Anything else, someone else's, it only fast-forwards.
            ls = ("ls-remote", "origin", branch)
            tip = git(self.cwd, *ls, secrets=True).split()[:1]
            ours = {before, task.previous.ref} if task.previous else {before}
            lease = [f"--force-with-lease={branch}:{ref}" for ref in tip if ref in ours]
            push = ("push", "--quiet", *lease, "origin", f"{after}:{branch}")
            git(self.cwd, *push, secrets=True)
        return Attempt(
            ref=after,
            diff=changed,
            claims=output.strip()[-4000:],
        )
