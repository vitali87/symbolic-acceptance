"""Run a coding agent as a command; the commit it leaves is the attempt.

Any agent with a command line works. With Claude Code, for example::

    CommandAgent(["claude", "-p", "--permission-mode", "auto"], cwd=checkout)

The task prompt is sent on standard input. If the agent leaves changes
uncommitted, they are committed for it, so every attempt is one commit; the
diff is computed here with git, from where the run started to that commit,
and the agent's own output is kept only as its claims.

The agent runs without the runner's GitHub token, which would let it report
a check run or comment as the runner. It can write to the checkout's git
configuration and attributes, so every git command here overrides what they
could change: hooks, replace refs, filters, textconv and external diff
drivers, path prefixes, quoting, ignored submodules, context lines.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.loop import Attempt, Escalate, Task

__all__ = [
    "RUNNER_SECRETS",
    "AgentError",
    "CommandAgent",
    "diff",
    "git",
    "local_diff",
    "without_secrets",
]

# Git's empty tree: reading attributes from it means none apply, so no
# smudge filter or diff attribute in the checkout can alter what git shows.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
_OVERRIDES = (
    "--no-replace-objects",
    f"--attr-source={EMPTY_TREE}",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.quotePath=true",
)
# Git without the user's or the system's configuration, which the agent may
# have written too.
_ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
# The variables the runner reads its GitHub token from (see github.GitHub).
RUNNER_SECRETS = frozenset({"GITHUB_TOKEN", "GH_TOKEN"})


def without_secrets(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without :data:`RUNNER_SECRETS`, for running code the agent wrote."""
    return {k: v for k, v in env.items() if k not in RUNNER_SECRETS}


_DIFF = (
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--text",
    "--no-relative",
    "--ignore-submodules=none",
    "--unified=3",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


class AgentError(RuntimeError):
    """A git command the runner ran failed."""


def git(cwd: str | Path, *args: str, isolated: bool = False) -> str:
    """Run git in ``cwd`` with the checkout's own settings overridden.

    ``isolated`` also leaves out the user's and the system's configuration.
    """
    done = subprocess.run(
        ["git", *_OVERRIDES, *args],
        cwd=cwd,
        env={**os.environ, **_ISOLATED} if isolated else None,
        capture_output=True,
        encoding="utf-8",
        errors="replace",  # binary and non-UTF-8 files in a diff
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        raise AgentError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def diff(cwd: str | Path, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as plain text."""
    return git(cwd, *_DIFF, base, head, isolated=True)


def local_diff(cwd: str | Path) -> Callable[[Attempt], str]:
    """A ``run(diff=...)`` source: ``git diff`` from the attempt's base to it."""

    def source(attempt: Attempt) -> str:
        if not attempt.base:
            raise AgentError("the attempt has no base commit to diff against")
        return diff(cwd, attempt.base, attempt.ref)

    return source


class CommandAgent:
    """Run ``argv`` in ``cwd`` once per attempt.

    ``base`` is the commit attempts are compared with; it defaults to the
    checkout's HEAD when the first attempt starts, so pass the recorded base
    when resuming a run. Each later attempt starts from the previous one,
    which the checkout must have checked out. With ``push``, each attempt is
    pushed to that branch of ``origin``, so CI runs on it; the agent may
    rewrite its earlier attempts, and the branch follows as long as nobody
    else has pushed to it. The agent gets ``env`` (by default this process's
    environment) without :data:`RUNNER_SECRETS`.

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
        self.base = base
        self.push = push
        self.env = env

    def _head(self) -> str:
        return git(self.cwd, "rev-parse", "HEAD").strip()

    def _attempt(self, task: Task) -> Attempt:
        if self.base is None:
            self.base = self._head()
        before = self._head()
        if task.previous is not None and before != task.previous.ref:
            raise Escalate(
                f"the checkout is at {before[:12]}, not at attempt "
                f"{task.number - 1} ({task.previous.ref[:12]}); check that out "
                "to continue"
            )
        try:
            done = subprocess.run(
                self.argv,
                cwd=self.cwd,
                input=task.prompt(),
                env=without_secrets(self.env),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise Escalate(f"the agent ran past {self.timeout:.0f}s") from None
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip()[-2000:]
            raise Escalate(f"the agent exited with {done.returncode}: {tail}")
        if git(self.cwd, "status", "--porcelain").strip():
            git(self.cwd, "add", "--all")
            git(self.cwd, "commit", "--quiet", "-m", f"Attempt {task.number}")
        after = self._head()
        if after == before:
            raise Escalate(f"attempt {task.number} changed nothing")
        if self.push is not None:
            branch = f"refs/heads/{self.push}"
            # The branch holds the previous attempt, which the agent may have
            # rewritten; anything else on it is someone else's and stays.
            lease = (
                [f"--force-with-lease={branch}:{task.previous.ref}"]
                if task.previous
                else []
            )
            git(self.cwd, "push", "--quiet", *lease, "origin", f"{after}:{branch}")
        return Attempt(
            ref=after,
            diff=diff(self.cwd, self.base, after),
            claims=done.stdout.strip()[-4000:],
            base=self.base,
        )

    async def __call__(self, task: Task) -> Attempt:
        return await asyncio.to_thread(self._attempt, task)
