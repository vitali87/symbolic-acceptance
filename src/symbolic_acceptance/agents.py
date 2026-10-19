"""Run a coding agent as a command; the commit it leaves is the attempt.

Any agent with a command line works. With Claude Code, for example::

    CommandAgent(["claude", "-p", "--permission-mode", "auto"], cwd=checkout)

The task prompt is sent on standard input. If the agent leaves changes
uncommitted, they are committed for it, so every attempt is one commit; the
diff is computed here with git, from where the run started to that commit,
and the agent's own output is kept only as its claims.

The agent can write to the checkout's git configuration, so every git
command here overrides what that configuration could change: hooks,
replace refs, textconv and external diff drivers, path prefixes, quoting.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from symbolic_acceptance.loop import Attempt, Task

__all__ = ["AgentError", "CommandAgent", "diff", "git", "local_diff"]

_OVERRIDES = (
    "--no-replace-objects",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.quotePath=true",
)
_DIFF = (
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--text",
    "--no-relative",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


class AgentError(RuntimeError):
    """The agent failed, or finished without changing anything."""


def git(cwd: str | Path, *args: str) -> str:
    """Run git in ``cwd`` with the checkout's own settings overridden."""
    done = subprocess.run(
        ["git", *_OVERRIDES, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        raise AgentError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def diff(cwd: str | Path, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as plain text."""
    return git(cwd, *_DIFF, base, head)


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
    when resuming a run. With ``push``, each attempt is pushed to that branch
    of ``remote``, so CI runs on it.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path = ".",
        timeout: float = 3600.0,
        commit: bool = True,
        base: str | None = None,
        push: str | None = None,
        remote: str = "origin",
    ) -> None:
        if not argv:
            raise ValueError("argv must name the agent's command")
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.timeout = timeout
        self.commit = commit
        self.base = base
        self.push = push
        self.remote = remote

    def _head(self) -> str:
        return git(self.cwd, "rev-parse", "HEAD").strip()

    def _attempt(self, task: Task) -> Attempt:
        if self.base is None:
            self.base = self._head()
        before = self._head()
        try:
            done = subprocess.run(
                self.argv,
                cwd=self.cwd,
                input=task.prompt(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise AgentError(f"the agent ran past {self.timeout:.0f}s") from None
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip()[-2000:]
            raise AgentError(f"the agent exited with {done.returncode}: {tail}")
        if self.commit and git(self.cwd, "status", "--porcelain").strip():
            git(self.cwd, "add", "--all")
            git(self.cwd, "commit", "--quiet", "-m", f"Attempt {task.number}")
        after = self._head()
        if after == before:
            raise AgentError(f"attempt {task.number} changed nothing")
        if self.push is not None:
            git(
                self.cwd,
                "push",
                "--quiet",
                self.remote,
                f"{after}:refs/heads/{self.push}",
            )
        return Attempt(
            ref=after,
            diff=diff(self.cwd, self.base, after),
            claims=done.stdout.strip()[-4000:],
            base=self.base,
        )

    async def __call__(self, task: Task) -> Attempt:
        return await asyncio.to_thread(self._attempt, task)
