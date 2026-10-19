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
could change: hooks, replace refs, the file monitor, and in the diff
textconv and external drivers, binary files, path prefixes, quoting,
renames, submodules and context lines. A filter can still rewrite what the
agent's changes are committed as, which is the agent's to decide anyway:
everyone judges the commit.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from symbolic_acceptance.diffs import parse
from symbolic_acceptance.loop import Attempt, Escalate, Task

__all__ = [
    "RUNNER_SECRETS",
    "CommandAgent",
    "GitError",
    "diff",
    "for_agent",
    "git",
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


_DIFF = (
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--text",
    "--no-relative",
    "--ignore-submodules=none",
    "--submodule=short",
    "--find-renames",
    "--unified=3",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


class GitError(RuntimeError):
    """A git command the runner ran failed."""


def git(cwd: str | Path, *args: str, isolated: bool = False) -> str:
    """Run git on the repository in ``cwd``, overriding its own settings.

    ``isolated`` also leaves out the user's and the system's configuration.
    """
    env = {k: v for k, v in os.environ.items() if k not in _ELSEWHERE}
    done = subprocess.run(
        ["git", *_OVERRIDES, *args],
        cwd=cwd,
        env=env | _ISOLATED if isolated else env,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        error = done.stderr.decode(errors="replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {error}")
    # Decoded here rather than in text mode, which would take a carriage
    # return inside a line for a line break: file content could then forge
    # lines of a diff. "replace" covers binary and non-UTF-8 files.
    return done.stdout.decode(errors="replace")


def diff(cwd: str | Path, base: str, head: str) -> str:
    """The unified diff from ``base`` to ``head``, as plain text."""
    return git(cwd, *_DIFF, base, head, isolated=True)


class CommandAgent:
    """Run ``argv`` in ``cwd`` once per attempt.

    ``base`` is the commit attempts are compared with; it defaults to the
    checkout's HEAD now, before the agent has run, so pass it to
    :func:`symbolic_acceptance.loop.run` to keep it with the run, and pass
    the recorded one back when resuming. Each later attempt starts from the
    previous one, which the checkout must have checked out. With ``push``,
    each attempt is
    pushed to that branch of ``origin``, so CI runs on it; the agent may
    rewrite its earlier attempts, and the branch follows as long as nobody
    else has pushed to it. An attempt that changes ``.github/`` is not
    pushed but handed to a person: pushing would run the workflows it holds,
    with whatever token they ask for, before any check could refuse them.
    The agent gets ``env`` (by default this process's environment) without
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

    def _attempt(self, task: Task) -> Attempt:
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
                env=for_agent(self.env),
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
        # Untracked files and submodules count, whatever the checkout says.
        status = ("status", "--porcelain", "--untracked-files=all")
        if git(self.cwd, *status, "--ignore-submodules=none").strip():
            git(self.cwd, "add", "--all")
            git(self.cwd, "commit", "--quiet", "-m", f"Attempt {task.number}")
        after = self._head()
        if after == before:
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
            diff=changed,
            claims=done.stdout.strip()[-4000:],
        )

    async def __call__(self, task: Task) -> Attempt:
        return await asyncio.to_thread(self._attempt, task)
