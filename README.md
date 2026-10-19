# symbolic-acceptance

Turn a bug report or feature request into an **acceptance checklist**, then
drive a coding agent until its change meets it.

A person approves the checklist before any code is written. Each attempt is
checked by CI, an LLM judge and people, in that order, and every failed
requirement goes back to the agent, with the reasons and evidence, until the
attempt is approved or a person has to step in.

**No dependencies.** The package, its build and its CI use only the Python
standard library (3.11+). There is nothing to download: not at install time,
not at build time (the build backend is in this repository), not in CI (no
third-party actions). Builds are reproducible byte for byte.

## A checklist

```toml
title = "Throttle brute-force logins (#342)"

[q.repro_fixed]
ask = "Does the reproduction script from the issue now get throttled?"
by = "ci"
check = "repro"

[q.locks_account]
ask = "Does the change lock an account after repeated failed logins?"
by = "llm"

[q.can_unlock]
ask = "Can a locked account be unlocked by its owner or an administrator?"
by = "llm"

[q.backs_off]
ask = "Does the change add an increasing delay between failed login attempts?"
by = "llm"

[q.sso_untouched]
ask = "Does the change leave the single sign-on code alone?"
by = "ci"
frozen = ["src/auth/sso/*"]

[q.sso_exempt]
ask = "Is this change to single sign-on acceptable without a security review?"
by = "human"

[[require]]
that = "repro_fixed"
fail = "The reproduction from the issue is still not throttled"

[[require]]
id = "protected"
that = { any = [{ all = ["locks_account", "can_unlock"] }, "backs_off"] }
fail = "Neither lockout with a way to unlock, nor back-off between attempts"

[[require]]
id = "sso"
when = { not = "sso_untouched" }
that = "sso_exempt"
fail = "Single sign-on changed without approval"
```

`[q.<id>]` declares a yes/no question and the oracle that answers it (`by`).
`[[require]]` declares a requirement: `that` must be YES, `fail` is what the
agent is told when it is not, and `when` makes it conditional. Expressions are
a question id or `{ all = [...] }`, `{ any = [...] }`, `{ not = ... }`.

Because every requirement carries its own failure, a checklist always
decides: once its questions are answered, an attempt is approved or has named
failures, never both and never neither. The full example is
[`examples/issue-342.toml`](examples/issue-342.toml); `symbolic-acceptance
template bug` and `template feature` print starting points.

## How an attempt is checked

1. **Automated questions first, all of them.** Every `ci` and `llm` question
   that could show a requirement unmet is asked, the oracles in parallel, so
   the agent hears about every failure at once, not just the first. A `when` condition
   is settled before the checks it gates.
2. **People last, and only when it matters.** `human` questions are asked
   only once nothing automated is left, no requirement has failed, and their
   answer can still change the result. A failure never waits for a person,
   and nobody is asked a question whose answer cannot matter.
3. **Feedback follows the logic.** A failed `all` lists the items to fix, a
   failed `any` lists the alternatives (fix any one), and `not` flips what was
   wanted. Each item carries the oracle's reasoning and evidence:

   ```text
   2. protected: Neither lockout with a way to unlock, nor back-off between attempts
      - any(all(locks_account, can_unlock), backs_off): needs YES; fix any one of:
        - all(locks_account, can_unlock): needs YES; fix each of:
          - can_unlock: needs YES, got NO [llm:claude-opus-5-5]
            Q: Can a locked account be unlocked by its owner or an administrator?
            Why: lock() has no inverse; add unlock() in src/auth.py
        - backs_off: needs YES, got NO [llm:claude-opus-5-5]
            ...
   ```

A run stops when an attempt is **approved**, and **escalates** to a person on
the attempt cap, on no progress (the same requirements fail twice in a row),
on a regression (a requirement that held now fails), or when an oracle hands
the attempt over. A run waiting on a person is **saved and resumed** later,
for example from the next GitHub event, rather than holding a process open.

## Oracles

| `by` | Built in | Answers from |
|---|---|---|
| `ci` | `Checks`, `GitHubChecks` | `run = "<command>"` in a checkout, `frozen = [<globs>]`, or the GitHub check run named by `check` |
| `llm` | `Judge` | Claude, reading the diff |
| `human` | `Reviews`, terminal | `/yes <id> <why>` replies on the pull request |

An oracle is any function `(questions, context) -> {id: Answer}`, sync or
async. It raises `Pending` when it has asked and must wait, and `Escalate` to
hand the attempt to a person.

**The agent is treated as adversarial.** It writes what the oracles read, so:

- The judge gets the diff, computed by the runner with git, inside
  a block with a boundary the agent cannot predict. The agent's own account
  of its change is never shown to it. The runner's git overrides whatever the
  checkout's configuration could change (hooks, replace refs, textconv and
  diff drivers, path prefixes), since the agent can write that configuration.
- An answer from the judge that helps the attempt pass must be backed by the
  diff wherever it can be: a YES quotes the lines that show it, whole and
  in order, at least one of them a line the change adds or removes, and the
  quotes are checked. An
  unbacked helpful answer gives way to the other one. Which answer helps
  follows from the checklist: YES inside `that`, NO inside a `when`
  condition. A question marked `kind = "absence"` ("does it leave the API
  unchanged?") has nothing to quote for its YES, so its NO quotes what
  breaks it. With `samples > 1`, every sample must give the helpful answer.
- `run` commands execute in a fresh clone of the attempt's commit that
  reads none of the checkout's git configuration or attributes, so nothing
  in the checkout outside the commit can change their result.
- Text in the change addressed to a reviewer escalates instead of being
  answered, and so does a diff too large to judge whole.
- `frozen = [<globs>]` fails any change to the checklist, CI configuration
  or existing tests; a new regression test can be allowed with `allow_new`.
- On GitHub, only owners, members and collaborators (or an allowlist) can
  answer, never bots or the accounts in `exclude` (put the agent's there if
  it has one), only after the question was asked about that commit, and
  never in an edited comment (anyone with write access can edit someone
  else's). The runner's own markers count only at the start of a comment
  it could have written.
- Freeze `.github/` when CI check runs answer questions; otherwise an attempt
  could add a workflow that reports a check of the same name. Lint warns.
  Only runs from GitHub Actions count, unless `GitHubChecks(app=...)` names
  another CI app, whose configuration should then be frozen too.
- Keep the run record outside the agent's checkout; inside it, the record
  would be committed with the attempt. The command line refuses to.
- The agent, and the `run` commands that execute its code, get no GitHub
  token from the runner, with which they could report a check run or
  comment as the runner. Give the agent no other credentials that can
  comment as a reviewer either: a reply posted with your token reads as
  yours. What local `run` checks cannot protect against, and how to run an
  agent that might attack the runner itself, is in
  [`docs/design.md`](docs/design.md).

The judge calls the Claude Messages API over HTTPS with `urllib`. It uses
`claude-opus-5-5` at `high` effort, structured outputs for the verdict, and
server-side refusal fallback (`fallbacks: "default"`), so a classifier false
positive is retried on another model instead of failing the check. It sends no
sampling parameters, which current models reject. Set `ANTHROPIC_API_KEY`.

## Use it

From Python:

```python
import asyncio
from pathlib import Path

import symbolic_acceptance as sa
from symbolic_acceptance.agents import CommandAgent
from symbolic_acceptance.checks import Checks, Commands
from symbolic_acceptance.github import GitHub, GitHubChecks, Reviews
from symbolic_acceptance.judge import Judge

checklist = sa.load("acceptance/issue-342.toml")
github = GitHub("owner/repo")

result = asyncio.run(
    sa.run(
        checklist,
        agent=CommandAgent(
            ["claude", "-p", "--permission-mode", "auto"],
            cwd="work",  # the agent's checkout
            push="agent/issue-342",  # so CI runs on every attempt
        ),
        oracles={
            "ci": Checks(Commands("work"), fallback=GitHubChecks(github, wait=1800)),
            "llm": Judge(),
            "human": Reviews(github, number=57, exclude=["agent-account"]),
        },
        issue=Path("issue-342.md").read_text(encoding="utf-8"),
        max_attempts=5,
        on_progress=lambda run: sa.record.save(run, checklist, "run.json"),
    )
)
result.status  # "approved" | "escalated" | "waiting"
result.reason
```

From the command line:

```sh
symbolic-acceptance template bug > acceptance.toml   # or: draft issue.md (Claude drafts it)
symbolic-acceptance lint acceptance.toml             # before a person approves it
symbolic-acceptance digest acceptance.toml           # pin the approved version
symbolic-acceptance run acceptance.toml --issue issue.md \
    --agent "claude -p --permission-mode auto" --cwd work \
    --push agent/issue-342 --record run.json --github owner/repo --pr 57
symbolic-acceptance run acceptance.toml --resume run.json \
    --agent "claude -p --permission-mode auto" --cwd work \
    --push agent/issue-342 --github owner/repo --pr 57
symbolic-acceptance report acceptance.toml run.json  # Markdown, for the PR
```

A resumed run needs the agent's checkout at its last attempt, and one run
must not be resumed twice at once: the resumes would share the checkout
and the record. An agent that fails, times out or changes nothing
escalates the run, which is saved like any other outcome. An error in the
runner itself (git, the network, a misconfigured oracle) stops it instead,
and the record keeps the run as it was, ready to resume.

`explain acceptance.toml answers.json` decides a checklist from answers you
supply, which is handy for checking a checklist by hand.

## Install

```sh
pip install git+https://github.com/vitali87/symbolic-acceptance
```

or from a checkout, `pip install .`. It needs Python 3.11 or later, and
git 2.40 or later for the runner. The build backend lives in
[`_build/backend.py`](_build/backend.py) and `pyproject.toml` requires
nothing, so `pip install --no-index .` works offline.

## Develop

```sh
PYTHONPATH=src:tests python3 -m unittest discover -s tests -t tests
```

The design, and how it changed from the original proposal, is in
[`docs/design.md`](docs/design.md).

## License

MIT
