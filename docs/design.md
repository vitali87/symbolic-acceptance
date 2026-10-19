# Design

This is the proposal of 15 October 2026, revised after review and a first
implementation. It keeps the proposal's shape: a definition of done fixed
before coding starts, split into yes/no checks routed to CI, an LLM judge and
people, with every failure fed back to the agent until approval or
escalation. What changed, and why, is below.

## Summary

symbolic-acceptance turns an issue into an **acceptance checklist** that a
person approves before any code is written, then drives a coding agent until
its attempt meets it. Each attempt is checked by CI and an LLM judge
together, then by people where it still matters. Every failed requirement
goes back to the agent with its reasons and evidence. The run escalates to a
person on the attempt cap, on no progress, on a regression, or when an
oracle hands the attempt over.

It has no dependencies: the package, its build and its CI use only the Python
standard library.

## Changes from the proposal

### 1. A checklist format instead of `.srg`

The proposal used a symbolic-reasoning `.srg` policy, with one `!=>` path per
failure to a `changes` outcome. `.srg` is built for guidelines: decision
graphs with many outcomes, whose solver gathers every fact. Checked against
symbolic-reasoning, the proposal's own example showed the problems that
causes for acceptance:

- **Silent attempts.** The example had no `!=>` path for `repro_fixed` or
  `suite_green`. With either failing, the graph reached *no* outcome, so the
  agent would get no feedback. Rule 1 and its lint existed to catch this.
- **People asked moot questions.** `Coverage.REACHABLE` asks every predicate
  inside a reached aggregator, so `sso_exempt`, inside
  `any(not(touches_sso), sso_exempt)`, went to a person on every passing
  attempt, even with SSO untouched; when unanswered, it blocked approval.
  Rules 3 and 4 existed to work around this.
- **Reasons inferred from NO answers.** Compiled graphs drop path ids and
  labels, hence the proposed upstream change to `OutcomePath`.

The checklist (TOML, read with the standard library's `tomllib`) makes each
failure part of the requirement: `[[require]]` has `that`, `fail` and an
optional `when`. A checklist therefore always decides, which the tests check
over every answer set of the example; the failure reason is the
requirement's own message, so no upstream change is needed; and there is no
second language to keep in step with symbolic-reasoning's grammar. Rules 1,
3 and 4 are gone; what is left for lint is listed in `lint.py`.

### 2. Asking order is the runner's job, not the author's

Instead of an authoring rule ("human predicates sit behind the automated
gate"), the planner (`logic.plan`) decides:

- `ci` and `llm` questions are asked with full coverage, the oracles in
  parallel, so all failures come back at once; parts that already hold are
  skipped.
- A `when` is settled before the `that` it gates.
- `human` questions come only when nothing automated is left but what they
  gate, nothing has failed, and the answer can still change a requirement.
  A failure never waits for a person, and a moot question is never asked,
  except where a requirement reads one question twice: three-valued logic
  cannot see that `all(not(q), any(r, q))` fails with `r` NO, whatever
  `q` is.

### 3. Feedback follows the logic, not the NO answers

"Every NO on a reached changes path" misreports any expression with `not` or
`any`: in `any(not(touches_sso), sso_exempt)` the cause of failure is a YES
(`touches_sso`), and in `any(all(locks_account, can_unlock), backs_off)` the
agent is told to fix three things when any one alternative would do.
Explanations (`feedback.explain`) walk the expression: the items that are NO
under a failed `all` (fix each), all alternatives under a failed `any` (fix
any one), with `not` flipping what is wanted.

### 4. The judge: evidence instead of temperature 0

"Temperature 0, every NO re-sampled" cannot be built on current models:
Claude Opus 5.5 rejects `temperature`, and Sonnet 5.5 and Haiku 5.5 reject
non-default values. It was also aimed the wrong way: the agent wants YES, so
the dangerous error is a false YES, which re-sampling NOs does nothing about.
The judge instead:

- returns structured JSON (`output_config.format`);
- must back the answer that helps the attempt pass wherever it can: each
  quote of a YES names a file of the diff and gives whole lines,
  consecutive within a hunk, and together the quotes hold more than a
  trivial line the change adds or removes, not merely respaces;
  they are checked. An unbacked
  helpful answer is asked for again, then gives way to the other one, or
  goes to a person when the checklist reads the question both ways. Which
  answer helps follows from where the question sits (YES in `that`, NO in
  `when`, flipped by `not`), and the judge is told it when there is one
  such answer, so that when unsure it gives the other; "an unsupported YES
  counts as NO" alone would have let a misquoted YES on a `when` gate skip
  a person's review. A question about absence (`kind = "absence"`) has
  nothing to quote for its YES, so its NO quotes what breaks it;
- with `samples > 1`, needs every sample to give the answer that helps;
- escalates on text addressed to the reviewer, on a refusal, on a reply cut
  off before it ends (thinking counts toward `max_tokens`), and on a diff
  too large to read whole (it never truncates);
- reads a diff the runner computed, never one the agent supplied, and never
  sees the agent's own description of its change.

It uses `claude-opus-5-5` at `high` effort, with server-side refusal
fallback (`fallbacks: "default"`).

### 5. Waiting on people without holding a process

`await sa.run(...)` as one long coroutine does not fit people who answer
hours later, or a GitHub Action with a time limit. An oracle raises
`Pending` once it has asked; the run returns `waiting` and is saved
(`record.save`), and the next event resumes it. Records keep answers, not
conclusions: statuses are re-derived on load and must match. A resume
needs the checkout at the last attempt, and one run must not be resumed
twice at once. Every outcome is saved, including an agent that fails or
changes nothing, which escalates rather than crashing the run; an error in
the runner itself, or Ctrl-C or SIGTERM, stops it with the record left as
it was.

### 6. Stop conditions on requirements, not raw answers

No progress and regression compare *requirement* statuses between attempts.
Raw answers flip for reasons that do not matter (a moot question, a judge's
wording); requirement statuses only change when the outcome does.
A regression is a requirement that passed and now fails; one that newly
applies (its `when` turned YES) goes back to the agent as ordinary feedback.
Escalating on a regression is the default, as proposed, and can be turned
off (`stop_on_regression=False`).

### 7. Integrity, made concrete

- `frozen = [<globs>]` fails any change to frozen paths; `allow_new` lets
  the new regression test in.
- The bug template's `new_test` tells the agent, and whoever writes the CI
  check behind it, that a test failing on the base branch for an unrelated
  reason (an import error) does not count; `test_targets_bug` asks the
  judge whether it exercises the reported behaviour, and `tests_intact`
  whether tests were skipped, deselected or loosened.
- On GitHub, only owners, members and collaborators (or an allowlist) answer,
  never bots or the accounts in `exclude`, and only after the question was
  asked about that commit. The runner cannot tell the agent's account from
  a person's, so an agent with a GitHub account of its own needs it in
  `exclude` (or answers limited to `reviewers`), and that account must not
  have write access, with which it could delete a reviewer's NO. A reply
  answers the latest question with its id posted before it, so a reply
  given for one commit never counts for another. The runner's markers count
  only at the start of a comment it could have written, and neither its
  own comments nor code blocks, which quote the agent, ever answer.
- A paused oracle's answers decide the attempt only if they can; otherwise
  it is asked again on resume, so a person's correction made meanwhile
  counts.
- The runner never pushes an attempt that changes `.github/`, since pushing
  runs its workflows at once (as it would another CI service's
  configuration, so push only where GitHub Actions is the only CI); give
  the workflows that run the attempt's code a read-only token
  (`permissions: contents: read`), so that code cannot report checks or
  comment.
- The runner's diff ignores what the agent's checkout configuration and
  attributes could change in it (hooks, replace refs, textconv, binary
  files, diff format); `run` commands execute in a fresh copy of the commit
  that reads none of the checkout's git configuration or attributes, nor
  the user's; and the run record, with the issue text, is kept outside the
  checkout.
- Edited comments never count as answers, and lint warns when CI check runs
  answer questions but `.github/workflows` is not frozen. Only check runs
  from GitHub Actions count by default, since another installed CI app
  could run a configuration the attempt added.
- The diff ignores what the checkout's or the user's git configuration says
  about submodules, renames and context, so a gitlink cannot drop out of it
  nor a new file pass for a changed one, and frozen paths match whatever
  their case. A new file whose name differs from a frozen one only in case
  still replaces it when checked out on a case-insensitive file system, so
  local `run` checks and CI belong on case-sensitive ones.
- The agent, and the `run` commands that execute its code, run without the
  runner's GitHub token, so they cannot report a check run or comment as
  the runner. Runner and agent still share a user and a machine, so what
  the agent writes outside its checkout (user site-packages, say, or a
  file a symlink in the commit points to), or a daemon it leaves running,
  can still change a local `run` check. Checks that must hold against an
  agent set on gaming them belong in CI, behind frozen workflows; against
  one attacking the runner itself, run the agent in a sandbox that holds
  none of the runner's credentials, with a checkout that has none
  persisted.
  The runner's own git (status, add, commit, push) reads the checkout's
  `.git`, whose configuration can name programs git runs (filters, a
  signing program, credential helpers) and where a push goes; only the
  push sees the runner's GitHub token. So a sandboxed agent must not be
  able to write `.git`: mount it read-only, and the runner commits for it.
- An `llm` question in a `when` condition helps the attempt with a NO,
  which has nothing to quote, so the judge's NO there stands unbacked.
  Where the condition is about files ("does it touch SSO?"), a frozen
  check over those paths decides it exactly instead.
- An oracle may only answer the questions it was asked.
- The checklist's digest is recorded; resuming with a changed checklist
  fails.

### 8. No dependencies

symbolic-reasoning needs pydantic, which brings a compiled extension and
three more packages, and it is not published on PyPI. A zero-dependency
package cannot build on it, so the three-valued evaluation it would have
provided is reimplemented here in about a hundred lines. The rest follows the
same rule: HTTP through `urllib`, an in-repository PEP 517/660 build backend
(`build-system.requires = []`), reproducible archives, and a CI workflow with
no third-party actions.

## Concepts

| Concept | What it is |
|---|---|
| Checklist | Questions and requirements for one issue, approved by a person, then frozen |
| Question | One yes/no fact, routed to an oracle by `by` |
| Requirement | `that` must be YES (when `when` is YES); `fail` is the message when not |
| Attempt | One commit: its ref, the runner-computed diff, the agent's claims |
| Evaluation | One attempt checked: every answer, each requirement's status, the failures |
| Oracle | `(questions, context) -> {id: Answer}`; may raise `Pending` or `Escalate` |
| Run | The evaluations of one issue until approval, escalation or a pause |

## Milestones

1. Loop core: `run`, `evaluate`, planning, feedback, stop conditions,
   records. **Done.**
2. Lint. **Done.**
3. Oracles: local commands and frozen files, GitHub check runs, the
   evidence-checked judge, people via pull request comments. **Done.**
4. Drafting: bug and feature templates; Claude drafts, lint gates, questions
   go back to the reporter. **Done.**
5. A GitHub Action that runs the loop on a labelled issue and posts the
   report. The pieces exist (`--resume`, `Reviews`, `post_report`); the
   workflow is next.
6. Dogfood on this repository's own issues.

## Open questions

- Which organisation hosts it, beside `cohelm/symbolic-reasoning` or
  separate? (It no longer depends on that package.)
- Which agent first: Claude Code (`claude -p`), the Agent SDK, or our own
  harness? `CommandAgent` works with any command line.
- Can a person's answer carry over to a later attempt that does not touch the
  files it concerned? Today every attempt is judged afresh.
- How does the agent dispute a NO without arguing the judge round? Today its
  claims reach people (in the report) but never the judge.
