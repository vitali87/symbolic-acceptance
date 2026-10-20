"""Checks a person should see before approving a checklist.

Loading already rejects anything malformed, and the format rules out the
silent cases (an attempt that is neither approved nor told what failed). Lint
looks for checklists that load but cannot mean what their author intended:

* errors: unknown oracles (``unknown-oracle``), options the built-in oracles
  cannot read or act on (``bad-option``), requirements that can never pass
  (``never-passes``; a warning if ``when`` can keep one from applying), and
  checklists that can never be approved (``never-approved``);
* warnings: requirements that can never fail (``never-fails``), ``when``
  conditions that are always or never YES (``when-always``, ``when-never``),
  checks held up behind a person (``gated-by-person``), questions nothing
  reads (``unused-question``) or asked twice (``duplicate-question``),
  options no built-in oracle reads, as a typo is (``unknown-option``), CI
  check runs an attempt could fake (``unfrozen-workflows``), commands and
  check runs whose NO helps the attempt pass, which one that never ran gives
  too (``no-helps``), commands any failure passes, a crash too
  (``any-failure``), and blanks a template left to fill in
  (``placeholder``);
* info: checklists too large for the exhaustive checks (``not-exhaustive``).
"""

from __future__ import annotations

import difflib
import itertools
import re
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Literal

from symbolic_acceptance.checklist import (
    Checklist,
    Expr,
    Question,
    Requirement,
    refs,
    render,
)
from symbolic_acceptance.checks import (
    admits,
    command,
    expected,
    frozen_paths,
    matches,
    patterns,
)
from symbolic_acceptance.logic import DEFAULT_LAST, FAIL, NA, status, value, wanted
from symbolic_acceptance.templates import FORMAT_GUIDE, ORACLES, TEMPLATES

__all__ = ["MAX_EXHAUSTIVE", "Finding", "lint"]

# Exhaustive checks enumerate 2**n answer sets; past this many questions they
# are skipped and reported as such.
MAX_EXHAUSTIVE = 16
# The options a built-in oracle acts on, and that oracle. On a question for
# another built-in oracle nothing acts on them, though the author may think
# so: frozen paths on an llm question are frozen for nobody.
_READERS = {
    "check": "ci",
    "run": "ci",
    "expect": "ci",
    "frozen": "ci",
    "allow_new": "ci",
    "kind": "llm",
}
# A question's other keys, which a typo in an option may have meant.
_KEYS = (*_READERS, "ask", "by", "context")
# What a shell's pipes, command lists and redirections are made of.
_SHELL = "|&;<>"
# The blanks the templates and the format guide leave to fill in: one of
# more than a word anywhere, numbered as it may be once copied ("<acceptance
# criterion 2, ...>"), and one of a word as a whole value (run = "<command>"),
# since in a sentence it may be text, as "<script>" is in HTML. Other text
# in angle brackets is text, as "x < lo or x > hi" is.
_BLANKS = {
    blank
    for text in (*TEMPLATES.values(), FORMAT_GUIDE)
    for blank in re.findall(r"<[^<>]+>", text)
}
_PLACEHOLDER = re.compile(
    "|".join(
        re.sub(r"\d+", r"\\d+", re.escape(blank))
        if " " in blank
        else rf"\A{re.escape(blank)}\Z"
        for blank in sorted(_BLANKS)
    )
)


@dataclass(frozen=True)
class Finding:
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    where: str | None = None

    def __str__(self) -> str:
        where = f" [{self.where}]" if self.where else ""
        return f"{self.level}: {self.code}{where}: {self.message}"


def _assignments(ids: tuple[str, ...]) -> Iterator[dict[str, bool]]:
    for values in itertools.product((True, False), repeat=len(ids)):
        yield dict(zip(ids, values))


def _constant(expr: Expr) -> bool | None:
    """The value ``expr`` has under every answer set, if it has one."""
    seen = {value(expr, facts) for facts in _assignments(refs(expr))}
    return seen.pop() if len(seen) == 1 else None


def _requirement_findings(requirement: Requirement) -> list[Finding]:
    findings: list[Finding] = []
    ids = requirement.questions()
    rid = requirement.id
    if len(ids) > MAX_EXHAUSTIVE:
        return [
            Finding(
                "info",
                "not-exhaustive",
                f"reads {len(ids)} questions; exhaustive checks skipped",
                rid,
            )
        ]
    if requirement.when is not None:
        constant = _constant(requirement.when)
        if constant is True:
            findings.append(
                Finding(
                    "warning",
                    "when-always",
                    f"`when` {render(requirement.when)} is always YES; drop it",
                    rid,
                )
            )
        elif constant is False:
            findings.append(
                Finding(
                    "warning",
                    "when-never",
                    f"`when` {render(requirement.when)} is never YES, so this "
                    "requirement never applies",
                    rid,
                )
            )
    results = [status(requirement, facts) for facts in _assignments(ids)]
    applicable = [result for result in results if result != NA]
    if applicable and all(result == FAIL for result in applicable):
        findings.append(
            Finding("error", "never-passes", "fails under every answer set", rid)
            if NA not in results
            else Finding(
                "warning",
                "never-passes",
                "fails whenever it applies, so it holds only by not applying",
                rid,
            )
        )
    elif FAIL not in results:
        findings.append(
            Finding(
                "warning",
                "never-fails",
                f"holds under every answer set, so it checks nothing: "
                f"{render(requirement.that)}",
                rid,
            )
        )
    return findings


def _shell_syntax(run: str) -> list[str]:
    """The pipes, command lists and redirections a shell would read in
    ``run``: its characters for them outside quotes and not escaped, as POSIX,
    and so the command's own splitting, reads them."""
    plain, quote, escaped = [], "", False
    for char in run:
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        else:
            plain.append(char)
            continue
        plain.append(" ")
    return re.findall(f"[{re.escape(_SHELL)}]+", "".join(plain))


def _run_command(question: Question) -> None:
    """Read the question's ``run`` command. It runs without a shell, so shell
    syntax in a command given as one string would reach the program as
    arguments, and ``prog | grep -q x`` would pass whatever ``prog`` says."""
    command(question)
    run = question.options["run"]
    if not isinstance(run, str):
        return
    if syntax := _shell_syntax(run):
        raise ValueError(
            f"q.{question.id}.run has {' '.join(syntax)} but runs without a shell;"
            ' for one, write run = ["sh", "-c", "<script>"]'
        )


def _option_findings(question: Question) -> list[Finding]:
    """Options the built-in oracles read, checked before any run needs them."""
    options = question.options
    reads: list[Callable[[], object]] = [
        lambda: patterns(question, "allow_new"),
        lambda: expected(question),
        lambda: question.choice("kind", "presence", "absence"),
    ]
    if "frozen" in options:
        reads.append(lambda: frozen_paths(question))
    if "run" in options:
        reads.append(lambda: _run_command(question))
    problems: list[str] = []
    for read in reads:
        try:
            read()
        except ValueError as error:
            problems.append(str(error))
    check = options.get("check")
    if "check" in options and (not isinstance(check, str) or not check):
        problems.append(f"q.{question.id}.check must name a check run")
    if "frozen" in options and "run" in options:
        problems.append(f"q.{question.id} has both frozen and run; split it in two")
    problems += [
        f"q.{question.id}.{key} works only with {partner}"
        for key, partner in (("expect", "run"), ("allow_new", "frozen"))
        if key in options and partner not in options
    ]
    problems += [
        f'q.{question.id}.{key} works only with by = "{reader}"'
        for key, reader in _READERS.items()
        if key in options and question.by != reader
    ]
    return [
        Finding("error", "bad-option", problem, question.id) for problem in problems
    ]


def _unread_option_findings(question: Question) -> list[Finding]:
    """Options on a built-in oracle's question that none of them reads."""
    findings = []
    for key in question.options:
        if key in _READERS:
            continue
        message = f"q.{question.id}.{key} is read by no built-in oracle"
        if close := difflib.get_close_matches(key, _KEYS, n=1):
            message += f"; did you mean {close[0]}?"
        findings.append(Finding("warning", "unknown-option", message, question.id))
    return findings


def _freezes(question: Question, path: str) -> bool:
    """Whether ``question`` is a frozen check that fails a new file at ``path``."""
    if question.by != "ci" or "frozen" not in question.options:
        return False
    try:
        frozen, allowed = frozen_paths(question), patterns(question, "allow_new")
    except ValueError:
        return False  # reported as a bad option
    return matches(path, frozen) and not admits(path, allowed)


def _check_runs_findings(checklist: Checklist) -> list[Finding]:
    """CI check runs can be faked by a workflow the attempt adds."""
    used = [checklist.questions[qid] for qid in checklist.used()]
    probes = [
        f".github/workflows/added-by-the-attempt.{ext}" for ext in ("yml", "yaml")
    ]
    if all(any(_freezes(q, probe) for q in used) for probe in probes):
        return []
    return [
        Finding(
            "warning",
            "unfrozen-workflows",
            f"answered by the CI check run {q.options['check']!r}, but no frozen "
            "check covers .github/workflows, so an attempt could add a workflow "
            "that reports a check of that name",
            q.id,
        )
        for q in used
        if q.by == "ci"
        and "run" not in q.options
        and "frozen" not in q.options
        and isinstance(q.options.get("check"), str)
    ]


def _no_helps_findings(checklist: Checklist) -> list[Finding]:
    """A command or check run that never ran answers NO, which proves nothing
    where a NO helps the attempt pass."""
    helps = wanted(checklist)
    findings = []
    for qid in checklist.used():
        question = checklist.questions[qid]
        if question.by != "ci" or False not in helps[qid]:
            continue
        if "run" in question.options:
            unrun = "a command that times out, cannot run or is never run"
        elif "check" in question.options:
            unrun = "a check run that is skipped, cancelled or neutral"
        else:
            continue
        findings.append(
            Finding(
                "warning",
                "no-helps",
                f"a NO helps the attempt pass (under not, or in when), but {unrun} "
                "answers NO too; ask it the other way round, so that a YES helps "
                "(with expect = <the status it exits with> for a command that "
                "must fail)",
                qid,
            )
        )
    return findings


def _any_failure_findings(checklist: Checklist) -> list[Finding]:
    """``expect = "fail"`` takes any failure, which an attempt need not earn."""
    return [
        Finding(
            "warning",
            "any-failure",
            'expect = "fail" takes any failure, a crash or a missing script\'s too; '
            "name the status the command must exit with (expect = 1, say)",
            question.id,
        )
        for question in checklist.questions.values()
        if question.by == "ci" and question.options.get("expect") == "fail"
    ]


def _texts(value: object, path: str) -> Iterator[tuple[str, str]]:
    """Every string in the checklist's data, with where it is."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _texts(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _texts(item, f"{path}[{index}]")


def _placeholder_findings(checklist: Checklist) -> list[Finding]:
    return [
        Finding(
            "warning",
            "placeholder",
            f"{path} still holds the template's {blank[0]}",
        )
        for path, text in _texts(checklist.to_dict(), "")
        for blank in _PLACEHOLDER.finditer(text)
    ]


def lint(
    checklist: Checklist,
    *,
    oracles: Collection[str] | None = None,
    last: Collection[str] = DEFAULT_LAST,
) -> list[Finding]:
    """Return findings, errors first. ``oracles`` lists the oracle names known;
    without it, every oracle is taken for one."""
    findings: list[Finding] = []
    used = set(checklist.used())

    for question in checklist.questions.values():
        if question.by in ORACLES:
            findings.extend(_option_findings(question))
            findings.extend(_unread_option_findings(question))
        if question.id not in used:
            findings.append(
                Finding(
                    "warning",
                    "unused-question",
                    "no requirement reads this question",
                    question.id,
                )
            )
        if oracles is not None and question.by not in oracles:
            findings.append(
                Finding(
                    "error",
                    "unknown-oracle",
                    f"by = {question.by!r}; known oracles: "
                    f"{', '.join(sorted(oracles))}",
                    question.id,
                )
            )
    asks: dict[str, str] = {}
    for question in checklist.questions.values():
        key = " ".join(question.ask.lower().split())
        if key in asks:
            findings.append(
                Finding(
                    "warning",
                    "duplicate-question",
                    f"asks the same as q.{asks[key]}",
                    question.id,
                )
            )
        else:
            asks[key] = question.id

    for requirement in checklist.requirements:
        findings.extend(_requirement_findings(requirement))
        if requirement.when is not None:
            gated_by = [
                qid
                for qid in refs(requirement.when)
                if checklist.questions[qid].by in last
            ]
            waiting = [
                qid
                for qid in refs(requirement.that)
                if checklist.questions[qid].by not in last
            ]
            if gated_by and waiting:
                findings.append(
                    Finding(
                        "warning",
                        "gated-by-person",
                        f"automated checks ({', '.join(waiting)}) wait for "
                        f"{', '.join(gated_by)} to be answered by a person",
                        requirement.id,
                    )
                )

    findings.extend(_check_runs_findings(checklist))
    findings.extend(_no_helps_findings(checklist))
    findings.extend(_any_failure_findings(checklist))
    findings.extend(_placeholder_findings(checklist))

    ordered = checklist.used()
    if len(ordered) <= MAX_EXHAUSTIVE:
        approvable = any(
            all(status(r, facts) != FAIL for r in checklist.requirements)
            for facts in _assignments(ordered)
        )
        if not approvable:
            findings.append(
                Finding(
                    "error",
                    "never-approved",
                    "no answer set satisfies every requirement at once",
                )
            )
    else:
        findings.append(
            Finding(
                "info",
                "not-exhaustive",
                f"{len(ordered)} questions; the whole-checklist check was skipped",
            )
        )
    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda finding: rank[finding.level])
