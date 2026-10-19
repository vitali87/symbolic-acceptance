"""Acceptance checklists: the definition of done for one issue.

A checklist is a TOML document with two parts:

* ``[q.<id>]`` questions: one yes/no fact each, routed to the oracle named in
  ``by``. Keys other than ``ask``, ``by`` and ``context`` are options for that
  oracle, such as the CI check that answers it.
* ``[[require]]`` requirements: each must hold for the attempt to be approved,
  and each one that fails is reported to the agent with its own ``fail``
  message. ``when`` makes a requirement conditional; it applies only when that
  expression is YES.

Expressions are a question id, or a table with one key: ``all`` or ``any``
(a list of expressions) or ``not`` (one expression).

Because every requirement carries its own failure, a checklist always decides:
once every question it needs is answered, the attempt is either approved or
has at least one named failure, never both and never neither.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "AllOf",
    "AnyOf",
    "Checklist",
    "ChecklistError",
    "Expr",
    "Not",
    "Question",
    "Ref",
    "Requirement",
    "load",
    "loads",
    "refs",
    "render",
]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOP_LEVEL = frozenset({"title", "context", "q", "require"})
_QUESTION_KEYS = frozenset({"ask", "by", "context"})
_REQUIREMENT_KEYS = frozenset({"id", "that", "when", "fail"})
_EMPTY: Mapping[str, object] = MappingProxyType({})


class ChecklistError(ValueError):
    """A checklist that cannot be loaded, with where the problem is."""


@dataclass(frozen=True)
class Ref:
    id: str


@dataclass(frozen=True)
class AllOf:
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class AnyOf:
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class Not:
    item: Expr


Expr = Ref | AllOf | AnyOf | Not


def render(expr: Expr) -> str:
    """Write an expression the way people read it: ``any(all(a, b), c)``."""
    if isinstance(expr, Ref):
        return expr.id
    if isinstance(expr, Not):
        return f"not({render(expr.item)})"
    name = "all" if isinstance(expr, AllOf) else "any"
    return f"{name}({', '.join(render(item) for item in expr.items)})"


def refs(expr: Expr) -> tuple[str, ...]:
    """Question ids in an expression, in reading order, once each."""
    if isinstance(expr, Ref):
        return (expr.id,)
    if isinstance(expr, Not):
        return refs(expr.item)
    seen: dict[str, None] = {}
    for item in expr.items:
        seen.update(dict.fromkeys(refs(item)))
    return tuple(seen)


def _to_data(expr: Expr) -> str | dict[str, Any]:
    if isinstance(expr, Ref):
        return expr.id
    if isinstance(expr, Not):
        return {"not": _to_data(expr.item)}
    key = "all" if isinstance(expr, AllOf) else "any"
    return {key: [_to_data(item) for item in expr.items]}


@dataclass(frozen=True)
class Question:
    """One yes/no fact. ``options`` are frozen: tables become read-only
    mappings and arrays become tuples, however the question was made."""

    id: str
    ask: str
    by: str
    context: str | None = None
    options: Mapping[str, object] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        frozen = MappingProxyType({k: _freeze(v) for k, v in self.options.items()})
        object.__setattr__(self, "options", frozen)


@dataclass(frozen=True)
class Requirement:
    id: str
    that: Expr
    fail: str
    when: Expr | None = None

    def questions(self) -> tuple[str, ...]:
        """Question ids this requirement reads, ``when`` first."""
        seen: dict[str, None] = {}
        if self.when is not None:
            seen.update(dict.fromkeys(refs(self.when)))
        seen.update(dict.fromkeys(refs(self.that)))
        return tuple(seen)


@dataclass(frozen=True)
class Checklist:
    questions: Mapping[str, Question]
    requirements: tuple[Requirement, ...]
    title: str | None = None
    context: str | None = None
    source: str | None = field(default=None, compare=False, repr=False)

    def requirement(self, requirement_id: str) -> Requirement:
        for requirement in self.requirements:
            if requirement.id == requirement_id:
                return requirement
        raise KeyError(requirement_id)

    def used(self) -> tuple[str, ...]:
        """Question ids some requirement reads, in first-use order."""
        seen: dict[str, None] = {}
        for requirement in self.requirements:
            seen.update(dict.fromkeys(requirement.questions()))
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """The checklist as plain data, in the shape the TOML was written in."""
        data: dict[str, Any] = {}
        if self.title is not None:
            data["title"] = self.title
        if self.context is not None:
            data["context"] = self.context
        data["q"] = {
            q.id: {
                "ask": q.ask,
                "by": q.by,
                **({"context": q.context} if q.context is not None else {}),
                **{key: _thaw(item) for key, item in q.options.items()},
            }
            for q in self.questions.values()
        }
        data["require"] = [
            {
                "id": r.id,
                **({"when": _to_data(r.when)} if r.when is not None else {}),
                "that": _to_data(r.that),
                "fail": r.fail,
            }
            for r in self.requirements
        ]
        return data

    @property
    def digest(self) -> str:
        """SHA-256 of the checklist's meaning; comments and layout don't count."""
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,  # TOML dates and times, which JSON has no type for
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class _Loader:
    def __init__(self, data: dict[str, object], where: str) -> None:
        self.data = data
        self.where = where

    def error(self, message: str) -> ChecklistError:
        return ChecklistError(f"{self.where}: {message}")

    def text(self, value: object, what: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self.error(f"{what} must be a non-empty string")
        return value

    def optional_text(self, value: object, what: str) -> str | None:
        return None if value is None else self.text(value, what)

    def identifier(self, value: object, what: str) -> str:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise self.error(
                f"{what} must be an identifier (letters, digits, '_'), got {value!r}"
            )
        return value

    def load(self, source: str) -> Checklist:
        unknown = sorted(set(self.data) - _TOP_LEVEL)
        if unknown:
            raise self.error(
                f"unknown top-level key(s) {', '.join(unknown)}; expected "
                "title, context, [q.<id>] and [[require]]"
            )
        questions = self.questions(self.data.get("q", {}))
        return Checklist(
            questions=MappingProxyType(questions),
            requirements=self.requirements(self.data.get("require"), questions),
            title=self.optional_text(self.data.get("title"), "title"),
            context=self.optional_text(self.data.get("context"), "context"),
            source=source,
        )

    def questions(self, table: object) -> dict[str, Question]:
        if not isinstance(table, dict):
            raise self.error("q must be a table of questions: [q.<id>]")
        questions: dict[str, Question] = {}
        for qid, body in table.items():
            self.identifier(qid, "a question id")
            if not isinstance(body, dict):
                raise self.error(f"q.{qid} must be a table with ask and by")
            options = {k: v for k, v in body.items() if k not in _QUESTION_KEYS}
            questions[qid] = Question(
                id=qid,
                ask=self.text(body.get("ask"), f"q.{qid}.ask"),
                by=self.identifier(body.get("by"), f"q.{qid}.by (the oracle's name)"),
                context=self.optional_text(body.get("context"), f"q.{qid}.context"),
                options=options,
            )
        return questions

    def requirements(
        self, items: object, questions: dict[str, Question]
    ) -> tuple[Requirement, ...]:
        if not isinstance(items, list) or not items:
            raise self.error("a checklist needs at least one [[require]]")
        requirements: list[Requirement] = []
        seen: set[str] = set()
        for index, body in enumerate(items, start=1):
            label = f"require #{index}"
            if not isinstance(body, dict):
                raise self.error(f"{label} must be a table")
            unknown = sorted(set(body) - _REQUIREMENT_KEYS)
            if unknown:
                raise self.error(
                    f"{label}: unknown key(s) {', '.join(unknown)}; "
                    "expected id, that, when and fail"
                )
            if "that" not in body:
                raise self.error(f"{label} needs `that`")
            that = self.expression(body["that"], questions, f"{label}.that")
            if "id" in body:
                rid = self.identifier(body["id"], f"{label}.id")
            elif isinstance(that, Ref):
                rid = that.id
            else:
                raise self.error(
                    f"{label} needs an id, because `that` is not a single question"
                )
            label = f"require {rid!r}"
            if rid in seen:
                raise self.error(f"{label} is defined twice")
            seen.add(rid)
            when = None
            if "when" in body:
                when = self.expression(body["when"], questions, f"{label}.when")
            fail = self.text(body.get("fail"), f"{label}.fail")
            requirements.append(Requirement(id=rid, that=that, fail=fail, when=when))
        return tuple(requirements)

    def expression(
        self, value: object, questions: dict[str, Question], at: str
    ) -> Expr:
        if isinstance(value, str):
            if value not in questions:
                raise self.error(f"{at}: {value!r} is not a question (q.{value})")
            return Ref(value)
        if not isinstance(value, dict) or len(value) != 1:
            raise self.error(
                f"{at}: expected a question id or a table with one of all, any or not"
            )
        ((key, inner),) = value.items()
        if key == "not":
            return Not(self.expression(inner, questions, f"{at}.not"))
        if key in ("all", "any"):
            if not isinstance(inner, list) or not inner:
                raise self.error(f"{at}.{key} must be a non-empty list")
            items = tuple(
                self.expression(item, questions, f"{at}.{key}[{n}]")
                for n, item in enumerate(inner)
            )
            return AllOf(items) if key == "all" else AnyOf(items)
        raise self.error(f"{at}: unknown operator {key!r}; use all, any or not")


def loads(text: str, *, name: str = "<checklist>") -> Checklist:
    """Load a checklist from TOML text."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ChecklistError(f"{name}: {error}") from None
    return _Loader(data, name).load(text)


def load(path: str | Path) -> Checklist:
    """Load a checklist from a UTF-8 TOML file."""
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ChecklistError(f"{path}: not valid UTF-8 ({error.reason})") from None
    return loads(text, name=str(path))
