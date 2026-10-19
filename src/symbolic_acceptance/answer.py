"""An oracle's answer to one question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Answer"]


@dataclass(frozen=True)
class Answer:
    """A strict yes/no, why, and what it rests on.

    ``evidence`` holds what an auditor can check: test ids and job URLs from
    CI, lines quoted from the diff by a judge, a reviewer's comment link.
    ``by`` names who answered: an oracle, a model, a reviewer's login.
    """

    is_met: bool
    reasoning: str = ""
    evidence: tuple[str, ...] = ()
    by: str = ""

    def __post_init__(self) -> None:
        # Oracles wrap models, APIs and people, so the types are checked here
        # rather than trusted: a "no" string must never read as a YES.
        if not isinstance(self.is_met, bool):
            raise TypeError(f"is_met must be a bool, got {type(self.is_met).__name__}")
        if not isinstance(self.reasoning, str) or not isinstance(self.by, str):
            raise TypeError("reasoning and by must be strings")
        if isinstance(self.evidence, str):
            raise TypeError("evidence must be a sequence of strings, not one string")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, str) for item in evidence):
            raise TypeError("evidence must be strings")
        object.__setattr__(self, "evidence", evidence)

    @property
    def word(self) -> str:
        return "YES" if self.is_met else "NO"

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"is_met": self.is_met}
        if self.reasoning:
            data["reasoning"] = self.reasoning
        if self.evidence:
            data["evidence"] = list(self.evidence)
        if self.by:
            data["by"] = self.by
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Answer:
        return cls(
            is_met=data["is_met"],
            reasoning=data.get("reasoning", ""),
            evidence=tuple(data.get("evidence", ())),
            by=data.get("by", ""),
        )
