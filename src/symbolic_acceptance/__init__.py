"""Turn an issue into an acceptance checklist and drive a coding agent to it.

A person approves the checklist before any code is written. Each attempt is
checked by CI and an LLM judge together, then by people where it still
matters, and every failed requirement goes back to the agent until the
attempt is approved or a person has to step in. Standard library only.
"""

from symbolic_acceptance import record, templates
from symbolic_acceptance.answer import Answer
from symbolic_acceptance.checklist import (
    AllOf,
    AnyOf,
    Checklist,
    ChecklistError,
    Not,
    Question,
    Ref,
    Requirement,
    load,
    loads,
)
from symbolic_acceptance.feedback import (
    Assessment,
    Fact,
    Failure,
    Group,
    Reason,
    assess,
    explain,
    render_failures,
)
from symbolic_acceptance.lint import Finding, lint
from symbolic_acceptance.loop import (
    APPROVED,
    CHANGES,
    ESCALATED,
    RUNNING,
    WAITING,
    Agent,
    Attempt,
    Context,
    Escalate,
    Evaluation,
    Oracle,
    OracleError,
    Pending,
    Result,
    Task,
    evaluate,
    run,
)

__version__ = "0.1.0"

__all__ = [
    "APPROVED",
    "CHANGES",
    "ESCALATED",
    "RUNNING",
    "WAITING",
    "Agent",
    "AllOf",
    "Answer",
    "AnyOf",
    "Assessment",
    "Attempt",
    "Checklist",
    "ChecklistError",
    "Context",
    "Escalate",
    "Evaluation",
    "Fact",
    "Failure",
    "Finding",
    "Group",
    "Not",
    "Oracle",
    "OracleError",
    "Pending",
    "Question",
    "Reason",
    "Ref",
    "Requirement",
    "Result",
    "Task",
    "__version__",
    "assess",
    "evaluate",
    "explain",
    "lint",
    "load",
    "loads",
    "record",
    "render_failures",
    "run",
    "templates",
]
