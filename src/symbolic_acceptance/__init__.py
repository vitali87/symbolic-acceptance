"""Turn an issue into an acceptance checklist and drive a coding agent to it.

A person approves the checklist before any code is written. Each attempt is
checked by CI, an LLM judge and people, in that order, and every failed
requirement goes back to the agent until the attempt is approved or a person
has to step in. Standard library only.
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
    Failure,
    Reason,
    assess,
    explain,
    render_failures,
)
from symbolic_acceptance.lint import Issue, lint
from symbolic_acceptance.loop import (
    APPROVED,
    CHANGES,
    ESCALATED,
    RUNNING,
    WAITING,
    Attempt,
    Context,
    Escalate,
    Evaluation,
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
    "Failure",
    "Issue",
    "Not",
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
