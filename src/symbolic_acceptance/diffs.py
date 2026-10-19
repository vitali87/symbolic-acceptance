"""Read unified diffs (``git diff`` output): touched files and located lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["DiffLine", "FileChange", "locate", "parse"]

_GIT_HEADER = re.compile(r'^diff --git (?P<old>"(?:[^"\\]|\\.)*"|\S+) (?P<new>.+)$')
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ESCAPES = {
    "a": "\a",
    "b": "\b",
    "t": "\t",
    "n": "\n",
    "v": "\v",
    "f": "\f",
    "r": "\r",
    '"': '"',
    "\\": "\\",
}
_OCTAL = re.compile(r"[0-7]{3}")
MIN_QUOTE = 8


Change = Literal["added", "deleted", "modified", "renamed"]


@dataclass(frozen=True)
class FileChange:
    path: str
    status: Change
    old_path: str | None = None


@dataclass(frozen=True)
class DiffLine:
    """One line of hunk content: ``+`` added, ``-`` removed, ``" "`` context.

    ``line`` is the new file's line number for added and context lines, and
    the old file's for removed lines.
    """

    path: str
    side: Literal["+", "-", " "]
    line: int
    text: str

    @property
    def where(self) -> str:
        return f"{self.path}:{'-' if self.side == '-' else ''}{self.line}"

    @property
    def kind(self) -> Literal["added", "removed", "context"]:
        if self.side == "+":
            return "added"
        return "removed" if self.side == "-" else "context"


def _unquote(path: str) -> str:
    """Undo git's C-style quoting of unusual paths (see ``core.quotePath``).

    Octal escapes are the path's UTF-8 bytes, so they are collected as bytes
    and decoded once.
    """
    if not (len(path) >= 2 and path[0] == path[-1] == '"'):
        return path
    body, raw, index = path[1:-1], bytearray(), 0
    while index < len(body):
        if body[index] != "\\":
            raw += body[index].encode("utf-8")
            index += 1
        elif _OCTAL.fullmatch(body, index + 1, index + 4):
            raw.append(int(body[index + 1 : index + 4], 8))
            index += 4
        else:
            escaped = body[index + 1 : index + 2]
            raw += _ESCAPES.get(escaped, escaped).encode("utf-8")
            index += 2
    return raw.decode("utf-8", errors="surrogateescape")


def _strip_prefix(path: str) -> str | None:
    path = _unquote(path.strip())
    if path == "/dev/null":
        return None
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


@dataclass
class _File:
    old: str | None = None
    new: str | None = None
    status: Change = "modified"

    def change(self) -> FileChange | None:
        path = self.new or self.old
        if path is None:
            return None
        return FileChange(
            path, self.status, self.old if self.status == "renamed" else None
        )


def parse(diff: str) -> tuple[list[FileChange], list[DiffLine]]:
    """Files the diff touches, and every hunk line with its location."""
    files: list[_File] = []
    lines: list[DiffLine] = []
    # Git ends lines with "\n" only; splitlines() would also break on form
    # feeds and Unicode separators inside a line and lose count of the hunk.
    rows = diff.split("\n")
    index = 0
    while index < len(rows):
        row = rows[index]
        index += 1
        if row.startswith("diff --git ") or not files:
            files.append(_File())
        current = files[-1]
        if row.startswith("diff --git "):
            if match := _GIT_HEADER.match(row):
                current.old = _strip_prefix(match["old"])
                current.new = _strip_prefix(match["new"])
        elif row.startswith("new file mode"):
            current.status = "added"
        elif row.startswith("deleted file mode"):
            current.status = "deleted"
        elif row.startswith("rename from "):
            current.status = "renamed"
            current.old = _unquote(row[len("rename from ") :])
        elif row.startswith("rename to "):
            current.new = _unquote(row[len("rename to ") :])
        elif row.startswith("--- "):
            old = _strip_prefix(row[4:])
            if old is None:
                current.status = "added"
            else:
                current.old = old
        elif row.startswith("+++ "):
            current.new = _strip_prefix(row[4:])
            if current.new is None:
                current.status = "deleted"
        elif (match := _HUNK.match(row)) is not None:
            old_line, new_line = int(match[1]), int(match[3])
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
            path = current.new or current.old or ""
            # Count the hunk out, so content that looks like a header
            # ("+++ x" as an added line) is still read as content.
            while (old_left > 0 or new_left > 0) and index < len(rows):
                content = rows[index]
                side, text = content[:1], content[1:]
                if side not in ("+", "-", " ", "\\", ""):
                    break  # the counts were wrong; this line starts what follows
                index += 1
                if side == "\\":
                    continue
                if side == "+":
                    lines.append(DiffLine(path, "+", new_line, text))
                    new_line += 1
                    new_left -= 1
                elif side == "-":
                    lines.append(DiffLine(current.old or path, "-", old_line, text))
                    old_line += 1
                    old_left -= 1
                else:
                    lines.append(DiffLine(path, " ", new_line, text))
                    new_line += 1
                    old_line += 1
                    new_left -= 1
                    old_left -= 1
    changes = [change for file in files if (change := file.change()) is not None]
    return changes, lines


def locate(quote: str, lines: list[DiffLine]) -> list[DiffLine] | None:
    """Where each line of ``quote`` appears in the diff, or ``None``.

    Every non-blank line of the quote must occur within one diff line,
    preferring added and removed lines; a leading ``+`` or ``-`` copied from
    the diff is tolerated. Quotes shorter than a few characters prove nothing
    and are rejected.
    """
    parts = [part.strip() for part in quote.splitlines() if part.strip()]
    if sum(len(part) for part in parts) < MIN_QUOTE:
        return None
    # A line that also appears unchanged elsewhere is credited to the change.
    ordered = sorted(lines, key=lambda line: line.side == " ")
    found: list[DiffLine] = []
    for part in parts:
        candidates = [part]
        if part[:1] in "+-" and len(part) > 1:
            candidates.append(part[1:].strip())
        match = next(
            (line for c in candidates for line in ordered if c and c in line.text),
            None,
        )
        if match is None:
            return None
        found.append(match)
    return found
