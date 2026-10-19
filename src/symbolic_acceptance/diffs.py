"""Read unified diffs (``git diff`` output): touched files and located lines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["DiffLine", "FileChange", "locate", "parse"]

_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"(?= )')
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

    ``path`` is the file's name after the change (before it, for a deleted
    file). ``line`` is the new file's line number for added and context
    lines, and the old file's for removed lines. ``hunk`` counts the hunks
    of the diff, so lines with the same one are consecutive in their file.
    """

    path: str
    side: Literal["+", "-", " "]
    line: int
    text: str
    hunk: int

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


def _header(paths: str) -> tuple[str | None, str | None]:
    """The two paths of a ``diff --git`` header.

    Git leaves a path with spaces unquoted, so the header is split where its
    two halves name the same file, which they do unless the file was renamed;
    a rename has ``rename from`` and ``rename to`` lines that give both paths.
    """
    half = len(paths) // 2
    if quoted := _QUOTED.match(paths):
        old, new = quoted[0], paths[quoted.end() + 1 :]
    elif paths[half : half + 1] == " " and paths[2:half] == paths[half + 3 :]:
        old, new = paths[:half], paths[half + 1 :]
    else:
        old, _, new = paths.partition(" ")
    return _strip_prefix(old), _strip_prefix(new)


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
    hunk = -1
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
            current.old, current.new = _header(row[len("diff --git ") :])
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
            hunk += 1
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
                    lines.append(DiffLine(path, "+", new_line, text, hunk))
                    new_line += 1
                    new_left -= 1
                elif side == "-":
                    lines.append(DiffLine(path, "-", old_line, text, hunk))
                    old_line += 1
                    old_left -= 1
                else:
                    lines.append(DiffLine(path, " ", new_line, text, hunk))
                    new_line += 1
                    old_line += 1
                    new_left -= 1
                    old_left -= 1
    changes = [change for file in files if (change := file.change()) is not None]
    return changes, lines


def _same(part: str, line: DiffLine) -> bool:
    text = line.text.strip()
    return part == text or (part[:1] in "+-" and part[1:].strip() == text)


def locate(quote: str, path: str, lines: list[DiffLine]) -> list[DiffLine] | None:
    """Where ``quote`` appears in the diff of the file at ``path``, or ``None``.

    The quote's non-blank lines must be whole non-blank lines of one hunk,
    one after another; surrounding whitespace and a leading ``+`` or ``-``
    copied from the diff don't count. A place where the change adds or
    removes one of them is preferred. Quotes shorter than a few characters
    prove nothing and are rejected.
    """
    parts = [part.strip() for part in quote.split("\n") if part.strip()]
    # The path as the diff names it, with or without the a/ or b/ before it.
    names = {path, path.removeprefix("a/"), path.removeprefix("b/")}
    candidates = [line for line in lines if line.path in names and line.text.strip()]
    unchanged = None
    for start in range(len(candidates) - len(parts) + 1):
        window = candidates[start : start + len(parts)]
        if len({line.hunk for line in window}) > 1 or not all(
            map(_same, parts, window)
        ):
            continue
        if sum(len(line.text.strip()) for line in window) < MIN_QUOTE:
            return None
        if any(line.side != " " for line in window):
            return window
        unchanged = unchanged or window
    return unchanged
