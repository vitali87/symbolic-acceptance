"""Read unified diffs (``git diff`` output): touched files and located lines."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["DiffLine", "FileChange", "locate", "parse"]

_GIT_HEADER = re.compile(r"^diff --git (?P<old>\S+|\"(?:[^\"\\]|\\.)*\") (?P<new>.+)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
MIN_QUOTE = 8


@dataclass(frozen=True)
class FileChange:
    path: str
    status: str  # "added", "deleted", "modified" or "renamed"
    old_path: str | None = None


@dataclass(frozen=True)
class DiffLine:
    """One line of hunk content: ``+`` added, ``-`` removed, ``" "`` context.

    ``line`` is the new file's line number for added and context lines, and
    the old file's for removed lines.
    """

    path: str
    side: str
    line: int
    text: str

    @property
    def where(self) -> str:
        return f"{self.path}:{'-' if self.side == '-' else ''}{self.line}"


def _unquote(path: str) -> str:
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        body = path[1:-1]
        out, index = [], 0
        while index < len(body):
            if body[index] == "\\" and index + 1 < len(body):
                out.append(_ESCAPES.get(body[index + 1], body[index + 1]))
                index += 2
            else:
                out.append(body[index])
                index += 1
        path = "".join(out)
    return path


def _strip_prefix(path: str) -> str | None:
    path = _unquote(path.strip())
    if path == "/dev/null":
        return None
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


def parse(diff: str) -> tuple[list[FileChange], list[DiffLine]]:
    """Files the diff touches, and every hunk line with its location."""
    files: list[FileChange] = []
    lines: list[DiffLine] = []
    current: dict[str, str | None] | None = None

    def flush() -> None:
        if current is None:
            return
        path = current["new"] or current["old"]
        if path is None:
            return
        status = current["status"] or "modified"
        old = current["old"] if status == "renamed" else None
        files.append(FileChange(path, status, old))

    rows = diff.splitlines()
    index = 0
    while index < len(rows):
        row = rows[index]
        index += 1
        if row.startswith("diff --git "):
            flush()
            current = {"old": None, "new": None, "status": None}
            match = _GIT_HEADER.match(row)
            if match:
                old, new = match["old"], match["new"]
                current["old"] = _strip_prefix(old)
                current["new"] = _strip_prefix(new)
            continue
        if current is None:
            current = {"old": None, "new": None, "status": None}
        if row.startswith("new file mode"):
            current["status"] = "added"
        elif row.startswith("deleted file mode"):
            current["status"] = "deleted"
        elif row.startswith("rename from "):
            current["status"] = "renamed"
            current["old"] = _unquote(row[len("rename from ") :])
        elif row.startswith("rename to "):
            current["new"] = _unquote(row[len("rename to ") :])
        elif row.startswith("--- "):
            old = _strip_prefix(row[4:])
            if old is None:
                current["status"] = "added"
            else:
                current["old"] = old
        elif row.startswith("+++ "):
            new = _strip_prefix(row[4:])
            if new is None:
                current["status"] = "deleted"
                current["new"] = None
            else:
                current["new"] = new
        elif (match := _HUNK.match(row)) is not None:
            old_line, new_line = int(match[1]), int(match[3])
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
            path = current["new"] or current["old"] or ""
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
                    lines.append(DiffLine(current["old"] or path, "-", old_line, text))
                    old_line += 1
                    old_left -= 1
                else:
                    lines.append(DiffLine(path, " ", new_line, text))
                    new_line += 1
                    old_line += 1
                    new_left -= 1
                    old_left -= 1
    flush()
    return files, lines


def locate(quote: str, lines: list[DiffLine]) -> list[DiffLine] | None:
    """Where each line of ``quote`` appears in the diff, or ``None``.

    Every non-blank line of the quote must occur within one diff line; a
    leading ``+`` or ``-`` copied from the diff is tolerated. Quotes shorter
    than a few characters prove nothing and are rejected.
    """
    parts = [part.strip() for part in quote.splitlines() if part.strip()]
    if sum(len(part) for part in parts) < MIN_QUOTE:
        return None
    found: list[DiffLine] = []
    for part in parts:
        candidates = [part]
        if part[:1] in "+-" and len(part) > 1:
            candidates.append(part[1:].strip())
        match = next(
            (line for c in candidates for line in lines if c and c in line.text),
            None,
        )
        if match is None:
            return None
        found.append(match)
    return found
