"""Read unified diffs (``git diff`` output): touched files and located lines."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

__all__ = ["DiffLine", "FileChange", "locate", "parse"]

_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"(?= )')
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


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
    ``respaced`` marks an added or removed line whose text is on the other
    side of its hunk too, spaced otherwise: only its whitespace changed.
    """

    path: str
    side: Literal["+", "-", " "]
    line: int
    text: str
    hunk: int
    respaced: bool = False

    @property
    def where(self) -> str:
        return f"{self.path}:{'-' if self.side == '-' else ''}{self.line}"

    @property
    def changed(self) -> bool:
        """Whether the change adds or removes the line, not just its spacing."""
        return self.side != " " and not self.respaced

    @property
    def kind(self) -> Literal["added", "removed", "respaced", "context"]:
        if self.respaced:
            return "respaced"
        if self.side == "+":
            return "added"
        return "removed" if self.side == "-" else "context"


def _unquote(path: str) -> str:
    """Undo git's C-style quoting of unusual paths (see ``core.quotePath``).

    Git's escapes are C's, which Python's include. Octal escapes are the
    path's UTF-8 bytes, so ``unicode_escape`` makes each escape, and each
    other byte, the character of that number, which Latin-1 maps back to
    the byte.
    """
    if not (len(path) >= 2 and path[0] == path[-1] == '"'):
        return path
    raw = path[1:-1].encode().decode("unicode_escape").encode("latin-1")
    return raw.decode("utf-8", errors="surrogateescape")


def _strip_prefix(path: str) -> str | None:
    # Git ends a ---/+++ line with a tab when the name has a space in it.
    path = _unquote(path.removesuffix("\t"))
    if path == "/dev/null":
        return None
    if path[:2] in ("a/", "b/"):
        return path[2:]
    return path


def _header(paths: str) -> tuple[str | None, str | None]:
    """The two paths of a ``diff --git`` header.

    Git leaves a path with spaces unquoted, so the header is split in the
    middle: its two halves name the same file, unless the file was renamed,
    and a rename has ``rename from`` and ``rename to`` lines that give both.
    """
    if quoted := _QUOTED.match(paths):
        return _strip_prefix(quoted[0]), _strip_prefix(paths[quoted.end() + 1 :])
    half = len(paths) // 2
    return _strip_prefix(paths[:half]), _strip_prefix(paths[half + 1 :])


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
    return changes, _respaced(lines)


def _respaced(lines: list[DiffLine]) -> list[DiffLine]:
    """``lines``, each one whose change is only its whitespace marked so."""
    spacings: dict[tuple[int, str], set[tuple[str, str]]] = {}
    for line in lines:
        key = line.hunk, line.text.strip()
        spacings.setdefault(key, set()).add((line.side, line.text))
    return [
        replace(line, respaced=True)
        if line.side != " "
        and any(
            side not in (line.side, " ") and text != line.text
            for side, text in spacings[line.hunk, line.text.strip()]
        )
        else line
        for line in lines
    ]


def _same(part: str, line: DiffLine) -> bool:
    text = line.text.strip()
    return part == text or (part[:1] in "+-" and part[1:].strip() == text)


def locate(quote: str, path: str, lines: list[DiffLine]) -> list[DiffLine] | None:
    """Where ``quote`` appears in the diff of the file at ``path``, or ``None``.

    The quote's non-blank lines must be whole non-blank lines of one hunk,
    one after another; surrounding whitespace and a leading ``+`` or ``-``
    copied from the diff don't count. A place where the change adds or
    removes one of them, rather than respacing it, is preferred.
    """
    parts = [part.strip() for part in quote.split("\n") if part.strip()]
    # The path as the diff names it, with or without the a/ or b/ before it;
    # read both ways it may name two files, and then it names neither.
    names = {path, path.removeprefix("a/"), path.removeprefix("b/")}
    files = {line.path for line in lines} & names
    if len(files) != 1:
        return None
    candidates = [line for line in lines if line.path in files and line.text.strip()]
    unchanged = None
    for start in range(len(candidates) - len(parts) + 1):
        window = candidates[start : start + len(parts)]
        if len({line.hunk for line in window}) > 1 or not all(
            map(_same, parts, window)
        ):
            continue
        if any(line.changed for line in window):
            return window
        unchanged = unchanged or window
    return unchanged
