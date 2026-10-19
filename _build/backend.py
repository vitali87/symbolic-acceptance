"""A PEP 517 / PEP 660 build backend in the standard library only.

``pyproject.toml`` names this module as its backend with an empty
``build-system.requires``, so building or installing the package downloads
nothing: no setuptools, no hatchling, no wheel. It builds one pure-Python
wheel, an sdist and an editable wheel. Archives are reproducible: entries are
sorted and timestamped from ``SOURCE_DATE_EPOCH`` (default 2020-01-01), so
the same tree always builds the same bytes.
"""

from __future__ import annotations

import base64
import csv
import functools
import gzip
import hashlib
import io
import os
import re
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path
from typing import TypedDict

Settings = dict[str, str | list[str]] | None
# The [project] table of this repository's pyproject.toml.
Project = TypedDict(
    "Project",
    {
        "name": str,
        "version": str,
        "description": str,
        "readme": str,
        "requires-python": str,
        "license": str,
        "license-files": list[str],
        "authors": list[dict[str, str]],
        "keywords": list[str],
        "classifiers": list[str],
        "dependencies": list[str],
        "import-names": list[str],
        "scripts": dict[str, str],
        "urls": dict[str, str],
    },
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "symbolic_acceptance"
SDIST_FILES = ("pyproject.toml", "README.md", "LICENSE")
SDIST_DIRS = ("_build", "src", "tests", "examples", "docs")
WHEEL_TAG = "py3-none-any"
DEFAULT_EPOCH = 1577836800  # 2020-01-01T00:00:00Z


@functools.cache
def _project() -> Project:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project: Project = tomllib.load(stream)["project"]
    return project


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _epoch() -> int:
    return max(int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_EPOCH)), 315532800)


def _metadata() -> str:
    project = _project()
    lines = [
        # 2.5 is the lowest version with Import-Name, which this declares.
        "Metadata-Version: 2.5",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Author: {', '.join(author['name'] for author in project['authors'])}",
        f"License-Expression: {project['license']}",
        *(f"License-File: {name}" for name in project["license-files"]),
        f"Requires-Python: {project['requires-python']}",
        f"Keywords: {','.join(project['keywords'])}",
        *(f"Classifier: {c}" for c in project["classifiers"]),
        *(f"Project-URL: {k}, {v}" for k, v in project["urls"].items()),
        *(f"Import-Name: {name}" for name in project["import-names"]),
        *(f"Requires-Dist: {requirement}" for requirement in project["dependencies"]),
        "Description-Content-Type: text/markdown",
    ]
    body = (ROOT / project["readme"]).read_text(encoding="utf-8")
    return "\n".join(lines) + "\n\n" + body


def _dist_info() -> str:
    project = _project()
    return f"{_normalized(project['name'])}-{project['version']}.dist-info"


def _wheel_name() -> str:
    project = _project()
    return f"{_normalized(project['name'])}-{project['version']}-{WHEEL_TAG}.whl"


def _entry_points() -> str:
    scripts = sorted(_project()["scripts"].items())
    return "\n".join(["[console_scripts]", *(f"{k} = {v}" for k, v in scripts)]) + "\n"


def _files(directory: Path) -> list[Path]:
    """The files under ``directory``, in order, without Python's caches."""
    return [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]


def _package_files() -> list[tuple[str, bytes]]:
    source = ROOT / "src" / PACKAGE
    return [
        (path.relative_to(source.parent).as_posix(), path.read_bytes())
        for path in _files(source)
    ]


def _digest(data: bytes) -> str:
    raw = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _write_wheel(directory: str, files: list[tuple[str, bytes]]) -> str:
    dist_info = _dist_info()
    wheel = (
        "Wheel-Version: 1.0\nGenerator: symbolic-acceptance _build\n"
        f"Root-Is-Purelib: true\nTag: {WHEEL_TAG}\n"
    )
    entries = list(files)
    entries.append((f"{dist_info}/METADATA", _metadata().encode("utf-8")))
    entries.append((f"{dist_info}/WHEEL", wheel.encode("utf-8")))
    entries.append((f"{dist_info}/entry_points.txt", _entry_points().encode("utf-8")))
    for name in _project()["license-files"]:
        entries.append((f"{dist_info}/licenses/{name}", (ROOT / name).read_bytes()))
    record_name = f"{dist_info}/RECORD"
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for name, data in entries:
        writer.writerow([name, _digest(data), len(data)])
    writer.writerow([record_name, "", ""])
    entries.append((record_name, rows.getvalue().encode("utf-8")))

    timestamp = time.gmtime(_epoch())[:6]
    target = Path(directory) / _wheel_name()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return target.name


# -- PEP 517 hooks ------------------------------------------------------------


def build_wheel(
    wheel_directory: str,
    config_settings: Settings = None,
    metadata_directory: str | None = None,
) -> str:
    return _write_wheel(wheel_directory, _package_files())


def build_editable(
    wheel_directory: str,
    config_settings: Settings = None,
    metadata_directory: str | None = None,
) -> str:
    pth = f"{(ROOT / 'src').as_posix()}\n".encode()
    return _write_wheel(wheel_directory, [(f"__editable__.{PACKAGE}.pth", pth)])


def build_sdist(sdist_directory: str, config_settings: Settings = None) -> str:
    project = _project()
    base = f"{_normalized(project['name'])}-{project['version']}"
    paths = [ROOT / name for name in SDIST_FILES]
    for directory in SDIST_DIRS:
        paths += _files(ROOT / directory)
    members = [(f"{base}/PKG-INFO", _metadata().encode("utf-8"))]
    members += [
        (f"{base}/{p.relative_to(ROOT).as_posix()}", p.read_bytes()) for p in paths
    ]
    members.sort()
    target = Path(sdist_directory) / f"{base}.tar.gz"
    epoch = _epoch()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = epoch
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed,
    ):
        compressed.write(buffer.getvalue())
    return target.name
