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
from typing import Any

Settings = dict[str, str | list[str]] | None

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "symbolic_acceptance"
SDIST_FILES = ("pyproject.toml", "README.md", "LICENSE")
SDIST_DIRS = ("_build", "src", "tests", "examples", "docs")
WHEEL_TAG = "py3-none-any"
DEFAULT_EPOCH = 1577836800  # 2020-01-01T00:00:00Z


@functools.cache
def _project() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


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
    ]
    authors = [a["name"] for a in project.get("authors", []) if "name" in a]
    if authors:
        lines.append(f"Author: {', '.join(authors)}")
    lines.append(f"License-Expression: {project['license']}")
    lines += [f"License-File: {name}" for name in project.get("license-files", [])]
    lines.append(f"Requires-Python: {project['requires-python']}")
    if project.get("keywords"):
        lines.append(f"Keywords: {','.join(project['keywords'])}")
    lines += [f"Classifier: {c}" for c in project.get("classifiers", [])]
    lines += [f"Project-URL: {k}, {v}" for k, v in project.get("urls", {}).items()]
    lines += [f"Import-Name: {name}" for name in project.get("import-names", [])]
    for requirement in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {requirement}")
    readme = project.get("readme")
    body = ""
    if readme:
        lines.append("Description-Content-Type: text/markdown")
        body = (ROOT / readme).read_text(encoding="utf-8")
    return "\n".join(lines) + "\n\n" + body


def _dist_info() -> str:
    project = _project()
    return f"{_normalized(project['name'])}-{project['version']}.dist-info"


def _wheel_name() -> str:
    project = _project()
    return f"{_normalized(project['name'])}-{project['version']}-{WHEEL_TAG}.whl"


def _entry_points() -> str:
    scripts = _project().get("scripts", {})
    if not scripts:
        return ""
    lines = ["[console_scripts]"] + [f"{k} = {v}" for k, v in sorted(scripts.items())]
    return "\n".join(lines) + "\n"


def _package_files() -> list[tuple[str, bytes]]:
    source = ROOT / "src" / PACKAGE
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append(
                (path.relative_to(source.parent).as_posix(), path.read_bytes())
            )
    return files


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
    if entry_points := _entry_points():
        entries.append((f"{dist_info}/entry_points.txt", entry_points.encode("utf-8")))
    for name in _project().get("license-files", []):
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


def get_requires_for_build_wheel(config_settings: Settings = None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings: Settings = None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings: Settings = None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: Settings = None
) -> str:
    target = Path(metadata_directory) / _dist_info()
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_text(_metadata(), encoding="utf-8")
    if entry_points := _entry_points():
        (target / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return target.name


prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel


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
        if (ROOT / directory).is_dir():
            paths += [
                p
                for p in sorted((ROOT / directory).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            ]
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
