import base64
import csv
import hashlib
import importlib.util
import io
import tarfile
import tempfile
import tomllib
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

from helpers import ROOT

import symbolic_acceptance as sa


def backend():
    spec = importlib.util.spec_from_file_location(
        "backend", ROOT / "_build" / "backend.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.backend = backend()

    def test_nothing_to_download(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)
        self.assertEqual(project["build-system"]["requires"], [])
        self.assertEqual(project["project"]["dependencies"], [])

    def test_the_package_reports_the_version_it_is_built_with(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            version = tomllib.load(stream)["project"]["version"]
        self.assertEqual(sa.__version__, version)

    def test_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = self.backend.build_wheel(tmp)
            self.assertEqual(
                name, f"symbolic_acceptance-{sa.__version__}-py3-none-any.whl"
            )
            with zipfile.ZipFile(Path(tmp, name)) as wheel:
                names = wheel.namelist()
                self.assertIn("symbolic_acceptance/__init__.py", names)
                self.assertIn("symbolic_acceptance/py.typed", names)
                self.assertFalse(
                    [n for n in names if "__pycache__" in n or n.startswith("tests/")]
                )
                dist = f"symbolic_acceptance-{sa.__version__}.dist-info"
                self.assertIn(f"{dist}/licenses/LICENSE", names)
                record = list(
                    csv.reader(io.StringIO(wheel.read(f"{dist}/RECORD").decode()))
                )
                self.assertEqual({row[0] for row in record}, set(names))
                for path, digest, size in record:
                    if path.endswith("RECORD"):
                        continue
                    data = wheel.read(path)
                    expected = base64.urlsafe_b64encode(
                        hashlib.sha256(data).digest()
                    ).rstrip(b"=")
                    self.assertEqual(digest, "sha256=" + expected.decode())
                    self.assertEqual(int(size), len(data))
                metadata = Parser().parsestr(wheel.read(f"{dist}/METADATA").decode())
                self.assertEqual(metadata["Metadata-Version"], "2.5")
                self.assertEqual(metadata["Name"], "symbolic-acceptance")
                self.assertEqual(metadata["License-Expression"], "MIT")
                self.assertEqual(metadata["Import-Name"], "symbolic_acceptance")
                self.assertEqual(metadata["Requires-Python"], ">=3.11")
                self.assertIsNone(metadata["Requires-Dist"], "no dependencies")
                entry_points = wheel.read(f"{dist}/entry_points.txt").decode()
                self.assertIn(
                    "symbolic-acceptance = symbolic_acceptance.cli:main", entry_points
                )

    def test_builds_are_reproducible(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            for build in (self.backend.build_wheel, self.backend.build_sdist):
                first, second = build(one), build(two)
                self.assertEqual(
                    Path(one, first).read_bytes(), Path(two, second).read_bytes()
                )

    def test_sdist(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = self.backend.build_sdist(tmp)
            self.assertEqual(name, f"symbolic_acceptance-{sa.__version__}.tar.gz")
            with tarfile.open(Path(tmp, name)) as sdist:
                names = sdist.getnames()
            base = f"symbolic_acceptance-{sa.__version__}"
            for required in (
                "PKG-INFO",
                "pyproject.toml",
                "_build/backend.py",
                "src/symbolic_acceptance/loop.py",
                "LICENSE",
            ):
                self.assertIn(f"{base}/{required}", names)

    def test_editable_points_at_the_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = self.backend.build_editable(tmp)
            with zipfile.ZipFile(Path(tmp, name)) as wheel:
                pth = (
                    wheel.read("__editable__.symbolic_acceptance.pth").decode().strip()
                )
            self.assertEqual(Path(pth), ROOT / "src")


if __name__ == "__main__":
    unittest.main()
