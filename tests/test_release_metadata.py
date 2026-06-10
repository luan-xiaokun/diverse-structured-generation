"""Release metadata consistency checks for artifact snapshots."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.2.0"
EXPECTED_LICENSE = "MIT"
EXPECTED_TITLE = (
    "DiverseGuide: Automata-Based Steering for Diverse Structured Generation"
)
EXPECTED_REPOSITORY = "https://github.com/luan-xiaokun/diverse-structured-generation"


def _read_toml(relative_path: str):
    with (ROOT / relative_path).open("rb") as f:
        return tomllib.load(f)


def _read_cff_top_level_scalars() -> dict[str, str]:
    fields = {}
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        if line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"')
        if value:
            fields[key] = value
    return fields


def _read_python_dunder_version(relative_path: str) -> str:
    module = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "__version__" in names and isinstance(node.value, ast.Constant):
            return str(node.value.value)
    raise AssertionError(f"__version__ not found in {relative_path}")


def test_release_versions_are_consistent():
    project = _read_toml("pyproject.toml")
    rust_crate = _read_toml("regex_dfa_guide/Cargo.toml")
    citation = _read_cff_top_level_scalars()
    rust_python_version = _read_python_dunder_version(
        "regex_dfa_guide/python/regex_dfa_guide/__init__.py"
    )

    assert project["project"]["version"] == EXPECTED_VERSION
    assert rust_crate["package"]["version"] == EXPECTED_VERSION
    assert citation["version"] == EXPECTED_VERSION
    assert rust_python_version == EXPECTED_VERSION


def test_release_license_metadata_is_consistent():
    project = _read_toml("pyproject.toml")
    rust_crate = _read_toml("regex_dfa_guide/Cargo.toml")
    citation = _read_cff_top_level_scalars()
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert project["project"]["license"] == EXPECTED_LICENSE
    assert rust_crate["package"]["license"] == EXPECTED_LICENSE
    assert citation["license"] == EXPECTED_LICENSE
    assert "Permission is hereby granted, free of charge" in license_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text


def test_release_identity_metadata_names_diverseguide():
    project = _read_toml("pyproject.toml")
    citation = _read_cff_top_level_scalars()

    assert project["project"]["name"] == "diverse-guide"
    assert "DiverseGuide" in project["project"]["description"]
    assert project["project"]["urls"]["Repository"] == EXPECTED_REPOSITORY
    assert project["project"]["urls"]["Homepage"] == EXPECTED_REPOSITORY
    assert project["project"]["urls"]["Issues"] == f"{EXPECTED_REPOSITORY}/issues"
    assert citation["title"] == EXPECTED_TITLE
    assert citation["repository-code"] == EXPECTED_REPOSITORY
    assert citation["url"] == EXPECTED_REPOSITORY
