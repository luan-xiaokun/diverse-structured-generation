"""Reference result checks for the v0.2.0 OSP artifact snapshot."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "artifact-results" / "v0.2.0"

EXPECTED_FILES = {
    REFERENCE_ROOT / "README.md",
    REFERENCE_ROOT / "primary" / "diversity.csv",
    REFERENCE_ROOT / "primary" / "runtime.csv",
    REFERENCE_ROOT / "optional" / "temperature_ablation.csv",
    REFERENCE_ROOT / "optional" / "component_ablation.csv",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_reference_result_files_are_tracked_in_expected_layout():
    actual_files = {path for path in REFERENCE_ROOT.rglob("*") if path.is_file()}
    assert actual_files == EXPECTED_FILES


def test_reference_csv_files_use_lf_line_endings():
    for path in REFERENCE_ROOT.glob("*/*.csv"):
        assert b"\r" not in path.read_bytes(), f"CR byte found in {path}"


def test_primary_reference_results_have_expected_schema_and_rows():
    diversity_rows = _read_rows(REFERENCE_ROOT / "primary" / "diversity.csv")
    runtime_rows = _read_rows(REFERENCE_ROOT / "primary" / "runtime.csv")

    assert len(diversity_rows) == 8
    assert len(runtime_rows) == 8
    assert {row["experiment"] for row in diversity_rows} == {"diversity"}
    assert {row["experiment"] for row in runtime_rows} == {"runtime"}
    assert {row["setting"] for row in diversity_rows} == {"baseline", "diverse"}
    assert {row["setting"] for row in runtime_rows} == {"baseline", "diverse"}
    assert {row["baseline_backend"] for row in runtime_rows} == {"internal"}
    assert {row["seed"] for row in runtime_rows} == {"42"}

    diversity_columns = set(diversity_rows[0])
    runtime_columns = set(runtime_rows[0])
    assert {
        "grammar",
        "state_coverage",
        "transition_coverage",
        "path_coverage",
        "vendi_score",
    }.issubset(diversity_columns)
    assert {
        "grammar",
        "baseline_backend",
        "generated_tokens",
        "seconds",
        "tokens_per_second",
        "seed",
    }.issubset(runtime_columns)


def test_optional_reference_results_have_expected_schema_and_rows():
    temperature_rows = _read_rows(
        REFERENCE_ROOT / "optional" / "temperature_ablation.csv"
    )
    component_rows = _read_rows(REFERENCE_ROOT / "optional" / "component_ablation.csv")

    assert len(temperature_rows) == 8
    assert len(component_rows) == 4
    assert {row["experiment"] for row in temperature_rows} == {"temperature_ablation"}
    assert {row["experiment"] for row in component_rows} == {"component_ablation"}
    assert {row["temperature"] for row in temperature_rows} == {"1.5"}
    assert {row["ablation_component"] for row in component_rows} == {
        "",
        "penalty",
        "range_scaling",
        "reward",
    }
