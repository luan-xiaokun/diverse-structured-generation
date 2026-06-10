import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.metrics_trend import (
    build_data_args,
    build_result,
    build_sample_counts,
    default_output_paths,
    metric_value,
    parse_args,
)


def _args(**overrides):
    values = {
        "grammar": "unit",
        "model": "test/model",
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "path_frequency_estimator": "harmonic",
        "ablation_component": None,
        "d": 2,
        "s": 1,
        "step": 5,
        "max_samples": None,
        "compare_all_diverse": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_data_args_preserves_path_frequency_estimator():
    args = _args(path_frequency_estimator="min")

    baseline_args = build_data_args(args, baseline=True)
    diverse_args = build_data_args(args, baseline=False)

    assert baseline_args.path_frequency_estimator == "min"
    assert diverse_args.path_frequency_estimator == "min"
    assert baseline_args.baseline is True
    assert diverse_args.baseline is False


def test_build_sample_counts_appends_limit_when_needed():
    assert build_sample_counts(limit=12, step=5) == [5, 10, 12]


def test_metric_value_reads_distinct_counts():
    result = {
        "metrics": {
            "state_coverage": 0.5,
            "distinct_2gram": [7, 10],
            "distinct_3gram": [9, 10],
        }
    }

    assert metric_value(result, "state_coverage") == 0.5
    assert metric_value(result, "distinct_2gram_count") == 7.0
    assert metric_value(result, "distinct_3gram_count") == 9.0


def test_build_result_includes_path_frequency_estimator(tmp_path):
    args = _args(path_frequency_estimator="min")
    result = build_result(
        args=args,
        baseline_path=tmp_path / "baseline.json",
        diverse_path=tmp_path / "diverse.json",
        baseline_series=[{"sample_count": 5, "state_coverage": 0.1}],
        diverse_series=[{"sample_count": 5, "state_coverage": 0.2}],
        limit=5,
    )

    assert result["experiment"] == "diversity_trend"
    assert result["parameters"]["path_frequency_estimator"] == "min"
    assert result["input_paths"]["baseline"].endswith("baseline.json")


def test_build_result_supports_compare_all_series(tmp_path):
    args = _args(compare_all_diverse=True)
    result = build_result(
        args=args,
        baseline_path=tmp_path / "baseline.json",
        diverse_path=None,
        baseline_series=[{"sample_count": 5, "state_coverage": 0.1}],
        diverse_series=None,
        limit=5,
        extra_input_paths={
            "diverse_harmonic": "harmonic.json",
            "diverse_min": "min.json",
        },
        extra_series={
            "diverse_harmonic": [{"sample_count": 5, "state_coverage": 0.2}],
            "diverse_min": [{"sample_count": 5, "state_coverage": 0.3}],
        },
    )

    assert result["parameters"]["compare_all_diverse"] is True
    assert "diverse_harmonic" in result["series"]
    assert "diverse_min" in result["input_paths"]


def test_default_output_paths_include_min_suffix():
    output_json, output_figure = default_output_paths(
        _args(path_frequency_estimator="min")
    )

    assert output_json == Path("results/metric_trends/model/unit-path_freq_min.json")
    assert output_figure == Path("results/metric_trends/model/unit-path_freq_min.png")


def test_default_output_paths_include_compare_all_suffix():
    output_json, output_figure = default_output_paths(_args(compare_all_diverse=True))

    assert output_json == Path("results/metric_trends/model/unit-compare_all.json")
    assert output_figure == Path("results/metric_trends/model/unit-compare_all.png")


def test_parse_args_accepts_path_frequency_estimator(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["metrics_trend.py", "unit", "--path-frequency-estimator", "min"],
    )

    args = parse_args()

    assert args.path_frequency_estimator == "min"


def test_parse_args_accepts_compare_all_diverse(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["metrics_trend.py", "unit", "--compare-all-diverse"],
    )

    args = parse_args()

    assert args.compare_all_diverse is True


def test_parse_args_rejects_non_positive_step(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["metrics_trend.py", "unit", "--step", "0"])

    with pytest.raises(SystemExit):
        parse_args()
