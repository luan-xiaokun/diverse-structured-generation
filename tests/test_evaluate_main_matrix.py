import json
from argparse import Namespace

import pytest

from scripts import evaluate_main_matrix as emm


def _write_generation(path, *, grammar="toy", samples=None, regex="[ab]+", **extra):
    samples = ["a", "b", "ab"] if samples is None else samples
    payload = {
        "grammar": grammar,
        "regex": regex,
        "prompt": "unit",
        "model": "test/model",
        "max_tokens": 8,
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "seed": 7,
        "device": "cpu",
        "model_parameter_device": "cpu",
        "gamma": 0.3,
        "completion_bias_eta": 0.1,
        "path_frequency_estimator": "harmonic",
        "baseline_backend": "internal",
        "samples": samples,
        "elapsed_seconds": 2.0,
        "attempted_batches": 1,
        "accepted_samples": len(samples),
        "rejected_samples": 1,
    }
    payload.update(extra)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path, **overrides):
    values = {
        "data_root": tmp_path / "data",
        "output_dir": tmp_path / "results",
        "models": "test/model",
        "grammars": "toy",
        "methods": "diverse_guide_plus",
        "expected_samples": 3,
        "max_samples": None,
        "d": 2,
        "s": 1,
        "step": 2,
        "strict": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validity_split_uses_saved_regex():
    valid, invalid = emm.validity_split(["a", "ab", "c"], "[ab]+")

    assert valid == ["a", "ab"]
    assert invalid == ["c"]


def test_evaluate_matrix_records_metrics_efficiency_and_trends(tmp_path):
    args = _args(tmp_path)
    path = (
        args.data_root
        / "diverse_guide_plus"
        / emm.model_slug("test/model")
        / "toy.json"
    )
    _write_generation(path, samples=["a", "b", "c"])

    result = emm.evaluate_matrix(args)

    assert result["metadata"]["model_loading"] == "none"
    assert result["errors"] == []
    record = result["records"][0]
    assert record["status"] == "complete"
    assert record["saved_sample_count"] == 3
    assert record["valid_sample_count"] == 2
    assert record["invalid_sample_count"] == 1
    assert record["validity_rate"] == pytest.approx(2 / 3)
    assert record["efficiency"]["accepted_per_second"] == pytest.approx(1.5)
    assert record["generation_parameters"]["gamma"] == pytest.approx(0.3)
    assert record["metrics"]["state_coverage"] >= 0.0

    series = result["trend"]["series"]["diverse_guide_plus/model/toy"]
    assert [point["sample_count"] for point in series] == [2, 3]
    assert series[-1]["valid_sample_count"] == 2
    assert series[-1]["validity_rate"] == pytest.approx(2 / 3)


def test_evaluate_matrix_tolerates_missing_cells_by_default(tmp_path):
    result = emm.evaluate_matrix(_args(tmp_path))

    assert result["records"][0]["status"] == "missing"
    assert result["errors"]


def test_flatten_summary_keeps_key_fields(tmp_path):
    args = _args(tmp_path)
    path = (
        args.data_root
        / "diverse_guide_plus"
        / emm.model_slug("test/model")
        / "toy.json"
    )
    _write_generation(path)
    result = emm.evaluate_matrix(args)

    summary = emm.flatten_summary(result["records"][0])

    assert summary["method"] == "diverse_guide_plus"
    assert summary["validity_rate"] == pytest.approx(1.0)
    assert summary["completion_bias_eta"] == pytest.approx(0.1)
    assert summary["accepted_per_second"] == pytest.approx(1.5)
