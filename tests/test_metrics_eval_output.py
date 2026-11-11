import json
import sys
from argparse import Namespace

import pytest

from scripts.metrics_eval import (
    add_perplexity_metrics,
    build_metrics_result,
    compute_metrics,
    parse_args,
    print_metrics_result,
    write_metrics_result,
)


def _args(**overrides):
    values = {
        "grammar": "unit",
        "model": "test/model",
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "baseline": False,
        "d": 2,
        "s": 1,
        "n": None,
        "ppl": False,
        "ppl_model": "test/ppl",
        "ablation_component": None,
    }
    values.update(overrides)
    return Namespace(**values)


METADATA = {"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None}


def test_compute_metrics_and_sample_limit(dfa_single, monkeypatch):
    result = compute_metrics(dfa_single, ["a", "b"], d=2, s=1)

    assert result["sample_count"] == 2
    assert result["average_length"] == pytest.approx(1.0)
    assert result["dfa"] == {"state_count": 2, "transition_count": 2}
    assert 0.0 <= result["metrics"]["state_coverage"] <= 1.0
    assert 0.0 <= result["metrics"]["transition_coverage"] <= 1.0
    assert 0.0 <= result["metrics"]["path_coverage"] <= 1.0
    assert result["metrics"]["distinct_2gram"] == [0, 2]
    assert result["metrics"]["distinct_3gram"] == [0, 2]
    assert result["metrics"]["average_perplexity"] is None
    assert result["metrics"]["perplexity_count"] == 0
    assert result["metrics"]["perplexity_error_count"] == 0

    sliced = compute_metrics(dfa_single, ["a", "b"], d=2, s=1, n=1)
    assert sliced["sample_count"] == 1
    assert sliced["metrics"]["distinct_2gram"] == [0, 1]

    with pytest.raises(ValueError, match="n must be positive"):
        compute_metrics(dfa_single, ["a", "b"], n=0)
    monkeypatch.setattr(sys, "argv", ["metrics_eval.py", "unit", "-n", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def test_build_metrics_result_records_setting_and_parameters(dfa_single, tmp_path):
    input_path = tmp_path / "samples.json"
    result = build_metrics_result(
        args=_args(grammar="unit", baseline=True, temperature=1.5),
        dfa=dfa_single,
        inputs=["a", "b"],
        input_path=input_path,
        experiment="temperature_ablation",
        metadata=METADATA,
    )

    assert result["schema_version"] == 1
    assert result["experiment"] == "temperature_ablation"
    assert result["setting"] == "baseline"
    assert result["grammar"] == "unit"
    assert result["model"] == "test/model"
    assert result["input_path"] == str(input_path)
    assert result["parameters"]["temperature"] == 1.5
    assert result["metadata"]["timestamp_utc"] == "2026-04-27T00:00:00Z"

    diverse = build_metrics_result(
        args=_args(),
        dfa=dfa_single,
        inputs=["a", "b"],
        input_path=input_path,
        metadata=METADATA,
    )
    assert diverse["setting"] == "diverse"
    assert set(diverse["parameters"]) == {
        "d",
        "s",
        "n",
        "temperature",
        "top_k",
        "top_p",
        "ablation_component",
        "ppl",
        "ppl_model",
    }


def test_metrics_result_print_perplexity_and_write(capsys, dfa_single, tmp_path):
    result = build_metrics_result(
        args=_args(ppl=True),
        dfa=dfa_single,
        inputs=["a", "b"],
        input_path=tmp_path / "samples.json",
        metadata=METADATA,
    )

    print_metrics_result(result)
    stdout = capsys.readouterr().out
    assert "- Number of samples: 2" in stdout
    assert "Average perplexity:" not in stdout

    original_metadata = result["metadata"]
    original_vendi_score = result["metrics"]["vendi_score"]
    add_perplexity_metrics(result, [4.0, 6.0], perplexity_error_count=1)
    assert result["metrics"]["average_perplexity"] == pytest.approx(5.0)
    assert result["metrics"]["perplexity_count"] == 2
    assert result["metrics"]["perplexity_error_count"] == 1
    assert result["metrics"]["vendi_score"] == original_vendi_score
    assert result["metadata"] is original_metadata

    output_path = tmp_path / "results" / "diversity" / "unit.json"
    write_metrics_result(output_path, result)
    saved = json.loads(output_path.read_text())
    assert saved["grammar"] == "unit"
    assert saved["metrics"] == result["metrics"]
