"""Tests for shared path utilities."""

import argparse
from pathlib import Path

from diverse_guide.evaluation.paths import get_data_dir_path


def _args(**kwargs):
    defaults = dict(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        top_k=None,
        top_p=None,
        temperature=None,
        baseline=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_model_name_is_last_segment_lowercased():
    assert get_data_dir_path(_args()) == Path("data/diverse/qwen2.5-0.5b-instruct")
    assert "llama-3.1-8b" in str(
        get_data_dir_path(_args(model="Meta-Llama/Llama-3.1-8B"))
    )
    path = str(get_data_dir_path(_args(model="openai/gpt-4")))
    assert "openai" not in path and "gpt-4" in path


def test_sampling_params_are_appended_when_set():
    base = "data/diverse/qwen2.5-0.5b-instruct"
    assert get_data_dir_path(_args(top_k=50)) == Path(f"{base}-top_k_50")
    assert get_data_dir_path(_args(top_p=0.9)) == Path(f"{base}-top_p_0.9")
    assert get_data_dir_path(_args(temperature=1.0)) == Path(f"{base}-temperature_1.0")
    assert get_data_dir_path(_args(top_k=50, top_p=0.95, temperature=0.8)) == Path(
        f"{base}-top_k_50-top_p_0.95-temperature_0.8"
    )


def test_baseline_and_ablation_directories():
    assert get_data_dir_path(_args(baseline=True)) == Path(
        "data/baseline/qwen2.5-0.5b-instruct"
    )
    assert get_data_dir_path(_args(ablation_component="reward")) == Path(
        "data/diverse/qwen2.5-0.5b-instruct-no_reward"
    )
