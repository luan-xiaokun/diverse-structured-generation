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


def test_model_name_only():
    path = get_data_dir_path(_args())
    assert path == Path("data/diverse/qwen2.5-0.5b-instruct")


def test_baseline_flag():
    path = get_data_dir_path(_args(baseline=True))
    assert path == Path("data/baseline/qwen2.5-0.5b-instruct")


def test_top_k():
    path = get_data_dir_path(_args(top_k=50))
    assert path == Path("data/diverse/qwen2.5-0.5b-instruct-top_k_50")


def test_top_p():
    path = get_data_dir_path(_args(top_p=0.9))
    assert path == Path("data/diverse/qwen2.5-0.5b-instruct-top_p_0.9")


def test_temperature():
    path = get_data_dir_path(_args(temperature=1.0))
    assert path == Path("data/diverse/qwen2.5-0.5b-instruct-temperature_1.0")


def test_all_sampling_params():
    path = get_data_dir_path(_args(top_k=50, top_p=0.95, temperature=0.8))
    assert path == Path(
        "data/diverse/qwen2.5-0.5b-instruct-top_k_50-top_p_0.95-temperature_0.8"
    )


def test_none_params_omitted():
    # None values should not appear in the path
    path = get_data_dir_path(_args(top_k=None, top_p=None))
    assert "top_k" not in str(path)
    assert "top_p" not in str(path)


def test_model_name_lowercased():
    path = get_data_dir_path(_args(model="Meta-Llama/Llama-3.1-8B"))
    assert "llama-3.1-8b" in str(path)


def test_model_name_last_segment():
    # Only the last component of org/model is used
    path = get_data_dir_path(_args(model="openai/gpt-4"))
    assert "openai" not in str(path)
    assert "gpt-4" in str(path)
