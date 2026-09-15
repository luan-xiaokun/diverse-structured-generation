import torch

from diverse_guide.guide_rust import DiverseRegexLogitsProcessor
from scripts import benchmark_fixed_trace_replay as replay


def test_profile_adjustments_uses_full_then_values_only_cache_path(mock_tokenizer):
    processor = DiverseRegexLogitsProcessor("[ab]", mock_tokenizer)
    state = processor.guide.initial_state
    device = torch.device("cpu")

    miss = replay.profile_adjustments(
        processor=processor,
        state=state,
        device=device,
    )
    hit = replay.profile_adjustments(
        processor=processor,
        state=state,
        device=device,
    )

    assert miss["path"] == "full-cache-miss"
    assert hit["path"] == "values-cache-hit"
    assert miss["allowed_count"] == hit["allowed_count"] == 2
    assert miss["rust_compute_ms"] >= 0.0
    assert hit["rust_compute_ms"] >= 0.0
    assert miss["pyo3_transfer_ms"] >= 0.0
    assert hit["pyo3_transfer_ms"] >= 0.0


def test_fixed_trace_replay_once_without_model(mock_tokenizer):
    trace_data = {
        "regex": "[ab]",
        "traces": [
            {
                "prompt_token_ids": [0, 0],
                "generated_token_ids": [10],
            }
        ],
    }

    class Args:
        gamma = 0.3
        completion_bias_eta = 0.1
        path_frequency_estimator = "harmonic"

    run = replay.replay_once(
        args=Args(),
        guide_rust=__import__("diverse_guide.guide_rust", fromlist=[""]),
        tokenizer=mock_tokenizer,
        trace_data=trace_data,
        model=None,
        score_device=torch.device("cpu"),
        run_index=0,
        warmup=False,
    )

    assert run["trace_count"] == 1
    assert run["step_count"] == 1
    assert run["metrics"]["adjust_rust_compute_ms"]["count"] == 1
    assert run["metrics"]["torch_scatter_ms"]["mean"] >= 0.0
