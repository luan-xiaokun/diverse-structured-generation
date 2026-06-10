import pytest

import scripts.benchmark_runtime_latency as bench


def test_token_timing_streamer_skips_prompt_and_tracks_gaps(monkeypatch):
    now = iter([10.0, 10.1, 10.25])
    monkeypatch.setattr(bench, "perf_counter", lambda: next(now))

    streamer = bench.TokenTimingStreamer(prompt_token_count=2)
    streamer.start(9.5)
    streamer.put([1, 2, 3, 4, 5])

    assert streamer.generated_token_count == 3
    assert streamer.first_token_delay_ms == 500.0
    assert streamer.inter_token_delays_ms == pytest.approx([100.0, 150.0])


def test_summary_reports_mean_and_std():
    summary = bench._summary([1.0, 2.0, 3.0])

    assert summary == {
        "mean": 2.0,
        "std": bench.pstdev([1.0, 2.0, 3.0]),
        "min": 1.0,
        "max": 3.0,
    }


def test_parse_args_accepts_require_cuda(tmp_path):
    args = bench.parse_args(
        [
            "json",
            "--require-cuda",
        ]
    )

    assert args.require_cuda is True
    assert args.repo_path.exists()
