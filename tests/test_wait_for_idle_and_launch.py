import pytest

from scripts import wait_for_idle_and_launch as launcher


def test_parse_nvidia_smi_csv_accepts_utilization_rows():
    rows = launcher.parse_nvidia_smi_csv(
        "0, 14000, 24576, 0\n"
        "1, 9000, 24576, 12\n"
    )

    assert rows[0] == launcher.GpuSnapshot(
        index=0,
        free_mib=14000,
        total_mib=24576,
        utilization_percent=0,
    )
    assert rows[1].free_mib == 9000


def test_decide_idle_selects_gpu_with_most_free_memory():
    snapshot = launcher.ResourceSnapshot(
        cpu_percent=5.0,
        gpus=[
            launcher.GpuSnapshot(0, 9000, 24576, 0),
            launcher.GpuSnapshot(1, 13000, 24576, 0),
        ],
    )

    decision = launcher.decide_idle(
        snapshot,
        min_free_vram_mib=12 * launcher.MIB_PER_GIB,
        max_cpu_percent=20.0,
        gpu_index="auto",
        skip_gpu_check=False,
    )

    assert decision.idle is True
    assert decision.selected_gpu is not None
    assert decision.selected_gpu.index == 1


def test_decide_idle_rejects_busy_cpu_before_gpu_selection():
    snapshot = launcher.ResourceSnapshot(
        cpu_percent=80.0,
        gpus=[launcher.GpuSnapshot(0, 20000, 24576, 0)],
    )

    decision = launcher.decide_idle(
        snapshot,
        min_free_vram_mib=12 * launcher.MIB_PER_GIB,
        max_cpu_percent=20.0,
        gpu_index="auto",
        skip_gpu_check=False,
    )

    assert decision.idle is False
    assert "cpu busy" in decision.reason
    assert decision.selected_gpu is None


def test_decide_idle_rejects_low_vram_on_requested_gpu():
    snapshot = launcher.ResourceSnapshot(
        cpu_percent=5.0,
        gpus=[
            launcher.GpuSnapshot(0, 9000, 24576, 0),
            launcher.GpuSnapshot(1, 13000, 24576, 0),
        ],
    )

    decision = launcher.decide_idle(
        snapshot,
        min_free_vram_mib=12 * launcher.MIB_PER_GIB,
        max_cpu_percent=20.0,
        gpu_index="0",
        skip_gpu_check=False,
    )

    assert decision.idle is False
    assert decision.selected_gpu is not None
    assert decision.selected_gpu.index == 0
    assert "free" in decision.reason


def test_sample_cpu_percent_from_proc_stat_deltas():
    values = iter([(100, 200), (150, 300)])

    percent = launcher.sample_cpu_percent(
        1.0,
        sleeper=lambda seconds: None,
        stat_reader=lambda: next(values),
    )

    assert percent == pytest.approx(50.0)
