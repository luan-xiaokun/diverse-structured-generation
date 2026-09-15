"""Wait for idle CPU/GPU resources, then launch a benchmark command.

The launcher is intentionally conservative by default: it waits for enough free
VRAM and low CPU utilization for several consecutive checks, writes all child
process output to a timestamped log file, and mirrors that output to stdout so
progress remains visible.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MIB_PER_GIB = 1024


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    free_mib: int
    total_mib: int
    utilization_percent: int | None


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    gpus: list[GpuSnapshot]


@dataclass(frozen=True)
class IdleDecision:
    idle: bool
    selected_gpu: GpuSnapshot | None
    reason: str


def parse_nvidia_smi_csv(output: str) -> list[GpuSnapshot]:
    gpus = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError(f"unexpected nvidia-smi row: {raw_line!r}")
        index = int(parts[0])
        free_mib = int(parts[1])
        total_mib = int(parts[2])
        utilization_percent = None
        if len(parts) >= 4 and parts[3] not in {"", "[N/A]", "N/A"}:
            utilization_percent = int(parts[3])
        gpus.append(
            GpuSnapshot(
                index=index,
                free_mib=free_mib,
                total_mib=total_mib,
                utilization_percent=utilization_percent,
            )
        )
    return gpus


def query_gpus_with_nvidia_smi() -> list[GpuSnapshot]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_csv(result.stdout)


def query_gpus_with_pynvml() -> list[GpuSnapshot]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("pynvml is not installed") from exc

    pynvml.nvmlInit()
    try:
        gpus = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except pynvml.NVMLError:
                utilization = None
            gpus.append(
                GpuSnapshot(
                    index=index,
                    free_mib=int(memory.free // (1024 * 1024)),
                    total_mib=int(memory.total // (1024 * 1024)),
                    utilization_percent=utilization,
                )
            )
        return gpus
    finally:
        pynvml.nvmlShutdown()


def query_gpus() -> list[GpuSnapshot]:
    try:
        return query_gpus_with_nvidia_smi()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return query_gpus_with_pynvml()


def read_proc_stat() -> tuple[int, int]:
    with open("/proc/stat", encoding="utf-8") as handle:
        fields = handle.readline().split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("cannot read aggregate CPU stats from /proc/stat")
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def sample_cpu_percent(
    sample_seconds: float,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    stat_reader: Callable[[], tuple[int, int]] = read_proc_stat,
) -> float:
    idle_1, total_1 = stat_reader()
    sleeper(sample_seconds)
    idle_2, total_2 = stat_reader()
    total_delta = total_2 - total_1
    idle_delta = idle_2 - idle_1
    if total_delta <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def choose_gpu(gpus: list[GpuSnapshot], gpu_index: str) -> GpuSnapshot | None:
    if not gpus:
        return None
    if gpu_index == "auto":
        return max(gpus, key=lambda gpu: gpu.free_mib)
    requested = int(gpu_index)
    for gpu in gpus:
        if gpu.index == requested:
            return gpu
    return None


def decide_idle(
    snapshot: ResourceSnapshot,
    *,
    min_free_vram_mib: int,
    max_cpu_percent: float,
    gpu_index: str,
    skip_gpu_check: bool,
) -> IdleDecision:
    if snapshot.cpu_percent > max_cpu_percent:
        return IdleDecision(
            idle=False,
            selected_gpu=None,
            reason=(
                f"cpu busy: {snapshot.cpu_percent:.1f}% > "
                f"{max_cpu_percent:.1f}%"
            ),
        )

    if skip_gpu_check:
        return IdleDecision(idle=True, selected_gpu=None, reason="cpu idle")

    selected_gpu = choose_gpu(snapshot.gpus, gpu_index)
    if selected_gpu is None:
        return IdleDecision(
            idle=False,
            selected_gpu=None,
            reason=f"gpu {gpu_index!r} not available",
        )
    if selected_gpu.free_mib < min_free_vram_mib:
        return IdleDecision(
            idle=False,
            selected_gpu=selected_gpu,
            reason=(
                f"gpu {selected_gpu.index} free "
                f"{selected_gpu.free_mib / MIB_PER_GIB:.1f}GiB < "
                f"{min_free_vram_mib / MIB_PER_GIB:.1f}GiB"
            ),
        )
    return IdleDecision(
        idle=True,
        selected_gpu=selected_gpu,
        reason=(
            f"gpu {selected_gpu.index} free "
            f"{selected_gpu.free_mib / MIB_PER_GIB:.1f}GiB, "
            f"cpu {snapshot.cpu_percent:.1f}%"
        ),
    )


def format_gpus(gpus: list[GpuSnapshot]) -> str:
    if not gpus:
        return "none"
    parts = []
    for gpu in gpus:
        util = (
            "n/a"
            if gpu.utilization_percent is None
            else f"{gpu.utilization_percent}%"
        )
        parts.append(
            f"{gpu.index}:free={gpu.free_mib / MIB_PER_GIB:.1f}GiB/"
            f"{gpu.total_mib / MIB_PER_GIB:.1f}GiB,util={util}"
        )
    return "; ".join(parts)


def wait_for_idle(args: argparse.Namespace) -> GpuSnapshot | None:
    min_free_vram_mib = int(args.free_vram_gb * MIB_PER_GIB)
    consecutive_idle = 0
    started = time.monotonic()

    while True:
        cpu_percent = sample_cpu_percent(args.cpu_sample_seconds)
        gpus = [] if args.skip_gpu_check else query_gpus()
        snapshot = ResourceSnapshot(cpu_percent=cpu_percent, gpus=gpus)
        decision = decide_idle(
            snapshot,
            min_free_vram_mib=min_free_vram_mib,
            max_cpu_percent=args.max_cpu_percent,
            gpu_index=args.gpu_index,
            skip_gpu_check=args.skip_gpu_check,
        )
        if decision.idle:
            consecutive_idle += 1
        else:
            consecutive_idle = 0

        print(
            "[idle-check] "
            f"idle={decision.idle} stable={consecutive_idle}/{args.stable_checks} "
            f"cpu={cpu_percent:.1f}% gpus={format_gpus(gpus)} "
            f"reason={decision.reason}",
            flush=True,
        )

        if consecutive_idle >= args.stable_checks:
            return decision.selected_gpu

        if args.timeout_seconds is not None:
            elapsed = time.monotonic() - started
            if elapsed >= args.timeout_seconds:
                raise TimeoutError(
                    f"resources did not become idle within {args.timeout_seconds}s"
                )

        time.sleep(args.poll_interval_seconds)


def build_log_path(log_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"engineering-benchmark-{timestamp}.log"


def launch_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[launch] command={' '.join(command)}", flush=True)
    print(f"[launch] log={log_path}", flush=True)
    with log_path.open("ab") as log_file:
        header = (
            f"[launcher] started={datetime.now().isoformat(timespec='seconds')}\n"
            f"[launcher] command={' '.join(command)}\n"
        )
        log_file.write(header.encode())
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_file.write(chunk)
            log_file.flush()
        return_code = process.wait()
        footer = (
            f"\n[launcher] finished={datetime.now().isoformat(timespec='seconds')} "
            f"return_code={return_code}\n"
        )
        log_file.write(footer.encode())
        log_file.flush()
    print(f"[launch] return_code={return_code}", flush=True)
    return return_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for idle CPU/GPU resources before launching benchmarks."
    )
    parser.add_argument(
        "--free-vram-gb",
        type=float,
        default=12.0,
        help="Required free VRAM on the selected GPU.",
    )
    parser.add_argument(
        "--max-cpu-percent",
        type=float,
        default=20.0,
        help="Maximum aggregate CPU utilization before launch.",
    )
    parser.add_argument(
        "--gpu-index",
        default="auto",
        help="'auto' selects the GPU with most free memory; otherwise a GPU index.",
    )
    parser.add_argument(
        "--stable-checks",
        type=int,
        default=3,
        help="Number of consecutive idle checks required before launch.",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--cpu-sample-seconds", type=float, default=1.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional maximum wait time. Omit to wait indefinitely.",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Only check CPU before launching.",
    )
    parser.add_argument(
        "--no-set-cuda-visible-devices",
        action="store_true",
        help="Do not set CUDA_VISIBLE_DEVICES to the selected GPU.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/engineering/optimization_benchmark"),
    )
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to launch after resources are idle. If omitted, launches "
            "experiments/run_engineering_benchmark_matrix.sh."
        ),
    )
    args = parser.parse_args(argv)
    if args.free_vram_gb <= 0:
        parser.error("--free-vram-gb must be positive")
    if args.max_cpu_percent < 0 or args.max_cpu_percent > 100:
        parser.error("--max-cpu-percent must be in [0, 100]")
    if args.stable_checks <= 0:
        parser.error("--stable-checks must be positive")
    if args.poll_interval_seconds < 0:
        parser.error("--poll-interval-seconds cannot be negative")
    if args.cpu_sample_seconds <= 0:
        parser.error("--cpu-sample-seconds must be positive")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.gpu_index != "auto":
        try:
            int(args.gpu_index)
        except ValueError:
            parser.error("--gpu-index must be 'auto' or an integer")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command or ["experiments/run_engineering_benchmark_matrix.sh"]
    selected_gpu = wait_for_idle(args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if (
        selected_gpu is not None
        and not args.no_set_cuda_visible_devices
        and not args.skip_gpu_check
    ):
        env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.index)
        print(
            f"[launch] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
    return launch_command(command, log_path=build_log_path(args.log_dir), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
