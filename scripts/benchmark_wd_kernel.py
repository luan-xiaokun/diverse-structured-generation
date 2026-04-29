"""
Benchmark: C extension vs pure NumPy WD-shift kernel.

Generates synthetic strings of varying lengths and counts, then times both
backends on the pairwise kernel matrix computation.

Usage:
    uv run python scripts/benchmark_wd_kernel.py
    uv run python scripts/benchmark_wd_kernel.py --n 100 200 500 --d 5 --s 1
"""

import argparse
import random
import string
import time

import numpy as np


def random_strings(n: int, min_len: int, max_len: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    chars = string.ascii_lowercase + string.digits
    return [
        "".join(rng.choices(chars, k=rng.randint(min_len, max_len))) for _ in range(n)
    ]


def css_color_strings(n: int, seed: int = 42) -> list[str]:
    """Simulate CSS hex color outputs (6-9 chars)."""
    rng = random.Random(seed)
    hexchars = "0123456789abcdef"
    results = []
    for _ in range(n):
        if rng.random() < 0.7:
            results.append("#" + "".join(rng.choices(hexchars, k=6)))
        else:
            results.append("#" + "".join(rng.choices(hexchars, k=3)))
    return results


def time_fn(fn, *args, repeat: int = 3) -> tuple[float, float]:
    """Return (min, median) elapsed seconds over *repeat* calls."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[0], times[len(times) // 2]


def run_benchmark(
    label: str,
    sequences: list[str],
    d: int,
    s: int,
    repeat: int = 3,
) -> None:
    n = len(sequences)
    avg_len = sum(len(x) for x in sequences) / n
    print(f"\n{'─' * 60}")
    print(f"  {label}  |  n={n}  avg_len={avg_len:.1f}  d={d}  s={s}")
    print(f"{'─' * 60}")

    import diverse_guide.evaluation.string_kernel_py as py_mod

    # Sequential (single-core) numpy — force sequential by setting threshold high
    orig_threshold = py_mod._PARALLEL_THRESHOLD
    py_mod._PARALLEL_THRESHOLD = 10**9
    py_seq_min, py_seq_med = time_fn(
        py_mod.compute_wd_kernel_matrix, sequences, d, s, repeat=repeat
    )
    py_mod._PARALLEL_THRESHOLD = orig_threshold
    print(f"  NumPy (1 core)  : min={py_seq_min:.3f}s  median={py_seq_med:.3f}s")

    # Parallel numpy — only if n > threshold
    if n > py_mod._PARALLEL_THRESHOLD:
        py_par_min, py_par_med = time_fn(
            py_mod.compute_wd_kernel_matrix, sequences, d, s, repeat=repeat
        )
        import os

        ncpu = os.cpu_count() or 1
        speedup_par = py_seq_med / py_par_med if py_par_med > 0 else float("inf")
        print(
            f"  NumPy ({ncpu} cores) : min={py_par_min:.3f}s"
            f"  median={py_par_med:.3f}s  ({speedup_par:.1f}× vs 1-core)"
        )
    else:
        print(
            f"  NumPy (parallel): n <= threshold ({py_mod._PARALLEL_THRESHOLD}),"
            " sequential used"
        )

    # Optionally benchmark C backend
    try:
        import ctypes
        from pathlib import Path

        lib_path = (
            Path(__file__).resolve().parent.parent
            / "build"
            / "native"
            / "wd_kernel"
            / "wd_kernel.so"
        )
        ctypes.CDLL(str(lib_path))  # check availability

        import diverse_guide.evaluation.string_kernel as sk_mod

        if sk_mod.STRING_KERNEL_BACKEND == "c":
            c_min, c_med = time_fn(
                sk_mod.compute_wd_kernel_matrix, sequences, d, s, repeat=repeat
            )
            speedup_c = py_seq_med / c_med if c_med > 0 else float("inf")
            print(
                f"  C+OpenMP        : min={c_min:.3f}s  median={c_med:.3f}s"
                f"  ({speedup_c:.0f}× vs NumPy 1-core)"
            )

            # Correctness check: compare outputs
            mtx_py = py_mod.compute_wd_kernel_matrix(sequences[:10], d, s)
            mtx_c = sk_mod.compute_wd_kernel_matrix(sequences[:10], d, s)
            max_err = np.abs(mtx_py - mtx_c).max()
            print(
                f"  Correctness (n=10): max err={max_err:.2e}  "
                f"{'✓' if max_err < 1e-9 else '✗ MISMATCH'}"
            )
        else:
            print(
                "  C backend not available "
                "(build/native/wd_kernel/wd_kernel.so not built)"
            )
    except OSError:
        print(
            "  C backend not available (build/native/wd_kernel/wd_kernel.so not built)"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark WD-shift kernel backends.")
    parser.add_argument(
        "--n", type=int, nargs="+", default=[100, 300, 500], help="Sample sizes to test"
    )
    parser.add_argument("--d", type=int, default=5, help="Max k-mer degree")
    parser.add_argument("--s", type=int, default=1, help="Max position shift")
    parser.add_argument("--repeat", type=int, default=3, help="Timing repetitions")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"WD-shift kernel benchmark  (d={args.d}, s={args.s}, repeat={args.repeat})")

    for n in args.n:
        run_benchmark(
            "CSS colors   ", css_color_strings(n), args.d, args.s, args.repeat
        )
        run_benchmark(
            "Random 5-30  ", random_strings(n, 5, 30), args.d, args.s, args.repeat
        )

    print(f"\n{'─' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
