"""
Pure Python/NumPy implementation of the WD-shift (Weighted Degree with Shifts) kernel.

This module is the cross-platform fallback for ``string_kernel.py``.  The native
C extension is faster, but requires a C compiler and OpenMP; when it is
unavailable this module is used automatically.

For the matrix computation, the Python bottleneck is the O(n²) Python-level loop
over pairs.  We compensate with ``concurrent.futures.ProcessPoolExecutor``
(stdlib, no extra deps): for n > ``_PARALLEL_THRESHOLD`` the computation is
split across all available CPU cores.

Algorithm
---------
The WD-shift kernel between strings s1, s2 is:

    K(s1, s2) = sum_{k=1}^{d}w_k * sum_{i,j: |i-j|<=s} [s1[i:i+k] == s2[j:j+k]]
                / (2*(|i-j|+1))

where w_k = 2*(d-k+1) / (d*(d+1)) are the degree weights.

Reference:
    Rätsch et al., "Accurate splice-site prediction using support vector machines",
in BMC Bioinformatics, 2007.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Switch to parallel computation above this number of sequences.
# Below this threshold the process-pool startup overhead dominates.
_PARALLEL_THRESHOLD = 50


def _kernel_bytes(b1: bytes, b2: bytes, d: int, s: int) -> float:
    """Compute WD-shift kernel between two pre-encoded byte strings."""
    l1, l2 = len(b1), len(b2)
    if l1 == 0 or l2 == 0:
        return 0.0

    arr1 = np.frombuffer(b1, dtype=np.uint8)
    arr2 = np.frombuffer(b2, dtype=np.uint8)
    norm = float(d * (d + 1))
    result = 0.0

    for k in range(1, min(d, l1, l2) + 1):
        w = 2.0 * (d - k + 1) / norm
        n1 = l1 - k + 1
        n2 = l2 - k + 1

        # Sliding windows: shapes (n1, k) and (n2, k)
        w1 = sliding_window_view(arr1, k)
        w2 = sliding_window_view(arr2, k)

        # Pairwise position differences: shape (n1, n2)
        pos1 = np.arange(n1, dtype=np.int32)
        pos2 = np.arange(n2, dtype=np.int32)
        dist = np.abs(pos1[:, None] - pos2[None, :])

        valid = dist <= s
        if not np.any(valid):
            continue

        # k-mer equality test: (n1, n2) bool array
        match = np.all(w1[:, None] == w2[None, :], axis=2)

        hits = valid & match
        if np.any(hits):
            result += (w / 2.0 / (dist[hits].astype(np.float64) + 1.0)).sum()

    return float(result)


# ---------------------------------------------------------------------------
# Module-level worker function (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _compute_row(args: tuple) -> tuple[int, list[float]]:
    """Compute one row of the upper-triangle kernel matrix.

    args = (row_idx, encoded_i, encoded_list, d, s)
    Returns (row_idx, list of kernel values for columns >= row_idx).
    """
    i, b_i, encoded, d, s = args
    row = []
    for j in range(i, len(encoded)):
        row.append(_kernel_bytes(b_i, encoded[j], d, s))
    return i, row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def wd_shift_kernel(s1: str, s2: str, d: int, s: int) -> float:
    """WD-shift kernel between two strings.

    Parameters
    ----------
    s1, s2:
        Input strings (UTF-8 encoded internally).
    d:
        Maximum k-mer length (degree).  Must be >= 1.
    s:
        Maximum allowed position shift.  Must be >= 0.
    """
    if d <= 0:
        raise ValueError(f"d must be >= 1, got {d}")
    return _kernel_bytes(s1.encode("utf-8"), s2.encode("utf-8"), d, s)


def compute_wd_kernel_matrix(sequences: list[str], d: int, s: int) -> np.ndarray:
    """Compute the pairwise WD-shift kernel matrix.

    For ``len(sequences) > _PARALLEL_THRESHOLD``, uses all available CPU cores
    via ``concurrent.futures.ProcessPoolExecutor``; otherwise runs sequentially.

    Parameters
    ----------
    sequences:
        List of input strings.
    d:
        Maximum k-mer length.
    s:
        Maximum position shift.

    Returns
    -------
    np.ndarray
        Symmetric n×n kernel matrix.
    """
    if not isinstance(sequences, list) or not all(
        isinstance(x, str) for x in sequences
    ):
        raise TypeError("sequences must be a list of str")
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    encoded = [seq.encode("utf-8") for seq in sequences]
    matrix = np.zeros((n, n), dtype=np.float64)

    if n <= _PARALLEL_THRESHOLD:
        # Sequential: no process-pool overhead
        for i in range(n):
            matrix[i, i] = _kernel_bytes(encoded[i], encoded[i], d, s)
            for j in range(i + 1, n):
                val = _kernel_bytes(encoded[i], encoded[j], d, s)
                matrix[i, j] = val
                matrix[j, i] = val
    else:
        # Parallel: distribute rows across CPU cores
        n_workers = os.cpu_count() or 1
        args_list = [(i, encoded[i], encoded, d, s) for i in range(n)]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, row_vals in pool.map(_compute_row, args_list):
                for offset, val in enumerate(row_vals):
                    j = i + offset
                    matrix[i, j] = val
                    matrix[j, i] = val

    return matrix


def compute_wd_kernel_multiple(
    x: str, sequences: list[str], d: int, s: int
) -> np.ndarray:
    """Compute the kernel between one string *x* and each string in *sequences*.

    Returns a 1-D array of length len(sequences).
    """
    x_enc = x.encode("utf-8")
    return np.array(
        [_kernel_bytes(x_enc, seq.encode("utf-8"), d, s) for seq in sequences],
        dtype=np.float64,
    )
