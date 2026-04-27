"""
String kernel module: WD-shift kernel with native extension or NumPy fallback.

Two backends are available, selected automatically at import time:

* **C + OpenMP**: fastest; requires a C compiler and OpenMP.
  Build with ``python scripts/build_wd_kernel.py``.
* **Pure Python/NumPy** (``string_kernel_py``): no build step; cross-platform;
  roughly 10–30× slower than the C version for large sample sets.

The active backend is reported in ``STRING_KERNEL_BACKEND``.

Public API (identical regardless of backend)
--------------------------------------------
- ``wd_shift_kernel(s1, s2, d, s) -> float``
- ``compute_wd_kernel_matrix(sequences, d, s) -> np.ndarray``
- ``compute_wd_kernel_multiple(x, sequences, d, s) -> np.ndarray``
"""

import ctypes
import errno as py_errno
import os
import platform
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Try C extension first
# ---------------------------------------------------------------------------

_system = platform.system()
_lib_name = "wd_kernel.dll" if _system == "Windows" else "wd_kernel.so"
_module_dir = Path(__file__).resolve().parent
_package_root = _module_dir.parent
_project_root = _package_root.parent.parent
_candidate_lib_paths = [
    # Preferred for packaged installs
    _package_root / "_native" / _lib_name,
    # Preferred for local development builds
    _project_root / "build" / "native" / "wd_kernel" / _lib_name,
    # Legacy fallback path
    _project_root / "native" / "wd_kernel" / _lib_name,
]

_c_char_p_array = ctypes.POINTER(ctypes.c_char_p)
_c_int_array = ctypes.POINTER(ctypes.c_int)
_c_double_array = ctypes.POINTER(ctypes.c_double)

_wd_lib = None
_loaded_lib_path: Path | None = None
for _path in _candidate_lib_paths:
    if not _path.exists():
        continue
    try:
        _wd_lib = ctypes.CDLL(str(_path))
        _loaded_lib_path = _path
        break
    except OSError:
        continue

if _wd_lib is not None:
    _wd_lib.wd_shift_kernel.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _wd_lib.wd_shift_kernel.restype = ctypes.c_double

    _wd_lib.wd_shift_kernel_matrix_omp.argtypes = [
        _c_char_p_array,
        ctypes.c_int,
        _c_int_array,
        ctypes.c_int,
        ctypes.c_int,
        _c_double_array,
    ]
    _wd_lib.wd_shift_kernel_matrix_omp.restype = ctypes.c_int

    _wd_lib.wd_shift_kernel_multiple_omp.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        _c_char_p_array,
        ctypes.c_int,
        _c_int_array,
        ctypes.c_int,
        ctypes.c_int,
        _c_double_array,
    ]
    _wd_lib.wd_shift_kernel_multiple_omp.restype = ctypes.c_int

    STRING_KERNEL_BACKEND: str = "c"
else:
    STRING_KERNEL_BACKEND = "python"


# ---------------------------------------------------------------------------
# C backend implementations
# ---------------------------------------------------------------------------


def _wd_shift_kernel_c(s1: str, s2: str, d: int, s: int) -> float:
    s1b, s2b = s1.encode("utf-8"), s2.encode("utf-8")
    return _wd_lib.wd_shift_kernel(s1b, len(s1b), s2b, len(s2b), d, s)


def _compute_wd_kernel_matrix_c(sequences: list[str], d: int, s: int) -> np.ndarray:
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    encoded = [seq.encode("utf-8") for seq in sequences]
    c_strings = (ctypes.c_char_p * n)(*encoded)
    lengths = [len(e) for e in encoded]
    c_lengths = (ctypes.c_int * n)(*lengths)

    out = np.zeros((n, n), dtype=np.float64)
    c_out = out.ctypes.data_as(_c_double_array)

    ctypes.set_errno(0)
    status = _wd_lib.wd_shift_kernel_matrix_omp(c_strings, n, c_lengths, d, s, c_out)
    if status != 0:
        c_errno = ctypes.get_errno()
        msg = (
            "wd_shift_kernel_matrix_omp failed "
            f"(errno {c_errno}: {os.strerror(c_errno)})"
        )
        if c_errno == py_errno.EINVAL:
            raise ValueError(msg)
        elif c_errno == py_errno.ENOMEM:
            raise MemoryError(msg)
        else:
            raise OSError(msg)
    return out


def _compute_wd_kernel_multiple_c(
    x: str, sequences: list[str], d: int, s: int
) -> np.ndarray:
    x_enc = x.encode("utf-8")
    n = len(sequences)
    encoded = [seq.encode("utf-8") for seq in sequences]
    c_strings = (ctypes.c_char_p * n)(*encoded)
    lengths = [len(e) for e in encoded]
    c_lengths = (ctypes.c_int * n)(*lengths)

    out = np.zeros(n, dtype=np.float64)
    c_out = out.ctypes.data_as(_c_double_array)

    ctypes.set_errno(0)
    status = _wd_lib.wd_shift_kernel_multiple_omp(
        x_enc, len(x_enc), c_strings, n, c_lengths, d, s, c_out
    )
    if status != 0:
        c_errno = ctypes.get_errno()
        msg = (
            "wd_shift_kernel_multiple_omp failed "
            f"(errno {c_errno}: {os.strerror(c_errno)})"
        )
        if c_errno == py_errno.EINVAL:
            raise ValueError(msg)
        elif c_errno == py_errno.ENOMEM:
            raise MemoryError(msg)
        else:
            raise OSError(msg)
    return out


# ---------------------------------------------------------------------------
# Public API: dispatch to C or Python backend
# ---------------------------------------------------------------------------

if STRING_KERNEL_BACKEND == "c":

    def wd_shift_kernel(s1: str, s2: str, d: int, s: int) -> float:
        """WD-shift kernel between two strings (C + OpenMP backend)."""
        return _wd_shift_kernel_c(s1, s2, d, s)

    def compute_wd_kernel_matrix(sequences: list[str], d: int, s: int) -> np.ndarray:
        """Pairwise WD-shift kernel matrix (C + OpenMP backend)."""
        if not isinstance(sequences, list) or not all(
            isinstance(x, str) for x in sequences
        ):
            raise TypeError("sequences must be a list of str")
        return _compute_wd_kernel_matrix_c(sequences, d, s)

    def compute_wd_kernel_multiple(
        x: str, sequences: list[str], d: int, s: int
    ) -> np.ndarray:
        """Kernel between one string and a list (C + OpenMP backend)."""
        return _compute_wd_kernel_multiple_c(x, sequences, d, s)

else:
    from .string_kernel_py import (
        compute_wd_kernel_matrix,
        compute_wd_kernel_multiple,
        wd_shift_kernel,
    )

    # Re-export with module-level names so callers can use
    # `from diverse_guide.evaluation.string_kernel import ...`
    __all__ = [
        "wd_shift_kernel",
        "compute_wd_kernel_matrix",
        "compute_wd_kernel_multiple",
        "STRING_KERNEL_BACKEND",
    ]


if __name__ == "__main__":
    print(f"Backend: {STRING_KERNEL_BACKEND}")
    seq1 = "ATGCGAT" * 5
    seq2 = "TGCGAAT" * 5
    print(f"K({seq1!r}, {seq2!r}, d=3, s=1) = {wd_shift_kernel(seq1, seq2, 3, 1):.6f}")
    mtx = compute_wd_kernel_matrix(["ATGCGAT", "TGCGAAT", "GCGATAG"], d=3, s=1)
    print("Kernel matrix:\n", mtx)
