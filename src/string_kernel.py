"""
String kernel module for computing WD-shift kernel using C implementation.
"""

import ctypes
import errno as py_errno
import os

import numpy as np

lib_name = "wd_kernel.so"
lib_path = f"wd_kernel/{lib_name}"

try:
    wd_lib = ctypes.CDLL(lib_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print(f"Please ensure '{lib_name}' is compiled and accessible at '{lib_path}'")
    exit(1)

c_char_p_array = ctypes.POINTER(ctypes.c_char_p)
c_int_array = ctypes.POINTER(ctypes.c_int)
c_double_array = ctypes.POINTER(ctypes.c_double)

wd_lib.wd_shift_kernel.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
wd_lib.wd_shift_kernel.restype = ctypes.c_double

wd_lib.wd_shift_kernel_matrix_omp.argtypes = [
    c_char_p_array,
    ctypes.c_int,
    c_int_array,
    ctypes.c_int,
    ctypes.c_int,
    c_double_array,
]
wd_lib.wd_shift_kernel_matrix_omp.restype = ctypes.c_int

wd_lib.wd_shift_kernel_multiple_omp.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    c_char_p_array,
    ctypes.c_int,
    c_int_array,
    ctypes.c_int,
    ctypes.c_int,
]
wd_lib.wd_shift_kernel_multiple_omp.restype = ctypes.c_int


def wd_shift_kernel(s1: str, s2: str, d: int, s: int) -> float:
    """Calculates WD-shift kernel using the C implementation."""
    s1_bytes = s1.encode("utf-8")
    s2_bytes = s2.encode("utf-8")
    l1 = len(s1_bytes)
    l2 = len(s2_bytes)

    result = wd_lib.wd_shift_kernel(s1_bytes, l1, s2_bytes, l2, d, s)
    return result


def compute_wd_kernel_matrix(sequences: list[str], d: int, s: int) -> np.ndarray:
    """
    Computes the pairwise WD-shift kernel matrix for a list of sequences
    using a parallel C implementation with OpenMP.

    Args:
        sequences (list[str]): A list of input sequences.
        d (int): Maximum k-mer length (degree).
        s (int): Maximum allowed shift.

    Returns:
        np.ndarray: An n x n numpy array containing the kernel matrix.

    Raises:
        ValueError: If input parameters are invalid.
        MemoryError: If the C code failed to allocate memory.
        OSError: For other C library errors or OpenMP runtime issues.
        TypeError: If input is not a list of strings.
    """
    if not isinstance(sequences, list) or not all(
        isinstance(seq, str) for seq in sequences
    ):
        raise TypeError("Input must be a list of strings")

    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0))  # Return empty matrix for empty list

    # Prepare arguments for C function
    # 1. Encode strings and create char* array
    encoded_seqs = [seq.encode("utf-8") for seq in sequences]
    c_strings_arr = (ctypes.c_char_p * n)(*encoded_seqs)

    # 2. Create lengths array
    lengths = [len(seq) for seq in encoded_seqs]
    c_lengths_arr = (ctypes.c_int * n)(*lengths)

    # 3. Create output buffer (using NumPy for convenience)
    # Ensure C double matches NumPy float64
    output_matrix_np = np.zeros((n, n), dtype=np.float64)
    c_output_matrix_ptr = output_matrix_np.ctypes.data_as(c_double_array)

    ctypes.set_errno(0)

    status = wd_lib.wd_shift_kernel_matrix_omp(
        c_strings_arr, n, c_lengths_arr, d, s, c_output_matrix_ptr
    )

    if status != 0:
        c_errno = ctypes.get_errno()
        error_msg = (
            "C function wd_shift_kernel_matrix_omp failed with status"
            f"{status} and errno {c_errno}: {os.strerror(c_errno)}"
        )

        if c_errno == py_errno.EINVAL:
            raise ValueError(f"Invalid argument provided to C function. {error_msg}")
        elif c_errno == py_errno.ENOMEM:
            raise MemoryError(f"C function failed to allocate memory. {error_msg}")
        elif c_errno == py_errno.EDOM:
            raise ValueError(f"Math domain error in C function. {error_msg}")
        else:
            raise OSError(error_msg)

    return output_matrix_np


def compute_wd_kernel_multiple(
    x: str, sequences: list[str], d: int, s: int
) -> np.ndarray:
    x_bytes = x.encode("utf-8")
    length = len(x_bytes)
    encoded_seqs = [seq.encode("utf-8") for seq in sequences]
    n = len(encoded_seqs)
    c_strings_arr = (ctypes.c_char_p * n)(*encoded_seqs)

    lengths = [len(seq) for seq in encoded_seqs]
    c_lengths_arr = (ctypes.c_int * n)(*lengths)

    output_vector = np.zeros(n, dtype=np.float64)
    c_output_vector_ptr = output_vector.ctypes.data_as(c_double_array)

    ctypes.set_errno(0)

    status = wd_lib.wd_shift_kernel_multiple_omp(
        x_bytes, length, c_strings_arr, n, c_lengths_arr, d, s, c_output_vector_ptr
    )
    if status != 0:
        c_errno = ctypes.get_errno()
        error_msg = (
            "C function wd_shift_kernel_matrix_omp failed with status"
            f"{status} and errno {c_errno}: {os.strerror(c_errno)}"
        )

        if c_errno == py_errno.EINVAL:
            raise ValueError(f"Invalid argument provided to C function. {error_msg}")
        elif c_errno == py_errno.ENOMEM:
            raise MemoryError(f"C function failed to allocate memory. {error_msg}")
        elif c_errno == py_errno.EDOM:
            raise ValueError(f"Math domain error in C function. {error_msg}")
        else:
            raise OSError(error_msg)

    return output_vector


if __name__ == "__main__":
    seq1 = "ATGCGAT" * 100
    seq2 = "TGCGAAT" * 100
    max_degree = 3
    max_shift = 1

    for _ in range(1000):
        kernel_val_1 = wd_shift_kernel(seq1, seq2, d=3, s=1)
    print(f"Python calling C (d=3, s=1, beta=0.8): {kernel_val_1}")

    for _ in range(1000):
        kernel_val_2 = wd_shift_kernel(seq1, seq2, d=20, s=1)
    print(f"Python calling C (d=2, s=1, beta=1.0): {kernel_val_2}")

    mtx = compute_wd_kernel_matrix(
        ["ATGCGAT", "TGCGAAT", "GCGATAG", "CGATAGC"], d=3, s=1
    )
    print("WD Kernel Matrix:")
    print(mtx)
