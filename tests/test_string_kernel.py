"""
Tests for the WD-shift kernel NumPy backend and dispatch layer.

The NumPy tests always run.
The C-backend test is skipped when wd_kernel.so / wd_kernel.dll is not built.
Numerical correctness is verified against manually computed reference values.
"""

import numpy as np
import pytest

import diverse_guide.evaluation.string_kernel as sk_mod
import diverse_guide.evaluation.string_kernel_py as py_mod

# ---------------------------------------------------------------------------
# Manual reference values
#
# For d=2, s=1:
#   w_k = 2*(d-k+1) / (d*(d+1))  →  w_1 = 2/3, w_2 = 1/3
#
# K("ab", "ab"):
#   k=1: 'a'@0↔'a'@0 (dist=0): (2/3)/2/1 = 1/3
#        'b'@1↔'b'@1 (dist=0): (2/3)/2/1 = 1/3
#        → 2/3
#   k=2: 'ab'@0↔'ab'@0 (dist=0): (1/3)/2/1 = 1/6
#        → 1/6
#   Total: 2/3 + 1/6 = 5/6
#
# K("ab", "ba"):
#   k=1: 'a'@0↔'a'@1 (dist=1≤1): (2/3)/2/2 = 1/6
#        'b'@1↔'b'@0 (dist=1≤1): (2/3)/2/2 = 1/6
#        → 1/3
#   k=2: 'ab' ≠ 'ba'  → 0
#   Total: 1/3
#
# K("a", "b", d=2, s=1):  no matching k-mers → 0
# K("a", "a", d=2, s=1):
#   k=1: 'a'@0↔'a'@0: (2/3)/2/1 = 1/3
#   k=2: k > len → 0
#   Total: 1/3
#
# K("abc", "xbc", d=1, s=1): w_1 = 1; 'b'↔'b' and 'c'↔'c' (dist=0) give
#   1/2 each → 1.0
# ---------------------------------------------------------------------------

REFERENCE_VALUES = [
    (("ab", "ab", 2, 1), 5.0 / 6.0),
    (("ab", "ba", 2, 1), 1.0 / 3.0),
    (("a", "b", 2, 1), 0.0),
    (("a", "a", 2, 1), 1.0 / 3.0),
    (("abc", "xbc", 1, 1), 1.0),
    (("aaa", "bbb", 3, 2), 0.0),
]


def test_numpy_kernel_matches_manual_reference_values():
    for (s1, s2, d, s), expected in REFERENCE_VALUES:
        assert py_mod.wd_shift_kernel(s1, s2, d=d, s=s) == pytest.approx(
            expected, abs=1e-12
        ), (s1, s2, d, s)
    with pytest.raises(ValueError):
        py_mod.wd_shift_kernel("a", "b", d=0, s=1)


def test_numpy_kernel_algebraic_properties():
    k = py_mod.wd_shift_kernel
    for s1, s2 in [("ab", "ba"), ("hello", "world"), ("a", "ab")]:
        assert k(s1, s2, d=3, s=1) == pytest.approx(k(s2, s1, d=3, s=1), abs=1e-12)
    assert k("abcde", "abcde", d=3, s=1) >= k("abcde", "abcdf", d=3, s=1)
    for s in ["a", "ab", "hello"]:
        assert k(s, s, d=3, s=1) > 0.0
    assert k("", "abc", d=3, s=1) == pytest.approx(0.0)
    assert k("abc", "", d=3, s=1) == pytest.approx(0.0)


def test_numpy_kernel_matrix_and_multiple_match_single_calls():
    seqs = ["a", "ab", "ba", "b"]
    mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
    assert mtx.shape == (4, 4)
    np.testing.assert_allclose(mtx, mtx.T, atol=1e-12)
    assert np.all(np.diag(mtx) > 0)
    for i, s1 in enumerate(seqs):
        for j, s2 in enumerate(seqs):
            assert mtx[i, j] == pytest.approx(py_mod.wd_shift_kernel(s1, s2, d=2, s=1))

    assert py_mod.compute_wd_kernel_matrix([], d=2, s=1).shape == (0, 0)
    single = py_mod.compute_wd_kernel_matrix(["abc"], d=2, s=1)
    assert single.shape == (1, 1)
    assert single[0, 0] == pytest.approx(py_mod.wd_shift_kernel("abc", "abc", d=2, s=1))
    with pytest.raises(TypeError):
        py_mod.compute_wd_kernel_matrix("not_a_list", d=2, s=1)

    result = py_mod.compute_wd_kernel_multiple("ab", seqs, d=2, s=1)
    assert result.shape == (4,)
    for i, s in enumerate(seqs):
        assert result[i] == pytest.approx(py_mod.wd_shift_kernel("ab", s, d=2, s=1))


def test_dispatch_layer_exposes_consistent_interface():
    """string_kernel.py exposes the same interface regardless of backend."""
    assert sk_mod.STRING_KERNEL_BACKEND in ("c", "python")
    v1 = sk_mod.wd_shift_kernel("abc", "bca", d=3, s=1)
    v2 = sk_mod.wd_shift_kernel("bca", "abc", d=3, s=1)
    assert v1 == pytest.approx(v2, abs=1e-10)

    seqs = ["ab", "ba", "a", "b"]
    mtx = sk_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
    np.testing.assert_allclose(mtx, mtx.T, atol=1e-10)
    assert np.all(np.diag(mtx) > 0)

    result = sk_mod.compute_wd_kernel_multiple("ab", seqs, d=2, s=1)
    for i, s in enumerate(seqs):
        expected = sk_mod.wd_shift_kernel("ab", s, d=2, s=1)
        assert result[i] == pytest.approx(expected, abs=1e-10)


@pytest.mark.skipif(sk_mod.STRING_KERNEL_BACKEND != "c", reason="C extension not built")
def test_c_backend_agrees_with_python():
    seqs = ["ab", "ba", "abc", "hello", "world"]
    for s1 in seqs:
        for s2 in seqs:
            c_val = sk_mod.wd_shift_kernel(s1, s2, d=3, s=1)
            py_val = py_mod.wd_shift_kernel(s1, s2, d=3, s=1)
            assert c_val == pytest.approx(py_val, abs=1e-9), (s1, s2)
    mtx_c = sk_mod.compute_wd_kernel_matrix(seqs, d=3, s=1)
    mtx_py = py_mod.compute_wd_kernel_matrix(seqs, d=3, s=1)
    np.testing.assert_allclose(mtx_c, mtx_py, atol=1e-9)
