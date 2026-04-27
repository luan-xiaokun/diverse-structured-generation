"""
Tests for the WD-shift kernel NumPy backend and dispatch layer.

The NumPy tests always run.
The C-backend tests are skipped when wd_kernel.so / wd_kernel.dll is not built.
Numerical correctness is verified against manually computed reference values.
"""

import numpy as np
import pytest

import diverse_guide.evaluation.string_kernel as sk_mod
import diverse_guide.evaluation.string_kernel_py as py_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_c_backend() -> bool:
    return sk_mod.STRING_KERNEL_BACKEND == "c"


c_only = pytest.mark.skipif(not _is_c_backend(), reason="C extension not built")


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
# ---------------------------------------------------------------------------

K_AB_AB = 5.0 / 6.0  # K("ab", "ab", d=2, s=1)
K_AB_BA = 1.0 / 3.0  # K("ab", "ba", d=2, s=1)
K_A_B = 0.0  # K("a",  "b",  d=2, s=1)
K_A_A = 1.0 / 3.0  # K("a",  "a",  d=2, s=1)


# ===========================================================================
# NumPy implementation tests
# ===========================================================================


class TestNumpyKernelCorrectness:
    """Verify the pure NumPy kernel against manually computed values."""

    def test_self_similarity_ab(self):
        val = py_mod.wd_shift_kernel("ab", "ab", d=2, s=1)
        assert val == pytest.approx(K_AB_AB, abs=1e-12)

    def test_shifted_match(self):
        val = py_mod.wd_shift_kernel("ab", "ba", d=2, s=1)
        assert val == pytest.approx(K_AB_BA, abs=1e-12)

    def test_no_match(self):
        val = py_mod.wd_shift_kernel("a", "b", d=2, s=1)
        assert val == pytest.approx(K_A_B, abs=1e-12)

    def test_single_char_self(self):
        val = py_mod.wd_shift_kernel("a", "a", d=2, s=1)
        assert val == pytest.approx(K_A_A, abs=1e-12)

    def test_shift_boundary(self):
        # "abc" vs "xbc" with s=1 — 'bc' substrings match at same offset (dist=0)
        # but 'a' and 'x' don't. With d=1, only k=1 matters.
        # w_1 = 1, k-mers: 'b'@1↔'b'@1 (dist=0): 1/2/1=0.5; 'c'@2↔'c'@2: 0.5
        # 'a'@0↔'x'@0: no match
        val = py_mod.wd_shift_kernel("abc", "xbc", d=1, s=1)
        assert val == pytest.approx(1.0, abs=1e-12)  # k=1, w=1: 'b' + 'c' = 0.5+0.5

    def test_zero_for_no_common_chars(self):
        val = py_mod.wd_shift_kernel("aaa", "bbb", d=3, s=2)
        assert val == pytest.approx(0.0, abs=1e-12)

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError):
            py_mod.wd_shift_kernel("a", "b", d=0, s=1)


class TestNumpyKernelProperties:
    """Algebraic properties of the WD-shift kernel."""

    def test_symmetry(self):
        for s1, s2 in [("ab", "ba"), ("hello", "world"), ("a", "ab")]:
            assert py_mod.wd_shift_kernel(s1, s2, d=3, s=1) == pytest.approx(
                py_mod.wd_shift_kernel(s2, s1, d=3, s=1), abs=1e-12
            )

    def test_self_similarity_ge_cross_similarity(self):
        s1, s2 = "abcde", "abcdf"
        kss = py_mod.wd_shift_kernel(s1, s1, d=3, s=1)
        kst = py_mod.wd_shift_kernel(s1, s2, d=3, s=1)
        assert kss >= kst

    def test_self_similarity_positive(self):
        for s in ["a", "ab", "hello"]:
            assert py_mod.wd_shift_kernel(s, s, d=3, s=1) > 0.0

    def test_empty_string_zero(self):
        assert py_mod.wd_shift_kernel("", "abc", d=3, s=1) == pytest.approx(0.0)
        assert py_mod.wd_shift_kernel("abc", "", d=3, s=1) == pytest.approx(0.0)


class TestNumpyKernelMatrix:
    """Tests for compute_wd_kernel_matrix."""

    def test_output_shape(self):
        seqs = ["a", "b", "ab"]
        mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        assert mtx.shape == (3, 3)

    def test_symmetric(self):
        seqs = ["a", "ab", "ba", "b"]
        mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        np.testing.assert_allclose(mtx, mtx.T, atol=1e-12)

    def test_diagonal_positive(self):
        seqs = ["a", "ab", "hello"]
        mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        assert np.all(np.diag(mtx) > 0)

    def test_diagonal_consistency_with_single_call(self):
        seqs = ["ab", "ba"]
        mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        assert mtx[0, 0] == pytest.approx(py_mod.wd_shift_kernel("ab", "ab", d=2, s=1))
        assert mtx[1, 1] == pytest.approx(py_mod.wd_shift_kernel("ba", "ba", d=2, s=1))

    def test_off_diagonal_consistency_with_single_call(self):
        seqs = ["ab", "ba"]
        mtx = py_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        expected = py_mod.wd_shift_kernel("ab", "ba", d=2, s=1)
        assert mtx[0, 1] == pytest.approx(expected)
        assert mtx[1, 0] == pytest.approx(expected)

    def test_empty_input(self):
        mtx = py_mod.compute_wd_kernel_matrix([], d=2, s=1)
        assert mtx.shape == (0, 0)

    def test_single_element(self):
        mtx = py_mod.compute_wd_kernel_matrix(["abc"], d=2, s=1)
        assert mtx.shape == (1, 1)
        assert mtx[0, 0] == pytest.approx(
            py_mod.wd_shift_kernel("abc", "abc", d=2, s=1)
        )

    def test_invalid_input_type_raises(self):
        with pytest.raises(TypeError):
            py_mod.compute_wd_kernel_matrix("not_a_list", d=2, s=1)


class TestNumpyKernelMultiple:
    """Tests for compute_wd_kernel_multiple."""

    def test_output_length(self):
        seqs = ["a", "b", "ab"]
        result = py_mod.compute_wd_kernel_multiple("ab", seqs, d=2, s=1)
        assert result.shape == (3,)

    def test_values_consistent_with_single(self):
        x, seqs = "ab", ["a", "b", "ab", "ba"]
        result = py_mod.compute_wd_kernel_multiple(x, seqs, d=2, s=1)
        for i, s in enumerate(seqs):
            assert result[i] == pytest.approx(py_mod.wd_shift_kernel(x, s, d=2, s=1))

    def test_self_in_list(self):
        x, seqs = "abc", ["abc", "xyz"]
        result = py_mod.compute_wd_kernel_multiple(x, seqs, d=3, s=1)
        expected_self = py_mod.wd_shift_kernel(x, x, d=3, s=1)
        assert result[0] == pytest.approx(expected_self)


# ===========================================================================
# string_kernel.py dispatch layer tests
# ===========================================================================


class TestDispatchLayer:
    """string_kernel.py should expose the same interface regardless of backend."""

    def test_backend_attribute_is_set(self):
        assert sk_mod.STRING_KERNEL_BACKEND in ("c", "python")

    def test_single_kernel_symmetry(self):
        v1 = sk_mod.wd_shift_kernel("abc", "bca", d=3, s=1)
        v2 = sk_mod.wd_shift_kernel("bca", "abc", d=3, s=1)
        assert v1 == pytest.approx(v2, abs=1e-10)

    def test_matrix_symmetric(self):
        seqs = ["ab", "ba", "a", "b"]
        mtx = sk_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        np.testing.assert_allclose(mtx, mtx.T, atol=1e-10)

    def test_matrix_diagonal_positive(self):
        seqs = ["ab", "ba", "a"]
        mtx = sk_mod.compute_wd_kernel_matrix(seqs, d=2, s=1)
        assert np.all(np.diag(mtx) > 0)

    def test_multiple_consistent_with_single(self):
        x, seqs = "ab", ["a", "ba", "ab"]
        result = sk_mod.compute_wd_kernel_multiple(x, seqs, d=2, s=1)
        for i, s in enumerate(seqs):
            expected = sk_mod.wd_shift_kernel(x, s, d=2, s=1)
            assert result[i] == pytest.approx(expected, abs=1e-10)


@c_only
class TestCBackendAgreesWithPython:
    """When C extension is available, C and Python outputs must agree numerically."""

    _seqs = ["ab", "ba", "abc", "hello", "world"]

    def test_single_kernel_agreement(self):
        for s1 in self._seqs:
            for s2 in self._seqs:
                c_val = sk_mod.wd_shift_kernel(s1, s2, d=3, s=1)
                py_val = py_mod.wd_shift_kernel(s1, s2, d=3, s=1)
                assert c_val == pytest.approx(py_val, abs=1e-9), (
                    f"Mismatch: K({s1!r}, {s2!r}) C={c_val} Python={py_val}"
                )

    def test_matrix_agreement(self):
        mtx_c = sk_mod.compute_wd_kernel_matrix(self._seqs, d=3, s=1)
        mtx_py = py_mod.compute_wd_kernel_matrix(self._seqs, d=3, s=1)
        np.testing.assert_allclose(mtx_c, mtx_py, atol=1e-9)
