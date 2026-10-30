"""Regression checks for binary string kernels and PSD-safe spectral entropy."""
import numpy as np
import pytest
from diverse_guide.evaluation import string_kernel as native
from diverse_guide.evaluation import string_kernel_py as reference
from diverse_guide.evaluation.metrics import vendi_score

@pytest.mark.skipif(native.STRING_KERNEL_BACKEND != 'c', reason='native kernel required')
def test_embedded_nul_is_a_byte_not_a_kmer_terminator():
    samples = ['a', 'a\0', '\0a', 'a\0b', 'a\0a', 'ab\0a', '５８\u3000']
    expected = reference.compute_wd_kernel_matrix(samples, d=5, s=1)
    actual = native.compute_wd_kernel_matrix(samples, d=5, s=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(native.compute_wd_kernel_multiple(samples[1], samples,5,1), expected[1], atol=1e-12)
    for i,a in enumerate(samples):
        for j,b in enumerate(samples):
            assert native.wd_shift_kernel(a,b,5,1) == pytest.approx(expected[i,j],abs=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= -1e-12

def test_substantial_negative_eigenvalue_is_reported():
    with pytest.raises(ValueError, match='positive semidefinite'):
        vendi_score(['a','b'], lambda _: np.array([[1.,2.],[2.,1.]]))

def test_roundoff_negative_eigenvalue_has_zero_entropy_contribution():
    score=vendi_score(['a','b'],lambda _:np.array([[1.,1.+1e-15],[1.+1e-15,1.]]))
    assert score == pytest.approx(1.,abs=1e-12)

def test_existing_trace_normalization_is_preserved():
    score=vendi_score(['a','b'],lambda _:np.diag([1.,3.]))
    assert score == pytest.approx(np.exp(-.25*np.log(.25)-.75*np.log(.75)))

def test_prefix_includes_exact_endpoint():
    assert vendi_score(['a','b','c','d'],lambda x:np.eye(len(x)),step_size=2)==pytest.approx([2.,4.])

def test_asymmetric_kernel_is_rejected():
    with pytest.raises(ValueError,match='symmetric'):
        vendi_score(['a','b'],lambda _:np.array([[1.,0.],[.5,1.]]))
