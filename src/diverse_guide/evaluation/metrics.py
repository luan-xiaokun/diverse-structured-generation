"""
Diversity evaluation metrics for regex-constrained generation.

All functions in this module are pure (no model, no I/O, no side effects) and
operate on a :class:`~regex_dfa_guide.DiverseGuideDFA` instance plus a list of
generated strings.  They are imported by ``scripts/metrics_eval.py`` for the CLI
and by the test suite directly.
"""

from collections.abc import Callable

import numpy as np

from regex_dfa_guide import DiverseGuideDFA

# ---------------------------------------------------------------------------
# DFA-based coverage metrics
# ---------------------------------------------------------------------------


def get_coverage_ratio(count_dict: dict, threshold: int) -> float:
    """Fraction of keys in *count_dict* whose value meets *threshold*.

    Parameters
    ----------
    count_dict:
        Mapping from any key to an integer visit count.
    threshold:
        Minimum count required to consider a key "covered".

    Returns
    -------
    float
        A value in [0, 1]; returns 0 for an empty dict.
    """
    if not count_dict:
        return 0.0
    covered = sum(1 for v in count_dict.values() if v >= threshold)
    return covered / len(count_dict)


def state_coverage(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    """Fraction of DFA states visited at least *cov_threshold* times.

    Parameters
    ----------
    dfa:
        Built DFA for the target regex.
    inputs:
        Generated strings (must all be valid regex matches).
    cov_threshold:
        Minimum number of visits for a state to count as "covered".
    step_size:
        If given, returns a list of coverage values computed after every
        *step_size* inputs instead of a single final value.
    """
    states = dfa.get_states()
    state_count = {state: 0 for state in states}
    coverage_trace = []

    for i, s in enumerate(inputs):
        for state in dfa.get_state_sequence(s):
            state_count[state] += 1
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(state_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(state_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(state_count, cov_threshold))
    return coverage_trace


def transition_coverage(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    """Fraction of DFA byte-transitions visited at least *cov_threshold* times."""
    transitions = set()
    for u, trans in dfa.get_transitions().items():
        for c, v in trans.items():
            transitions.add((u, c, v))
    transition_count = {transition: 0 for transition in transitions}
    coverage_trace = []

    for i, s in enumerate(inputs):
        trans_seq = dfa.get_transition_sequence(s)
        _, prev_state = trans_seq[0]
        for char, state in trans_seq[1:]:
            transition_count[(prev_state, char, state)] += 1
            prev_state = state
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(transition_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(transition_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(transition_count, cov_threshold))
    return coverage_trace


def path_coverage(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    """Fraction of DFA state-pair paths visited at least *cov_threshold* times."""
    paths = set()
    for u, trans in dfa.get_transitions().items():
        for _, v in trans.items():
            paths.add((u, v))
    path_count = {path: 0 for path in paths}
    coverage_trace = []

    for i, s in enumerate(inputs):
        trans_seq = dfa.get_transition_sequence(s)
        _, prev_state = trans_seq[0]
        for _, state in trans_seq[1:]:
            path_count[(prev_state, state)] += 1
            prev_state = state
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(path_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(path_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(path_count, cov_threshold))
    return coverage_trace


# ---------------------------------------------------------------------------
# String-diversity metrics
# ---------------------------------------------------------------------------


def vendi_score(
    inputs: list[str],
    kernel_mtx_func: Callable[[list[str]], np.ndarray],
    step_size: int | None = None,
) -> float | list[float]:
    """Vendi diversity score computed from a kernel matrix.

    Parameters
    ----------
    inputs:
        List of generated strings.
    kernel_mtx_func:
        Callable ``f(sequences) -> ndarray`` that returns a pairwise kernel matrix.
        The matrix must be symmetric and positive semi-definite.
    step_size:
        If given, returns a list of scores computed on growing prefixes of *inputs*
        (in steps of *step_size*) instead of a single final score.
    """
    if step_size is not None and step_size <= 0:
        raise ValueError("step_size must be positive")
    if not inputs:
        return 0.0 if step_size is None else []
    gram_mtx = np.asarray(kernel_mtx_func(inputs), dtype=np.float64)
    if gram_mtx.shape != (len(inputs), len(inputs)):
        raise ValueError("kernel matrix shape does not match inputs")
    if step_size is None or step_size == len(inputs):
        return vendi_from_kernel(gram_mtx)
    endpoints = list(range(step_size, len(inputs), step_size)) + [len(inputs)]
    return [vendi_from_kernel(gram_mtx[:i, :i]) for i in endpoints]


def vendi_from_kernel(gram_mtx: np.ndarray) -> float:
    """Effective rank of a PSD kernel, retaining historical trace normalization.

    This is spectral entropy of K / trace(K). It equals standard Vendi for
    unit-diagonal kernels; raw WD kernels also weight samples by self-similarity.
    Do not add epsilon to the spectrum: 0 log 0 is exactly zero. Materially
    indefinite kernels are errors, never a zero-diversity measurement.
    """
    matrix = np.asarray(gram_mtx, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel matrix must be square")
    n = len(matrix)
    if not n:
        return 0.0
    if not np.isfinite(matrix).all():
        raise ValueError("kernel matrix must be finite")
    scale = float(np.max(np.abs(matrix)))
    tolerance = 128 * np.finfo(np.float64).eps * n * scale
    if np.max(np.abs(matrix - matrix.T)) > tolerance:
        raise ValueError("kernel matrix must be symmetric")
    spectrum = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
    spectral_scale = float(np.max(np.abs(spectrum)))
    tolerance = 128 * np.finfo(np.float64).eps * n * spectral_scale
    if spectrum[0] < -tolerance:
        raise ValueError(
            f"kernel matrix must be positive semidefinite: minimum eigenvalue "
            f"{spectrum[0]:.6g}, roundoff tolerance {tolerance:.6g}"
        )
    spectrum = np.maximum(spectrum, 0.0)
    mass = float(spectrum.sum())
    if mass <= 0:
        raise ValueError("kernel matrix must have positive trace")
    probabilities = spectrum[spectrum > 0] / mass
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def distinct_ngram(inputs: list[str], n: int) -> tuple[int, int]:
    """Count distinct character n-grams across all *inputs*.

    Returns
    -------
    tuple[int, int]
        ``(num_distinct_ngrams, num_inputs)``.
    """
    ngram_count: dict[str, int] = {}
    for s in inputs:
        for i in range(len(s) - n + 1):
            ngram = s[i : i + n]
            ngram_count[ngram] = ngram_count.get(ngram, 0) + 1
    return len(ngram_count), len(inputs)
