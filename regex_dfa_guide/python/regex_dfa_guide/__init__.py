"""This package provides an interface to construct diverse guide DFAs from regexes."""

from .regex_dfa_guide_rs import DiverseGuideDFA as _RsDiverseGuideDFA


class DiverseGuideDFA(_RsDiverseGuideDFA):
    """Python-facing DFA wrapper with the current public adjustment API."""

    def compute_adjustments(
        self,
        state: int,
        no_reward: bool,
        no_penalty: bool,
        completion_bias_eta: float,
    ) -> tuple[list[int], list[float]]:
        return super().compute_adjustments(
            state,
            1.0,
            no_reward,
            no_penalty,
            completion_bias_eta,
        )

    def compute_adjustments_with_estimator(
        self,
        state: int,
        no_reward: bool,
        no_penalty: bool,
        completion_bias_eta: float,
        path_frequency_estimator: str,
    ) -> tuple[list[int], list[float]]:
        return super().compute_adjustments_with_estimator(
            state,
            1.0,
            no_reward,
            no_penalty,
            completion_bias_eta,
            path_frequency_estimator,
        )


__all__ = ["DiverseGuideDFA"]
__version__ = "0.2.0"
