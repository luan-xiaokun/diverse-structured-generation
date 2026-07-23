"""AIAP baseline witness generation package."""

from .baseline import construct_one_witness, get_coverage_obligations
from .obligations import CoverageObligation, ObligationCatalog

__all__ = [
    "CoverageObligation",
    "ObligationCatalog",
    "construct_one_witness",
    "get_coverage_obligations",
]
