"""AIAP baseline witness generation package."""

from .baseline import construct_one_witness, get_coverage_obligations
from .obligations import CoverageObligation, ObligationCatalog
from .replay import (
    ByteStep,
    CandidateCertificate,
    CoveredObligation,
    FailureDiagnostic,
    ReplayPayload,
    TokenAction,
    accepted_certificates,
    aggregate_certificates,
    coverage_ratios,
    replay_candidate,
)

__all__ = [
    "CoverageObligation",
    "ObligationCatalog",
    "construct_one_witness",
    "get_coverage_obligations",
    "ByteStep",
    "TokenAction",
    "FailureDiagnostic",
    "CoveredObligation",
    "CandidateCertificate",
    "replay_candidate",
    "aggregate_certificates",
    "coverage_ratios",
    "accepted_certificates",
    "ReplayPayload",
]
