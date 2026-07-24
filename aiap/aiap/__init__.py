"""AIAP baseline witness generation package."""

from .baseline import (
    WitnessSearchResult,
    construct_one_witness,
    get_coverage_obligations,
)
from .obligations import CoverageObligation, ObligationCatalog
from .reachability import (
    ClassifiedObligation,
    ReachabilityEvidence,
    TokenGraphEdge,
    classify_reachability,
)
from .replay import (
    ByteStep,
    CandidateCertificate,
    CoveredObligation,
    FailureDiagnostic,
    ReplayPayload,
    TokenAction,
    WitnessOutcome,
    WitnessResult,
    accepted_certificates,
    aggregate_certificates,
    coverage_ratios,
    replay_candidate,
    witness_result_from_attempt,
)

__all__ = [
    "CoverageObligation",
    "ObligationCatalog",
    "construct_one_witness",
    "WitnessSearchResult",
    "get_coverage_obligations",
    "ByteStep",
    "TokenAction",
    "FailureDiagnostic",
    "CoveredObligation",
    "CandidateCertificate",
    "WitnessOutcome",
    "WitnessResult",
    "replay_candidate",
    "witness_result_from_attempt",
    "aggregate_certificates",
    "coverage_ratios",
    "accepted_certificates",
    "ReplayPayload",
    "TokenGraphEdge",
    "ReachabilityEvidence",
    "ClassifiedObligation",
    "classify_reachability",
]
