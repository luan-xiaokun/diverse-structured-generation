"""JSONL persistence for replay certificates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from .reachability import ReachabilityEvidence
from .replay import (
    ByteStep,
    CandidateCertificate,
    CoveredObligation,
    FailureDiagnostic,
    TokenAction,
    WitnessResult,
)

SCHEMA_VERSION = 1


def certificate_to_dict(certificate: CandidateCertificate) -> dict[str, object]:
    payload = asdict(certificate)
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def certificate_from_dict(payload: dict[str, object]) -> CandidateCertificate:
    actions = [
        TokenAction(
            int(item["token_id"]),
            int(item["from_state"]),
            None if item.get("to_state") is None else int(item["to_state"]),
            tuple(
                ByteStep(
                    int(step["byte"]), int(step["from_state"]), int(step["to_state"])
                )
                for step in item.get("byte_steps", [])
            ),
        )
        for item in payload.get("token_actions", [])
    ]
    failure_payload = payload.get("failure")
    failure = FailureDiagnostic(**failure_payload) if failure_payload else None
    witness_payload = payload.get("witness_result")
    witness_result = None
    if witness_payload:
        reachability_payload = witness_payload["reachability"]
        witness_result = WitnessResult(
            obligation_id=str(witness_payload["obligation_id"]),
            obligation_kind=str(witness_payload["obligation_kind"]),
            obligation_meaning=witness_payload["obligation_meaning"],
            outcome=witness_payload["outcome"],
            reachability=ReachabilityEvidence(**reachability_payload),
            failure_reason=witness_payload.get("failure_reason"),
            max_expansions=witness_payload.get("max_expansions"),
            expansions=witness_payload.get("expansions"),
        )
    covered = [
        CoveredObligation(str(item["id"]), str(item["kind"]), item["meaning"])
        for item in payload.get("covered_obligations", [])
    ]
    return CandidateCertificate(
        token_ids=[int(value) for value in payload.get("token_ids", [])],
        initial_state=int(payload["initial_state"]),
        final_state=None
        if payload.get("final_state") is None
        else int(payload["final_state"]),
        token_actions=actions,
        accepted=bool(payload.get("accepted", False)),
        aggregate_eligible=bool(payload.get("aggregate_eligible", False)),
        covered_obligations=covered,
        failure=failure,
        source_text=payload.get("source_text"),
        decoded_text=payload.get("decoded_text"),
        metadata=dict(payload.get("metadata", {})),
        state_events=[int(value) for value in payload.get("state_events", [])],
        transition_events=[
            tuple(int(value) for value in item)
            for item in payload.get("transition_events", [])
        ],
        path_events=[
            tuple(int(value) for value in item)
            for item in payload.get("path_events", [])
        ],
        witness_result=witness_result,
    )


def write_certificates_jsonl(
    path: str | Path, certificates: Iterable[CandidateCertificate]
) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for certificate in certificates:
            handle.write(
                json.dumps(certificate_to_dict(certificate), sort_keys=True) + "\n"
            )


def read_certificates_jsonl(path: str | Path) -> Iterator[CandidateCertificate]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield certificate_from_dict(json.loads(line))


def accepted_certificates_jsonl(path: str | Path) -> Iterator[CandidateCertificate]:
    return (
        certificate
        for certificate in read_certificates_jsonl(path)
        if certificate.aggregate_eligible and certificate.accepted
    )
