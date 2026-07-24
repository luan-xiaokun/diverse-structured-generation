"""Command-first one-case baseline runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from diverse_guide.guide_rust import DiverseRegexGuide

from .baseline import construct_one_witness
from .case_io import CaseSpec, load_case
from .obligations import ObligationCatalog
from .replay import (
    CandidateCertificate,
    FailureDiagnostic,
    aggregate_certificates,
    coverage_ratios,
    replay_candidate,
)
from .reporting import write_certificates_jsonl


def _json_safe_aggregates(aggregates: dict[str, dict[Any, int]]) -> dict[str, list[dict[str, Any]]]:
    """Encode tuple-keyed replay counters as portable JSON records."""
    return {
        name: [
            {
                "key": list(key) if isinstance(key, tuple) else key,
                "count": count,
            }
            for key, count in counts.items()
        ]
        for name, counts in aggregates.items()
    }


def _failed_proposal_certificate(dfa: Any, obligation: Any) -> CandidateCertificate:
    """Retain an obligation whose BFS found no accepting witness."""
    return CandidateCertificate(
        token_ids=[],
        initial_state=int(dfa.get_initial_state()),
        final_state=None,
        token_actions=[],
        accepted=False,
        aggregate_eligible=False,
        failure=FailureDiagnostic(
            "no_accepting_witness",
            f"BFS found no accepting witness for {obligation.obligation_id}",
        ),
        metadata={
            "proposal": obligation.obligation_id,
            "obligation": {
                "id": obligation.obligation_id,
                "kind": obligation.kind,
                "meaning": obligation.raw_meaning,
            },
        },
    )


def adapt_source(
    tokenizer: Any, source_text: str
) -> tuple[list[int], str, FailureDiagnostic | None]:
    ids = [
        int(x) for x in tokenizer(source_text, add_special_tokens=False)["input_ids"]
    ]
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return (
        ids,
        decoded,
        None
        if decoded == source_text
        else FailureDiagnostic(
            "decoded_text_mismatch", f"decoded text differs from source ({decoded!r})"
        ),
    )


def _load_tokenizer(case: CaseSpec):
    return AutoTokenizer.from_pretrained(
        case.tokenizer_repository, revision=case.tokenizer_revision, use_fast=True
    )


def run_case(
    case: CaseSpec,
    tokenizer: Any,
    sample_num: int | None = None,
    obligation_num: int | None = None,
) -> tuple[dict[str, Any], list[CandidateCertificate]]:
    if sample_num is not None and sample_num <= 0:
        raise ValueError("sample_num must be positive")
    if obligation_num is not None and obligation_num <= 0:
        raise ValueError("obligation_num must be positive")
    dfa = DiverseRegexGuide(case.regex, tokenizer).dfa
    catalog = ObligationCatalog.from_dfa(dfa)
    certs = []
    samples = case.samples[:sample_num] if sample_num is not None else case.samples
    for source in samples:
        ids, decoded, mismatch = adapt_source(tokenizer, source)
        cert = replay_candidate(
            dfa, ids, catalog, source_text=source, decoded_text=decoded
        )
        if mismatch:
            cert.accepted = cert.aggregate_eligible = False
            cert.failure = mismatch
            cert.covered_obligations = []
        certs.append(cert)
    covered_ids = {
        obligation_id
        for certificate in certs
        if certificate.aggregate_eligible and certificate.accepted
        for obligation_id in certificate.covered_obligation_ids
    }
    # In AIAP terminology an obligation is only a currently unvisited DFA
    # structure.  The catalog is merely the stable DFA structure universe used
    # to resolve replay meanings; proposals must come from its uncovered part.
    missing_obligations = [
        item for item in catalog if item.obligation_id not in covered_ids
    ]
    obligations = (
        missing_obligations[:obligation_num]
        if obligation_num is not None
        else missing_obligations
    )
    for obligation in obligations:
        witness = construct_one_witness(dfa, obligation)
        if witness is None:
            certs.append(_failed_proposal_certificate(dfa, obligation))
            continue
        decoded = tokenizer.decode(
            witness, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        certs.append(
            replay_candidate(
                dfa,
                witness,
                catalog,
                decoded_text=decoded,
                metadata={"proposal": obligation.obligation_id},
            )
        )
    aggregates = aggregate_certificates(certs)
    summary = {
        "sample_num": sample_num,
        "obligation_num": obligation_num,
        "case": {
            "path": str(case.path),
            "grammar": case.grammar,
            "tokenizer": {
                "repository": case.tokenizer_repository,
                "revision": case.tokenizer_revision,
            },
        },
        "candidates": {
            "total": len(certs),
            "accepted": sum(c.aggregate_eligible and c.accepted for c in certs),
            "failed": sum(not (c.aggregate_eligible and c.accepted) for c in certs),
        },
        "coverage": coverage_ratios(catalog, aggregates),
        "aggregates": _json_safe_aggregates(aggregates),
        "parity_diagnostics": [c.failure.category for c in certs if c.failure],
    }
    return summary, certs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the AIAP one-case replay baseline."
    )
    parser.add_argument("case", type=Path, help="explicit case-study JSON path")
    parser.add_argument(
        "--sample-num",
        type=int,
        metavar="N",
        help="number of initial samples used to compute missing obligations",
    )
    parser.add_argument(
        "--obligation-num",
        "--ob-num",
        dest="obligation_num",
        type=int,
        metavar="N",
        help="maximum number of missing obligations used for witness generation",
    )
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--jsonl", dest="jsonl_path", type=Path)
    args = parser.parse_args(argv)
    if args.sample_num is not None and args.sample_num <= 0:
        parser.error("--sample-num must be positive")
    if args.obligation_num is not None and args.obligation_num <= 0:
        parser.error("--obligation-num must be positive")
    try:
        case = load_case(args.case)
        summary, certs = run_case(
            case, _load_tokenizer(case), args.sample_num, args.obligation_num
        )
    except Exception as exc:
        parser.error(str(exc))
    print(f"Case: {summary['case']['grammar']} ({summary['case']['path']})")
    print(
        f"Tokenizer: {summary['case']['tokenizer']['repository']}"
        f"@{summary['case']['tokenizer']['revision']}"
    )
    print(
        f"Candidates: {summary['candidates']['total']} accepted="
        f"{summary['candidates']['accepted']} failed={summary['candidates']['failed']}"
    )
    print(
        "Coverage: " + ", ".join(f"{k}={v:.3f}" for k, v in summary["coverage"].items())
    )
    if args.json_path:
        args.json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.jsonl_path:
        # JSONL is the generated-witness artifact. Initial samples are used
        # internally to compute missing obligations, but are not emitted.
        write_certificates_jsonl(
            args.jsonl_path,
            [certificate for certificate in certs if certificate.metadata.get("proposal")],
        )
    return 0
