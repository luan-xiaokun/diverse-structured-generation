"""Command-first one-case baseline runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from diverse_guide.guide_rust import DiverseRegexGuide

from .baseline import WitnessSearchResult, construct_one_witness
from .case_io import CaseSpec, load_case
from .obligations import ObligationCatalog
from .reachability import ClassifiedObligation, classify_reachability
from .replay import (
    CandidateCertificate,
    FailureDiagnostic,
    aggregate_certificates,
    coverage_ratios,
    replay_candidate,
    witness_result_from_attempt,
)
from .reporting import write_certificates_jsonl


def _json_safe_aggregates(
    aggregates: dict[str, dict[Any, int]],
) -> dict[str, list[dict[str, Any]]]:
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


def _attempt_failure_certificate(
    dfa: Any,
    classified: ClassifiedObligation,
    search: WitnessSearchResult | None,
) -> CandidateCertificate:
    """Record a searched obligation whose search produced no replayable tokens."""
    result = witness_result_from_attempt(classified, search, None)
    return CandidateCertificate(
        token_ids=[],
        initial_state=int(dfa.get_initial_state()),
        final_state=None,
        token_actions=[],
        accepted=False,
        aggregate_eligible=False,
        failure=FailureDiagnostic(
            result.outcome,
            result.failure_reason
            or f"witness attempt failed for {result.obligation_id}",
        ),
        metadata={
            "proposal": result.obligation_id,
            "obligation": {
                "id": result.obligation_id,
                "kind": result.obligation_kind,
                "meaning": result.obligation_meaning,
            },
        },
        witness_result=result,
    )


def _result_json(classified: ClassifiedObligation, outcome: Any) -> dict[str, Any]:
    """Serialize an obligation result while keeping reachability evidence explicit."""
    payload = asdict(outcome)
    payload.update(
        {
            "id": classified.obligation.obligation_id,
            "kind": classified.obligation.kind,
            "meaning": classified.obligation.raw_meaning,
        }
    )
    return payload


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
    max_expansions: int | None = None,
) -> tuple[dict[str, Any], list[CandidateCertificate]]:
    if sample_num is not None and sample_num <= 0:
        raise ValueError("sample_num must be positive")
    if obligation_num is not None and obligation_num <= 0:
        raise ValueError("obligation_num must be positive")
    if max_expansions is not None and max_expansions <= 0:
        raise ValueError("max_expansions must be positive")
    dfa = DiverseRegexGuide(case.regex, tokenizer).dfa
    catalog = ObligationCatalog.from_dfa(dfa)
    baseline_certificates = []
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
        baseline_certificates.append(cert)
    covered_ids = {
        obligation_id
        for certificate in baseline_certificates
        if certificate.aggregate_eligible and certificate.accepted
        for obligation_id in certificate.covered_obligation_ids
    }
    # In AIAP terminology an obligation is only a currently unvisited DFA
    # structure.  The catalog is merely the stable DFA structure universe used
    # to resolve replay meanings; proposals must come from its uncovered part.
    missing_obligations = [
        item for item in catalog if item.obligation_id not in covered_ids
    ]
    classified_obligations = classify_reachability(dfa, missing_obligations)
    unreachable = [item for item in classified_obligations if not item.reachable]
    reachable = [item for item in classified_obligations if item.reachable]
    selected_reachable = (
        reachable[:obligation_num] if obligation_num is not None else reachable
    )

    results_by_id = {
        item.obligation.obligation_id: _result_json(
            item, witness_result_from_attempt(item, None, None)
        )
        for item in unreachable
    }
    attempt_certificates: list[CandidateCertificate] = []
    for classified in selected_reachable:
        try:
            search = construct_one_witness(
                dfa, classified.obligation, max_expansions=max_expansions
            )
        except Exception as exc:
            search = WitnessSearchResult(None, 0, max_expansions, False, str(exc))
        if search is None or search.token_ids is None:
            certificate = _attempt_failure_certificate(dfa, classified, search)
        else:
            decoded = tokenizer.decode(
                search.token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            certificate = replay_candidate(
                dfa,
                search.token_ids,
                catalog,
                decoded_text=decoded,
                metadata={"proposal": classified.obligation.obligation_id},
            )
            certificate.witness_result = witness_result_from_attempt(
                classified, search, certificate
            )
        attempt_certificates.append(certificate)
        results_by_id[classified.obligation.obligation_id] = _result_json(
            classified, certificate.witness_result
        )

    obligation_results = [
        results_by_id[item.obligation.obligation_id]
        for item in classified_obligations
        if item.obligation.obligation_id in results_by_id
    ]

    all_certificates = baseline_certificates + attempt_certificates
    aggregates = aggregate_certificates(all_certificates)
    outcome_names = (
        "success",
        "byte_level_unreachable",
        "search_limit",
        "invalid_witness",
        "internal_error",
    )
    outcome_counts = {
        name: sum(result["outcome"] == name for result in obligation_results)
        for name in outcome_names
    }
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
            "total": len(all_certificates),
            "accepted": sum(
                c.aggregate_eligible and c.accepted for c in all_certificates
            ),
            "failed": sum(
                not (c.aggregate_eligible and c.accepted) for c in all_certificates
            ),
        },
        "coverage": coverage_ratios(catalog, aggregates),
        "aggregates": _json_safe_aggregates(aggregates),
        "parity_diagnostics": [
            c.failure.category for c in all_certificates if c.failure
        ],
        "max_expansions": max_expansions,
        "obligations": {
            "missing_total": len(missing_obligations),
            "reachable_total": len(reachable),
            "unreachable_total": len(unreachable),
            "selected_reachable": len(selected_reachable),
            "results": obligation_results,
            "outcome_counts": outcome_counts,
        },
        "attempt_certificates": attempt_certificates,
    }
    return summary, all_certificates


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
        "--max-expansions",
        type=int,
        metavar="N",
        help="positive BFS expansion cap per selected reachable obligation",
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
    if args.max_expansions is not None and args.max_expansions <= 0:
        parser.error("--max-expansions must be positive")
    try:
        case = load_case(args.case)
        summary, all_certificates = run_case(
            case,
            _load_tokenizer(case),
            args.sample_num,
            args.obligation_num,
            args.max_expansions,
        )
    except Exception as exc:
        parser.error(str(exc))
    attempt_certificates = summary.pop("attempt_certificates", all_certificates)
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
    print(
        "Outcomes: "
        + ", ".join(
            f"{name}={count}"
            for name, count in summary["obligations"]["outcome_counts"].items()
        )
    )
    if args.json_path:
        args.json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.jsonl_path:
        write_certificates_jsonl(args.jsonl_path, attempt_certificates)
    return 0
