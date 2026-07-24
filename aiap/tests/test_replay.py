from importlib.metadata import version
from pathlib import Path

import regex_dfa_guide.regex_dfa_guide_rs as native_extension
from aiap.baseline import WitnessSearchResult
from aiap.obligations import CoverageObligation, ObligationCatalog
from aiap.reachability import ClassifiedObligation, ReachabilityEvidence
from aiap.replay import (
    aggregate_certificates,
    replay_candidate,
    witness_result_from_attempt,
)
from aiap.reporting import read_certificates_jsonl, write_certificates_jsonl
from regex_dfa_guide import DiverseGuideDFA


def make_dfa():
    return DiverseGuideDFA("(?:ab)?$", 99, {1: b"a", 2: b"b", 3: b"ab"})


def test_active_native_extension_runs_bytes_only_trace():
    extension_path = Path(native_extension.__file__).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    source_package = (
        repository_root / "regex_dfa_guide" / "python" / "regex_dfa_guide"
    ).resolve()

    assert version("regex-dfa-guide") == "0.3.0"
    assert extension_path.parent == source_package, (
        extension_path,
        source_package,
    )

    dfa = DiverseGuideDFA("a", 0, {1: b"a"})
    trace = dfa.get_byte_transition_sequence_from_state(dfa.get_initial_state(), 1)
    assert [int(step[0]) for step in trace[1:]] == [97]


def test_empty_input_credits_initial_state_once():
    dfa = make_dfa()
    cert = replay_candidate(dfa, [], ObligationCatalog.from_dfa(dfa))
    assert cert.accepted
    assert cert.state_events == [dfa.get_initial_state()]
    assert cert.transition_events == []


def test_multibyte_token_replays_every_interior_byte():
    dfa = make_dfa()
    cert = replay_candidate(dfa, [3], ObligationCatalog.from_dfa(dfa))
    assert cert.accepted
    assert len(cert.token_actions[0].byte_steps) == 2
    assert len(cert.transition_events) == 2
    assert len(cert.path_events) == 2
    assert [item.kind for item in cert.covered_obligations].count("transition") == 2


def test_eos_is_zero_byte_action():
    dfa = make_dfa()
    cert = replay_candidate(dfa, [99], ObligationCatalog.from_dfa(dfa))
    assert cert.accepted
    assert cert.token_actions[0].byte_steps == ()
    assert cert.transition_events == []


def test_rejected_candidate_isolated_from_aggregates():
    dfa = make_dfa()
    catalog = ObligationCatalog.from_dfa(dfa)
    accepted = replay_candidate(dfa, [3], catalog)
    rejected = replay_candidate(dfa, [1], catalog)
    assert not rejected.accepted
    assert rejected.failure is not None
    assert rejected.covered_obligations == []
    aggregate = aggregate_certificates([accepted, rejected])
    assert sum(aggregate["transition_counts"].values()) == 2


def test_jsonl_round_trip_preserves_semantic_payload(tmp_path):
    dfa = make_dfa()
    catalog = ObligationCatalog.from_dfa(dfa)
    certificates = [
        replay_candidate(dfa, [3], catalog, source_text="ab", decoded_text="ab"),
        replay_candidate(dfa, [1], catalog, source_text="a", decoded_text="a"),
    ]
    path = tmp_path / "certificates.jsonl"
    write_certificates_jsonl(path, certificates)
    recovered = list(read_certificates_jsonl(path))
    assert len(recovered) == 2
    assert recovered[0].covered_obligations == certificates[0].covered_obligations
    assert recovered[0].token_actions == certificates[0].token_actions
    assert recovered[1].failure is not None
    assert recovered[1].aggregate_eligible is False


def test_witness_result_requires_exact_replay_credit_for_success():
    dfa = make_dfa()
    catalog = ObligationCatalog.from_dfa(dfa)
    target = next(item for item in catalog if item.kind == "transition")
    classified = ClassifiedObligation(
        target,
        True,
        None,
        ReachabilityEvidence("annotated_token_graph_1", 3, 3, 1),
    )
    search = WitnessSearchResult([3], 2, None, False, None)

    accepted_without_target = replay_candidate(dfa, [], catalog)
    invalid = witness_result_from_attempt(classified, search, accepted_without_target)
    assert invalid.outcome == "invalid_witness"
    assert invalid.obligation_id == target.obligation_id

    accepted_with_target = replay_candidate(dfa, [3], catalog)
    success_target = CoverageObligation(
        "transition", accepted_with_target.transition_events[0]
    )
    success_classified = ClassifiedObligation(
        success_target,
        True,
        None,
        ReachabilityEvidence("annotated_token_graph_1", 3, 3, 1),
    )
    success = witness_result_from_attempt(
        success_classified, search, accepted_with_target
    )
    assert success.outcome == "success"
    assert success.max_expansions is None


def test_witness_result_reports_search_limit_and_round_trips_jsonl(tmp_path):
    target = CoverageObligation("state", 5)
    classified = ClassifiedObligation(
        target,
        True,
        None,
        ReachabilityEvidence("annotated_token_graph_1", 1, 1, 1),
    )
    search = WitnessSearchResult(None, 3, 3, True, "expansion_limit")
    dfa = make_dfa()
    certificate = replay_candidate(dfa, [], ObligationCatalog.from_dfa(dfa))
    certificate.witness_result = witness_result_from_attempt(
        classified, search, certificate
    )
    path = tmp_path / "attempt.jsonl"

    write_certificates_jsonl(path, [certificate])
    recovered = list(read_certificates_jsonl(path))

    assert recovered[0].witness_result == certificate.witness_result
    assert recovered[0].witness_result is not None
    assert recovered[0].witness_result.outcome == "search_limit"
    assert recovered[0].witness_result.failure_reason == "expansion_limit"


def test_witness_result_keeps_unreachable_and_internal_errors_distinct():
    target = CoverageObligation("state", 5)
    evidence = ReachabilityEvidence("annotated_token_graph_1", 1, 1, 0)
    unreachable = ClassifiedObligation(
        target, False, "byte_level_unreachable", evidence
    )
    reachable = ClassifiedObligation(target, True, None, evidence)

    assert witness_result_from_attempt(unreachable, None, None).outcome == (
        "byte_level_unreachable"
    )
    assert (
        witness_result_from_attempt(reachable, None, None).outcome == "internal_error"
    )
