from regex_dfa_guide import DiverseGuideDFA

from aiap.obligations import ObligationCatalog
from aiap.reporting import read_certificates_jsonl, write_certificates_jsonl
from aiap.replay import aggregate_certificates, replay_candidate


def make_dfa():
    return DiverseGuideDFA("(?:ab)?$", 99, {1: "a", 2: "b", 3: "ab"})


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
