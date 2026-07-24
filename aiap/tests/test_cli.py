import json

import pytest

from aiap.baseline import WitnessSearchResult
from aiap.case_io import load_case
from aiap.cli import adapt_source, main, run_case
from aiap.obligations import CoverageObligation, ObligationCatalog
from aiap.reachability import ClassifiedObligation, ReachabilityEvidence
from aiap.replay import CandidateCertificate, CoveredObligation


class TinyTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def test_source_adapter_is_lossless_and_no_special_tokens():
    ids, decoded, failure = adapt_source(TinyTokenizer(), "ab")
    assert ids == [97, 98] and decoded == "ab" and failure is None


def test_cli_requires_explicit_case():
    with pytest.raises(SystemExit):
        main([])


def test_case_loader_rejects_unknown_tokenizer(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps({"regex": "a", "samples": ["a"], "model": "unknown"}))
    with pytest.raises(ValueError, match="unrecognized tokenizer"):
        load_case(path)


def test_run_case_rejects_non_positive_sample_num():
    from aiap.cli import run_case

    with pytest.raises(ValueError, match="sample_num must be positive"):
        run_case(object(), object(), 0)


def test_run_case_rejects_non_positive_obligation_num():
    from aiap.cli import run_case

    with pytest.raises(ValueError, match="obligation_num must be positive"):
        run_case(object(), object(), None, 0)


def test_json_safe_aggregates_encode_tuple_keys():
    from aiap.cli import _json_safe_aggregates

    payload = _json_safe_aggregates({"path_counts": {(1, 2): 3}})
    assert payload == {"path_counts": [{"key": [1, 2], "count": 3}]}


def test_jsonl_output_is_reserved_for_generated_proposals(tmp_path):
    from aiap.cli import write_certificates_jsonl
    from aiap.replay import CandidateCertificate

    certificates = [
        CandidateCertificate([], 0, 0, [], True, True, metadata={}),
        CandidateCertificate(
            [], 0, 0, [], True, True, metadata={"proposal": "state:2"}
        ),
    ]
    output = tmp_path / "witnesses.jsonl"
    write_certificates_jsonl(
        output, [c for c in certificates if c.metadata.get("proposal")]
    )
    assert output.read_text(encoding="utf-8").count("\n") == 1


def test_failed_proposal_certificate_is_explicit():
    from aiap.cli import _failed_proposal_certificate

    class Dfa:
        def get_initial_state(self):
            return 7

    class Obligation:
        obligation_id = "state:9"
        kind = "state"
        raw_meaning = 9

    certificate = _failed_proposal_certificate(Dfa(), Obligation())
    assert certificate.metadata["proposal"] == "state:9"
    assert certificate.failure.category == "no_accepting_witness"
    assert not certificate.aggregate_eligible


def test_run_case_classifies_selected_baseline_missing_then_selects_reachable(
    monkeypatch,
):
    """Unreachable diagnostics never take a reachable generation slot."""
    import aiap.cli as cli

    covered = CoverageObligation("state", 0)
    unreachable = CoverageObligation("state", 1)
    reachable_one = CoverageObligation("state", 2)
    reachable_two = CoverageObligation("state", 3)
    catalog = ObligationCatalog((covered, unreachable, reachable_one, reachable_two))
    case = type(
        "Case",
        (),
        {
            "regex": "fake",
            "samples": ["baseline", "ignored"],
            "path": "case.json",
            "grammar": "fake",
            "tokenizer_repository": "fake/tokenizer",
            "tokenizer_revision": "revision",
        },
    )()
    evidence = ReachabilityEvidence("test", 1, 1, 0)
    classified_inputs = []
    searched = []

    monkeypatch.setattr(
        cli, "DiverseRegexGuide", lambda *_: type("Guide", (), {"dfa": object()})()
    )
    monkeypatch.setattr(
        cli.ObligationCatalog,
        "from_dfa",
        classmethod(lambda _cls, _dfa: catalog),
    )
    monkeypatch.setattr(cli, "adapt_source", lambda _tok, text: ([1], text, None))

    def replay(_dfa, _ids, _catalog, **kwargs):
        proposal = kwargs.get("metadata", {}).get("proposal")
        covered_ids = (
            [CoveredObligation.from_obligation(covered)]
            if not proposal
            else [CoveredObligation(proposal, "state", int(proposal.split(":")[1]))]
        )
        return CandidateCertificate(
            [1],
            0,
            0,
            [],
            True,
            True,
            covered_obligations=covered_ids,
            source_text=kwargs.get("source_text"),
            decoded_text=kwargs.get("decoded_text"),
            metadata=kwargs.get("metadata", {}),
        )

    def classify(_dfa, missing):
        classified_inputs.extend(item.obligation_id for item in missing)
        return [
            ClassifiedObligation(
                unreachable, False, "byte_level_unreachable", evidence
            ),
            ClassifiedObligation(reachable_one, True, None, evidence),
            ClassifiedObligation(reachable_two, True, None, evidence),
        ]

    def search(_dfa, obligation, *, max_expansions=None):
        searched.append((obligation.obligation_id, max_expansions))
        return WitnessSearchResult([1], 1, max_expansions, False, None)

    monkeypatch.setattr(cli, "replay_candidate", replay)
    monkeypatch.setattr(cli, "classify_reachability", classify)
    monkeypatch.setattr(cli, "construct_one_witness", search)

    summary, attempts = run_case(
        case, TinyTokenizer(), sample_num=1, obligation_num=2, max_expansions=7
    )

    assert classified_inputs == ["state:1", "state:2", "state:3"]
    assert searched == [("state:2", 7), ("state:3", 7)]
    assert len(attempts) == 2
    assert summary["obligations"]["missing_total"] == 3
    assert summary["obligations"]["reachable_total"] == 2
    assert summary["obligations"]["unreachable_total"] == 1
    assert summary["obligations"]["selected_reachable"] == 2
    results = summary["obligations"]["results"]
    assert [item["id"] for item in results] == ["state:1", "state:2", "state:3"]
    assert results[0]["outcome"] == "byte_level_unreachable"
    assert results[0]["max_expansions"] is None
    assert summary["obligations"]["outcome_counts"] == {
        "success": 2,
        "byte_level_unreachable": 1,
        "search_limit": 0,
        "invalid_witness": 0,
        "internal_error": 0,
    }


def test_cli_writes_only_attempt_certificates_and_prints_all_outcomes(
    monkeypatch, tmp_path, capsys
):
    import aiap.cli as cli

    attempt = CandidateCertificate(
        [], 0, 0, [], False, False, metadata={"proposal": "state:2"}
    )
    summary = {
        "case": {
            "grammar": "fake",
            "path": "case.json",
            "tokenizer": {"repository": "fake", "revision": "rev"},
        },
        "candidates": {"total": 2, "accepted": 1, "failed": 1},
        "coverage": {"state": 1.0, "transition": 0.0, "path": 0.0},
        "obligations": {
            "outcome_counts": {
                "success": 0,
                "byte_level_unreachable": 1,
                "search_limit": 1,
                "invalid_witness": 0,
                "internal_error": 0,
            }
        },
    }
    output = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(cli, "load_case", lambda path: object())
    monkeypatch.setattr(cli, "_load_tokenizer", lambda case: object())
    monkeypatch.setattr(cli, "run_case", lambda *args, **kwargs: (summary, [attempt]))

    assert main(["case.json", "--jsonl", str(output), "--max-expansions", "3"]) == 0
    assert output.read_text(encoding="utf-8").count("\n") == 1
    terminal = capsys.readouterr().out
    for outcome in summary["obligations"]["outcome_counts"]:
        assert outcome in terminal


def test_cli_rejects_non_positive_max_expansions():
    with pytest.raises(SystemExit):
        main(["case.json", "--max-expansions", "0"])
