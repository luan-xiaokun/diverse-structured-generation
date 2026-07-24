from pathlib import Path

import pytest
from transformers import AutoTokenizer

from aiap.case_io import load_case
from aiap.cli import run_case
from diverse_guide.evaluation.metrics import (
    path_coverage,
    state_coverage,
    transition_coverage,
)
from diverse_guide.guide_rust import DiverseRegexGuide


@pytest.mark.real_case_parity
@pytest.mark.parametrize("name", ["diverse-css-color.json", "diverse-email.json"])
def test_real_case_parity(name):
    lock = Path(__file__).parents[1] / "tokenizer-lock.json"
    case = load_case(
        Path(__file__).parents[2] / "experiments/case_study/data" / name, lock
    )
    try:
        tok = AutoTokenizer.from_pretrained(
            case.tokenizer_repository,
            revision=case.tokenizer_revision,
            local_files_only=True,
        )
    except Exception:
        pytest.fail(
            "locked tokenizer missing; run: uv run python "
            "aiap/scripts/provision_real_case_tokenizers.py "
            "--lock aiap/tokenizer-lock.json ({exc})"
        )
    summary, certs = run_case(case, tok)
    assert summary["candidates"]["total"] >= len(case.samples)
    assert len(certs) == summary["candidates"]["total"]
    assert all(c.source_text is not None or c.metadata.get("proposal") for c in certs)
    dfa = DiverseRegexGuide(case.regex, tok).dfa
    accepted_texts = []
    for cert in certs:
        if cert.failure:
            assert cert.failure.category in {
                "decoded_text_mismatch",
                "illegal_token",
                "byte_trace",
                "destination_mismatch",
                "non_final",
            }
            assert not cert.aggregate_eligible
        else:
            assert cert.accepted and cert.aggregate_eligible
            assert cert.state_events or cert.transition_events or cert.path_events
            assert cert.decoded_text is not None
            accepted_texts.append(cert.decoded_text)
            # Independent legacy traversal must reproduce every ordered event
            # retained by the canonical replay certificate for this item.
            assert list(dfa.get_state_sequence(cert.decoded_text)) == cert.state_events
            transition_seq = dfa.get_transition_sequence(cert.decoded_text)
            expected_transitions = []
            expected_paths = []
            _, previous = transition_seq[0]
            for byte, state in transition_seq[1:]:
                expected_transitions.append((int(previous), int(byte), int(state)))
                expected_paths.append((int(previous), int(state)))
                previous = state
            assert expected_transitions == cert.transition_events
            assert expected_paths == cert.path_events

    # Legacy metric ratios over the same accepted decoded inputs must equal
    # the replay-derived aggregate ratios; rejected certificates are excluded.
    assert summary["coverage"]["state"] == pytest.approx(
        state_coverage(dfa, accepted_texts)
    )
    assert summary["coverage"]["transition"] == pytest.approx(
        transition_coverage(dfa, accepted_texts)
    )
    assert summary["coverage"]["path"] == pytest.approx(
        path_coverage(dfa, accepted_texts)
    )
