from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from aiap.case_io import CaseSpec, load_case
from aiap.cli import run_case
from diverse_guide.evaluation.metrics import (
    path_coverage,
    state_coverage,
    transition_coverage,
)
from diverse_guide.guide_rust import DiverseRegexGuide
from diverse_guide.vocab import build_token_id_map


def _load_locked_tokenizer(case: CaseSpec) -> Any:
    """Load only the immutable cached snapshot, with an actionable failure."""
    try:
        return AutoTokenizer.from_pretrained(
            case.tokenizer_repository,
            revision=case.tokenizer_revision,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.fail(
            "locked tokenizer snapshot missing for "
            f"{case.tokenizer_key} ({case.tokenizer_repository}"
            f"@{case.tokenizer_revision}); run: uv run python "
            "aiap/scripts/provision_real_case_tokenizers.py "
            f"--lock aiap/tokenizer-lock.json ({exc})"
        )


def _token_driven_trace(
    dfa: Any, token_ids: list[int]
) -> tuple[int, bool, list[int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Expand token actions into the ordered byte-level traversal evidence."""
    state = int(dfa.get_initial_state())
    states: list[int] = [state]
    transitions: list[tuple[int, int, int]] = []
    paths: list[tuple[int, int]] = []

    for token_id in token_ids:
        byte_trace = list(dfa.get_byte_transition_sequence_from_state(state, token_id))
        assert byte_trace
        _, current = byte_trace[0]
        assert int(current) == state
        for byte, target in byte_trace[1:]:
            source = int(current)
            current = int(target)
            transitions.append((source, int(byte), current))
            paths.append((source, current))
            states.append(current)
        token_target = int(dfa.get_next_token_state(state, token_id))
        assert int(current) == token_target
        state = token_target

    return state, bool(dfa.is_final_state(state)), states, transitions, paths


def _direct_trace(
    dfa: Any, source_text: str
) -> tuple[int, bool, list[int], list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Collect direct UTF-8 traversal evidence in the token-trace shape."""
    states = [int(state) for state in dfa.get_state_sequence(source_text)]
    byte_trace = list(dfa.get_transition_sequence(source_text))
    assert byte_trace
    _, current = byte_trace[0]
    transitions: list[tuple[int, int, int]] = []
    paths: list[tuple[int, int]] = []
    for byte, target in byte_trace[1:]:
        source = int(current)
        current = int(target)
        transitions.append((source, int(byte), current))
        paths.append((source, current))
    final_state = int(current)
    return (
        final_state,
        bool(dfa.is_final_state(final_state)),
        states,
        transitions,
        paths,
    )


def _assert_full_trace_parity(
    dfa: Any, vocabulary: dict[int, bytes], tokenizer: Any, source_text: str
) -> None:
    token_ids = [
        int(token_id)
        for token_id in tokenizer(source_text, add_special_tokens=False)["input_ids"]
    ]
    assert b"".join(
        vocabulary[token_id] for token_id in token_ids
    ) == source_text.encode("utf-8")
    assert _token_driven_trace(dfa, token_ids) == _direct_trace(dfa, source_text)


@pytest.mark.real_case_parity
@pytest.mark.parametrize("name", ["diverse-css-color.json", "diverse-email.json"])
def test_real_case_parity(name):
    lock = Path(__file__).parents[1] / "tokenizer-lock.json"
    case = load_case(
        Path(__file__).parents[2] / "experiments/case_study/data" / name, lock
    )
    tok = _load_locked_tokenizer(case)
    vocabulary = build_token_id_map(tok)
    dfa = DiverseRegexGuide(case.regex, tok).dfa
    for source_text in case.samples:
        _assert_full_trace_parity(dfa, vocabulary, tok, source_text)

    summary, certs = run_case(case, tok)
    assert summary["candidates"]["total"] >= len(case.samples)
    assert len(certs) == summary["candidates"]["total"]
    assert all(c.source_text is not None or c.metadata.get("proposal") for c in certs)
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


@pytest.mark.real_case_parity
def test_qwen_split_u1680_bytes_and_trace_parity():
    case = load_case(
        Path(__file__).parents[2]
        / "experiments/case_study/data/diverse-css-color.json",
        Path(__file__).parents[1] / "tokenizer-lock.json",
    )
    tok = _load_locked_tokenizer(case)
    vocabulary = build_token_id_map(tok)
    source_text = "rgb(0\u16800\u16800)"
    token_ids = [
        int(token_id)
        for token_id in tok(source_text, add_special_tokens=False)["input_ids"]
    ]

    assert token_ids == [16509, 7, 15, 124999, 222, 15, 124999, 222, 15, 8]
    assert vocabulary[124999] == b"\xe1\x9a"
    assert vocabulary[222] == b"\x80"
    fragment_indexes = [
        index
        for index in range(len(token_ids) - 1)
        if token_ids[index : index + 2] == [124999, 222]
    ]
    assert fragment_indexes == [3, 6]
    for index in fragment_indexes:
        assert (
            b"".join(vocabulary[token_id] for token_id in token_ids[index : index + 2])
            == b"\xe1\x9a\x80"
        )

    _assert_full_trace_parity(
        DiverseRegexGuide(case.regex, tok).dfa, vocabulary, tok, source_text
    )


def test_missing_locked_tokenizer_error_is_actionable(monkeypatch):
    case = load_case(
        Path(__file__).parents[2]
        / "experiments/case_study/data/diverse-css-color.json",
        Path(__file__).parents[1] / "tokenizer-lock.json",
    )
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cache absent")),
    )

    with pytest.raises(pytest.fail.Exception) as failure:
        _load_locked_tokenizer(case)

    message = str(failure.value)
    assert "qwen2.5-1.5b-instruct" in message
    assert "Qwen/Qwen2.5-1.5B-Instruct" in message
    assert "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" in message
    assert (
        "uv run python aiap/scripts/provision_real_case_tokenizers.py "
        "--lock aiap/tokenizer-lock.json"
    ) in message
