import os, json
from pathlib import Path
import pytest
from transformers import AutoTokenizer
from aiap.case_io import load_case
from aiap.cli import run_case

@pytest.mark.real_case_parity
@pytest.mark.parametrize("name", ["diverse-css-color.json", "diverse-email.json"])
def test_real_case_parity(name):
    lock = Path(__file__).parents[1] / "tokenizer-lock.json"
    case = load_case(Path(__file__).parents[2] / "experiments/case_study/data" / name, lock)
    try:
        tok = AutoTokenizer.from_pretrained(case.tokenizer_repository, revision=case.tokenizer_revision, local_files_only=True)
    except Exception as exc:
        pytest.fail(f"locked tokenizer missing; run: uv run python aiap/scripts/provision_real_case_tokenizers.py --lock aiap/tokenizer-lock.json ({exc})")
    summary, certs = run_case(case, tok)
    assert summary["candidates"]["total"] >= len(case.samples)
    assert len(certs) == summary["candidates"]["total"]
    assert all(c.source_text is not None or c.metadata.get("proposal") for c in certs)
    for cert in certs:
        if cert.failure:
            assert cert.failure.category in {"decoded_text_mismatch", "illegal_token", "byte_trace", "destination_mismatch", "non_final"}
            assert not cert.aggregate_eligible
        else:
            assert cert.accepted and cert.aggregate_eligible
            assert cert.state_events or cert.transition_events or cert.path_events
