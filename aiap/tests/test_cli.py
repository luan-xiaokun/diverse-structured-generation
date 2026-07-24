import json

import pytest

from aiap.case_io import load_case
from aiap.cli import adapt_source, main


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
