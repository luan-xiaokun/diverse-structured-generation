import json
from pathlib import Path
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
    path = tmp_path / "case.json"; path.write_text(json.dumps({"regex": "a", "samples": ["a"], "model": "unknown"}))
    with pytest.raises(ValueError, match="unrecognized tokenizer"):
        load_case(path)
