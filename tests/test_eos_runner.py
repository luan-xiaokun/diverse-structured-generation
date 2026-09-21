import json
from pathlib import Path

import pytest

from scripts.run_group1_task import completed_is_valid, digest, file_hash


def test_frozen_matrix_and_mode_fingerprints():
    from scripts.eos_reentry import build_tasks

    base = json.loads(
        (Path(__file__).parents[1] / "configs/eos-reentry/base.json").read_text()
    )
    tasks = build_tasks(base, "/models/snapshot", "source-fingerprint")
    assert len(tasks) == len({t["id"] for t in tasks}) == 49
    for stage, count in [("smoke", 4), ("B", 20), ("C", 5), ("D", 20)]:
        assert sum(t["stage"] == stage for t in tasks) == count
    assert sum(t["step_probes"] for t in tasks if t["stage"] != "smoke") == 3
    assert all(
        t["target_valid_samples"] == (20 if t["stage"] == "smoke" else 500)
        for t in tasks
    )
    for t in tasks:
        assert t["fingerprint"] == digest(
            {k: v for k, v in t.items() if k != "fingerprint"}
        )
        changed = dict(t, eos_completion_mode="other")
        assert t["fingerprint"] != digest(
            {k: v for k, v in changed.items() if k != "fingerprint"}
        )


def test_completed_task_requires_mode_and_journal_integrity(tmp_path):
    result = tmp_path / "samples.json"
    journal = tmp_path / "attempts.jsonl"
    journal.write_text("original")
    result.write_text(
        json.dumps(
            dict(
                configuration_fingerprint="x",
                samples=["a"],
                eos_completion_mode="entry_count",
            )
        )
    )
    task = dict(
        fingerprint="x", target_valid_samples=1, eos_completion_mode="entry_count"
    )
    status = dict(
        status="complete",
        fingerprint="x",
        result_sha256=file_hash(result),
        artifacts_sha256={str(journal): file_hash(journal)},
    )
    assert completed_is_valid(status, task, result)
    assert not completed_is_valid(
        status, dict(task, eos_completion_mode="legacy"), result
    )
    journal.write_text("damaged")
    assert not completed_is_valid(status, task, result)


def test_attempt_denominators_token_costs_and_strata(tmp_path):
    from scripts.eos_reentry import attempt_statistics

    rows = [
        dict(
            event="finish",
            attempt=1,
            valid=False,
            token_ids=[[1, 2]],
            diagnostics=dict(
                ever_accepting=False, eos_opportunities={}, eos_selected={}
            ),
        ),
        dict(
            event="finish",
            attempt=2,
            valid=False,
            token_ids=[[1, 2, 3]],
            diagnostics=dict(
                ever_accepting=True, eos_opportunities={"1": 1}, eos_selected={}
            ),
        ),
        dict(
            event="finish",
            attempt=3,
            valid=True,
            token_ids=[[1, 99]],
            diagnostics=dict(
                ever_accepting=True, eos_opportunities={"1": 1}, eos_selected={"1": 1}
            ),
        ),
    ]
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    stats = attempt_statistics(path)
    assert stats["reject_ratio"] == 2 / 3
    assert (
        stats["never_accepting_ratio"] == stats["lost_after_accepting_ratio"] == 1 / 3
    )
    assert stats["tokens_per_valid"] == 7
    assert stats["accepted_tokens"] == 2 and stats["rejected_tokens"] == 5
    assert stats["eos_opportunities_1"] == 2 and stats["eos_rate_1"] == 0.5
    assert stats["eos_rate_2"] is None
    path.write_text("\n".join(map(json.dumps, rows + [rows[-1]])))
    with pytest.raises(ValueError, match="duplicates or gaps"):
        attempt_statistics(path)
