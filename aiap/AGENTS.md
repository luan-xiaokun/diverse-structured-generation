# AIAP project collaboration

## Working method

GSD is retired for this project. Do not invoke its skills, commands, agents, hooks, or configuration, or recreate `.planning/`. Work directly from the user's request: inspect relevant code, implement within agreed scope, run proportionate checks, and report the result.

Use [HANDOFF.md](HANDOFF.md) to recover context, [STATE.md](STATE.md) for progress, and [ROADMAP.md](ROADMAP.md) for research direction. Update these plain Markdown files when material decisions or progress change. Framework-specific plans, checkpoints, and generated status metadata are not required.

`docs/archive/gsd/` is historical reference only. Archived instructions and configuration must not govern current work. Consult its research decisions and verification evidence when useful.

## Project and correctness

- This exploratory research demo extends ICFEM 2025 diverse structured generation with multi-obligation witnesses, LLM naturalness guidance, and complementary test-suite selection.
- Keep AIAP work under `aiap/` by default; coordinate parent Python/Rust DFA changes when required.
- A coverage obligation is only a DFA structure unvisited by the selected baseline sample replay. The complete structure catalog is not the obligation set.
- `--sample-num` selects baseline samples; `--obligation-num`/`--ob-num` caps reachable targets attempted. Unreachable missing targets remain aggregate diagnostics and do not consume generation slots.
- JSONL contains attempted witness certificates only. Validate all acceptance and coverage claims by canonical DFA replay.
- Preserve bytes-only vocabulary and strict self-contained artifacts. Lock tokenizer revisions; offline tests fail actionably when snapshots are missing.
- Use heuristics where appropriate; do not claim global optimality without evidence.

## Implementation and verification

Python code lives in `aiap/`; tests in `tests/`. Use the parent repository `.venv`, Rust extension, and case-study assets. Preserve user changes. Keep code typed, format with Ruff, and retain the repository's Ruff pre-commit checks.

From this directory:

```bash
../.venv/bin/python -m pytest -m 'not real_case_parity'
HF_HUB_OFFLINE=1 ../.venv/bin/python -m pytest \
  tests/test_case_parity.py -m real_case_parity
```

Use targeted checks for the change at hand. Historical verification results are not evidence of a fresh test run.
