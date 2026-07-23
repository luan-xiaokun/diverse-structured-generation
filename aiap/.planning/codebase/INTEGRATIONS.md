# External Integrations

**Analysis Date:** 2026-07-23

## APIs & External Services

**Hugging Face Hub:**
- Model/tokenizer distribution for `Qwen/Qwen2.5-0.5B`, referenced in `aiap/cov_test.py`.
  - Client: `transformers.AutoTokenizer`.
  - Auth: none shown; public-model access is assumed.
  - Offline behavior: possible only when tokenizer assets are already cached.

**Parent research library:**
- `regex_dfa_guide` and `diverse_guide` are local parent-repository packages, not remote services.
  - Integration method: direct Python imports in `aiap/cov_test.py`.
  - Contract: provides DFA traversal and coverage metric methods.

## Data Storage

**Input files:**
- JSON case-study samples at `experiments/case_study/data/diverse-css-color.json` and `experiments/case_study/data/diverse-email.json`.
- Each input is expected to provide at least `regex` and `samples` fields.

**Persistent state:**
- None. Witnesses and coverage data are computed in memory and printed.

## Authentication & Identity

- No application authentication, user identities, API keys, or OAuth integrations are defined.

## Monitoring & Observability

- Console progress output only (`print` in `aiap/cov_test.py`).
- No structured logs, telemetry, error tracking, or metrics sink.

## CI/CD & Deployment

- No AIAP-specific CI workflow, packaging, container, or deployment target.
- Parent repository tooling is the likely execution environment until the demo becomes a standalone experiment package.

## Environment Configuration

- No `.env` file or required environment variables are documented for the demo.
- Hugging Face cache location and the parent package installation are implicit external prerequisites.

---
*Integration audit: 2026-07-23*
