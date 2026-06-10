# External Integrations

**Analysis Date:** 2026-05-19

## APIs & External Services

**Model Hosting / Downloads:**
- Hugging Face Hub - Model and tokenizer artifacts are loaded through `transformers.from_pretrained` in `scripts/generate_re.py`, `scripts/eval_runtime.py`, `scripts/metrics_eval.py`, `README.md`, and `REPRODUCIBILITY.md`.
  - SDK/Client: `transformers` 5.3.0 with transitive `huggingface-hub` 1.7.1 from `uv.lock`.
  - Auth: Not referenced by code. Private or gated models may use Hugging Face client environment such as `HF_TOKEN`, but no repository code reads or requires it.
  - Default generation model: `Qwen/Qwen2.5-1.5B-Instruct` in `scripts/generate_re.py`, `scripts/eval_runtime.py`, `README.md`, and `REPRODUCIBILITY.md`.
  - Optional perplexity model: `microsoft/Phi-4-mini-instruct` in `scripts/metrics_eval.py`, `README.md`, and `REPRODUCIBILITY.md`.
  - Reproducibility revisions: `REPRODUCIBILITY.md` records `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` and `microsoft/Phi-4-mini-instruct` revision `cfbefacb99257ffa30c83adab238a50856ac3083`; scripts do not pass `revision=`.

**Package Registries and Build Downloads:**
- PyPI - Python dependencies resolve from the default registry in `uv.lock` and `experiments/case_study/uv.lock`.
  - SDK/Client: `uv` from root workflows in `pyproject.toml`, `Dockerfile`, `README.md`, and `.github/workflows/test.yml`.
  - Auth: Not detected.
- crates.io - Rust dependencies resolve from `registry+https://github.com/rust-lang/crates.io-index` in `regex_dfa_guide/Cargo.lock`.
  - SDK/Client: Cargo using `regex_dfa_guide/Cargo.toml`.
  - Auth: Not detected.
- rustup installer - Docker build downloads Rust via `https://sh.rustup.rs` in `Dockerfile`.
  - SDK/Client: `curl` in `Dockerfile`.
  - Auth: Not detected.
- Astral uv installer - Docker build downloads uv via `https://astral.sh/uv/install.sh` in `Dockerfile`; CI uses `astral-sh/setup-uv@v5` in `.github/workflows/test.yml`.
  - SDK/Client: `curl` in `Dockerfile`, GitHub Action in `.github/workflows/test.yml`.
  - Auth: Not detected.
- Docker base image registry - `Dockerfile` uses build arg `CUDA_BASE=nvidia/cuda:13.1.1-runtime-ubuntu24.04`.
  - SDK/Client: Docker.
  - Auth: Not detected.

**Source Repository Fetches:**
- GitHub - Case-study fetch scripts clone pinned upstream projects for optional coverage experiments.
  - SDK/Client: `git` in `experiments/case_study/fetch_packages.sh` and `experiments/case_study/fetch_packages.ps1`.
  - Repositories: `https://github.com/JoshData/python-email-validator.git` and `https://github.com/ubernostrum/webcolors.git`.
  - Auth: Not detected.
- GitHub Actions Marketplace - CI uses `actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2`, and `astral-sh/setup-uv@v5` in `.github/workflows/test.yml`.
  - SDK/Client: GitHub Actions runner.
  - Auth: GitHub-hosted runner token implicit to Actions; no repository secrets referenced.

**Research / Metadata Links:**
- GitHub repository URLs and issue tracker are declared in `[project.urls]` of `pyproject.toml` and validated by `tests/test_release_metadata.py`.
- Paper DOI `https://doi.org/10.1007/978-981-95-4213-0_2` is declared in `[project.urls]` of `pyproject.toml`.

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable.
  - Client: Not applicable.

**File Storage:**
- Local filesystem only.
- Generated samples are written under `data/diverse/{model}/` and `data/baseline/{model}/` by `scripts/generate_re.py` and path helpers in `src/diverse_guide/evaluation/paths.py`.
- Evaluation JSON and CSV outputs are written under `results/` and `results/tables/` by `scripts/metrics_eval.py`, `scripts/eval_runtime.py`, and `scripts/collect_results.py`.
- Reference artifact outputs are committed under `artifact-results/v0.2.0/`.
- Optional case-study input data is committed under `experiments/case_study/data/`; optional case-study outputs are generated under `experiments/case_study/`.
- Native WD kernel artifacts are loaded from `src/diverse_guide/_native/`, `build/native/wd_kernel/`, or `native/wd_kernel/` by `src/diverse_guide/evaluation/string_kernel.py`.

**Caching:**
- Hugging Face model cache is managed externally by `transformers` / `huggingface-hub`; repository code does not set cache directories.
- Cargo cache is managed externally by Cargo and by `Swatinem/rust-cache@v2` in `.github/workflows/test.yml`.
- Python dependency cache is managed by uv and CI runner defaults; repository code does not define a custom cache service.

## Authentication & Identity

**Auth Provider:**
- None.
  - Implementation: No application users, sessions, JWTs, OAuth, API keys, or auth middleware are present in `src/`, `scripts/`, `experiments/`, or `.github/workflows/test.yml`.
- Hugging Face access for gated/private models is delegated to the external Hugging Face client environment; no repository env var is read directly.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- Console/stdout logging via `print` in CLI scripts such as `scripts/generate_re.py`, `scripts/eval_runtime.py`, `scripts/metrics_eval.py`, and `scripts/build_wd_kernel.py`.
- Progress bars via `tqdm` in `scripts/eval_runtime.py`.
- CI logs are GitHub Actions step output from `.github/workflows/test.yml`.
- Structured run metadata is written into result JSON by `scripts/repro_results.py` and consumed by `scripts/collect_results.py`.

## CI/CD & Deployment

**Hosting:**
- No hosted application or service deployment is configured.
- Docker images are buildable locally from `Dockerfile` with targets `latest` and `latest-cov`; no registry push workflow is present.
- Repository/package metadata points to GitHub in `pyproject.toml`.

**CI Pipeline:**
- GitHub Actions workflow `.github/workflows/test.yml`.
- Matrix: `ubuntu-latest` and `macos-latest`, Python `3.12`.
- Main job steps: checkout, stable Rust toolchain, Rust cache, uv setup, `uv sync --group dev`, `uv run maturin develop --release -m regex_dfa_guide/Cargo.toml`, `uv run poe test`, `uv run poe test-rust`, and `uv run poe lint`.
- Additional job parses PowerShell scripts under `experiments/` and validates shared helper output in `experiments/common.ps1`.

## Environment Configuration

**Required env vars:**
- Not detected for normal package import, tests, or internal scripts.

**Optional env vars and build args:**
- `SEED` - Optional experiment-level seed consumed by `experiments/*.sh` and `experiments/*.ps1`, then passed to `scripts/generate_re.py` and `scripts/eval_runtime.py`.
- `CUDA_BASE` - Docker build arg in `Dockerfile`, default `nvidia/cuda:13.1.1-runtime-ubuntu24.04`.
- `PYTHON_VERSION` - Docker build arg in `Dockerfile`, default `3.12`.
- `DEBIAN_FRONTEND`, `UV_LINK_MODE`, `UV_INSTALL_DIR`, `RUSTUP_HOME`, `CARGO_HOME`, and `PATH` - Non-secret Docker environment values in `Dockerfile`.
- Hugging Face client auth/cache env vars are not referenced directly; use standard Hugging Face tooling outside the repository when needed for private models.

**Secrets location:**
- Not detected.
- No `.env`, `.env.*`, `.npmrc`, `.pypirc`, `.netrc`, credential, secret, or key files were found in the scanned repository scope.

## Webhooks & Callbacks

**Incoming:**
- None. No HTTP server, route handlers, webhooks, callbacks, or API endpoints are present in `src/`, `scripts/`, or `experiments/`.

**Outgoing:**
- Model artifact downloads through Hugging Face are initiated by `transformers.from_pretrained` in `scripts/generate_re.py`, `scripts/eval_runtime.py`, and optional perplexity logic in `scripts/metrics_eval.py`.
- Optional GitHub repository clones are initiated by `experiments/case_study/fetch_packages.sh` and `experiments/case_study/fetch_packages.ps1`.
- Build-time downloads are initiated by `Dockerfile` for rustup and uv installation.
- Package downloads are initiated by uv/Cargo workflows from `pyproject.toml`, `uv.lock`, `regex_dfa_guide/Cargo.toml`, and `regex_dfa_guide/Cargo.lock`.

---

*Integration audit: 2026-05-19*
