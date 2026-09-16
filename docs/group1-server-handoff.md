# Group 1 server handoff

This release prepares the FAC five-repeat experiment for server deployment.
The local agent develops and exports commits; the server agent installs,
validates, and runs experiments. The task specification is included verbatim
as `experiment-group-1-task.md` in the transfer package.

## Release contents and readiness

- `source.bundle`: full `fac-revision` history, including the historical
  `cf7189e` and preservation snapshot `671ec48`. Clone this to obtain source.
- `imported-seed42/`: read-only copies of all 54 generation files (27,000
  samples), original evaluation and refresh outputs, summaries, generation
  logs, and available library coverage results.
- `manifest.json`: 270 task identities for seeds 42–46, historical parameters,
  input hashes, candidate model revisions and explicit unresolved issues.
- `evidence/`: historical source differences, library snapshot candidates,
  historical library fetch script, model/tokenizer configuration files.
- `SHA256SUMS`: checksums of every payload file; the outer archive has a
  separate `.sha256` file.

User decision on 2026-09-16 supersedes the task's provisional reuse plan:
use temperature=1.0 and rerun all 36 Qwen seed=42 configurations. This means
252 new runs (126,000 samples) plus 18 Phi seed=42 reuse candidates. If Phi's
historical parameters or semantics cannot be confirmed, rerun those too.
All 54 old runs remain preserved as historical evidence.

This is a deployment/evidence package, not a production-ready five-repeat
runner. Phi reuse candidates await compatibility confirmation; new tasks await
preflight. Do not use the old matrix script to launch the production jobs
yet: it inherits model sampling defaults, uses non-atomic sample writes, and
does not provide the required attempt journal or safe task recovery.

Model weights, Python environments, CUDA binaries, build caches and unrelated
worktrees are excluded. Reserve approximately 60 GB of free server disk for
models, installation and results (100 GB for longer-term use). Download the
three model snapshots into one shared cache once their revisions are approved.

## Install and verify

Upload both `group1-transfer-20260916.tar.gz` and its `.sha256` file. From the
upload directory, using a fresh extraction destination:

```bash
sha256sum -c group1-transfer-20260916.tar.gz.sha256
tar -xzf group1-transfer-20260916.tar.gz
cd group1-transfer-20260916
sha256sum -c SHA256SUMS
git clone --branch fac-revision source.bundle ../diverse-guide-server
git -C ../diverse-guide-server rev-parse HEAD
```

The commit must equal `manifest.json`'s `source_commit`. Keep the imported
directory read-only. Preserve the original archive and checksums on a second
persistent storage location. Use separate writable directories for new runs
and reevaluation. A Git clone does not copy the ignored experiment data.

Install Python 3.12, `uv`, Rust/Cargo, a C compiler and the server's NVIDIA
driver using the server's normal provisioning procedure. In the clone:

```bash
uv sync --locked --group dev --group outlines
uv run --no-sync python scripts/build_wd_kernel.py
uv run --no-sync python -c 'import torch, transformers, importlib.metadata as m; print(torch.__version__, transformers.__version__, m.version("outlines")); print(torch.cuda.is_available(), torch.version.cuda)'
nvidia-smi
uv run --no-sync pytest tests/test_dfa.py tests/test_guide.py tests/test_guide_equivalence.py tests/test_evaluate_main_matrix.py -q
```

Record these outputs plus Python/Rust versions, OS, GPU model/count, RAM,
available disk and the checksums of `uv.lock` and `regex_dfa_guide/Cargo.lock`.
Stop on a lockfile mismatch rather than silently updating dependencies. These
commands validate the deployment; no local GPU test is claimed by this release.

## Historical compatibility findings

1. The old records save `temperature`, `top_k`, and `top_p` as null. The
   historical generator only passes values when supplied, so null means model
   defaults, not temperature=1.0. The cached configurations of both Qwen models
   specify temperature=0.7, top_p=0.8, top_k=20 and repetition_penalty=1.1.
   The code requests `do_sample=True`, uses float16 on CUDA and defaults to
   batch size 1. Treat this as evidence-based reconstruction, not a historical
   runtime dump. The user has chosen temperature=1.0 and Qwen seed=42 reruns;
   keep old Qwen samples outside the formal five-run statistics.
2. Saved method parameters are: Outlines gamma=0, eta=0, estimator=min;
   original DiverseGuide gamma=1/6, eta=0, estimator=min; DiverseGuide+
   gamma=0.3, eta=1.0, estimator=harmonic. The current matrix script defaults
   to eta=0.1, so its defaults do not reproduce the saved + configuration.
3. Cached model revision candidates are listed in `manifest.json`; old sample
   JSON did not record revisions. Model and tokenizer must use the same
   approved snapshot. Evidence includes configuration and tokenizer hashes,
   but no claim that cache state alone proves historical provenance.
4. Three library-curve directories have different statement/branch
   denominators. The task names `case_study_coverage_curves`; preserve that
   set alongside the alternatives. The current fetch script selects newer
   libraries than the historical `cf7189e` script. Do not run it as an implicit
   update to Figure 1. Identify the manuscript source and matching library
   snapshots before library evaluation; fetched library candidates are evidence
   only, not approved historical fixtures.
5. `671ec48` changes Python/Rust steering code relative to the recorded
   evaluation commit `cf7189e`. The bundle includes both revisions and the
   package includes the diff. Confirm generation semantics rather than assuming
   that the snapshot exactly reproduces June's runs.
6. Historical generation retains aggregate acceptance/rejection counts but
   not individual attempt traces or RNG checkpoints. Preserve this missing
   evidence explicitly. New production tasks require incremental attempt
   journaling and atomic output; restarting a partial run must not append a
   fresh history to it.

The complete main evaluation dated 2026-06-13T12:08:42Z includes 54 records.
The `evaluation_outlines_json_refresh` directory is an earlier three-record
evaluation; keep it as provenance, not as an extra independent repetition.

## Server-side recheck and pilot

After installation, use an absolute path to the extracted package below, and
a fresh writable output directory:

```bash
uv run --no-sync python scripts/evaluate_main_matrix.py \
  --data-root /ABS/PACKAGE/imported-seed42/data/extension/main_matrix_500 \
  --output-dir /ABS/NEW-RESULTS/seed42-reevaluation \
  --expected-samples 500 --step 50 -d 5 -s 1 --strict
```

Compare all 54 metric records with the imported `evaluation/metrics.json`;
record any differences, evaluator commit and native-kernel build. Never
overwrite the original evaluation. Main metric trends use the original step
50; library curves use budgets `1,2,5,10,25,50,100,200,500`.

The next development bundle must supply the production runner and explicit
configuration freeze. Then pilot all three models and methods, including JSON,
in an isolated directory. Measure GPU memory, RAM and disk before choosing
concurrency. Existing library coverage code uses a shared temporary filename;
use isolated case-study directories for concurrent tasks. Pilot outputs must
not enter the formal repeats. Table 5 hypothesis testing remains undecided;
generation and descriptive statistics do not depend on selecting that test.

## Bundle synchronization

Keep server experiment outputs outside tracked source files. Send the local
agent the server's installed commit (`git rev-parse HEAD`) after every update.
Local development continues on `fac-revision`. For an incremental delivery,
replace `SERVER_COMMIT` with that confirmed commit and use a new bundle name:

```bash
git bundle create /ABS/DELIVERY/group1-update.bundle SERVER_COMMIT..fac-revision
git bundle verify /ABS/DELIVERY/group1-update.bundle
sha256sum /ABS/DELIVERY/group1-update.bundle
```

The server verifies the communicated SHA-256, pauses workers before changing
their source/environment, and applies the update in its clone:

```bash
git status --short
git bundle verify /ABS/UPLOAD/group1-update.bundle
git fetch /ABS/UPLOAD/group1-update.bundle refs/heads/fac-revision
git merge --ff-only FETCH_HEAD
uv sync --locked --group dev --group outlines
git rev-parse HEAD
```

If source changes or divergent commits prevent a fast-forward, preserve them
and coordinate with the local agent. Each run records the exact commit; a
source update that changes experiment semantics requires a new configuration
version and compatibility review. Git bundles transport code and history;
raw results travel separately with SHA-256 manifests.

## Re-export this package locally

Commit the intended source on `fac-revision`, then run from that worktree:

```bash
python3 scripts/export_group1_transfer.py \
  --legacy-root /home/lxk/projects/diverse-dfa-gen/.worktrees/master-counter-fix \
  --task /home/lxk/reports/diverse-guide-plus-report/docs/experiment-group-1-task.md \
  --hf-cache /home/lxk/.cache/huggingface/hub \
  --output /home/lxk/projects/diverse-dfa-gen/exports/group1-transfer-20260916
```

The output directory must not exist. For subsequent releases, choose a new
directory name and adjust the extraction commands to match it.
