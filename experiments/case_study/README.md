# Case Study

This directory is an optional automated impact check for DiverseGuide-generated
samples. It compares how baseline and diverse generated input sets affect
coverage percentage in two open-source Python libraries:

- `email_validator`
- `webcolors`

The study runs the same harness on two datasets per domain:

- `baseline`
- `diverse`

For OSP review, this case study is supplementary evidence that diverse
structured samples can be useful as downstream test inputs. It is not required
for the primary reproduction path, which is documented in the root
`REPRODUCIBILITY.md`.

The current observed total coverage percentages from the official workflow are:

- `email_validator`: `46.16%` vs `59.14%`
- `webcolors`: `78.04%` vs `83.18%`

Use the repository root [README.md](../../README.md)
for main project setup, and
[experiments/README.md](../README.md)
for the full paper-reproduction map across all tables.

## Layout

- `data/`: generated samples used as inputs for the experiments
- `tests/`: experiment harness and regression checks
- `run_case_study.sh`: Linux / macOS experiment runner
- `run_case_study.ps1`: Windows PowerShell experiment runner
- `report_case_study.py`: summarizes per-run coverage outputs
- `fetch_packages.sh`: Linux / macOS package refresh script
- `fetch_packages.ps1`: Windows PowerShell package refresh script
- `Makefile`: convenience commands for common Linux / macOS workflows
- `.coveragerc`: coverage configuration for the copied libraries

The `email_validator/` and `webcolors/` directories are pinned snapshots of
upstream packages and should be refreshed via the fetch script for your
platform, not edited by hand.

## Environment

This case study is intentionally isolated from the repository root project.
Create and use the local virtual environment inside this directory.

### Linux / macOS

From a clean clone, create the local virtual environment and fetch the pinned
package snapshots:

```bash
cd experiments/case_study
uv sync
bash fetch_packages.sh
```

### Windows

From a clean clone, create the local virtual environment and fetch the pinned
package snapshots:

```powershell
cd experiments/case_study
uv sync
.\fetch_packages.ps1
```

## Run

### Linux / macOS

Use either of the following:

```bash
cd experiments/case_study
bash run_case_study.sh
```

```bash
make -C experiments/case_study run
```

### Windows

Use PowerShell:

```powershell
cd experiments/case_study
.\run_case_study.ps1
```

The runner executes four coverage jobs:

1. baseline email samples against `email_validator`
2. diverse email samples against `email_validator`
3. baseline CSS color samples against `webcolors`
4. diverse CSS color samples against `webcolors`

It writes per-run coverage data to `.coverage.*`, JSON reports to
`.coverage.*.json`, HTML reports to `htmlcov_*`, and a consolidated summary to
`case_study_summary.json`.

## Refresh Pinned Packages

The fetch scripts recreate the ignored `email_validator/` and `webcolors/`
directories from pinned upstream revisions. Run them before the case study in a
clean clone.

### Linux / macOS

```bash
make -C experiments/case_study refresh
```

Or run the script directly:

```bash
bash fetch_packages.sh
```

### Windows

```powershell
.\fetch_packages.ps1
```

## Test Semantics

The tests in `tests/` are not conventional correctness tests for the copied
libraries. They are experiment harnesses with regression assertions:

- they execute the exact same library entrypoints used to collect coverage
- they verify that the observed success and error counts match the current
  behavior of the full datasets
- they still support `TEST_DATA_SAMPLE_SIZE` for quick exploratory runs, but
  exact count assertions are only enforced for the full datasets

This makes the study reproducible without changing the coverage behavior being
measured.

## Convenience Commands

For Linux / macOS:

```bash
cd experiments/case_study
make refresh
make test
make run
make clean
```

For Windows:

```powershell
cd experiments/case_study
uv sync
.\fetch_packages.ps1
.\run_case_study.ps1
.venv\Scripts\python.exe -m pytest -q tests
```
