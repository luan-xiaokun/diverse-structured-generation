# Case Study

This directory is a self-contained case study used in the research project to
compare how different generated input sets affect coverage percentage in two
open-source Python libraries:

- `email_validator`
- `webcolors`

The study runs the same harness on two datasets per domain:

- `baseline`
- `diverse`

The current observed total coverage percentages from the official workflow are:

- `email_validator`: `46.16%` vs `59.14%`
- `webcolors`: `78.04%` vs `83.18%`

Use the repository root [README.md](/home/lxk/projects/diverse-dfa-gen/README.md)
for main project setup, and
[experiments/README.md](/home/lxk/projects/diverse-dfa-gen/experiments/README.md)
for the full paper-reproduction map across all tables.

## Layout

- `data/`: generated samples used as inputs for the experiments
- `tests/`: experiment harness and regression checks
- `run_case_study.sh`: Linux and macOS experiment runner
- `run_case_study.ps1`: Windows PowerShell experiment runner
- `report_case_study.py`: summarizes per-run coverage outputs
- `fetch_packages.sh`: Linux and macOS package refresh script
- `fetch_packages.ps1`: Windows PowerShell package refresh script
- `Makefile`: convenience commands for common Linux and macOS workflows
- `.coveragerc`: coverage configuration for the copied libraries

The `email_validator/` and `webcolors/` directories are pinned snapshots of
upstream packages and should be refreshed via the fetch script for your
platform, not edited by hand.

## Environment

This case study is intentionally isolated from the repository root project.
Create and use the local virtual environment inside this directory.

### Linux / macOS

These instructions expect `make` to be installed. On most Linux distributions
it is already available or provided by packages such as `build-essential` or
`base-devel`. On macOS, `make` is usually available after installing the Xcode
Command Line Tools.

```bash
cd experiments/case_study
uv sync
make test
```

### Windows

Windows does not require `make`. Use PowerShell from this directory:

```powershell
cd experiments/case_study
uv sync
.venv\Scripts\python.exe -m pytest -q tests
```

## Run

### Linux / macOS

Use either of the following:

```bash
cd experiments/case_study
bash run_case_study.sh
```

```bash
make run
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

### Linux / macOS

```bash
make refresh
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

For Linux and macOS:

```bash
make test
make run
make refresh
make clean
```

For Windows:

```powershell
uv sync
.\run_case_study.ps1
.\fetch_packages.ps1
.venv\Scripts\python.exe -m pytest -q tests
```
