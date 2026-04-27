# Standardize Reproduction Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make root reproduction scripts write stable JSON outputs under `results/` and generate CSV summaries from those JSON files.

**Architecture:** Keep the existing `poe gen/eval/eval-runtime` and shell/PowerShell entrypoints. Add a small script-side result helper for metadata and JSON writing, refactor the evaluation CLIs just enough to return dictionaries, and add a collector that reads result JSON files without rerunning experiments.

**Tech Stack:** Python 3.12, pytest, Ruff, Bash, PowerShell, existing `uv`/`poe` tasks, existing `regex_dfa_guide` DFA extension.

---

## File Structure

- Create `scripts/repro_results.py`: shared metadata and JSON writing helpers for script CLIs.
- Modify `scripts/metrics_eval.py`: add pure result construction, `--output`, metadata, and JSON serialization.
- Modify `scripts/eval_runtime.py`: add runtime result construction, `--output`, metadata, and JSON serialization.
- Create `scripts/collect_results.py`: read result JSON files and write CSV tables.
- Modify `experiments/common.sh`: add result path helpers and pass `--output` to evaluation suites.
- Modify `experiments/common.ps1`: PowerShell equivalent of result path helpers and `--output`.
- Modify `experiments/5_component_ablation.sh`: write component ablation evaluation JSON files.
- Modify `experiments/5_component_ablation.ps1`: PowerShell equivalent for component ablation JSON files.
- Modify `experiments/README.md`: document `results/` and table collection.
- Create `tests/test_repro_results.py`: test shared helper behavior.
- Create `tests/test_metrics_eval_output.py`: test metric result dictionary and output writing without HuggingFace models.
- Create `tests/test_eval_runtime_output.py`: test runtime result dictionary and output writing without HuggingFace models.
- Create `tests/test_collect_results.py`: test CSV collection from fixture JSON.

## Task 1: Shared Reproduction Result Helpers

**Files:**
- Create: `scripts/repro_results.py`
- Create: `tests/test_repro_results.py`

- [ ] **Step 1: Write failing tests for metadata and JSON writing**

Create `tests/test_repro_results.py`:

```python
import json

from scripts.repro_results import build_metadata, write_json


def test_build_metadata_includes_stable_keys():
    metadata = build_metadata()

    assert set(metadata) >= {"timestamp_utc", "git_commit", "python", "platform"}
    assert isinstance(metadata["timestamp_utc"], str)
    assert metadata["timestamp_utc"].endswith("Z")
    assert isinstance(metadata["python"], str)
    assert isinstance(metadata["platform"], str)


def test_build_metadata_merges_extra_values():
    metadata = build_metadata({"string_kernel_backend": "python"})

    assert metadata["string_kernel_backend"] == "python"


def test_write_json_creates_parent_directories(tmp_path):
    output_path = tmp_path / "nested" / "result.json"

    write_json(output_path, {"schema_version": 1, "value": 3})

    with output_path.open() as f:
        data = json.load(f)
    assert data == {"schema_version": 1, "value": 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_repro_results.py -q
```

Expected: FAIL because `scripts.repro_results` does not exist.

- [ ] **Step 3: Implement helper module**

Create `scripts/repro_results.py`:

```python
"""Helpers for structured reproduction result files."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    """Return the current git commit if the repository metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def build_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_commit": get_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
uv run pytest tests/test_repro_results.py -q
```

Expected: PASS.

- [ ] **Step 5: Run lint for new helper**

Run:

```bash
uv run ruff check scripts/repro_results.py tests/test_repro_results.py
```

Expected: PASS.

- [ ] **Step 6: Commit helper module**

```bash
git add scripts/repro_results.py tests/test_repro_results.py
git commit -m "feat: add reproduction result helpers"
```

## Task 2: Metrics Evaluation JSON Output

**Files:**
- Modify: `scripts/metrics_eval.py`
- Create: `tests/test_metrics_eval_output.py`

- [ ] **Step 1: Write failing tests for metrics result construction and output writing**

Create `tests/test_metrics_eval_output.py`:

```python
import json
from argparse import Namespace

import pytest

from scripts.metrics_eval import (
    build_metrics_result,
    compute_metrics,
    write_metrics_result,
)


def _args(**overrides):
    values = {
        "grammar": "unit",
        "model": "test/model",
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "baseline": False,
        "d": 2,
        "s": 1,
        "n": None,
        "ppl": False,
        "ppl_model": "test/ppl",
        "ablation_component": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_compute_metrics_returns_numeric_result(dfa_single):
    result = compute_metrics(dfa_single, ["a", "b"], d=2, s=1)

    assert result["sample_count"] == 2
    assert result["average_length"] == pytest.approx(1.0)
    assert result["dfa"] == {"state_count": 2, "transition_count": 2}
    assert 0.0 <= result["metrics"]["state_coverage"] <= 1.0
    assert 0.0 <= result["metrics"]["transition_coverage"] <= 1.0
    assert 0.0 <= result["metrics"]["path_coverage"] <= 1.0
    assert result["metrics"]["distinct_2gram"] == [0, 2]
    assert result["metrics"]["distinct_3gram"] == [0, 2]
    assert result["metrics"]["average_perplexity"] is None
    assert result["metrics"]["perplexity_count"] == 0
    assert result["metrics"]["perplexity_error_count"] == 0


def test_build_metrics_result_includes_reproduction_context(dfa_single, tmp_path):
    input_path = tmp_path / "samples.json"
    args = _args(grammar="unit", baseline=True, temperature=1.5)

    result = build_metrics_result(
        args=args,
        dfa=dfa_single,
        inputs=["a", "b"],
        input_path=input_path,
        experiment="temperature_ablation",
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["schema_version"] == 1
    assert result["experiment"] == "temperature_ablation"
    assert result["setting"] == "baseline"
    assert result["grammar"] == "unit"
    assert result["input_path"] == str(input_path)
    assert result["parameters"]["temperature"] == 1.5
    assert result["metadata"]["timestamp_utc"] == "2026-04-27T00:00:00Z"


def test_write_metrics_result_creates_json(tmp_path, dfa_single):
    output_path = tmp_path / "results" / "diversity" / "unit.json"
    result = build_metrics_result(
        args=_args(),
        dfa=dfa_single,
        inputs=["a", "b"],
        input_path=tmp_path / "samples.json",
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    write_metrics_result(output_path, result)

    with output_path.open() as f:
        saved = json.load(f)
    assert saved["grammar"] == "unit"
    assert saved["metrics"]["state_coverage"] == result["metrics"]["state_coverage"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_metrics_eval_output.py -q
```

Expected: FAIL because `compute_metrics`, `build_metrics_result`, and
`write_metrics_result` do not exist.

- [ ] **Step 3: Refactor `scripts/metrics_eval.py` imports**

Add these imports near the top of `scripts/metrics_eval.py`:

```python
from pathlib import Path
from typing import Any

from diverse_guide.evaluation import string_kernel

try:
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.repro_results import build_metadata, write_json
```

Keep existing imports and remove no current CLI arguments.

- [ ] **Step 4: Replace print-only `evaluation()` with pure metric construction plus printing**

In `scripts/metrics_eval.py`, replace `evaluation(...)` with these functions:

```python
def compute_metrics(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    d: int = 5,
    s: int = 3,
    n: int | None = None,
    perplexities: list[float] | None = None,
    perplexity_error_count: int = 0,
) -> dict[str, Any]:
    def f(inputs):
        return compute_wd_kernel_matrix(inputs, d=d, s=s)

    if n:
        inputs = inputs[:n]

    average_length = float(np.mean([len(x) for x in inputs])) if inputs else 0.0
    state_num = len(dfa.get_states())
    transition_num = sum(map(len, dfa.get_transitions().values()))
    ppls = perplexities or []

    return {
        "sample_count": len(inputs),
        "average_length": average_length,
        "dfa": {
            "state_count": state_num,
            "transition_count": transition_num,
        },
        "metrics": {
            "state_coverage": state_coverage(dfa, inputs),
            "transition_coverage": transition_coverage(dfa, inputs),
            "path_coverage": path_coverage(dfa, inputs),
            "distinct_2gram": list(distinct_ngram(inputs, 2)),
            "distinct_3gram": list(distinct_ngram(inputs, 3)),
            "vendi_score": float(vendi_score(inputs, f)),
            "average_perplexity": float(np.mean(ppls)) if ppls else None,
            "perplexity_count": len(ppls),
            "perplexity_error_count": perplexity_error_count,
        },
    }


def build_metrics_result(
    args,
    dfa: DiverseGuideDFA,
    inputs: list[str],
    input_path: str | Path,
    experiment: str = "diversity",
    perplexities: list[float] | None = None,
    perplexity_error_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = compute_metrics(
        dfa,
        inputs,
        d=args.d,
        s=args.s,
        n=args.n,
        perplexities=perplexities,
        perplexity_error_count=perplexity_error_count,
    )
    result.update(
        {
            "schema_version": 1,
            "experiment": experiment,
            "setting": "baseline" if args.baseline else "diverse",
            "grammar": args.grammar,
            "model": args.model,
            "input_path": str(input_path),
            "parameters": {
                "d": args.d,
                "s": args.s,
                "n": args.n,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "ablation_component": args.ablation_component,
                "ppl": args.ppl,
                "ppl_model": args.ppl_model,
            },
            "metadata": metadata
            if metadata is not None
            else build_metadata(
                {"string_kernel_backend": string_kernel.STRING_KERNEL_BACKEND}
            ),
        }
    )
    return result


def print_metrics_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    dfa_info = result["dfa"]
    print(f"- Number of samples: {result['sample_count']}")
    print(f"- Average length: {result['average_length']:.2f}")
    print(f"- Number of states: {dfa_info['state_count']}")
    print(f"- Number of transitions: {dfa_info['transition_count']}")
    print(f"- State Coverage: {100 * metrics['state_coverage']:.2f}%")
    print(f"- Transition Coverage: {100 * metrics['transition_coverage']:.2f}%")
    print(f"- Path Coverage: {100 * metrics['path_coverage']:.2f}%")
    print(f"- Distinct 2 gram: {tuple(metrics['distinct_2gram'])}")
    print(f"- Distinct 3 gram: {tuple(metrics['distinct_3gram'])}")
    print(f"- Vendi Score: {metrics['vendi_score']:.2f}")
    if metrics["average_perplexity"] is not None:
        print(f"Average perplexity: {metrics['average_perplexity']:.4f}")


def write_metrics_result(path: str | Path, result: dict[str, Any]) -> None:
    write_json(path, result)
```

- [ ] **Step 5: Add CLI arguments**

In `parse_args()`, add:

```python
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write structured evaluation results to this JSON path.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="diversity",
        choices=["diversity", "temperature_ablation", "component_ablation"],
        help="Experiment group name to record in structured output.",
    )
```

- [ ] **Step 6: Update `main()` to build, print, and optionally write JSON**

In `main()`, replace the call to `evaluation(...)` and the final perplexity
print with this flow:

```python
    perplexities = []
    perplexity_error_count = 0
    if args.ppl:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
        model = AutoModelForCausalLM.from_pretrained(args.ppl_model, dtype=dtype).to(
            device
        )
        model.eval()
        for text in gen_data["samples"]:
            try:
                ppl = calculate_perplexity(text, model, tokenizer)
                perplexities.append(ppl)
            except Exception as e:
                perplexity_error_count += 1
                print(f"Error calculating perplexity for text: {text}\n{e}")

    result = build_metrics_result(
        args=args,
        dfa=dfa,
        inputs=gen_data["samples"],
        input_path=json_path,
        experiment=args.experiment,
        perplexities=perplexities,
        perplexity_error_count=perplexity_error_count,
    )
    print_metrics_result(result)
    if args.output is not None:
        write_metrics_result(args.output, result)
        print(f"Structured results saved to: {args.output}")
```

- [ ] **Step 7: Run focused metrics tests**

Run:

```bash
uv run pytest tests/test_metrics_eval_output.py -q
```

Expected: PASS.

- [ ] **Step 8: Run existing metrics tests**

Run:

```bash
uv run pytest tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 9: Run lint for metrics changes**

Run:

```bash
uv run ruff check scripts/metrics_eval.py tests/test_metrics_eval_output.py
```

Expected: PASS.

- [ ] **Step 10: Commit metrics JSON output**

```bash
git add scripts/metrics_eval.py tests/test_metrics_eval_output.py
git commit -m "feat: write structured metrics results"
```

## Task 3: Runtime Evaluation JSON Output

**Files:**
- Modify: `scripts/eval_runtime.py`
- Create: `tests/test_eval_runtime_output.py`

- [ ] **Step 1: Write failing runtime result tests**

Create `tests/test_eval_runtime_output.py`:

```python
import json
from argparse import Namespace

from scripts.eval_runtime import build_runtime_result, write_runtime_result


def _args(**overrides):
    values = {
        "grammar": "css-color",
        "model": "test/model",
        "n": 2000,
        "max_tokens": 60,
        "top_k": None,
        "top_p": None,
        "temperature": None,
        "baseline": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_runtime_result_includes_timing_and_context():
    result = build_runtime_result(
        args=_args(temperature=1.5),
        total_token_num=2031,
        total_time=123.45,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["schema_version"] == 1
    assert result["experiment"] == "runtime"
    assert result["setting"] == "baseline"
    assert result["grammar"] == "css-color"
    assert result["parameters"]["temperature"] == 1.5
    assert result["tokens"] == {"generated": 2031, "target": 2000}
    assert result["timing"]["seconds"] == 123.45
    assert result["timing"]["tokens_per_second"] == 2031 / 123.45


def test_build_runtime_result_handles_zero_time():
    result = build_runtime_result(
        args=_args(baseline=False),
        total_token_num=0,
        total_time=0.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    assert result["setting"] == "diverse"
    assert result["timing"]["tokens_per_second"] is None


def test_write_runtime_result_creates_json(tmp_path):
    output_path = tmp_path / "results" / "runtime" / "css-color.json"
    result = build_runtime_result(
        args=_args(),
        total_token_num=2000,
        total_time=100.0,
        metadata={"timestamp_utc": "2026-04-27T00:00:00Z", "git_commit": None},
    )

    write_runtime_result(output_path, result)

    with output_path.open() as f:
        saved = json.load(f)
    assert saved["timing"]["tokens_per_second"] == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_eval_runtime_output.py -q
```

Expected: FAIL because `build_runtime_result` and `write_runtime_result` do not
exist.

- [ ] **Step 3: Add runtime helper imports**

In `scripts/eval_runtime.py`, add:

```python
from pathlib import Path
from typing import Any

try:
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.repro_results import build_metadata, write_json
```

- [ ] **Step 4: Add runtime result helpers**

Add these functions above `main()`:

```python
def build_runtime_result(
    args,
    total_token_num: int,
    total_time: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens_per_second = total_token_num / total_time if total_time > 0 else None
    return {
        "schema_version": 1,
        "experiment": "runtime",
        "setting": "baseline" if args.baseline else "diverse",
        "grammar": args.grammar,
        "model": args.model,
        "parameters": {
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "tokens": {
            "generated": total_token_num,
            "target": args.n,
        },
        "timing": {
            "seconds": total_time,
            "tokens_per_second": tokens_per_second,
        },
        "metadata": metadata if metadata is not None else build_metadata(),
    }


def write_runtime_result(path: str | Path, result: dict[str, Any]) -> None:
    write_json(path, result)
```

- [ ] **Step 5: Add runtime CLI output argument**

In `parse_args()`, add:

```python
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write structured runtime results to this JSON path.",
    )
```

- [ ] **Step 6: Update `main()` to optionally write JSON**

Replace the two final print statements with:

```python
    result = build_runtime_result(args, total_token_num, total_time)
    print(
        f"Generated {result['tokens']['generated']} tokens in "
        f"{result['timing']['seconds']:.2f} seconds."
    )
    tokens_per_second = result["timing"]["tokens_per_second"]
    if tokens_per_second is None:
        print("Tokens per second: unavailable.")
    else:
        print(f"Tokens per second: {tokens_per_second:.2f}.")
    if args.output is not None:
        write_runtime_result(args.output, result)
        print(f"Structured results saved to: {args.output}")
```

- [ ] **Step 7: Run focused runtime tests**

Run:

```bash
uv run pytest tests/test_eval_runtime_output.py -q
```

Expected: PASS.

- [ ] **Step 8: Run lint for runtime changes**

Run:

```bash
uv run ruff check scripts/eval_runtime.py tests/test_eval_runtime_output.py
```

Expected: PASS.

- [ ] **Step 9: Commit runtime JSON output**

```bash
git add scripts/eval_runtime.py tests/test_eval_runtime_output.py
git commit -m "feat: write structured runtime results"
```

## Task 4: Experiment Scripts Write Standard Result Paths

**Files:**
- Modify: `experiments/common.sh`
- Modify: `experiments/common.ps1`
- Modify: `experiments/4_temperature_ablation.sh`
- Modify: `experiments/4_temperature_ablation.ps1`
- Modify: `experiments/5_component_ablation.sh`
- Modify: `experiments/5_component_ablation.ps1`

- [ ] **Step 1: Update Bash common helpers**

In `experiments/common.sh`, add result path helpers after `grammar_extra_args()`:

```bash
result_setting_dir() {
    local baseline_flag=${1:-false}
    if [[ "$baseline_flag" == "true" ]]; then
        printf '%s\n' "baseline"
    else
        printf '%s\n' "diverse"
    fi
}

metric_result_path() {
    local experiment=$1
    local baseline_flag=$2
    local grammar=$3
    local suffix=${4:-}
    local setting
    setting="$(result_setting_dir "$baseline_flag")"
    if [[ -n "$suffix" ]]; then
        printf 'results/%s/%s/%s/%s.json\n' "$experiment" "$setting" "$suffix" "$grammar"
    else
        printf 'results/%s/%s/%s.json\n' "$experiment" "$setting" "$grammar"
    fi
}

runtime_result_path() {
    local baseline_flag=$1
    local grammar=$2
    local setting
    setting="$(result_setting_dir "$baseline_flag")"
    printf 'results/runtime/%s/%s.json\n' "$setting" "$grammar"
}
```

- [ ] **Step 2: Update Bash eval suites to pass output paths**

Change `run_eval_suite()` signature and command construction:

```bash
run_eval_suite() {
    local baseline_flag=${1:-false}
    local experiment=${2:-diversity}
    local result_suffix=${3:-}
    shift 3 || true
    local extra_args=("$@")

    for grammar in "${GRAMMARS[@]}"; do
        local output_path
        output_path="$(metric_result_path "$experiment" "$baseline_flag" "$grammar" "$result_suffix")"
        local cmd=("eval" "$grammar" "--model" "$DEFAULT_MODEL" "--experiment" "$experiment" "--output" "$output_path")
        cmd+=("${extra_args[@]}")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}
```

Change `run_runtime_suite()` command construction:

```bash
run_runtime_suite() {
    local baseline_flag=${1:-false}

    for grammar in "${GRAMMARS[@]}"; do
        local output_path
        output_path="$(runtime_result_path "$baseline_flag" "$grammar")"
        local cmd=("eval-runtime" "$grammar" "--model" "$DEFAULT_MODEL" "--output" "$output_path")
        if [[ "$baseline_flag" == "true" ]]; then
            cmd+=("--baseline")
        fi
        run_poe "${cmd[@]}"
    done
}
```

- [ ] **Step 3: Update Bash temperature ablation calls**

In `experiments/4_temperature_ablation.sh`, change eval calls to:

```bash
run_eval_suite false "temperature_ablation" "temperature-$TEMPERATURE" "${PPL_ARGS[@]}"
run_eval_suite true "temperature_ablation" "temperature-$TEMPERATURE" "${PPL_ARGS[@]}"
```

- [ ] **Step 4: Update Bash component ablation eval calls**

In `experiments/5_component_ablation.sh`, change the evaluation section to:

```bash
run_poe eval css-color \
    --model "$DEFAULT_MODEL" \
    --experiment component_ablation \
    --output results/component_ablation/default/css-color.json
for component in "${ABLATION_COMPONENTS[@]}"; do
    run_poe eval css-color \
        --model "$DEFAULT_MODEL" \
        --experiment component_ablation \
        --ablation-component "$component" \
        --output "results/component_ablation/$component/css-color.json"
done
```

- [ ] **Step 5: Update PowerShell common helpers**

In `experiments/common.ps1`, add helpers after `Get-GrammarExtraArgs`:

```powershell
function Get-ResultSettingDir {
    param([switch]$Baseline)

    if ($Baseline) {
        return "baseline"
    }
    return "diverse"
}

function Get-MetricResultPath {
    param(
        [string]$Experiment,
        [switch]$Baseline,
        [string]$Grammar,
        [string]$Suffix = ""
    )

    $setting = Get-ResultSettingDir -Baseline:$Baseline
    if ($Suffix) {
        return "results/$Experiment/$setting/$Suffix/$Grammar.json"
    }
    return "results/$Experiment/$setting/$Grammar.json"
}

function Get-RuntimeResultPath {
    param(
        [switch]$Baseline,
        [string]$Grammar
    )

    $setting = Get-ResultSettingDir -Baseline:$Baseline
    return "results/runtime/$setting/$Grammar.json"
}
```

- [ ] **Step 6: Update PowerShell eval suites**

Change `Invoke-EvalSuite` to:

```powershell
function Invoke-EvalSuite {
    param(
        [switch]$Baseline,
        [string]$Experiment = "diversity",
        [string]$ResultSuffix = "",
        [string[]]$ExtraArgs = @()
    )

    foreach ($grammar in $script:Grammars) {
        $outputPath = Get-MetricResultPath -Experiment $Experiment -Baseline:$Baseline -Grammar $grammar -Suffix $ResultSuffix
        $cmd = @("eval", $grammar, "--model", $script:DefaultModel, "--experiment", $Experiment, "--output", $outputPath)
        $cmd += $ExtraArgs
        if ($Baseline) {
            $cmd += "--baseline"
        }
        Invoke-Poe @cmd
    }
}
```

Change `Invoke-RuntimeSuite` to:

```powershell
function Invoke-RuntimeSuite {
    param([switch]$Baseline)

    foreach ($grammar in $script:Grammars) {
        $outputPath = Get-RuntimeResultPath -Baseline:$Baseline -Grammar $grammar
        $cmd = @("eval-runtime", $grammar, "--model", $script:DefaultModel, "--output", $outputPath)
        if ($Baseline) {
            $cmd += "--baseline"
        }
        Invoke-Poe @cmd
    }
}
```

- [ ] **Step 7: Update PowerShell temperature ablation calls**

In `experiments/4_temperature_ablation.ps1`, change eval calls to:

```powershell
Invoke-EvalSuite -Experiment "temperature_ablation" -ResultSuffix "temperature-$temperature" -ExtraArgs $pplArgs
Invoke-EvalSuite -Baseline -Experiment "temperature_ablation" -ResultSuffix "temperature-$temperature" -ExtraArgs $pplArgs
```

- [ ] **Step 8: Update PowerShell component ablation eval calls**

In `experiments/5_component_ablation.ps1`, change the evaluation section to:

```powershell
Invoke-Poe eval css-color `
    --model $script:DefaultModel `
    --experiment component_ablation `
    --output results/component_ablation/default/css-color.json
foreach ($component in $ablationComponents) {
    Invoke-Poe eval css-color `
        --model $script:DefaultModel `
        --experiment component_ablation `
        --ablation-component $component `
        --output "results/component_ablation/$component/css-color.json"
}
```

- [ ] **Step 9: Syntax-check Bash scripts**

Run:

```bash
bash -n experiments/common.sh experiments/1_generation.sh experiments/2_diversity_evaluation.sh experiments/3_efficiency_evaluation.sh experiments/4_temperature_ablation.sh experiments/5_component_ablation.sh
```

Expected: PASS with no output.

- [ ] **Step 10: Run lint and focused tests**

Run:

```bash
uv run ruff check scripts tests
uv run pytest tests/test_repro_results.py tests/test_metrics_eval_output.py tests/test_eval_runtime_output.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit experiment script output paths**

```bash
git add experiments/common.sh experiments/common.ps1 experiments/4_temperature_ablation.sh experiments/4_temperature_ablation.ps1 experiments/5_component_ablation.sh experiments/5_component_ablation.ps1
git commit -m "feat: write experiment results to standard paths"
```

## Task 5: CSV Table Collection

**Files:**
- Create: `scripts/collect_results.py`
- Create: `tests/test_collect_results.py`

- [ ] **Step 1: Write failing collector tests**

Create `tests/test_collect_results.py`:

```python
import csv
import json

from scripts.collect_results import collect_all


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def _metrics_result(experiment, setting, grammar):
    return {
        "schema_version": 1,
        "experiment": experiment,
        "setting": setting,
        "grammar": grammar,
        "model": "test/model",
        "sample_count": 2,
        "average_length": 1.5,
        "dfa": {"state_count": 3, "transition_count": 4},
        "metrics": {
            "state_coverage": 1.0,
            "transition_coverage": 0.5,
            "path_coverage": 0.25,
            "distinct_2gram": [1, 2],
            "distinct_3gram": [0, 2],
            "vendi_score": 1.75,
            "average_perplexity": None,
            "perplexity_count": 0,
            "perplexity_error_count": 0,
        },
        "parameters": {"temperature": None, "ablation_component": None},
        "metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
    }


def _runtime_result(setting, grammar):
    return {
        "schema_version": 1,
        "experiment": "runtime",
        "setting": setting,
        "grammar": grammar,
        "model": "test/model",
        "parameters": {"n": 2000, "max_tokens": 60, "temperature": None},
        "tokens": {"generated": 2000, "target": 2000},
        "timing": {"seconds": 100.0, "tokens_per_second": 20.0},
        "metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
    }


def test_collect_all_writes_csv_tables(tmp_path):
    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "diversity" / "diverse" / "css-color.json",
        _metrics_result("diversity", "diverse", "css-color"),
    )
    _write_json(
        results_dir / "runtime" / "baseline" / "css-color.json",
        _runtime_result("baseline", "css-color"),
    )
    _write_json(
        results_dir
        / "temperature_ablation"
        / "diverse"
        / "temperature-1.5"
        / "css-color.json",
        _metrics_result("temperature_ablation", "diverse", "css-color"),
    )
    _write_json(
        results_dir / "component_ablation" / "default" / "css-color.json",
        _metrics_result("component_ablation", "diverse", "css-color"),
    )

    collect_all(results_dir)

    diversity_csv = results_dir / "tables" / "diversity.csv"
    runtime_csv = results_dir / "tables" / "runtime.csv"
    assert diversity_csv.exists()
    assert runtime_csv.exists()
    with diversity_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment"] == "diversity"
    assert rows[0]["setting"] == "diverse"
    assert rows[0]["grammar"] == "css-color"
    assert rows[0]["state_coverage"] == "1.0"
    with runtime_csv.open() as f:
        runtime_rows = list(csv.DictReader(f))
    assert runtime_rows[0]["tokens_per_second"] == "20.0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_collect_results.py -q
```

Expected: FAIL because `scripts.collect_results` does not exist.

- [ ] **Step 3: Implement CSV collector**

Create `scripts/collect_results.py`:

```python
"""Collect structured reproduction JSON files into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "experiment",
    "setting",
    "grammar",
    "model",
    "sample_count",
    "average_length",
    "state_count",
    "transition_count",
    "state_coverage",
    "transition_coverage",
    "path_coverage",
    "distinct_2gram_count",
    "distinct_2gram_samples",
    "distinct_3gram_count",
    "distinct_3gram_samples",
    "vendi_score",
    "average_perplexity",
    "perplexity_count",
    "perplexity_error_count",
    "temperature",
    "ablation_component",
]

RUNTIME_FIELDS = [
    "experiment",
    "setting",
    "grammar",
    "model",
    "target_tokens",
    "generated_tokens",
    "seconds",
    "tokens_per_second",
    "max_tokens",
    "temperature",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def metric_row(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data["metrics"]
    dfa = data["dfa"]
    parameters = data.get("parameters", {})
    distinct_2gram = metrics["distinct_2gram"]
    distinct_3gram = metrics["distinct_3gram"]
    return {
        "experiment": data["experiment"],
        "setting": data["setting"],
        "grammar": data["grammar"],
        "model": data["model"],
        "sample_count": data["sample_count"],
        "average_length": data["average_length"],
        "state_count": dfa["state_count"],
        "transition_count": dfa["transition_count"],
        "state_coverage": metrics["state_coverage"],
        "transition_coverage": metrics["transition_coverage"],
        "path_coverage": metrics["path_coverage"],
        "distinct_2gram_count": distinct_2gram[0],
        "distinct_2gram_samples": distinct_2gram[1],
        "distinct_3gram_count": distinct_3gram[0],
        "distinct_3gram_samples": distinct_3gram[1],
        "vendi_score": metrics["vendi_score"],
        "average_perplexity": metrics["average_perplexity"],
        "perplexity_count": metrics["perplexity_count"],
        "perplexity_error_count": metrics["perplexity_error_count"],
        "temperature": parameters.get("temperature"),
        "ablation_component": parameters.get("ablation_component"),
    }


def runtime_row(data: dict[str, Any]) -> dict[str, Any]:
    parameters = data.get("parameters", {})
    tokens = data["tokens"]
    timing = data["timing"]
    return {
        "experiment": data["experiment"],
        "setting": data["setting"],
        "grammar": data["grammar"],
        "model": data["model"],
        "target_tokens": tokens["target"],
        "generated_tokens": tokens["generated"],
        "seconds": timing["seconds"],
        "tokens_per_second": timing["tokens_per_second"],
        "max_tokens": parameters.get("max_tokens"),
        "temperature": parameters.get("temperature"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_metric_table(results_dir: Path, experiment: str) -> list[dict[str, Any]]:
    rows = []
    paths = sorted((results_dir / experiment).glob("**/*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON result files found under {results_dir / experiment}"
        )
    for path in paths:
        rows.append(metric_row(read_json(path)))
    return rows


def collect_runtime_table(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    paths = sorted((results_dir / "runtime").glob("**/*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON result files found under {results_dir / 'runtime'}"
        )
    for path in paths:
        rows.append(runtime_row(read_json(path)))
    return rows


def collect_all(results_dir: str | Path = "results") -> None:
    root = Path(results_dir)
    tables_dir = root / "tables"
    write_csv(
        tables_dir / "diversity.csv",
        collect_metric_table(root, "diversity"),
        METRIC_FIELDS,
    )
    write_csv(
        tables_dir / "runtime.csv",
        collect_runtime_table(root),
        RUNTIME_FIELDS,
    )
    write_csv(
        tables_dir / "temperature_ablation.csv",
        collect_metric_table(root, "temperature_ablation"),
        METRIC_FIELDS,
    )
    write_csv(
        tables_dir / "component_ablation.csv",
        collect_metric_table(root, "component_ablation"),
        METRIC_FIELDS,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect structured reproduction JSON files into CSV tables."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing structured reproduction JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_all(args.results_dir)
    print(f"CSV tables written to: {args.results_dir / 'tables'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run collector tests**

Run:

```bash
uv run pytest tests/test_collect_results.py -q
```

Expected: PASS.

- [ ] **Step 5: Run lint for collector**

Run:

```bash
uv run ruff check scripts/collect_results.py tests/test_collect_results.py
```

Expected: PASS.

- [ ] **Step 6: Commit collector**

```bash
git add scripts/collect_results.py tests/test_collect_results.py
git commit -m "feat: collect reproduction results into tables"
```

## Task 6: Documentation and Final Verification

**Files:**
- Modify: `experiments/README.md`

- [ ] **Step 1: Update output documentation**

In `experiments/README.md`, replace the "How to Save Results" section with:

````markdown
## Standard Result Outputs

The evaluation scripts write structured JSON files under `results/`.
Generated samples remain under `data/`.

The standard output layout is:

```text
results/
  diversity/
    diverse/<grammar>.json
    baseline/<grammar>.json
  runtime/
    diverse/<grammar>.json
    baseline/<grammar>.json
  temperature_ablation/
    diverse/temperature-1.5/<grammar>.json
    baseline/temperature-1.5/<grammar>.json
  component_ablation/
    default/css-color.json
    reward/css-color.json
    penalty/css-color.json
    range_scaling/css-color.json
  tables/
    diversity.csv
    runtime.csv
    temperature_ablation.csv
    component_ablation.csv
```

After running the experiment scripts, collect table-oriented CSV summaries with:

```bash
uv run python scripts/collect_results.py
```

PowerShell uses the same command:

```powershell
uv run python scripts/collect_results.py
```

Console logs are still useful for debugging. They are no longer the primary
reproduction artifact; the JSON and CSV files under `results/` are.
````

- [ ] **Step 2: Update Expected Runtime and Outputs section**

In the "Outputs" list, replace the evaluation summary bullet with:

```markdown
- Evaluation summaries are written as structured JSON files under `results/`.
- Table-oriented CSV summaries are written under `results/tables/` by
  `uv run python scripts/collect_results.py`.
```

- [ ] **Step 3: Run Markdown link check for changed docs**

Run the existing lightweight local link check style used in prior work:

```bash
uv run python - <<'PY'
from pathlib import Path
import re

for path in [Path("experiments/README.md")]:
    text = path.read_text()
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        local = (path.parent / target).resolve()
        if not local.exists():
            raise SystemExit(f"Broken link in {path}: {target}")
print("All checked local Markdown links resolve")
PY
```

Expected: prints `All checked local Markdown links resolve`.

- [ ] **Step 4: Run full Python test suite**

Run:

```bash
uv run poe test
```

Expected: all tests pass.

- [ ] **Step 5: Run lint**

Run:

```bash
uv run poe lint
```

Expected: PASS.

- [ ] **Step 6: Run a smoke command for metrics JSON without model loading**

Create a small temporary input by running a short generation-free Python command:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("data/diverse/test_model")
path.mkdir(parents=True, exist_ok=True)
(path / "css-color.json").write_text(json.dumps({
    "grammar": "css-color",
    "regex": "#[0-9a-fA-F]{6}",
    "prompt": "Give me a CSS color code.",
    "model": "test/model",
    "max_tokens": 18,
    "top_k": None,
    "top_p": None,
    "temperature": None,
    "samples": ["#aabbcc", "#112233"]
}, indent=2))
PY
uv run poe eval css-color --model test/model --output /tmp/diverse-guide-metrics-smoke.json
```

Expected: command exits 0 and prints `Structured results saved to:
/tmp/diverse-guide-metrics-smoke.json`.

- [ ] **Step 7: Re-run collector test as the collector smoke check**

Run:

```bash
uv run pytest tests/test_collect_results.py -q
```

Expected: PASS. The test creates representative JSON files and verifies the CSV
collector output without requiring a full experiment run.

- [ ] **Step 8: Remove smoke data created under ignored `data/`**

Run:

```bash
rm -rf data/diverse/test_model
```

Expected: no tracked file changes are removed.

- [ ] **Step 9: Check git status**

Run:

```bash
git status --short
```

Expected: only intended tracked changes plus the pre-existing untracked
`.codex` and `archive/` entries.

- [ ] **Step 10: Commit docs**

```bash
git add experiments/README.md
git commit -m "docs: document standardized reproduction outputs"
```

## Task 7: Final Integration Check

**Files:**
- Inspect: all files changed in Tasks 1-6

- [ ] **Step 1: Run full verification**

Run:

```bash
uv run poe lint
uv run poe test
bash -n experiments/common.sh experiments/1_generation.sh experiments/2_diversity_evaluation.sh experiments/3_efficiency_evaluation.sh experiments/4_temperature_ablation.sh experiments/5_component_ablation.sh
```

Expected: lint passes, tests pass, and Bash syntax check has no output.

- [ ] **Step 2: Confirm commits**

Run:

```bash
git log --oneline -7
```

Expected: recent commits include the design commit, this plan commit, and the
implementation commits from Tasks 1-6.

- [ ] **Step 3: Confirm no accidental artifact tracking**

Run:

```bash
git status --short
```

Expected: `.codex` and `archive/` may remain untracked. `results/` and `data/`
may appear only if smoke commands created ignored or untracked files; do not add
them to git unless the user explicitly requests committed fixture outputs.
