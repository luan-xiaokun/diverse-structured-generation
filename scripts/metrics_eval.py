"""
CLI script: evaluate diversity metrics for generated samples.

Pure evaluation functions live in ``diverse_guide.evaluation.metrics``; this
script handles argument parsing, data loading, and optional perplexity
computation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from diverse_guide.evaluation import string_kernel
from diverse_guide.evaluation.metrics import (
    distinct_ngram,
    path_coverage,
    state_coverage,
    transition_coverage,
    vendi_score,
)
from diverse_guide.evaluation.paths import get_data_dir_path
from diverse_guide.evaluation.perplexity import calculate_perplexity
from regex_dfa_guide import DiverseGuideDFA

try:
    from repro_results import build_metadata, write_json
except ModuleNotFoundError:
    from scripts.repro_results import build_metadata, write_json


def make_metric_line_plot(
    dfa: DiverseGuideDFA, inputs: list[str], fig_name: str
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(layout="constrained")
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    def f1(inputs):
        return string_kernel.compute_wd_kernel_matrix(inputs, d=3, s=1)

    state_cov = state_coverage(dfa, inputs, step_size=1)
    transition_cov = transition_coverage(dfa, inputs, step_size=1)
    vendi_trace = vendi_score(inputs, f1, step_size=20)

    lns1 = ax1.plot(range(0, len(inputs)), state_cov, label="State Coverage")
    lns2 = ax1.plot(range(0, len(inputs)), transition_cov, label="Transition Coverage")
    lns3 = ax2.plot(
        range(0, len(inputs), 20),
        vendi_trace,
        label="Vendi Score",
        color="orange",
        linestyle="--",
    )
    lns = lns1 + lns2 + lns3
    labs = [ln.get_label() for ln in lns]
    ax1.grid()
    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Coverage Ratio")
    ax2.set_ylabel("Vendi Score")
    ax1.legend(lns, labs, loc=0)
    fig.savefig(f"figures/{fig_name}.png")


def compute_metrics(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    d: int = 5,
    s: int = 3,
    n: int | None = None,
    perplexities: list[float] | None = None,
    perplexity_error_count: int = 0,
) -> dict[str, Any]:
    def f(values):
        return string_kernel.compute_wd_kernel_matrix(values, d=d, s=s)

    if n is not None:
        if n <= 0:
            raise ValueError("n must be positive")
        inputs = inputs[:n]
        if perplexities is not None:
            perplexities = perplexities[:n]

    average_length = float(np.mean([len(x) for x in inputs])) if inputs else 0.0
    state_num = len(dfa.get_states())
    transition_num = sum(map(len, dfa.get_transitions().values()))
    average_perplexity = None
    if perplexities is not None and perplexities:
        average_perplexity = float(np.mean(perplexities))

    return {
        "sample_count": len(inputs),
        "average_length": average_length,
        "dfa": {
            "state_count": state_num,
            "transition_count": transition_num,
        },
        "metrics": {
            "state_coverage": float(state_coverage(dfa, inputs)),
            "transition_coverage": float(transition_coverage(dfa, inputs)),
            "path_coverage": float(path_coverage(dfa, inputs)),
            "distinct_2gram": list(distinct_ngram(inputs, 2)),
            "distinct_3gram": list(distinct_ngram(inputs, 3)),
            "vendi_score": float(vendi_score(inputs, f)),
            "average_perplexity": average_perplexity,
            "perplexity_count": len(perplexities) if perplexities is not None else 0,
            "perplexity_error_count": perplexity_error_count,
        },
    }


def add_perplexity_metrics(
    result: dict[str, Any],
    perplexities: list[float],
    perplexity_error_count: int = 0,
) -> None:
    metrics = result["metrics"]
    metrics["average_perplexity"] = (
        float(np.mean(perplexities)) if perplexities else None
    )
    metrics["perplexity_count"] = len(perplexities)
    metrics["perplexity_error_count"] = perplexity_error_count


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
    if metadata is None:
        metadata = build_metadata(
            {"string_kernel_backend": string_kernel.STRING_KERNEL_BACKEND}
        )

    result = {
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
        "metadata": metadata,
    }
    result.update(
        compute_metrics(
            dfa,
            inputs,
            d=args.d,
            s=args.s,
            n=args.n,
            perplexities=perplexities,
            perplexity_error_count=perplexity_error_count,
        )
    )
    return result


def print_metrics_result(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(f"- Number of samples: {result['sample_count']}")
    print(f"- Average length: {result['average_length']:.2f}")
    print(f"- Number of states: {result['dfa']['state_count']}")
    print(f"- Number of transitions: {result['dfa']['transition_count']}")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diversity metrics of generated samples."
    )
    parser.add_argument("grammar", type=str, help="Grammar name")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--baseline", "-b", action="store_true", help="Use baseline")
    parser.add_argument(
        "-d", type=int, default=5, help="Max k-mer degree for Vendi score"
    )
    parser.add_argument("-s", type=int, default=1, help="Max shift for Vendi score")
    parser.add_argument(
        "-n", type=int, default=None, help="Number of samples to evaluate"
    )
    parser.add_argument("--ppl", action="store_true", help="Compute perplexity")
    parser.add_argument(
        "--ppl-model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Model for perplexity calculation",
    )
    parser.add_argument(
        "--ablation-component",
        type=str,
        choices=["reward", "penalty", "range_scaling"],
        help="Component to ablate (for ablation studies).",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output path.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["diversity", "temperature_ablation", "component_ablation"],
        default="diversity",
        help="Experiment type for structured output.",
    )
    args = parser.parse_args()
    if args.n is not None and args.n <= 0:
        parser.error("-n must be positive")
    return args


def main():
    args = parse_args()
    is_baseline = args.baseline
    print(f"Grammar: {args.grammar}" + (" (baseline)" if is_baseline else ""))
    json_path = get_data_dir_path(args) / f"{args.grammar}.json"
    print(f"Loading generated samples from: {json_path}")
    with open(json_path) as f:
        gen_data = json.load(f)
    regex = "(?:" + gen_data["regex"] + ")$"
    print(regex)
    dfa = DiverseGuideDFA(regex, 2**32 - 1, {})

    result = build_metrics_result(
        args=args,
        dfa=dfa,
        inputs=gen_data["samples"],
        input_path=json_path,
        experiment=args.experiment,
    )
    print_metrics_result(result)

    ppls = None
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
        ppls = []
        samples_for_ppl = (
            gen_data["samples"][: args.n]
            if args.n is not None
            else gen_data["samples"]
        )
        for text in samples_for_ppl:
            try:
                ppl = calculate_perplexity(text, model, tokenizer)
                ppls.append(ppl)
            except Exception as e:
                perplexity_error_count += 1
                print(f"Error calculating perplexity for text: {text}\n{e}")

        add_perplexity_metrics(result, ppls, perplexity_error_count)
        average_perplexity = result["metrics"]["average_perplexity"]
        if average_perplexity is not None:
            print(f"Average perplexity: {average_perplexity:.4f}")

    if args.output:
        write_metrics_result(args.output, result)
        print(f"Structured results saved to: {args.output}")


if __name__ == "__main__":
    main()
