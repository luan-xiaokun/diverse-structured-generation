"""
CLI script: evaluate diversity metrics for generated samples.

Pure evaluation functions live in ``src/metrics.py``; this script handles
argument parsing, data loading, and optional perplexity computation.
"""

import argparse
import json

import numpy as np

from metrics import (
    distinct_ngram,
    path_coverage,
    state_coverage,
    transition_coverage,
    vendi_score,
)
from paths import get_data_dir_path
from perplexity import calculate_perplexity
from regex_dfa_guide import DiverseGuideDFA
from string_kernel import compute_wd_kernel_matrix


def make_metric_line_plot(
    dfa: DiverseGuideDFA, inputs: list[str], fig_name: str
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(layout="constrained")
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    def f1(inputs):
        return compute_wd_kernel_matrix(inputs, d=3, s=1)

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


def evaluation(
    dfa: DiverseGuideDFA,
    inputs: list[str],
    d: int = 5,
    s: int = 3,
    n: int | None = None,
):
    def f(inputs):
        return compute_wd_kernel_matrix(inputs, d=d, s=s)

    if n:
        inputs = inputs[:n]

    average_length = np.mean([len(x) for x in inputs])
    print(f"- Number of samples: {len(inputs)}")
    print(f"- Average length: {average_length:.2f}")

    state_num = len(dfa.get_states())
    transition_num = sum(map(len, dfa.get_transitions().values()))
    print(f"- Number of states: {state_num}")
    print(f"- Number of transitions: {transition_num}")

    print(f"- State Coverage: {100 * state_coverage(dfa, inputs):.2f}%")
    print(f"- Transition Coverage: {100 * transition_coverage(dfa, inputs):.2f}%")
    print(f"- Path Coverage: {100 * path_coverage(dfa, inputs):.2f}%")
    print(f"- Distinct 2 gram: {distinct_ngram(inputs, 2)}")
    print(f"- Distinct 3 gram: {distinct_ngram(inputs, 3)}")
    print(f"- Vendi Score: {vendi_score(inputs, f):.2f}")


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
    return parser.parse_args()


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
    evaluation(dfa, gen_data["samples"], d=args.d, s=args.s, n=args.n)

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
        for text in gen_data["samples"]:
            try:
                ppl = calculate_perplexity(text, model, tokenizer)
                ppls.append(ppl)
            except Exception as e:
                print(f"Error calculating perplexity for text: {text}\n{e}")
        print(f"Average perplexity: {np.mean(ppls):.4f}")


if __name__ == "__main__":
    main()
