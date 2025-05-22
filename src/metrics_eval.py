"""
Evaluation of diversity metrics for generated samples.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.stats import entropy

from minimal_dfa import MinDivDFA
from perplexity import calculate_perplexity
from string_kernel import compute_wd_kernel_matrix


def get_coverage_ratio(count_dict: dict[str, int], threshold: int):
    covered = 0
    for count in count_dict.values():
        if count >= threshold:
            covered += 1
    return covered / len(count_dict)


def state_coverage(
    dfa: MinDivDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    states = dfa.get_states()
    state_count = {state: 0 for state in states}
    coverage_trace = []

    for i, s in enumerate(inputs):
        state = dfa.get_initial_state()
        for state in dfa.get_state_sequence(s):
            state_count[state] += 1
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(state_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(state_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(state_count, cov_threshold))
    return coverage_trace


def transition_coverage(
    dfa: MinDivDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    transitions = set()
    for u, trans in dfa.get_transitions().items():
        for c, v in trans.items():
            transitions.add((u, c, v))
    transition_count = {transition: 0 for transition in transitions}
    coverage_trace = []

    for i, s in enumerate(inputs):
        trans_seq = dfa.get_transition_sequence(s)
        _, prev_state = trans_seq[0]
        for char, state in trans_seq[1:]:
            transition_count[(prev_state, char, state)] += 1
            prev_state = state
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(transition_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(transition_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(transition_count, cov_threshold))
    return coverage_trace


def path_coverage(
    dfa: MinDivDFA,
    inputs: list[str],
    cov_threshold: int = 1,
    step_size: int | None = None,
) -> float | list[float]:
    paths = set()
    for u, trans in dfa.get_transitions().items():
        for _, v in trans.items():
            paths.add((u, v))
    path_count = {path: 0 for path in paths}
    coverage_trace = []

    for i, s in enumerate(inputs):
        trans_seq = dfa.get_transition_sequence(s)
        _, prev_state = trans_seq[0]
        for _, state in trans_seq[1:]:
            path_count[(prev_state, state)] += 1
            prev_state = state
        if step_size and (i + 1) % step_size == 0:
            coverage_trace.append(get_coverage_ratio(path_count, cov_threshold))

    if step_size is None:
        return get_coverage_ratio(path_count, cov_threshold)
    if len(inputs) % step_size != 0:
        coverage_trace.append(get_coverage_ratio(path_count, cov_threshold))
    return coverage_trace


def vendi_score(
    inputs: list[str],
    kernel_mtx_func: Callable[[list[str]], np.ndarray],
    step_size: int | None = None,
) -> float:
    gram_mtx = kernel_mtx_func(inputs)
    if step_size is None or step_size == len(inputs):
        assert kernel_mtx_func is not None, "kernel_mtx_func is None"
        eigenvalues = np.linalg.eigvalsh(gram_mtx / len(inputs))
        return np.exp(entropy(eigenvalues + 1e-10))

    vendi_trace = []
    for i in range(step_size, len(inputs), step_size):
        vendi_trace.append(
            np.exp(entropy(np.linalg.eigvalsh(gram_mtx[:i, :i]) + 1e-10))
        )
    if len(inputs) % step_size != 0:
        vendi_trace.append(
            np.exp(entropy(np.linalg.eigvalsh(gram_mtx / len(inputs)) + 1e-10))
        )
    return vendi_trace


def distinct_ngram(inputs: list[str], n: int):
    ngram_count = {}
    for s in inputs:
        for i in range(len(s) - n + 1):
            ngram = s[i : i + n]
            if ngram not in ngram_count:
                ngram_count[ngram] = 0
            ngram_count[ngram] += 1
    return len(ngram_count), len(inputs)


def make_metric_line_plot(dfa: MinDivDFA, inputs: list[str], fig_name: str) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(layout="constrained")
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()

    def f1(inputs):
        return compute_wd_kernel_matrix(inputs, d=3, s=1)

    state_cov = state_coverage(dfa, inputs, step_size=1)
    transition_cov = transition_coverage(dfa, inputs, step_size=1)
    vendi_trace = vendi_score(inputs, f1, step_size=20)

    lns1 = ax1.plot(
        range(0, len(inputs)),
        state_cov,
        label="State Coverage",
    )
    lns2 = ax1.plot(
        range(0, len(inputs)),
        transition_cov,
        label="Transition Coverage",
    )
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
    dfa: MinDivDFA, inputs: list[str], d: int = 5, s: int = 3, n: int | None = None
):
    def f(inputs):
        return compute_wd_kernel_matrix(inputs, d=d, s=s)

    if n:
        inputs = inputs[:n]

    average_length = np.mean([len(s) for s in inputs])
    print(f"- Number of samples: {len(inputs)}")
    print(f"- Average length: {average_length:.4f}")

    state_num = len(dfa.get_states())
    transition_num = sum(map(len, dfa.get_transitions().values()))
    print(f"- Number of states: {state_num}")
    print(f"- Number of transitions: {transition_num}")

    state_cov = state_coverage(dfa, inputs)
    print(f"- State Coverage: {100 * state_cov:.4f}")

    transition_cov = transition_coverage(dfa, inputs)
    print(f"- Transition Coverage: {100 * transition_cov:.4f}")

    path_cov = path_coverage(dfa, inputs)
    print(f"- Path Coverage: {100 * path_cov:.4f}")

    print(f"- Distinct 2 gram: {distinct_ngram(inputs, 2)}")
    print(f"- Distinct 3 gram: {distinct_ngram(inputs, 3)}")

    vendi = vendi_score(inputs, f)
    print(f"- Vendi Score: {vendi:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate diversity metrics of generated samples."
    )
    parser.add_argument("task", type=str, help="Task name")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None, help="The top-k value for sampling."
    )
    parser.add_argument(
        "--top-p", type=float, default=None, help="The top-p value for sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="The temperature for sampling."
    )
    parser.add_argument("--baseline", "-b", action="store_true", help="Use baseline")
    parser.add_argument(
        "-d",
        type=int,
        default=5,
        help="Distance parameter for Vendi score (default: 5)",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=1,
        help="Step size for Vendi score (default: 3)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: None)",
    )
    parser.add_argument("--ppl", action="store_true", help="Compute perplexity")
    return parser.parse_args()


def get_data_dir_path(args) -> Path:
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    is_baseline = args.baseline
    data_dir = Path("data" if is_baseline else "data/diverse")
    segments = []
    model = args.model.split("/")[-1]
    segments.append(model.lower())
    if top_k is not None:
        segments.append(f"top_k_{top_k}")
    if top_p is not None:
        segments.append(f"top_p_{top_p}")
    if temperature is not None:
        segments.append(f"temperature_{temperature}")
    return data_dir / "-".join(segments)


def main():
    args = parse_args()
    is_baseline = args.baseline
    print(f"Task: {args.task}" + (" (baseline)" if is_baseline else ""))
    json_path = get_data_dir_path(args) / f"{args.task}.json"
    with open(json_path, "r") as f:
        gen_data = json.load(f)
    regex = "(?:" + gen_data["regex"] + ")$"
    print(regex)
    dfa = MinDivDFA(regex, 2**32 - 1, {})
    # make_metric_line_plot(dfa, gen_data["samples"], args.task)
    evaluation(dfa, gen_data["samples"], d=args.d, s=args.s, n=args.n)

    if args.ppl:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "models/microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        model.eval()
        ppls = []
        for text in gen_data["samples"]:
            try:
                ppl = calculate_perplexity(text, model, tokenizer)
                ppls.append(ppl)
            except Exception as e:
                print(f"Error calculating perplexity for text: {text}")
                print(e)
        print(f"Average perplexity: {np.mean(ppls):.4f}")


if __name__ == "__main__":
    main()
