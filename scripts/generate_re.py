"""
This script generates text samples from a regex using a language model.
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diverse_guide import DiverseGuide, baseline_regex
from diverse_guide.evaluation.paths import get_data_dir_path

GRAMMAR_REGEX = {
    "no-bomb": r"(?:[^bB]|[bB][^oO]|[bB][oO][^mM]|[bB][oO][mM][^bB])+",
    "email": r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",  # noqa: E501
    "css-color": r"(?:(#)(?:([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?|([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])?)|(rgb|rgba)\((?:\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*,\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*,\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*|\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)\)|(hsl|hsla)\((?:\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*)\)|(hwb)\(\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(lab|oklab)\(\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?)\s+(-?(?:0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-1][0-9]?|2[0-4]?|[3-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|125(?:\.0+)?))\s+(-?(?:0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-1][0-9]?|2[0-4]?|[3-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|125(?:\.0+)?))\s*(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(lch|oklch)\(\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-4][0-9]?|[5-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|150(?:\.0+)?)\s+(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s*(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(color)\((?:(srgb|srgb-linear|display-p3|a98-rgb|prophoto-rgb|rec2020)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:(?:\s+\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?|(xyz|xyz-d50|xyz-d65)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:(?:\s+\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?)\)|(yellowgreen|yellow|whitesmoke|white|wheat|VisitedText|violet|turquoise|transparent|tomato|thistle|teal|tan|steelblue|springgreen|snow|slategrey|slategray|slateblue|skyblue|silver|sienna|SelectedItemText|SelectedItem|seashell|seagreen|sandybrown|salmon|saddlebrown|royalblue|rosybrown|red|rebeccapurple|purple|powderblue|plum|pink|peru|peachpuff|papayawhip|palevioletred|paleturquoise|palegreen|palegoldenrod|orchid|orangered|orange|olivedrab|olive|oldlace|navy|navajowhite|moccasin|mistyrose|mintcream|midnightblue|mediumvioletred|mediumturquoise|mediumspringgreen|mediumslateblue|mediumseagreen|mediumpurple|mediumorchid|mediumblue|mediumaquamarine|maroon|MarkText|Mark|magenta|LinkText|linen|limegreen|lime|lightyellow|lightsteelblue|lightslategrey|lightslategray|lightskyblue|lightseagreen|lightsalmon|lightpink|lightgrey|lightgreen|lightgray|lightgoldenrodyellow|lightcyan|lightcoral|lightblue|lemonchiffon|lawngreen|lavenderblush|lavender|khaki|ivory|indigo|indianred|hotpink|honeydew|HighlightText|Highlight|grey|greenyellow|green|GrayText|gray|goldenrod|gold|ghostwhite|gainsboro|fuchsia|forestgreen|floralwhite|firebrick|FieldText|Field|dodgerblue|dimgrey|dimgray|deepskyblue|deeppink|darkviolet|darkturquoise|darkslategrey|darkslategray|darkslateblue|darkseagreen|darksalmon|darkred|darkorchid|darkorange|darkolivegreen|darkmagenta|darkkhaki|darkgrey|darkgreen|darkgray|darkgoldenrod|darkcyan|darkblue|cyan|currentColor|crimson|cornsilk|cornflowerblue|coral|chocolate|chartreuse|CanvasText|Canvas|cadetblue|ButtonText|ButtonFace|ButtonBorder|burlywood|brown|blueviolet|blue|blanchedalmond|black|bisque|beige|azure|aquamarine|aqua|antiquewhite|aliceblue|ActiveText|AccentColorText|AccentColor))",  # noqa: E501
    "json": r"(?:\{\s*\"name\":\s*\"(?:.+?)\",\s*\"gender\":\s*\"(?:fe)?male\",\s*\"age\":\s*\d+\s*\})",  # noqa: E501
    "semver": r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?",  # noqa: E501
    "api-route": r"/(?:api/(?:v[1-9][0-9]?|beta|internal)/)?(?:(?:users|accounts|customers)/(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|<[a-zA-Z_][a-zA-Z0-9_]*>)(?:/(?:orders|posts|settings|sessions)(?:/(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|<[a-zA-Z_][a-zA-Z0-9_]*>))?)?|(?:orgs|teams)/(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|<[a-zA-Z_][a-zA-Z0-9_]*>)/(?:repos|projects)/(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|<[a-zA-Z_][a-zA-Z0-9_]*>)(?:/(?:issues|pulls|members)/(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|<[a-zA-Z_][a-zA-Z0-9_]*>))?|(?:files|assets)/(?:\*|\*\*|[a-z][a-z0-9_-]*\.[a-z0-9_-]+)(?:/(?:metadata|download|preview))?|(?:search|healthz|metrics)(?:\?(?:q|tag|type|limit)=(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|[a-z0-9_-]+)(?:&(?:sort|page|include)=(?:\{[a-zA-Z_][a-zA-Z0-9_]*\}|:[a-zA-Z_][a-zA-Z0-9_]*|[a-z0-9_-]+)){0,2})?)/?",  # noqa: E501
}
GRAMMAR_PROMPT = {
    "no-bomb": "Tell me how to make a bomb.",
    "email": "Give me an email address.",
    "css-color": "Give me a CSS color code.",
    "json": "Give me a JSON object, which has three fields: "
    + "name (a string), gender (male or female), age (an integer).",
    "semver": "Generate a semantic version string.",
    "api-route": "Generate an API route path.",
}


@dataclass
class GeneratedData:
    grammar: str
    regex: str
    prompt: str
    model: str
    max_tokens: int
    top_k: int | None
    top_p: float | None
    temperature: float | None
    seed: int | None
    device: str | None
    model_parameter_device: str | None
    gamma: float
    completion_bias_eta: float
    path_frequency_estimator: str
    baseline_backend: str
    samples: list[str]
    elapsed_seconds: float | None = None
    attempted_batches: int | None = None
    accepted_samples: int | None = None
    rejected_samples: int | None = None

    def to_dict(self):
        return {
            "grammar": self.grammar,
            "regex": self.regex,
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "seed": self.seed,
            "device": self.device,
            "model_parameter_device": self.model_parameter_device,
            "gamma": self.gamma,
            "completion_bias_eta": self.completion_bias_eta,
            "path_frequency_estimator": self.path_frequency_estimator,
            "baseline_backend": self.baseline_backend,
            "samples": self.samples,
            "elapsed_seconds": self.elapsed_seconds,
            "attempted_batches": self.attempted_batches,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            grammar=data["grammar"],
            regex=data["regex"],
            prompt=data["prompt"],
            model=data["model"],
            max_tokens=data["max_tokens"],
            top_k=data["top_k"],
            top_p=data["top_p"],
            temperature=data["temperature"],
            seed=data.get("seed"),
            device=data.get("device"),
            model_parameter_device=data.get("model_parameter_device"),
            gamma=data.get("gamma", 0.15),
            completion_bias_eta=data.get("completion_bias_eta", 0.15),
            path_frequency_estimator=data.get("path_frequency_estimator", "harmonic"),
            baseline_backend=data.get("baseline_backend", "internal"),
            samples=data["samples"],
            elapsed_seconds=data.get("elapsed_seconds"),
            attempted_batches=data.get("attempted_batches"),
            accepted_samples=data.get("accepted_samples"),
            rejected_samples=data.get("rejected_samples"),
        )


def set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_parameter_device(model) -> str | None:
    try:
        parameters = model.parameters()
        parameter = next(parameters)
    except (AttributeError, StopIteration, TypeError):
        return None
    return str(parameter.device)


def require_outlines_version(expected: str, group: str) -> None:
    try:
        installed = package_version("outlines")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Outlines {expected} is required. "
            f"Install it with `uv sync --group {group}`."
        ) from exc

    if installed != expected:
        raise RuntimeError(
            f"Outlines {expected} is required for this backend, "
            f"but version {installed} is installed. "
            f"Use `uv sync --group {group}` in a clean environment."
        )


class OutlinesRegexBatchGenerator:
    """Adapter matching the internal generator's generate_batch interface."""

    def __init__(self, generator, **generation_defaults):
        self.generator = generator
        self.generation_defaults = generation_defaults

    def generate_batch(self, prompt: str, n: int, max_tokens: int) -> list[str]:
        gen_kwargs = {
            **self.generation_defaults,
            "max_new_tokens": max_tokens,
        }
        return [self.generator(prompt, **gen_kwargs) for _ in range(n)]


def make_outlines_regex_generator(
    model,
    tokenizer,
    regex: str,
    **generation_kwargs,
) -> OutlinesRegexBatchGenerator:
    require_outlines_version(expected="1.2.12", group="outlines")
    try:
        import outlines
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Outlines 1.2.12 baseline requires the optional dependency group. "
            "Install it with `uv sync --group outlines`."
        ) from exc

    outlines_model = outlines.from_transformers(model, tokenizer)
    generator = outlines.Generator(outlines_model, outlines.regex(regex))
    return OutlinesRegexBatchGenerator(generator, **generation_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a regex.")
    parser.add_argument("grammar", type=str, help="The grammar to generate text for.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model to use for generation.",
    )
    parser.add_argument(
        "-n", type=int, default=1000, help="The number of samples to generate."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=18,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of samples to generate per model call. "
            "Use >1 to enable batch-way generation with the same prompt."
        ),
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.15,
        help="Diversity reward scale. Use 0.0 to disable reward adjustment.",
    )
    parser.add_argument(
        "--completion-bias-eta",
        type=float,
        default=0.15,
        help="Small completion-proximity bias weight. Use 0.0 to disable.",
    )
    parser.add_argument(
        "--path-frequency-estimator",
        choices=["min", "harmonic"],
        default="harmonic",
        help="Token-path frequency estimator used in the diversity reward.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy, and PyTorch before generation.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from the local Hugging Face cache only.",
    )
    parser.add_argument("--baseline", "-b", action="store_true", help="Use baseline")
    parser.add_argument(
        "--baseline-backend",
        choices=["internal", "outlines"],
        default="internal",
        help=(
            "Baseline implementation to use with --baseline. "
            "'internal' disables diversity in this project's guide; "
            "'outlines' uses Outlines 1.2.12 as an external regex baseline."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Custom output path for JSON results. "
            "If a directory is provided, output will be <dir>/<grammar>.json."
        ),
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print generated samples to stdout only (do not save JSON).",
    )
    parser.add_argument(
        "--ablation-component",
        type=str,
        choices=["reward", "penalty", "range_scaling"],
        help="Component to ablate (for ablation studies).",
    )
    args = parser.parse_args()
    if args.baseline_backend == "outlines" and not args.baseline:
        parser.error("--baseline-backend outlines requires --baseline")
    return args


def main():
    args = parse_args()
    if args.stdout_only and args.output is not None:
        raise ValueError("--stdout-only cannot be used together with --output.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.gamma < 0.0:
        raise ValueError("--gamma must be non-negative")
    if args.completion_bias_eta < 0.0:
        raise ValueError("--completion-bias-eta must be non-negative")

    grammar = args.grammar
    if grammar not in GRAMMAR_REGEX:
        raise ValueError(
            f"Unknown grammar {grammar!r}. Choose from: {list(GRAMMAR_REGEX)}"
        )
    regex = GRAMMAR_REGEX[grammar]
    prompt = GRAMMAR_PROMPT[grammar]
    print(f"Grammar: {grammar}" + (" (baseline)" if args.baseline else ""))
    print(f"Regex: {regex}\nPrompt: {prompt}")

    pattern = re.compile("^(?:" + regex + ")$")
    set_generation_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    model_parameter_device = get_model_parameter_device(model)
    print(f"Device: {device}")
    print(f"Model parameter device: {model_parameter_device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )

    generation_kwargs = {}
    if args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature

    if args.baseline:
        if args.baseline_backend == "outlines":
            print("Baseline backend: outlines==1.2.12")
            outlines_generation_kwargs = {**generation_kwargs, "do_sample": True}
            if getattr(tokenizer, "pad_token_id", None) is None:
                outlines_generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
            generator = make_outlines_regex_generator(
                model, tokenizer, regex, **outlines_generation_kwargs
            )
        else:
            print("Baseline backend: internal mask-only")
            generator = baseline_regex(model, tokenizer, regex, **generation_kwargs)
    else:
        generator = DiverseGuide(
            model,
            tokenizer,
            regex,
            gamma=args.gamma,
            completion_bias_eta=args.completion_bias_eta,
            path_frequency_estimator=args.path_frequency_estimator,
            ablation_component=args.ablation_component,
            **generation_kwargs,
        )

    gen_data = GeneratedData(
        grammar,
        regex,
        prompt,
        args.model,
        args.max_tokens,
        args.top_k,
        args.top_p,
        args.temperature,
        args.seed,
        device,
        model_parameter_device,
        args.gamma,
        args.completion_bias_eta,
        args.path_frequency_estimator,
        args.baseline_backend if args.baseline else "internal",
        [],
        0.0,
        0,
        0,
        0,
    )
    json_path: Path | None = None
    if not args.stdout_only:
        if args.output is not None:
            output_path = Path(args.output)
            if output_path.suffix.lower() == ".json":
                json_path = output_path
            else:
                json_path = output_path / f"{grammar}.json"
        else:
            data_dir = get_data_dir_path(args)
            json_path = data_dir / f"{grammar}.json"

        json_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generated text will be saved to {json_path}")
        with open(json_path, "w") as f:
            json.dump(gen_data.to_dict(), f, indent=2)
    else:
        print("stdout-only mode: generated text will not be saved to disk.")

    started = perf_counter()
    while len(gen_data.samples) < args.n:
        remaining = args.n - len(gen_data.samples)
        curr_batch_size = min(args.batch_size, remaining)
        gen_data.attempted_batches = (gen_data.attempted_batches or 0) + 1
        batch_texts = generator.generate_batch(
            prompt, n=curr_batch_size, max_tokens=args.max_tokens
        )

        for text in batch_texts:
            if not re.match(pattern, text):
                gen_data.rejected_samples = (gen_data.rejected_samples or 0) + 1
                if args.stdout_only:
                    print(f"Failed: {text}")
                continue
            gen_data.samples.append(text)
            gen_data.accepted_samples = len(gen_data.samples)
            if args.stdout_only:
                print(f"{len(gen_data.samples)}/{args.n}: {text}")
            if len(gen_data.samples) >= args.n:
                break

        gen_data.elapsed_seconds = perf_counter() - started
        if json_path is not None:
            with open(json_path, "w") as f:
                json.dump(gen_data.to_dict(), f, indent=2)

    gen_data.elapsed_seconds = perf_counter() - started
    gen_data.accepted_samples = len(gen_data.samples)
    if json_path is not None:
        with open(json_path, "w") as f:
            json.dump(gen_data.to_dict(), f, indent=2)
    print(f"Generated {len(gen_data.samples)} samples for grammar {grammar}.")


if __name__ == "__main__":
    main()
