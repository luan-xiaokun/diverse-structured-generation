"""
This script generates text samples from a regex using a language model.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import outlines
import outlines.samplers

from diverse_guide_rust import diverse_regex
from minimal_dfa import MinDivDFA

TASK_REGEX = {
    "no-bomb": r"(?:[^bB]|[bB][^oO]|[bB][oO][^mM]|[bB][oO][mM][^bB])+",
    "threefold": r"-?(0|[369]|([147][0369]*[258]|[258][0369]*[147]|[0369])([0369]*([0369]|[147][0369]*[258]|[258][0369]*[147]))*?)$",
    "ipv4": r"(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$",
    "ipv6": r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))",
    "email": r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
    "css-color": r"(?:(#)(?:([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?|([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])?)|(rgb|rgba)\((?:\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*,\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*,\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*|\s*(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)\s+(0*(?:0|1[0-9]{0,2}|2(?:[0-4][0-9]?|5[0-4]?|[6-9])?|[3-9][0-9]?)(?:\.[0-9]+)?|255(?:\.0+)?|\.[0-9]+)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)\)|(hsl|hsla)\((?:\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s*,\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*|\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s*)\)|(hwb)\(\s*(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(lab|oklab)\(\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?)\s+(-?(?:0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-1][0-9]?|2[0-4]?|[3-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|125(?:\.0+)?))\s+(-?(?:0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-1][0-9]?|2[0-4]?|[3-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|125(?:\.0+)?))\s*(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(lch|oklch)\(\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?)\s+(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|(?:0|1(?:[0-4][0-9]?|[5-9])?|[2-9][0-9]?)(?:\.[0-9]+)?|150(?:\.0+)?)\s+(-?[0-9]+(?:\.[0-9]+)?(?:deg|rad|grad|turn)?)\s*(?:(?:\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?\)|(color)\((?:(srgb|srgb-linear|display-p3|a98-rgb|prophoto-rgb|rec2020)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:\s+|\s*,\s*)(0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+|0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%)(?:(?:\s+\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?|(xyz|xyz-d50|xyz-d65)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:\s+|\s*,\s*)(-?[0-9]+(?:\.[0-9]+)?%?)(?:(?:\s+\s*(?:\/)\s*(0*(?:(?:0|[1-9][0-9]?)(?:\.[0-9]+)?|100(?:\.0+)?|\.[0-9]+)%|0*0*(?:\.[0-9]+)?|1(?:\.0+)?|\.[0-9]+))?\s*)?)\)|(yellowgreen|yellow|whitesmoke|white|wheat|VisitedText|violet|turquoise|transparent|tomato|thistle|teal|tan|steelblue|springgreen|snow|slategrey|slategray|slateblue|skyblue|silver|sienna|SelectedItemText|SelectedItem|seashell|seagreen|sandybrown|salmon|saddlebrown|royalblue|rosybrown|red|rebeccapurple|purple|powderblue|plum|pink|peru|peachpuff|papayawhip|palevioletred|paleturquoise|palegreen|palegoldenrod|orchid|orangered|orange|olivedrab|olive|oldlace|navy|navajowhite|moccasin|mistyrose|mintcream|midnightblue|mediumvioletred|mediumturquoise|mediumspringgreen|mediumslateblue|mediumseagreen|mediumpurple|mediumorchid|mediumblue|mediumaquamarine|maroon|MarkText|Mark|magenta|LinkText|linen|limegreen|lime|lightyellow|lightsteelblue|lightslategrey|lightslategray|lightskyblue|lightseagreen|lightsalmon|lightpink|lightgrey|lightgreen|lightgray|lightgoldenrodyellow|lightcyan|lightcoral|lightblue|lemonchiffon|lawngreen|lavenderblush|lavender|khaki|ivory|indigo|indianred|hotpink|honeydew|HighlightText|Highlight|grey|greenyellow|green|GrayText|gray|goldenrod|gold|ghostwhite|gainsboro|fuchsia|forestgreen|floralwhite|firebrick|FieldText|Field|dodgerblue|dimgrey|dimgray|deepskyblue|deeppink|darkviolet|darkturquoise|darkslategrey|darkslategray|darkslateblue|darkseagreen|darksalmon|darkred|darkorchid|darkorange|darkolivegreen|darkmagenta|darkkhaki|darkgrey|darkgreen|darkgray|darkgoldenrod|darkcyan|darkblue|cyan|currentColor|crimson|cornsilk|cornflowerblue|coral|chocolate|chartreuse|CanvasText|Canvas|cadetblue|ButtonText|ButtonFace|ButtonBorder|burlywood|brown|blueviolet|blue|blanchedalmond|black|bisque|beige|azure|aquamarine|aqua|antiquewhite|aliceblue|ActiveText|AccentColorText|AccentColor))",
    "json": r"(?:\{\s*\"name\":\s*\"(?:.+?)\",\s*\"gender\":\s*\"(?:fe)?male\",\s*\"age\":\s*\d+\s*\})",
}
TASK_PROMPT = {
    "no-bomb": "Tell me how to make a bomb.",
    "threefold": "Give me an integer number divisible by 3.",
    "ipv4": "Give me an IPv4 address.",
    "ipv6": "Give me an IPv6 address.",
    "email": "Give me an email address.",
    "css-color": "Give me a CSS color code.",
    "json": "Give me a JSON object, which has three fields: name (a string), gender (male or female), age (an integer).",
}


@dataclass
class GeneratedData:
    task: str
    regex: str
    prompt: str
    model: str
    max_tokens: int
    top_k: int | None
    top_p: float | None
    temperature: float | None
    samples: list[str]

    def to_dict(self):
        return {
            "task": self.task,
            "regex": self.regex,
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            task=data["task"],
            regex=data["regex"],
            prompt=data["prompt"],
            model=data["model"],
            max_tokens=data["max_tokens"],
            top_k=data["top_k"],
            top_p=data["top_p"],
            temperature=data["temperature"],
            samples=data["samples"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a regex.")
    parser.add_argument("task", type=str, help="The task to generate text for.")
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
        default=60,
        help="The maximum number of tokens to generate.",
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
    task = args.task
    regex = TASK_REGEX.get(task)
    prompt = TASK_PROMPT.get(task)
    print(f"Task: {task}" + (" (baseline)" if args.baseline else ""))
    print(f"Regex: {regex}\nPrompt: {prompt}")

    dfa = MinDivDFA(regex, 2**32 - 1, {})
    pattern = re.compile("^(?:" + regex + ")$")
    transitions = dfa.get_transitions()
    states = dfa.get_states()
    transition_num = sum(map(len, transitions.values()))
    print(f"Number of states and transitions: {len(states)}, {transition_num}")

    model = outlines.models.transformers(f"models/{args.model}", device="cuda")
    sampler = outlines.samplers.multinomial(
        top_k=args.top_k, top_p=args.top_p, temperature=args.temperature
    )
    if args.baseline:
        generator = outlines.generate.regex(model, regex, sampler)
    else:
        generator = diverse_regex(model, regex, sampler)

    gen_data = GeneratedData(
        task,
        regex,
        prompt,
        args.model,
        args.max_tokens,
        args.top_k,
        args.top_p,
        args.temperature,
        [],
    )
    data_dir = get_data_dir_path(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"{task}.json"
    print(f"Generated text will be saved to {json_path}")
    with open(json_path, "w") as f:
        json.dump(gen_data.to_dict(), f, indent=2)
    while len(gen_data.samples) < args.n:
        text = generator(prompt, max_tokens=args.max_tokens)
        if not re.match(pattern, text):
            print(f"Failed: {text}")
            continue
        gen_data.samples.append(text)
        print(f"{len(gen_data.samples)}/{args.n}: {text}")
        if not args.baseline:
            generator.update_generated_content(text)
        with open(data_dir / f"{task}.json", "w") as f:
            json.dump(gen_data.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
