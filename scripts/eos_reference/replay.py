"""Replay historical JSON attempts with an explicitly recorded native build.

This diagnoses old trajectories, not sampling under the new dynamic EOS bonus.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = None
SOURCE = Path(__file__).resolve().parents[2]
NATIVE = None
PROVENANCE = None


def add_arguments(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--native-module", type=Path)
    parser.add_argument("--output", type=Path, required=True)


def configure(args):
    global ROOT, SOURCE, NATIVE, PROVENANCE
    ROOT = args.data_root.resolve()
    SOURCE = args.source_root.resolve()
    NATIVE = args.native_module
    report = {
        "source_root": str(SOURCE),
        "commit": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "-C", str(SOURCE), "status", "--short"], text=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE = args.output.with_suffix(".source.json")
    PROVENANCE.write_text(json.dumps(report, indent=2) + "\n")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup():
    vocabmod = load("runtime_vocab", SOURCE / "src/diverse_guide/vocab.py")
    tj = json.loads(
        (
            ROOT
            / "group1-transfer-20260916/evidence/models/Qwen--Qwen2.5-1.5B-Instruct"
            / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306/tokenizer.json"
        ).read_text()
    )
    inverse = vocabmod.gpt2_unicode_to_bytes()
    raw_vocab = dict(tj["model"]["vocab"])
    raw_vocab.update({t["content"]: t["id"] for t in tj["added_tokens"]})
    special = {t["content"] for t in tj["added_tokens"] if t["special"]}

    class Tokenizer:
        eos_token_id = 151645
        all_special_tokens = special

        def get_vocab(self):
            return raw_vocab

        def convert_tokens_to_string(self, tokens):
            return bytes(inverse[c] for t in tokens for c in t).decode(
                "utf8", errors="replace"
            )

    vocab = vocabmod.build_token_id_map(Tokenizer())
    raw_bytes = {
        i: bytes(inverse[c] for c in t)
        for t, i in raw_vocab.items()
        if t not in special
    }
    if NATIVE is None:
        import regex_dfa_guide.regex_dfa_guide_rs as mod
    else:
        mod = load("regex_dfa_guide_rs", NATIVE)
    native_info = {
        "native_module": mod.__file__,
        "native_sha256": hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
    }
    report = json.loads(PROVENANCE.read_text())
    report.update(native_info)
    PROVENANCE.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(native_info), file=sys.stderr)
    manifest = json.loads(
        (ROOT / "results/group2-20260919-v1/manifest.json").read_text()
    )
    regex = next(t["regex"] for t in manifest["tasks"] if t["grammar"] == "json")
    dfa = mod.DiverseGuideDFA("(?:" + regex + ")$", 151645, vocab)
    return dfa, raw_bytes


def journals(gamma, eta, seed):
    if gamma == 0.3 and eta == 1:
        p = (
            ROOT
            / "results/group1-20260916-v2/production/tasks"
            / f"qwen2.5-1.5b-instruct__json__diverse_guide_plus__{seed}"
        )
    else:
        tag = f"gamma-{gamma:g}__eta-{eta:g}"
        p = (
            ROOT
            / f"results/group2-20260919-v1/parameters/gamma-{gamma:g}_eta-{eta:g}"
            / "production/tasks"
            / f"qwen2.5-1.5b-instruct__json__diverse_guide_plus__{seed}__{tag}"
        )
    paths = list(p.glob("attempt-*/attempts.jsonl"))
    assert len(paths) == 1, (p, paths)
    return paths[0]


def replay(dfa, raw_bytes, gamma, eta, seed):
    stats = collections.Counter()
    examples = []
    path = journals(gamma, eta, seed)
    for line in path.read_text().split("\n"):
        if not line:
            continue
        r = json.loads(line)
        if r["event"] != "finish":
            continue
        ids = r["token_ids"][0]
        decoded = b"".join(raw_bytes.get(i, b"") for i in ids).decode(
            "utf8", errors="replace"
        )
        assert decoded == r["text"], (r["attempt"], "decode mismatch")
        q = dfa.get_initial_state()
        seen_final = False
        first_final = None
        first_exit = None
        states = [q]
        for token_index, tok in enumerate(ids):
            was_final = dfa.is_final_state(q)
            next_q = dfa.get_next_token_state(q, tok)
            assert next_q is not None, (r["attempt"], q, tok)
            q = next_q
            seen_final |= dfa.is_final_state(q)
            if dfa.is_final_state(q) and first_final is None:
                first_final = token_index + 1
            if was_final and not dfa.is_final_state(q):
                stats["accepting_exit_steps"] += 1
                if first_exit is None:
                    first_exit = token_index
            states.append(q)
        final = dfa.is_final_state(q)
        assert final == r["valid"], (r["attempt"], "native/validator disagreement")
        stats["attempts"] += 1
        stats["accepted" if final else "rejected"] += 1
        if first_final is not None:
            stats["first_accepting_token_sum"] += first_final
            stats["ever_accepting"] += 1
        if not final:
            stats[
                "rejected_at_budget" if len(ids) == 72 else "rejected_before_budget"
            ] += 1
            stats[
                "rejected_after_visiting_accepting"
                if seen_final
                else "rejected_without_visiting_accepting"
            ] += 1
            stats["final_distance_" + str(dfa.get_distance_to_accepting(q))] += 1
            if len(examples) < 2:
                prefix = (
                    b"".join(raw_bytes.get(i, b"") for i in ids[:first_final]).decode(
                        "utf8", errors="replace"
                    )
                    if first_final
                    else None
                )
                examples.append(
                    {
                        "attempt": r["attempt"],
                        "text": r["text"],
                        "tokens": len(ids),
                        "visited_accepting": seen_final,
                        "first_final_token": first_final,
                        "first_valid_prefix": prefix,
                        "first_exit_token": raw_bytes.get(ids[first_exit], b"").decode(
                            "utf8", errors="replace"
                        )
                        if first_exit is not None
                        else None,
                        "states": states,
                        "distances": [dfa.get_distance_to_accepting(x) for x in states],
                    }
                )
    assert stats["accepted"] == 500
    return {
        "gamma": gamma,
        "eta": eta,
        "seed": seed,
        "journal": str(path.relative_to(ROOT)),
        **stats,
        "reject_rate": stats["rejected"] / stats["attempts"],
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--assert-no-reversal", action="store_true")
    parser.add_argument("--all-gammas", action="store_true")
    args = parser.parse_args()
    configure(args)
    dfa, raw_bytes = setup()
    results = [
        replay(dfa, raw_bytes, gamma, eta, seed)
        for gamma in ([0.1, 0.3, 0.5] if args.all_gammas else [0.3])
        for eta in [0, 0.5, 1, 2]
        for seed in range(42, 47)
    ]
    dest = args.output
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for gamma in sorted({r["gamma"] for r in results}):
        rates = {}
        for eta in [0, 0.5, 1, 2]:
            rs = [r for r in results if r["eta"] == eta and r["gamma"] == gamma]
            rates[eta] = sum(r["reject_rate"] for r in rs) / 5
            print(
                f"gamma={gamma} eta={eta}: reject={rates[eta] * 100:.3f}%, "
                f"rejected={sum(r['rejected'] for r in rs)}, "
                f"after_accepting={sum(r.get('rejected_after_visiting_accepting', 0) for r in rs)}"  # noqa: E501
            )
        if args.assert_no_reversal:
            assert rates[2] <= rates[0], (
                "REPRODUCED: eta=2 has higher JSON rejection than eta=0 "
                "in native-DFA replay"
            )
