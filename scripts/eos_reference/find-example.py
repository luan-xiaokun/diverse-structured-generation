import argparse
import json

from replay import add_arguments, configure, journals, setup

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
configure(args)
dfa, raw = setup()
found = []
for seed in range(42, 47):
    for line in journals(0.3, 2, seed).read_text().split("\n"):
        if not line:
            continue
        r = json.loads(line)
        if r["event"] != "finish" or r["valid"]:
            continue
        q = dfa.get_initial_state()
        ids = r["token_ids"][0]
        first = None
        for i, t in enumerate(ids):
            q = dfa.get_next_token_state(q, t)
            if dfa.is_final_state(q):
                first = i + 1
                break
        if first is None:
            continue
        prefix = b"".join(raw.get(t, b"") for t in ids[:first]).decode(
            "utf8", errors="replace"
        )
        try:
            obj = json.loads(prefix)
        except ValueError:
            continue
        if not isinstance(obj, dict) or set(obj) != {"name", "gender", "age"}:
            continue
        if (
            not isinstance(obj["name"], str)
            or obj["gender"] not in ["male", "female"]
            or not isinstance(obj["age"], int)
        ):
            continue
        if first >= len(ids):
            continue
        nq = dfa.get_next_token_state(q, ids[first])
        found.append(
            {
                "seed": seed,
                "attempt": r["attempt"],
                "complete_after_tokens": first,
                "prefix": prefix,
                "next_token_id": ids[first],
                "next_token": raw.get(ids[first], b"").decode("utf8", errors="replace"),
                "distance_before": dfa.get_distance_to_accepting(q),
                "distance_after": dfa.get_distance_to_accepting(nq),
                "text": r["text"],
                "total_tokens": len(ids),
            }
        )
        if len(found) >= 3:
            break
    if len(found) >= 3:
        break
print(json.dumps(found, ensure_ascii=False, indent=2))
args.output.write_text(json.dumps(found, ensure_ascii=False, indent=2) + "\n")
