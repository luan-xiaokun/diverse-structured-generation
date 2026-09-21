"""Probe distance/bonus invariants and replay actual counter state; no model logits."""

import argparse
import json

from replay import add_arguments, configure, journals, setup

parser = argparse.ArgumentParser()
add_arguments(parser)
parser.add_argument("--attempt", type=int, default=1403)
args = parser.parse_args()
configure(args)
dfa, raw_bytes = setup()
tested = 0
max_formula_error = 0
final_states = []
for q in dfa.get_states():
    try:
        d = dfa.get_distance_to_accepting(q)
        ids, a0 = dfa.compute_adjustments_with_estimator(
            q, False, False, 0.0, "harmonic"
        )
    except ValueError:
        continue
    ids2, a2 = dfa.compute_adjustments_with_estimator(q, False, False, 2.0, "harmonic")
    cached = dfa.compute_adjustment_values_with_estimator(
        q, False, False, 2.0, "harmonic"
    )
    assert ids == ids2
    ds = [dfa.get_distance_to_accepting(dfa.get_next_token_state(q, t)) for t in ids]
    assert d == (0 if dfa.is_final_state(q) else 1 + min(ds)), (q, d, min(ds))
    for t, n, x, y, z in zip(ids, ds, a0, a2, cached):
        expected = 2 * max(0, d - n) / (1 + d)
        err = abs(y - x - expected)
        max_formula_error = max(max_formula_error, err)
        assert err < 1e-5, (q, t, err)
        assert abs(y - z) < 1e-5, (q, t, "cached order mismatch")
        tested += 1
    if d == 0:
        final_states.append(
            {
                "state": q,
                "allowed_tokens": len(ids),
                "outgoing_to_nonfinal": sum(n > 0 for n in ds),
                "eos_allowed": 151645 in ids,
                "max_completion_bonus": max(y - x for x, y in zip(a0, a2)),
            }
        )

# Rebuild cross-sample counters in journal order; local counters reset each call.
# Inspect actual choices around the first rejected run that visited acceptance.
probe_steps = []
target = 42
dfa.reset_path_counter()
for line in journals(0.3, 2, target).read_text().split("\n"):
    if not line:
        continue
    r = json.loads(line)
    if r["event"] != "finish":
        continue
    dfa.reset_local_state_counter()
    q = dfa.get_initial_state()
    ids = r["token_ids"][0]
    states = [q]
    for t in ids:
        q = dfa.get_next_token_state(q, t)
        states.append(q)
    is_target = r["attempt"] == args.attempt
    if is_target:
        assert not r["valid"] and any(dfa.is_final_state(q) for q in states)
    q = states[0]
    for i, t in enumerate(ids):
        nq = states[i + 1]
        d = dfa.get_distance_to_accepting(q)
        nd = dfa.get_distance_to_accepting(nq)
        if is_target and (d == 0 or nd == 0):
            tids, a0 = dfa.compute_adjustments_with_estimator(
                q, False, False, 0.0, "harmonic"
            )
            _, a2 = dfa.compute_adjustments_with_estimator(
                q, False, False, 2.0, "harmonic"
            )
            j = tids.index(t)
            row = {
                "attempt": r["attempt"],
                "step": i + 1,
                "token": raw_bytes.get(t, b"<EOS>").decode("utf8", errors="replace"),
                "distance": d,
                "next_distance": nd,
                "base_adjustment": a0[j],
                "eta2_adjustment": a2[j],
                "completion_bonus": a2[j] - a0[j],
            }
            if 151645 in tids:
                k = tids.index(151645)
                row.update(eos_base=a0[k], eos_eta2=a2[k])
            probe_steps.append(row)
        dfa.update_local_state_counter(q, t)
        q = nq
    if r["valid"]:
        dfa.commit_valid_token_path(ids)
    if is_target:
        break

report = {
    "tested_token_records": tested,
    "max_formula_error": max_formula_error,
    "accepting_states": final_states,
    "actual_counter_replay_steps": probe_steps,
    "limitations": (
        "Uses the explicitly selected/rebuilt native module. "
        "No recorded model logits: these are legacy steering "
        "adjustments, not a counterfactual new generation."
    ),
}
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
