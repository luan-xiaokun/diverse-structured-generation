"""Read-only token-boundary diagnostics; never selects or commits tokens."""

from __future__ import annotations

from collections import Counter


def trajectory(dfa, ids, eos_token_id):
    """Replay one attempt without mutating global or penalty counters."""
    q = dfa.get_initial_state()
    entries = int(dfa.is_final_state(q))
    first = 0 if entries else None
    exits = 0
    opportunities, stops = Counter(), Counter()
    for index, token in enumerate(ids, 1):
        accepting = q != -1 and dfa.is_final_state(q)
        if accepting:
            bucket = str(min(entries, 4))
            opportunities[bucket] += 1
            if token == eos_token_id:
                stops[bucket] += 1
        nxt = dfa.get_next_token_state(q, token)
        if nxt is None:
            raise ValueError(f"invalid token {token} at step {index}")
        final = nxt != -1 and dfa.is_final_state(nxt)
        if token != eos_token_id:
            if not accepting and final:
                entries += 1
                if first is None:
                    first = index
            exits += int(accepting and not final)
        q = nxt
    return {
        "first_accepting_token": first,
        "accepting_entries": entries,
        "accepting_exits": exits,
        "final_accepting": q != -1 and dfa.is_final_state(q),
        "ever_accepting": first is not None,
        "generated_tokens": len(ids),
        "termination": "eos" if ids and ids[-1] == eos_token_id else "budget",
        "eos_opportunities": dict(opportunities),
        "eos_selected": dict(stops),
    }


class StepProbeProcessor:
    """Single-row observer; finalize the last chosen token before counter commit.

    A pending step keeps only one step's candidate scores. Output is restricted
    to the chosen token and guide EOS when either endpoint is accepting.
    Logits are at the steering processor boundary (before temperature/top-k/p).
    """

    def __init__(self, processor, emit):
        self.processor = processor
        self.emit = emit
        self.pending = None
        self.step = 0

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def reset(self):
        self.pending = None
        self.step = 0
        self.processor.reset()

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError("step probes require batch=1")
        if self.pending and input_ids.shape[1] > self.pending["length"]:
            self.finish(int(input_ids[0, self.pending["length"]]))
        raw = scores.detach().clone()
        result = self.processor(input_ids, scores)
        p = self.processor
        q = p.sequence_states[0]
        if q == -1:
            self.pending = None
            return result
        ids, bases = p.dfa.compute_adjustments_with_estimator(
            q, p.no_reward, p.no_penalty, 0.0, p.path_frequency_estimator
        )
        allowed = raw[0, ids]
        self.pending = {
            "length": input_ids.shape[1],
            "q": q,
            "entries": p.accepting_entries[0],
            "raw": raw[0, ids].cpu().tolist(),
            "raw_eos": float(raw[0, p.guide.eos_token_id]),
            "final": result[0, ids].detach().cpu().tolist(),
            "ids": ids,
            "base": bases,
            "range": float(allowed.max() - allowed.min()),
        }
        return result

    def finish(self, token):
        if self.pending is None:
            return
        row, self.pending = self.pending, None
        self.step += 1
        p = self.processor
        q = row["q"]
        nxt = p.dfa.get_next_token_state(q, token)
        if nxt is None:
            raise ValueError("probe selected a masked token")
        if not (p.dfa.is_final_state(q) or p.dfa.is_final_state(nxt)):
            return

        def candidate(t):
            if t not in row["ids"]:
                # Preserve the raw EOS logit on the transition into F, even
                # though EOS was masked at the decision's starting state.
                return {
                    "raw_logit": row["raw_eos"],
                    "final_logit": None,
                    "diversity_base": None,
                    "completion_signal": None,
                    "completion_bonus": None,
                    "masked": True,
                }
            i = row["ids"].index(t)
            d = p.dfa.get_distance_to_accepting(q)
            nd = p.dfa.get_distance_to_accepting(p.dfa.get_next_token_state(q, t))
            signal = max(0, d - nd) / (1 + d)
            if t == p.guide.eos_token_id and p.dfa.is_final_state(q):
                if p.eos_completion_mode == "entry_count":
                    signal = row["entries"]
                elif p.eos_completion_mode == "constant":
                    signal = 1
            return {
                "raw_logit": row["raw"][i],
                "final_logit": row["final"][i],
                "diversity_base": row["base"][i],
                "completion_signal": signal,
                "completion_bonus": p.completion_bias_eta * signal,
            }

        self.emit(
            {
                "step": self.step,
                "q": q,
                "q_next": nxt,
                "accepting_entries": row["entries"],
                "selected_token": token,
                "range": row["range"],
                "gamma": p.gamma,
                "no_range_scaling": p.no_range_scaling,
                "eos_completion_mode": p.eos_completion_mode,
                "selected": candidate(token),
                "eos": candidate(p.guide.eos_token_id),
            }
        )
