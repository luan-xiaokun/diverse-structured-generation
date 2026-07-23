# Technology Stack

**Project:** AIAP Multi-Obligation Witness Generation  
**Researched:** 2026-07-23  
**Confidence:** MEDIUM-HIGH — library capabilities and reproducibility claims are verified against official documentation; the recommended multi-obligation search is an engineering design derived from the project DFA API and needs empirical validation.

## Recommended Stack

### Core implementation

| Technology | Version policy | Purpose | Why |
|---|---:|---|---|
| Python | Existing project Python 3.12+ | Experiment orchestration, algorithms, analysis | Keep the prototype beside `aiap/cov_test.py` and reuse its Python/Rust boundary. |
| `regex_dfa_guide` + `diverse_guide` | Parent-worktree pinned revision | Authoritative DFA/token transitions, acceptance, and coverage trace | It is the existing correctness substrate. Do not reimplement token-to-byte transition semantics in a second automaton. |
| Direct `heapq` / `collections.deque` | stdlib | Search frontier | The current BFS is already direct and preserves per-token byte traces. It is more transparent than introducing a graph framework into the hot loop. |
| `numpy` | Lock with `uv.lock` | Coverage bitsets, score tables, aggregate statistics | Compact boolean/`uint64` coverage masks and deterministic aggregation. |
| `pytest` | Existing dev dependency | Algorithm and invariant tests | Add trace/acceptance/claimed-coverage properties plus deterministic golden fixtures. |

### Graph-based multi-obligation witness search

**Use a bounded label-setting search over product labels** `(dfa_state, covered_obligations_mask)`. A label records: token prefix, byte-level trace, covered-obligation bitset, length, and optional LLM score. Expand only DFA-allowed token IDs; update the mask by scanning *all* byte transitions inside that token. Keep a Pareto frontier per DFA state, discarding a label when another has a superset mask at no worse symbolic cost (and, when applicable, no worse naturalness bound).

This directly generalizes the one-obligation two-BFS baseline, retains mechanical acceptance/coverage validation, and avoids the full `|Q| × 2^|O|` product graph. Bound each state frontier and candidate goal group (`K` obligations), then sweep `K`, beam/frontier cap, and score weights in ablations. Use reverse BFS from accepting states once to precompute symbolic distance-to-acceptance; use it as an admissible pruning/lower-bound component. This is the research method, not merely an implementation detail.

Use ordinary BFS when all symbolic edges have unit cost; if an explicit non-negative symbolic cost is introduced (e.g., token count plus a non-negative coverage penalty), use Dijkstra. NetworkX officially documents BFS for unweighted graphs and Dijkstra for non-negative weights, but **do not put NetworkX in the main search path**: its graph nodes/edges cannot conveniently retain this project's byte-in-token trace and per-label dominance logic. It is appropriate only as a small-instance oracle/visualization helper. [NetworkX shortest paths](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/index.html) — HIGH confidence (official source).

### Constrained LLM decoding and naturalness scoring

| Technology | Version policy | Purpose | When to use |
|---|---:|---|---|
| `torch` | Keep parent pin: `2.8.0`; record CUDA build | Batched model inference and reproducibility controls | Required only for LLM-guided conditions. |
| `transformers` | Keep parent lock: `5.3.0`; pin model revision | `AutoTokenizer`, causal-LM scores, beam generation | Use its `generate` API as an adapter, not as the DFA authority. |
| Hugging Face Hub | Pin immutable commit SHA, cache snapshot | Model/tokenizer artifact identity | Required for every reported LLM result. |
| Optional `outlines[transformers]` | Existing optional lock only | Comparator baseline | Use it only as a separately reported constrained-generation baseline; do not let it replace the project DFA's trace validator. |

**Use DFA-masked deterministic beam search, then rerank**, initially. For each live beam, map its token prefix to the DFA state and admit exactly `dfa.get_allowed_token_ids(state)` through a custom logits processor (or the documented `prefix_allowed_tokens_fn` adapter). The Transformers API states that `prefix_allowed_tokens_fn` constrains every beam step; `GenerationConfig` controls beam parameters, and it recommends `renormalize_logits=True` when logits processors alter scores. [Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) — HIGH confidence (official source).

Score candidates with a logged, additive objective such as `log P_LLM(tokens) - λ_len·length + λ_cov·new_obligations`; normalize consistently (e.g., mean log-probability per generated token) before comparing different lengths. Initially treat LLM likelihood as the naturalness proxy; add external fluency/judge models only as explicitly named, separately reproducible evaluation conditions. After every decode, independently replay the candidate through the DFA and recompute the byte trace and coverage set. **Do not use rejection sampling, post-hoc repair, unconstrained text then validation, or `trust_remote_code=True` by default**: they either waste generation budget, weaken attribution of the formal guarantee, or add avoidable supply-chain variability.

### Test-suite selection

**Use deterministic greedy weighted set cover as the primary selector.** The universe is obligations; each existing/new accepted witness maps to the obligations in its independently verified trace. At each round choose the witness maximizing newly covered obligation weight divided by witness cost (default cost = one test; add length/runtime cost in sensitivity analysis). Break equal scores by stable candidate ID. Stop at target coverage or exhaustion; report the full coverage-versus-suite-size curve.

Use **OR-Tools CP-SAT** as an *exact small-pool oracle* and diagnostic baseline: binary `x_w` per witness, `sum(x_w for w covering o) >= 1` for each target obligation, and minimize `sum(cost_w*x_w)`. The official Python documentation supports declarative variables, constraints, objective, and solving; CP-SAT works over integers, which matches binary selection. Impose a recorded time limit and seed, and report `OPTIMAL` vs `FEASIBLE` vs timeout. Do not make CP-SAT the default for full candidate pools, because global exactness is expressly out of scope and search cost can dominate the experiment. [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver) — HIGH confidence (official source).

### Reproducibility infrastructure

| Item | Required record | Rationale |
|---|---|---|
| Environment | `uv.lock`, Python/Rust/package versions, OS, CPU/GPU, CUDA/driver | The parent already has a lockfile; extend it rather than create an unpinned `aiap` environment. |
| Inputs | Case-study JSON checksum, regex, sample-set checksum, obligation definition and threshold | Makes coverage claims replayable. |
| Model | Hub repo ID **and immutable commit SHA**, tokenizer revision/hash, dtype/device | Hub's `revision` can select a commit; a moving `main` name cannot reproduce a paper result. [Hub CLI revisions](https://huggingface.co/docs/huggingface_hub/en/guides/cli) |
| Decoding | Prompt/template hash, model mode, beam size, max tokens, logits/naturalness formula and weights, EOS policy | These choices alter both witness distribution and cost. |
| Randomness | Python, NumPy, Torch CPU/CUDA seeds; deterministic-algorithm setting; per-run seed list | PyTorch documents that exact reproducibility is not guaranteed across releases/platforms, but prescribes these controls. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness) |
| Outputs | JSONL per candidate: token IDs, decoded text, full DFA trace, verified coverage mask, raw/normalized LLM scores, timing, selection decision | Enables independent replay and separates generation failure from selection failure. |
| Analysis | One command that reads saved artifacts; seed-level results with mean/median and dispersion; fixed plot scripts | Prevents accidental recomputation under changed models or parameters. |

For symbolic-only runs, the search must be deterministic without a seed. For LLM runs, run a fixed multi-seed protocol; do not claim bitwise reproducibility across hardware. Enable `torch.use_deterministic_algorithms(True)` for a reproducibility mode and document any error/slowdown, since PyTorch explicitly notes the performance trade-off and platform limits. Pinning model revisions is non-negotiable. [PyTorch deterministic algorithms](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)

## Alternatives Considered

| Category | Recommended | Alternative | Why not as default |
|---|---|---|---|
| Multi-objective search | Bounded product-label search + dominance | Full subset-product shortest path | Exact but exponential in obligations; retain only as a tiny-instance oracle. |
| Graph library | Direct DFA traversal | NetworkX | Adds an unnecessary representation layer and obscures byte-inside-token accounting; use only as an oracle. |
| Naturalness | DFA-masked beam + logged LLM likelihood/rerank | Free generation then filter/repair | Loses valid search budget and no longer demonstrates formal control during generation. |
| Constraint adapter | Project DFA adapter/custom processor | Generic grammar package as authority | Existing DFA implementation is the accepted semantics and can emit the trace needed for coverage. |
| Suite selection | Greedy weighted set cover | Exact solver for all pools | Does not meet the prototype's bounded, scalable research scope. |
| Evaluation | Saved candidate artifacts + replay | Only aggregate metrics | Cannot audit invalid witnesses, coverage attribution, or selection decisions. |

## Installation / Locking Plan

Keep the parent `uv` environment and add experimental-only dependencies deliberately:

```bash
uv add --group aiap ortools
# NetworkX is already transitive through torch; add it explicitly only if an oracle script uses it.
uv add --group aiap networkx
uv lock
```

Do not add a database, web service, vector store, training framework, or distributed workflow engine. They do not advance the submission's experimental question and enlarge the reproducibility surface.

## Implementation Sequence Implied by This Stack

1. Extract a single pure DFA-trace/coverage layer and verify it against `cov_test.py` fixtures.
2. Implement one-obligation BFS as frozen baseline, then bounded product-label search with symbolic-only scores and trace replay.
3. Add greedy set cover and exact small-pool CP-SAT oracle before LLM guidance, so coverage gains are separable from naturalness gains.
4. Add DFA-masked beam/reranking behind a model adapter, plus full run manifests and seed protocol.

## Sources

- [NetworkX shortest-path algorithms](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/index.html) — official, HIGH.
- [Hugging Face Transformers generation API](https://huggingface.co/docs/transformers/main_classes/text_generation) — official, HIGH.
- [Hugging Face Hub revision-pinned downloads](https://huggingface.co/docs/huggingface_hub/en/guides/cli) — official, HIGH.
- [Google OR-Tools CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver) — official, HIGH.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness) — official, HIGH.
- Existing local evidence: `aiap/cov_test.py`, `aiap/.planning/codebase/STACK.md`, and root `pyproject.toml`/`uv.lock` — HIGH for the current repository state.

## Confidence and Open Validation

- **HIGH:** Direct DFA replay, Transformers constraint hook, OR-Tools modeling role, and artifact/seed/revision logging are supported by official documentation and the current repository layout.
- **MEDIUM:** The exact dominance ordering and the best frontier/beam caps are project-specific hypotheses. Validate on small DFA instances against exhaustive subset-product search, then pre-register the caps/ablations used in the paper.
- **MEDIUM:** LLM likelihood is a practical *proxy*, not a demonstrated human-naturalness measure. The evaluation phase must define a human or independently specified automatic quality protocol before making naturalness claims.
