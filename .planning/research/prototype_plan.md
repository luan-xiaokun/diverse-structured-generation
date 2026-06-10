# Prototype Plan

**Date:** 2026-05-19

## One-Week Minimal Prototype

Goal: produce evidence for backend selection, not a complete backend.

Deliverables:
- `research_summary.md`
- `tool_matrix.md`
- `architecture_proposal.md`
- `prototype_plan.md`
- `links.md`
- small benchmark script for llguidance and/or XGrammar if dependencies are available
- Lark prototype showing accepted terminals to tokenizer-trie mask construction

Tasks:
1. Install or vendor-free test llguidance and XGrammar in isolated optional dependency groups.
2. Write one shared JSON-like grammar in each accepted grammar syntax.
3. Load a Hugging Face tokenizer already compatible with the existing project.
4. For a fixed prefix sequence, collect:
   - compile time;
   - mask compute time;
   - allowed token count;
   - token advance time;
   - visible parser stats/state dump fields;
   - cache stats if exposed.
5. Implement a small Python tokenizer trie.
6. Use Lark `InteractiveParser.accepts()`/`choices()` to map accepted terminals to trie traversal for a toy LALR grammar.
7. Write a decision note: llguidance primary, XGrammar backup, or reverse.

Exit criteria:
- At least one purpose-built backend produces correct token masks for the toy grammar.
- Prototype records per-step mask timing and allowed-token count.
- Observability gaps are listed with exact modules to inspect or fork.

## Two-to-Four Week Research Version

Goal: implement a research-grade CFG constrained decoding backend skeleton.

Deliverables:
- Rust crate/module sketch for `GrammarRuntime`.
- Python binding or Python wrapper around selected baseline.
- Tokenizer trie with byte-level traversal and bitset output.
- Counter schema and JSON export.
- LR/GLR table extraction spike using RustyLR and/or Rustemo.
- Tests for token boundary cases.

Tasks:
1. Select primary baseline: llguidance unless XGrammar wins on integration/performance evidence.
2. Add wrapper-level instrumentation:
   - mask time;
   - allowed token histogram;
   - advance count;
   - cache visible stats;
   - state dump sampling.
3. Build Lark/parglare prototype for algorithm validation.
4. Generate a toy grammar with RustyLR.
5. Inspect generated parser tables and context stack.
6. Implement deterministic LR `Config`:
   - persistent stack;
   - shift/reduce/goto;
   - closure;
   - tokenizer trie traversal.
7. Add GLR/GSS only after deterministic LR works.
8. Compare outputs against llguidance/XGrammar for a grammar subset.

Exit criteria:
- Deterministic LR runtime computes token masks for at least one grammar with terminals that cross tokenizer boundaries.
- Counters distinguish physical visits, logical paths, and unique configs.
- Baseline backend and research backend agree on allowed-token masks for the shared test set.

## Long-Term Engineering Version

Goal: production-ready backend option for local package and serving integration.

Deliverables:
- Rust backend crate integrated with existing PyO3 package structure or a sibling crate.
- Stable Python API:
  - compile grammar;
  - create session/config;
  - compute mask;
  - advance token;
  - read stats;
  - reset/clone/fork config.
- Hugging Face `LogitsProcessor` integration.
- Optional adapters for vLLM/SGLang-style loops.
- Benchmark suite and regression corpus.
- Documentation for grammar restrictions and instrumentation semantics.

Tasks:
1. Harden grammar normalization and diagnostics.
2. Implement batching and cache policy.
3. Add memory bounds and parser limits.
4. Add GLR ambiguity controls.
5. Add JSON/regex/EBNF frontend compatibility where practical.
6. Add exhaustive tests for:
   - nullable rules;
   - left recursion;
   - ambiguous grammar;
   - skipped whitespace;
   - tokens crossing terminals;
   - EOS/accepting behavior;
   - invalid-prefix recovery.
7. Benchmark against llguidance/XGrammar on realistic tokenizers.
8. Publish artifacts:
   - runtime metrics;
   - mask latency histograms;
   - state/transition/path counter plots;
   - engineering decision report.

Exit criteria:
- Local generation loop can enforce CFG constraints with measured overhead.
- Instrumentation is detailed enough for research claims.
- Integration path is clear for either Hugging Face-only use or serving engines.

## Risk Burn-Down Order

1. Tokenizer boundary correctness.
2. Visibility of baseline stats and state.
3. LR table extraction from RustyLR/Rustemo.
4. Reduce/epsilon closure correctness.
5. GLR path explosion and counter semantics.
6. Cache invalidation and memory bounds.
7. Serving integration.

## Minimal Code Experiment

First experiment name: `cfg_mask_backend_probe`.

Inputs:
- grammar: small JSON-like CFG;
- tokenizer: one HF tokenizer already supported by local environment;
- prefixes: empty, `{`, `{"`, `{"a"`, `{"a":`, `{"a":1`;
- candidate backends: llguidance, XGrammar, Lark prototype.

Outputs:
- CSV or JSON rows:

```text
backend, prefix, step, allowed_token_count, mask_time_ns, advance_time_ns,
cache_hit, cache_miss, parser_state_summary, error
```

Success:
- all backends agree on obvious valid/invalid next structural tokens;
- at least one backend exposes enough stats to decide instrumentation patch points.
