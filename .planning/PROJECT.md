# LLM CFG Structured Decoding Backend

## What This Is

This project evaluates and then prototypes a grammar runtime backend for LLM structured output and constrained decoding. The target runtime accepts a CFG or constrained CFG family, advances incrementally during generation, computes token masks against an LLM tokenizer vocabulary, and exposes enough instrumentation to study parser states, transitions, stacks, paths, and cache behavior.

It builds on the existing `diverse-dfa-gen` artifact, which already has a Python/Rust constrained decoding path for regex/DFA masks, Hugging Face `LogitsProcessor` integration, tokenizer vocabulary conversion, DFA coverage metrics, and reproducibility scripts.

## Core Value

Compute correct token viability for CFG-constrained LLM decoding while making parser/runtime behavior measurable enough for research.

## Requirements

### Validated

- ✓ Existing regex/DFA constrained decoding through a Rust backend and Python Hugging Face integration — existing
- ✓ Existing tokenizer vocabulary conversion and token-level transition compilation for regex constraints — existing
- ✓ Existing runtime and diversity evaluation scripts, metrics, tests, and reproducibility artifacts — existing
- ✓ Existing Python 3.12, Rust, PyO3, uv, pytest, and CI development stack — existing

### Active

- [ ] Research and compare llguidance, XGrammar, RustyLR, Rustemo, Menhir, Lark, and parglare for LLM CFG structured decoding.
- [ ] Produce a capability matrix covering grammar support, incremental runtime APIs, token mask APIs, tokenizer integration, state/stack/table exposure, instrumentation feasibility, performance expectations, risks, and recommendation.
- [ ] Propose a backend architecture with a `GrammarRuntime` trait, token viability algorithm, tokenizer trie design, reduce/epsilon closure handling, cache strategy, and counter model.
- [ ] Define a phased implementation plan: high-performance baseline, Python viability prototype, and Rust LR/GLR instrumented backend.
- [ ] Run the first minimal experiment against llguidance or XGrammar using a small grammar and Hugging Face tokenizer to measure mask time, allowed-token counts, and observability gaps.

### Out of Scope

- Full parser-generator implementation from textbook CFG to naive NPDA — too much risk for v1 and duplicates mature parser-generator work.
- Building a complete production serving stack before the backend selection is validated — integration should follow runtime evidence.
- Prioritizing parse-tree construction as the primary deliverable — the core deliverable is constrained decoding state, token viability, and instrumentation.
- Replacing the existing regex/DFA backend immediately — CFG work should coexist and reuse tested vocabulary/logits patterns where possible.

## Context

The existing codebase is a research software artifact for regex-constrained diverse generation. It exposes `DiverseGuide`, `diverse_regex`, `baseline_regex`, and logits processors in `src/diverse_guide/guide_rust.py`, while the Rust backend in `regex_dfa_guide/src/diverse_guide_dfa.rs` builds minimized byte DFAs, token transitions, allowed-token tables, and path/counter metrics.

The new research direction generalizes from regular languages to context-free grammars. The central runtime shape is not a traditional parse-tree API, but:

```text
config -> allowed_tokens(vocab)
config + token -> next_config / next_config_set
```

Candidate `config` representations include LR stack, GLR graph-structured stack, PDA control state plus stack, Earley chart frontier, or grammar residual state. The research must distinguish high-performance structured decoding backends from parser generators that require custom token-mask machinery.

## Constraints

- **Correctness**: Token masks must be sound for tokenizer tokens that cross grammar terminal boundaries.
- **Observability**: The backend must support or be patchable for counters over states, transitions, stacks/configs, masks, caches, and GLR paths.
- **Performance**: The baseline should target per-step mask overhead suitable for LLM sampling loops.
- **Integration**: Python and Rust are preferred because the existing project already uses Python package APIs, PyO3, Rust backends, and Hugging Face logits processors.
- **Scope**: Avoid implementing a parser generator or PDA constructor from scratch unless all candidate tools fail key requirements.
- **Serving**: vLLM/SGLang integration matters, but should be evaluated after the local tokenizer-mask API and instrumentation model are validated.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with llguidance or XGrammar for baseline evidence | They already target LLM structured generation and expose token-mask sampling loops | Pending |
| Use Lark or parglare for Python prototype only | They expose parser interaction quickly, but are not designed as high-performance token-mask engines | Pending |
| Use RustyLR or Rustemo for custom LR/GLR research backend | They expose LR/GLR concepts closer to state/stack/path instrumentation than black-box structured-output engines | Pending |
| Structure roadmap as horizontal layers | User selected horizontal roadmap organization for research, architecture, implementation, instrumentation, and integration layers | Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition**:
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone**:
1. Full review of all sections
2. Core Value check -- still the right priority?
3. Audit Out of Scope -- reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-19 after initialization*
