# Requirements: LLM CFG Structured Decoding Backend

**Defined:** 2026-05-19
**Core Value:** Compute correct token viability for CFG-constrained LLM decoding while making parser/runtime behavior measurable enough for research.

## v1 Requirements

### Research

- [ ] **RSCH-01**: The project compares llguidance, XGrammar, RustyLR, Rustemo, Menhir, Lark, and parglare against the requested backend capabilities.
- [ ] **RSCH-02**: The project identifies a recommended primary path, backup path, prototype path, and do-not-use path.
- [ ] **RSCH-03**: The project records source links for all repositories, documentation, and papers used in the decision.

### Matrix

- [ ] **MTRX-01**: The project provides a capability matrix covering language, license, maintenance, grammar formats, CFG/LR/LALR/IELR/GLR support, incremental parsing, token masks, tokenizer integration, state/stack/transition exposure, ACTION/GOTO export, performance expectation, risks, and recommendation.
- [ ] **MTRX-02**: The project explicitly answers llguidance versus XGrammar for high-performance token viability, instrumentation, and SGLang/vLLM integration.
- [ ] **MTRX-03**: The project explicitly answers RustyLR versus Rustemo versus Menhir for table extraction, custom runtime, LR/GSS counters, Rust fit, and correctness reference use.

### Architecture

- [ ] **ARCH-01**: The project defines a `GrammarRuntime`-style interface with initial config, allowed tokens, token advance, accepting state, and stats.
- [ ] **ARCH-02**: The project explains why `(state, token) -> next_state` is insufficient and why `(config, token) -> next_config_set` is required.
- [ ] **ARCH-03**: The project defines a tokenizer trie traversal design that handles token bytes/chars, terminal boundary crossing, reduce closure, and epsilon closure.
- [ ] **ARCH-04**: The project defines grammar-level, runtime-level, mask-level, and GLR-specific counters, including physical visits, logical path counts, and unique configuration counts.
- [ ] **ARCH-05**: The project defines a cache strategy for mask computation, trie traversal, and parser closure.

### Prototype

- [ ] **PRTO-01**: The project defines a one-week minimal prototype using llguidance or XGrammar and Lark/parglare.
- [ ] **PRTO-02**: The project defines a two-to-four week research prototype using RustyLR or Rustemo.
- [ ] **PRTO-03**: The project defines a long-term engineering path for a production-style Rust/Python backend.
- [ ] **PRTO-04**: The project specifies a minimal code experiment to run first, including grammar, tokenizer, prefixes, metrics, and success criteria.

## v2 Requirements

### Implementation

- **IMPL-01**: Implement the llguidance or XGrammar benchmark wrapper in the local repository.
- **IMPL-02**: Implement a Lark/parglare tokenizer-trie viability prototype.
- **IMPL-03**: Implement a Rust deterministic LR runtime using tables from RustyLR or Rustemo.
- **IMPL-04**: Implement GLR/GSS path counting.
- **IMPL-05**: Integrate the selected backend into a Hugging Face `LogitsProcessor`.
- **IMPL-06**: Evaluate vLLM or SGLang integration.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Naive CFG-to-NPDA implementation from scratch | High complexity and avoidable risk around closure, nullable rules, left recursion, and tokenizer boundaries |
| Full serving integration in v1 | Requires backend selection and local evidence first |
| Parse-tree-first API | The target product is constrained decoding and instrumentation, not parse tree construction |
| Replacing existing regex backend | Existing DFA backend remains valuable baseline and should coexist |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| RSCH-01 | Phase 1 | Pending |
| RSCH-02 | Phase 1 | Pending |
| RSCH-03 | Phase 1 | Pending |
| MTRX-01 | Phase 1 | Pending |
| MTRX-02 | Phase 1 | Pending |
| MTRX-03 | Phase 1 | Pending |
| ARCH-01 | Phase 2 | Pending |
| ARCH-02 | Phase 2 | Pending |
| ARCH-03 | Phase 2 | Pending |
| ARCH-04 | Phase 2 | Pending |
| ARCH-05 | Phase 2 | Pending |
| PRTO-01 | Phase 3 | Pending |
| PRTO-02 | Phase 3 | Pending |
| PRTO-03 | Phase 3 | Pending |
| PRTO-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-05-19*
*Last updated: 2026-05-19 after initial definition*
