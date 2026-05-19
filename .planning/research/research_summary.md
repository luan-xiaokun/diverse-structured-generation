# Research Summary: LLM CFG Structured Decoding Backends

**Date:** 2026-05-19

## Overall Conclusion

Use **llguidance** as the first high-performance baseline and **XGrammar** as the serving/framework baseline. Both are purpose-built for structured generation, compile grammars with tokenizer information, and expose token-mask sampling-loop APIs. llguidance is attractive for local Rust/Python experimentation because its public Rust API exposes `TokenParser::compute_mask`, `consume_token`, `dump_state`, `parser_stats`, `last_step_stats`, and `max_step_stats`. XGrammar is attractive for SGLang/vLLM-style integration because it is already the default or integrated backend in major serving engines and its API is built around `GrammarCompiler`, `CompiledGrammar`, and `GrammarMatcher.fill_next_token_bitmask`.

For canonical LR/GLR research, use **RustyLR first**, then evaluate **Rustemo** if GLR internals are easier to instrument in practice. RustyLR explicitly generates IELR/LALR tables, deterministic LR and nondeterministic GLR parsers, context objects, expected-token queries, and state-machine debug output. Rustemo exposes runtime types such as `LRParser`, `GlrParser`, `GssHead`, `LRContext`, and `ParserDefinition`; this is promising for GLR instrumentation but likely requires source inspection and custom hooks.

Use **Lark** for a one-week Python prototype of "accepted terminals -> tokenizer trie -> token mask." Lark's `InteractiveParser` has `feed_token`, `choices`, and `accepts`, but only for LALR interactive parsing. Use **parglare** as a GLR-capable Python alternative if ambiguity behavior must be explored early.

## Recommendation by Use Case

| Use case | Recommendation | Why |
|----------|----------------|-----|
| Fastest structured-decoding backend | llguidance | Direct token-mask loop, arbitrary CFG with Lark-like syntax, Rust API, Earley + regex-derivative lexer + token trie |
| SGLang/vLLM-aligned baseline | XGrammar | Default/recommended in SGLang, integrated with vLLM/TensorRT-LLM/MLC-LLM, high-performance C++ backend |
| Instrumentable LR/GLR research runtime | RustyLR | IELR/LALR/GLR parser generator, generated state/table concepts, context stack, expected-token queries |
| GLR GSS instrumentation backup | Rustemo | Runtime exposes GLR parser, GSS head, LR context, parser definition traits |
| Correctness reference | Menhir | Mature LR(1), incremental and inspection APIs, automaton dump; OCaml integration cost makes it poor production fit here |
| Python prototype | Lark first, parglare second | Fast exploration of accepted terminals and grammar behavior before Rust work |

## Tool Notes

### llguidance

Recommended as **primary baseline**.

Strengths:
- Purpose-built for LLM constrained decoding.
- Supports JSON schema, regex, and CFG in a Lark-like grammar format.
- Rust docs describe `ParserFactory`, `TokenParser`, and `Constraint`; the sampling loop is `compute_mask` then `commit_token`.
- `TokenParser` exposes `parser_stats`, `last_step_stats`, `max_step_stats`, `dump_state`, `is_accepting`, `validate_token`, `rollback`, and `consume_token`.
- Internals use Earley parsing, regex derivative lexing, and tokenizer trie traversal.
- MIT license and usable from Rust, C/C++, and Python.

Gaps:
- It is not an LR/GLR stack-machine reference.
- Fine-grained state/transition/stack counters may require patching Earley parser internals rather than public API-only wrapping.
- Need to verify whether Python bindings expose the same stats surface as Rust.

Likely patch points:
- `parser/src/tokenparser.rs` for per-step stats and mask/cache instrumentation.
- `parser/src/earley/*` for Earley item/state counters and closure metrics.
- `toktrie/*` for trie traversal and token acceptance/rejection counters.

### XGrammar

Recommended as **serving/framework baseline** and backup primary backend.

Strengths:
- Supports JSON schema, regex, custom EBNF CFG, tokenizer info, compiled grammars, CPU bitset masks, and per-step `GrammarMatcher`.
- SGLang docs state XGrammar is the default backend and suggest using it for performance and utility.
- GitHub README states broad integration with vLLM, SGLang, TensorRT-LLM, MLC-LLM, and more.
- Uses an adaptive token mask cache and PDA/persistent-stack approach in its technical material.
- Apache-2.0 license and C++/Python/JavaScript APIs.

Gaps:
- Public Python API is optimized around mask generation, not parser introspection.
- Instrumenting PDA state, persistent stack, cache hits, and path counters likely requires modifying C++ internals.
- XGrammar-2 has been released as of 2026-05; the project should pin which generation is evaluated.

### RustyLR

Recommended as **first LR/GLR research backend candidate**.

Strengths:
- Generates optimized IELR(1)/LALR(1) tables and supports deterministic LR plus nondeterministic GLR parsing.
- Generated code includes parser, context, state, rule, and nonterminal types.
- Context exposes expected-token and can-feed style queries, useful for viability prototypes.
- The executable has state-machine debug output.
- MIT/Apache dual license.

Gaps:
- It is a parser generator, not a constrained decoding engine.
- Tokenizer trie traversal, reduce closure across token bytes, persistent stacks, and bitset masks must be custom.
- Need to inspect generated table layout and whether ACTION/GOTO are stable enough for custom runtime extraction.

### Rustemo

Recommended as **GLR instrumentation backup**.

Strengths:
- Rust LR/GLR parser generator.
- Runtime docs expose `LRParser`, `GlrParser`, `GssHead`, `LRContext`, `Action`, and `ParserDefinition`.
- RNGLR design may be valuable for ambiguous grammars and GSS path counting.

Gaps:
- Performance is likely not as optimized for LLM per-token mask generation.
- Need source inspection to determine how much of GSS/action/goto is public versus generated/private.
- More suited for research runtime instrumentation than immediate serving baseline.

### Menhir

Recommended as **correctness reference**, not production backend.

Strengths:
- Mature OCaml LR(1) parser generator.
- `--table` provides incremental API; `--inspection` adds inspection API; `--dump` exports automaton descriptions.
- Good reference for LR automaton behavior and accepted terminals.

Gaps:
- OCaml integration cost is high for this Python/Rust project.
- Not designed for LLM tokenizer trie masks or serving integration.

### Lark and parglare

Recommended for **Phase 2 prototype only**.

Strengths:
- Lark `InteractiveParser` exposes `feed_token`, `choices`, and `accepts` for LALR.
- Lark supports Earley and LALR; its contextual lexer demonstrates parser feedback to lexer.
- parglare is pure Python LR/GLR with table calculation and GLR support.

Gaps:
- Python performance is not appropriate for production token masks.
- Lark interactive mode is LALR-only.
- Neither directly solves tokenizer-byte boundary viability.

## Direct Answers

### Q1: llguidance vs XGrammar

If the only goal is high-performance token viability, start with **llguidance** locally and benchmark **XGrammar** immediately after. llguidance exposes a Rust API surface closer to the requested loop and stats; XGrammar is better when matching SGLang/vLLM production behavior.

For observability and instrumentation, **llguidance appears easier to modify first** because Rust docs expose parser stats and state-dump methods. XGrammar has relevant concepts, but much of the desired instrumentation likely sits behind its C++ backend.

For SGLang/vLLM integration, **XGrammar has lower integration cost** because it is already documented as SGLang's default backend and has vLLM integration history. llguidance is still relevant because SGLang also supports it as an alternate backend.

### Q2: RustyLR vs Rustemo vs Menhir

RustyLR is the best first Rust target for ACTION/GOTO/table-driven runtime investigation. Rustemo is the best backup if its `GssHead`/`ParserDefinition` runtime types are more ergonomic for GSS instrumentation. Menhir is the best correctness reference but not a practical production backend for this repository.

### Q3: token viability design

Do not model the runtime as `(state, token) -> next_state`. CFG parsing state depends on stack history, reduce closure, nullable productions, and ambiguity. The correct shape is usually `(config, token) -> next_config_set`, where `config` includes a stack, GSS, Earley frontier, or residual grammar state.

Build a tokenizer trie over token bytes. To compute a mask, traverse trie nodes while simulating byte/char advances through the grammar runtime. When a trie path reaches a token leaf and at least one resulting config remains viable, mark that token as allowed. Keep partial lexer/terminal state in the traversal so tokens can cross grammar terminal boundaries. Apply reduce/epsilon closure before and after consuming terminal boundaries.

### Q4: counters

Separate:
- physical visits: how many engine operations actually ran;
- logical paths: how many parse paths a compact GSS/Earley representation denotes;
- unique configurations: deduplicated runtime states after normalization.

Collect grammar-level, runtime-level, mask-level, and GLR-specific counters as first-class runtime outputs, not log parsing.

## Final Decision Suggestion

Recommended primary path: **llguidance baseline + instrumented wrapper/fork**.

Recommended backup path: **XGrammar baseline for SGLang/vLLM parity**.

Recommended prototype path: **Lark InteractiveParser for LALR viability, parglare for GLR ambiguity experiments**.

Do-not-use path: **naive textbook CFG-to-NPDA implementation from scratch**.

Main implementation risks:
- tokenizer tokens crossing terminal boundaries;
- reduce/epsilon closure explosion;
- ambiguity path counting versus physical compact representation;
- public APIs hiding the exact states/stacks needed for counters;
- serving-engine integration constraints on CPU mask latency and batching.

Estimated engineering effort:
- 1 week: benchmark llguidance/XGrammar masks and Lark viability prototype.
- 2-4 weeks: research-grade Rust LR/GLR prototype with tokenizer trie and counters.
- 6-10+ weeks: robust production-style backend with batching, cache policy, Python bindings, and serving integration.

Files/modules to inspect first:
- llguidance: `parser/src/tokenparser.rs`, `parser/src/earley/*`, `toktrie/*`, Python bindings.
- XGrammar: `cpp/`, `include/xgrammar/`, `python/xgrammar/`, `GrammarMatcher`, `GrammarCompiler`, token mask cache internals.
- RustyLR: generated parser output for a toy grammar, `rusty_lr_core`, `rusty_lr_parser`.
- Rustemo: generated parser output, `GssHead`, `GlrParser`, `LRContext`, `ParserDefinition`.
- local repo: `regex_dfa_guide/src/diverse_guide_dfa.rs`, `src/diverse_guide/guide_rust.py`, `src/diverse_guide/vocab.py`.

Minimal code experiment:
- compile the same small JSON-like grammar in llguidance and XGrammar;
- load a local Hugging Face tokenizer;
- for a fixed prefix sequence, record mask time, allowed token count, accepted token advance, parser stats/state dump availability, and cache behavior if exposed.
