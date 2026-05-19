# Tool Matrix: CFG Structured Decoding Backends

**Date:** 2026-05-19

| Capability | llguidance | XGrammar | RustyLR | Rustemo | Menhir | Lark | parglare |
|------------|------------|----------|---------|---------|--------|------|----------|
| Language | Rust core, C/C++, Python | C++ core, Python, JS | Rust | Rust | OCaml | Python | Python |
| License | MIT | Apache-2.0 | MIT/Apache-2.0 | Apache-2.0 or MIT/Expat reported | permissive OCaml ecosystem license, verify before vendoring | MIT | MIT |
| Maintenance status | Active; docs.rs 1.7.x in 2026 | Active; v0.2.1 and XGrammar-2 news in 2026 | Active; latest GitHub release shown 2025 | Active but smaller ecosystem | Mature and active | Mature Python parser | Active, smaller Python parser |
| Grammar format | JSON schema, regex, Lark-like CFG, internal JSON | JSON schema, regex, EBNF/GGML BNF/custom CFG | Bison/Yacc-like Rust grammar | Rustemo grammar syntax | `.mly` Menhir grammar | Lark EBNF | parglare grammar |
| Supports CFG | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Supports LR(1) | No, Earley-based | PDA-style, not LR table API | IELR/minimal LR family | LR | Canonical LR(1) | LALR, Earley, CYK | LR |
| Supports LALR/IELR | No | No public LR table model | LALR and IELR/minimal LR | LR family; verify exact variants | Canonical LR(1), not IELR focus | LALR | LR table generation |
| Supports GLR | No; arbitrary CFG via Earley | General CFG via PDA/Earley in XGrammar-2 material | Yes, nondeterministic GLR | Yes, RNGLR | Yes via `--GLR` | Ambiguity via Earley, not GLR | Yes |
| Supports arbitrary CFG | Yes, stated by docs | Yes, stated for general CFG | Via GLR for nondeterministic grammars | Via GLR | GLR can handle conflicts | Earley can parse CFG; interactive LALR limited | GLR can handle conflicts |
| Supports incremental parsing | Yes, token parser session | Yes, `GrammarMatcher.accept_token` | Yes, feed token into context | Yes, parser runtime; verify API granularity | Yes, incremental API | Yes for LALR interactive parser | Parser runtime; verify interactive granularity |
| Supports token mask directly | Yes, `compute_mask` | Yes, `fill_next_token_bitmask` | No | No | No | No | No |
| Supports byte/char-level advance | Yes internally, byte/token trie | Yes internally, token/byte mask logic | If grammar tokens model bytes/chars; custom needed | If grammar tokens model bytes/chars; custom needed | Lexer-token level; custom needed | String/token level; bytes option exists | String/token level |
| Supports tokenizer integration | Yes, tokenizer trie and HF tokenizer bytes | Yes, `TokenizerInfo` from HF/tiktoken/SentencePiece | No | No | No | No | No |
| Can expose parser state | Partial: `dump_state`, stats; internals patchable | Limited public API; C++ patch likely | Yes via generated state/context/debug | Likely via runtime/context/GSS types | Yes via inspection API | Parser state object accessible in interactive parser | Likely via parser internals/tracing |
| Can expose stack | Not LR stack; Earley frontier internals | PDA/persistent stack internally, public exposure uncertain | Yes via context/custom runtime inspection | Yes, `GssHead` for GLR | Yes via inspection/incremental stack APIs | Internal LALR parser state stack | Likely with parser internals |
| Can expose transition | Patch internals | Patch internals | Generated table/state likely inspectable | `ParserDefinition` actions/gotos promising | Automaton dump and inspection | `choices` actions, internal table | Tables module and traces |
| Can instrument counters | Yes, best via fork; some stats already public | Yes, but likely C++ fork | Yes, custom runtime recommended | Yes, custom hooks/fork | Yes for research, awkward integration | Yes for prototype only | Yes for prototype only |
| Can export ACTION/GOTO | Not applicable | Not applicable | Likely from generated code or core tables; verify | ParserDefinition exposes actions/GOTOs; verify generated visibility | Yes via automaton/table artifacts | Internals, not stable public export | Tables module |
| Production backend fit | High | High | Medium after custom work | Medium-low after custom work | Low for this repo | Low | Low |
| Integration difficulty | Low-medium | Low for Python/serving, medium for deep instrumentation | Medium-high | Medium-high | High | Low | Low-medium |
| Performance expectation | High; docs cite around 50 us/token typical for 128k tokenizer | Very high for supported paths; near-zero JSON overhead claims | Unknown for token masks; LR tables fast but mask traversal custom | Unknown; not primarily LLM mask optimized | Good parser performance, poor stack fit for Python/Rust serving | Good prototype speed, not production | Prototype only |
| Main risks | Earley internals not LR-stack counters; Python stats exposure | Public API hides PDA/cache details; C++ patch effort | Generated API may not expose raw tables cleanly; custom GSS needed | Visibility/performance uncertainty | OCaml integration and non-serving focus | LALR-only interactive mode; no tokenizer support | Python speed and API stability |
| Recommendation | Primary baseline | Serving baseline/backup | Primary LR/GLR research candidate | GLR instrumentation backup | Correctness reference | Phase 2 prototype | Phase 2 GLR prototype backup |

## Specific Matrix Notes

### llguidance

Current API evidence:
- `ParserFactory` compiles grammars and holds tokenizer state.
- `TokenParser` drives a single session and exposes `compute_mask`, `consume_token`, `validate_token`, `rollback`, `is_accepting`, `parser_stats`, `last_step_stats`, `max_step_stats`, and `dump_state`.
- `Constraint` wraps the sampling loop with `compute_mask` and `commit_token`.

Instrumentation fit:
- Mask timing, allowed-token counts, parser stats, and state dumps can be measured quickly.
- Fine-grained state/transition counters need a fork of Earley/token-trie internals.

### XGrammar

Current API evidence:
- `Grammar.from_json_schema`, `Grammar.from_regex`, `Grammar.from_ebnf`.
- `TokenizerInfo.from_huggingface`, with fast tokenizer, tiktoken, and SentencePiece support.
- `GrammarCompiler` creates `CompiledGrammar` and adaptive token mask cache.
- `GrammarMatcher.accept_token` and `fill_next_token_bitmask` implement the generation loop.

Instrumentation fit:
- Wrapper can measure mask time and allowed-token counts immediately.
- PDA state, stack, transition, and cache internals likely require C++ changes.

### RustyLR

Current API evidence:
- Generates parser, context, state, rule, and nonterminal types.
- Context supports expected-token/can-feed/backtrace-style introspection.
- Command-line state-machine debug output exists.

Instrumentation fit:
- Best candidate for a custom table-driven LR/GLR runtime if generated table layouts are usable.
- Requires custom tokenizer trie, reduce closure, bitset mask, persistent stack/GSS, and counters.

### Rustemo

Current API evidence:
- Runtime exposes `LRParser`, `GlrParser`, `GssHead`, `LRContext`, `Action`, `ParserDefinition`.

Instrumentation fit:
- Strong candidate for GLR/GSS observation.
- Need source spike to verify generated table visibility and runtime hook points.

### Menhir

Current API evidence:
- `--table` enables incremental API.
- `--inspection` enables inspection API.
- `--dump` writes automaton description.
- `--GLR` supports nondeterministic parsing.

Instrumentation fit:
- Good reference for state/stack correctness and accepted-terminal behavior.
- Poor fit for production in this Python/Rust LLM decoding repo.

### Lark

Current API evidence:
- `InteractiveParser.feed_token`, `choices`, `accepts`.
- `choices` returns accepted token types and their actions for current state.
- LALR parser has contextual lexer feedback.

Instrumentation fit:
- Ideal for quickly validating the accepted-terminal-to-tokenizer-trie algorithm.
- Not a production backend.

### parglare

Current API evidence:
- Pure Python LR/GLR parser, table calculation, integrated scanning, CLI visualization, MIT license.

Instrumentation fit:
- Useful for ambiguity and GLR concepts in Python.
- Not production due to Python runtime and lack of direct tokenizer mask API.
