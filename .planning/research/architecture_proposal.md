# Architecture Proposal: Instrumented CFG Token-Mask Runtime

**Date:** 2026-05-19

## Recommended System Architecture

```text
Grammar source
  -> grammar compiler / parser generator
  -> grammar automaton or parser runtime
  -> tokenizer trie
  -> incremental runtime config
  -> allowed token bitset
  -> logits processor / serving adapter
  -> counters and traces
```

The project should support two backend families:

1. **Purpose-built structured decoding backend**: llguidance or XGrammar.
   - Used to get a fast working baseline.
   - Instrument by wrapping public APIs first, then forking internals where needed.

2. **Research LR/GLR backend**: RustyLR or Rustemo plus custom runtime.
   - Used to study canonical LR/GLR stacks, reductions, transitions, GSS paths, and cache behavior.
   - Token mask logic is custom.

## Runtime Interface

```rust
trait GrammarRuntime {
    type Config;

    fn initial_config(&self) -> Self::Config;

    fn allowed_tokens(&mut self, config: &Self::Config) -> BitSet;

    fn advance_token(
        &mut self,
        config: &Self::Config,
        token_id: u32,
    ) -> Vec<Self::Config>;

    fn is_accepting(&self, config: &Self::Config) -> bool;

    fn stats(&self) -> RuntimeStats;
}
```

For deterministic LR, `Config` can be a persistent LR stack plus lexer substate. For GLR, `Config` should be a compact GSS handle plus active heads. For Earley-style engines, `Config` can be a chart/frontier plus lexer state. For a black-box baseline such as llguidance/XGrammar, `Config` may initially be an opaque session object while wrappers collect mask-level stats.

## Why `(state, token) -> next_state` Is Wrong

A CFG parser state alone is insufficient. The legal next terminals after a prefix depend on:

- the LR stack history;
- reductions that pop multiple stack entries;
- gotos after reductions;
- nullable and epsilon productions;
- ambiguous paths;
- partial terminal/lexer state;
- tokenizer tokens that span multiple grammar terminals or cross terminal boundaries.

The correct shape is:

```text
(config, token) -> next_config_set
```

where `config` includes stack/GSS/chart/residual state. Even deterministic LR has a stack-dependent future; GLR and ambiguous grammars naturally return a set or compact representation of many configurations.

## Tokenizer Trie Design

Build a trie over token byte sequences:

```rust
struct TrieNode {
    byte_edges: FxHashMap<u8, TrieNodeId>,
    token_ids: SmallVec<[u32; 2]>,
}
```

Mask computation:

1. Start from `(root_trie_node, closure(config))`.
2. For each outgoing byte edge, advance grammar runtime over that byte or over terminal fragments.
3. If any resulting config is viable, continue traversal.
4. When a trie node has token IDs and at least one viable config exists, mark those token IDs allowed.
5. Store resulting configs for `advance_token` so the sampled token does not repeat all work when possible.

This trie traversal is required because LLM tokenizer tokens are not grammar terminals. A token may be:

- a complete terminal;
- a prefix of a terminal;
- multiple terminals;
- a suffix of one terminal plus prefix of another;
- whitespace or skipped lexeme plus syntax content;
- invalid only after reduce closure is considered.

## Handling Terminal Boundaries

The runtime must separate byte advance from grammar-symbol advance:

```text
token bytes -> lexer/terminal recognizer -> terminal completions -> parser shift/reduce/goto closure
```

For LR/GLR:
- maintain a lexer substate while scanning bytes;
- when a terminal completes, perform all valid shifts;
- after each shift, apply reduce closure until no more reductions are possible or until conflicts branch;
- carry partial lexeme states through trie traversal.

For Earley-style engines:
- scanning bytes updates lexer derivatives;
- completed lexemes scan Earley terminals;
- prediction/completion maintains closure.

## Reduce and Epsilon Closure

Before checking token leaves, normalize configs:

```text
closure(config):
  repeat:
    apply epsilon productions
    apply reductions available without consuming more input
    apply gotos after reductions
    merge equivalent configs
  until fixed point or resource limit
```

For deterministic LR, closure usually follows a single chain of reductions for a lookahead. For GLR, reductions can fork logical paths and then merge physical GSS nodes. Counters must record both logical and physical behavior.

## Cache Design

Use tiered caches:

### Mask Cache

Key:

```text
grammar_id, tokenizer_id, normalized_config_hash, parser_limits_version
```

Value:

```text
allowed_token_bitset, token_id -> next_config_set summary, stats snapshot
```

### Trie Transition Cache

Key:

```text
normalized_config_hash, trie_node_id
```

Value:

```text
viable next config set, terminal boundary events
```

### LR/GLR Closure Cache

Key:

```text
stack_or_gss_handle, lookahead_or_byte_class
```

Value:

```text
closed active heads, reductions applied, gotos applied
```

Cache invalidation must include grammar version, tokenizer version, parser mode, and instrumentation mode if instrumentation changes runtime behavior.

## Counter Design

### Grammar-level

```rust
struct GrammarStats {
    num_nonterminals: usize,
    num_terminals: usize,
    num_rules: usize,
    num_lr_states: Option<usize>,
    num_pda_states: Option<usize>,
    num_action_entries: Option<usize>,
    num_goto_entries: Option<usize>,
}
```

### Runtime-level

```rust
struct RuntimeStats {
    step_count: u64,
    visited_config_count: u64,
    unique_config_count: u64,
    visited_state_count: Vec<u64>,
    visited_transition_count: Vec<u64>,
    shift_count: Counter<(StateId, TerminalId)>,
    reduce_count: Counter<(StateId, RuleId)>,
    goto_count: Counter<(StateId, NonterminalId)>,
    stack_node_count: u64,
    stack_node_visit_count: Counter<StackNodeId>,
    max_stack_depth: usize,
    avg_stack_depth: f64,
    mask: MaskStats,
    glr: Option<GlrStats>,
}
```

### Mask-level

```rust
struct MaskStats {
    mask_compute_count: u64,
    mask_compute_time_ns: u128,
    allowed_token_count_histogram: Histogram,
    trie_nodes_visited: u64,
    tokens_checked: u64,
    tokens_rejected: u64,
    cache_hit_count: u64,
    cache_miss_count: u64,
}
```

### GLR-specific

```rust
struct GlrStats {
    active_stack_count: u64,
    gss_node_count: u64,
    ambiguity_count: u64,
    path_merge_count: u64,
    logical_path_count: u128,
    physical_visit_count: u64,
}
```

## Count Semantics

**Physical visit count** is the number of runtime operations actually performed. If one GSS node compactly represents 100 logical paths and the runtime touches the node once, physical visits increase by one.

**Logical path count** is the number of parse paths represented by compact structures. It may grow exponentially and should use saturating or big integer counters.

**Unique configuration count** is the number of distinct normalized configs observed. It deduplicates equivalent stacks/frontiers after closure and merge.

These counters answer different questions and must not be conflated.

## Backend-Specific Instrumentation Plan

### llguidance wrapper

Collect immediately:
- compile time;
- `compute_mask` time;
- allowed token count;
- `last_step_stats`;
- `max_step_stats`;
- `dump_state` snapshots for selected prefixes;
- token advance/backtrack/fast-forward behavior.

Fork later:
- Earley item visits;
- lexer derivative state visits;
- trie nodes visited;
- cache hit/miss;
- token rejection reasons.

### XGrammar wrapper

Collect immediately:
- grammar compile time;
- bitmask fill time;
- allowed token count from bitset;
- matcher termination/reset behavior.

Fork later:
- PDA state and persistent stack counters;
- adaptive token mask cache hit/miss;
- context-independent versus context-dependent token split;
- rejected token cause.

### Rust LR/GLR backend

Implement:
- table extraction from RustyLR or Rustemo generated code;
- persistent LR stack for deterministic mode;
- GSS for GLR mode;
- tokenizer byte trie;
- reduce/epsilon closure;
- bitset masks;
- full counters.

## Recommended First Experiment

Use the same grammar and tokenizer across llguidance, XGrammar, and a Lark prototype:

```ebnf
start ::= "{" pair ("," pair)* "}"
pair ::= "\"a\"" ":" NUMBER | "\"b\"" ":" STRING
NUMBER ::= /[0-9]+/
STRING ::= /"[^"]*"/
```

Measure:
- compile time;
- per-step mask time;
- allowed token count;
- number of trie nodes visited if available;
- stats/state visibility;
- behavior when a token crosses punctuation/string/number boundaries.
