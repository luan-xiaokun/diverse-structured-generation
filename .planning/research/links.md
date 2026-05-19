# Links

**Date:** 2026-05-19

## Primary Structured Decoding Backends

- llguidance GitHub: https://github.com/guidance-ai/llguidance
- llguidance docs.rs crate: https://docs.rs/llguidance/latest/llguidance/
- llguidance `TokenParser` docs: https://docs.rs/llguidance/latest/llguidance/struct.TokenParser.html
- llguidance `Constraint` docs: https://docs.rs/llguidance/latest/llguidance/struct.Constraint.html
- llguidance GitHub Pages: https://guidance-ai.github.io/llguidance/

- XGrammar GitHub: https://github.com/mlc-ai/xgrammar
- XGrammar docs: https://xgrammar.mlc.ai/
- XGrammar workflow tutorial: https://xgrammar.mlc.ai/docs/tutorials/workflow_of_xgrammar.html
- XGrammar constrained decoding tutorial: https://xgrammar.mlc.ai/docs/tutorials/constrained_decoding.html
- XGrammar Python API: https://xgrammar.mlc.ai/docs/api/python/index.html
- XGrammar arXiv paper: https://arxiv.org/abs/2411.15100
- XGrammar MLSys paper: https://proceedings.mlsys.org/paper_files/paper/2025/file/5c20ca4b0b20b0bd2f1d839dc605e70f-Paper-Conference.pdf
- XGrammar 2 arXiv: https://arxiv.org/abs/2601.04426

## Serving Integration Context

- SGLang structured outputs: https://docs.sglang.io/docs/advanced_features/structured_outputs
- vLLM project: https://github.com/vllm-project/vllm
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM

## Rust LR/GLR Candidates

- RustyLR GitHub: https://github.com/ehwan/RustyLR
- RustyLR docs.rs: https://docs.rs/rusty_lr
- RustyLR crate page: https://docs.rs/crate/rustylr/latest
- RustyLR lib.rs: https://lib.rs/crates/rustylr

- Rustemo GitHub: https://github.com/igordejanovic/rustemo
- Rustemo book introduction: https://igordejanovic.net/rustemo/introduction.html
- Rustemo docs.rs runtime: https://docs.rs/rustemo/latest/rustemo/
- Rustemo parsing docs: https://igordejanovic.net/rustemo/parsing/parsing.html

## Menhir

- Menhir homepage: https://gallium.inria.fr/~fpottier/menhir/
- Menhir manual: https://gallium.inria.fr/~fpottier/menhir/manual.html

## Python Prototype Candidates

- Lark GitHub: https://github.com/lark-parser/lark
- Lark docs: https://lark-parser.readthedocs.io/
- Lark API reference, `InteractiveParser`: https://lark-parser.readthedocs.io/en/stable/classes.html
- Lark parser overview: https://lark-parser.readthedocs.io/en/latest/parsers.html

- parglare GitHub: https://github.com/igordejanovic/parglare
- parglare docs: https://igordejanovic.net/parglare/
- parglare paper: https://www.sciencedirect.com/science/article/pii/S0167642321001271
- parglare PyPI: https://pypi.org/project/parglare/

## Local Codebase Reference Points

- Existing Rust DFA backend: `regex_dfa_guide/src/diverse_guide_dfa.rs`
- Existing PyO3 binding layer: `regex_dfa_guide/src/python_bindings/mod.rs`
- Existing Hugging Face logits processors: `src/diverse_guide/guide_rust.py`
- Existing tokenizer vocabulary conversion: `src/diverse_guide/vocab.py`
- Existing metrics: `src/diverse_guide/evaluation/metrics.py`
- Existing runtime benchmark script: `scripts/eval_runtime.py`
