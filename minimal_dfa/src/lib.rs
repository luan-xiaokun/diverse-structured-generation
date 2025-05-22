//! # Minimal_DFA
//!
//! `minimal_dfa` crate provides a convenient way to build minimal DFAs from regexes.

pub mod error;
pub mod minimal_dfa;
pub mod mindiv_dfa;
pub mod diverse_guide_dfa;

pub use error::{Error, Result};

#[cfg(feature = "python-bindings")]
mod python_bindings;
