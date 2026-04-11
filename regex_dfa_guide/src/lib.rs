//! # Regex_DFA_Guide
//!
//! `regex_dfa_guide` crate provides a convenient way to build minimal DFAs from regexes.

pub mod error;
pub mod diverse_guide_dfa;

pub use error::{Error, Result};

#[cfg(feature = "python-bindings")]
mod python_bindings;
