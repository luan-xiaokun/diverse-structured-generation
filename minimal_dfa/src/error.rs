//! The Errors that may occur within the crate.

use thiserror::Error;

pub type Result<T, E = crate::Error> = std::result::Result<T, E>;

#[derive(Error, Debug)]
pub enum Error {
    // Index Errors
    #[error("Failed to build DFA {0}")]
    IndexDfaError(#[from] Box<regex_automata::dfa::dense::BuildError>),
    #[error("Index failed since anchored universal start state doesn't exist")]
    DfaHasNoStartState,
    #[error("Ref recursion limit reached: {0}")]
    RefRecursionLimitReached(usize),
    #[error("Invalid state: {0}")]
    InvalidState(usize),
    #[error("Invalid token id: {0}")]
    InvalidTokenId(usize),
    #[error("No transition found for state {0} and input byte {1}")]
    NoTransitionFound(usize, usize),
    #[error("No token transition found for state {0} and input token {1}")]
    NoTokenTransitionFound(usize, usize),
}

impl Error {
    pub fn is_recursion_limit(&self) -> bool {
        matches!(self, Self::RefRecursionLimitReached(_))
    }
}

#[cfg(feature = "python-bindings")]
impl From<Error> for pyo3::PyErr {
    fn from(e: Error) -> Self {
        use pyo3::exceptions::PyValueError;
        use pyo3::PyErr;
        PyErr::new::<PyValueError, _>(e.to_string())
    }
}
