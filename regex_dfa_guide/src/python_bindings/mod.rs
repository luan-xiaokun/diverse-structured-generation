//! Provides tools and interfaces to integrate the crate's functionality with Python.

use std::sync::{Arc, PoisonError, RwLock};

use bincode::config;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap as HashMap;

use crate::diverse_guide_dfa::DiverseGuideDFA;

#[pyclass(
    name = "DiverseGuideDFA",
    module = "regex_dfa_guide.regex_dfa_guide_rs",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyDiverseGuideDFA(Arc<RwLock<DiverseGuideDFA>>);

fn map_poison_error_to_pyerr<G>(_: PoisonError<G>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RwLock poisoned")
}

#[pymethods]
impl PyDiverseGuideDFA {
    #[new]
    fn __new__(
        regex: &str,
        eos_token_id: u64,
        vocabulary: HashMap<u64, String>,
    ) -> PyResult<Self> {
        DiverseGuideDFA::new(regex, eos_token_id, vocabulary)
            .map(|x| PyDiverseGuideDFA(Arc::new(RwLock::new(x))))
            .map_err(Into::into)
    }

    fn fork(&self) -> PyResult<PyDiverseGuideDFA> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(PyDiverseGuideDFA(Arc::new(RwLock::new(dfa_guard.fork()))))
    }

    fn is_initial_state(&self, state: u32) -> PyResult<bool> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.is_initial_state(state))
    }

    fn is_final_state(&self, state: u32) -> PyResult<bool> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.is_final_state(state))
    }

    fn get_initial_state(&self) -> PyResult<u32> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_initial_state())
    }

    fn get_final_states(&self) -> PyResult<Vec<u32>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_final_states())
    }

    fn get_states(&self) -> PyResult<Vec<u32>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_states())
    }

    fn get_transitions(&self) -> PyResult<HashMap<u32, HashMap<u8, u32>>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_transitions())
    }

    fn get_state_sequence(&self, string: &str) -> PyResult<Vec<u32>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_state_sequence(string)?)
    }

    fn get_transition_sequence(&self, string: &str) -> PyResult<Vec<(u8, u32)>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_transition_sequence(string)?)
    }

    fn get_allowed_bytes(&self, state: u32) -> PyResult<Vec<u8>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_allowed_bytes(state)?)
    }

    fn get_allowed_token_ids(&self, state: u32) -> PyResult<Vec<u64>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_allowed_token_ids(state)?)
    }

    fn get_next_byte_state(&self, state: u32, input: u8) -> PyResult<u32> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_next_byte_state(state, input)?)
    }

    fn get_next_token_state(&self, state: u32, token_id: u64) -> PyResult<u32> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_next_token_state(state, token_id)?)
    }

    fn get_byte_state_sequence(&self, state: u32, token_id: u64) -> PyResult<Vec<u32>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_byte_state_sequence(state, token_id)?)
    }

    fn get_byte_transition_sequence(&self, string: &str) -> PyResult<Vec<(u8, u32)>> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(dfa_guard.get_byte_transition_sequence(string)?)
    }

    fn update_path_counter(&self, string: &str) -> PyResult<()> {
        let mut dfa_guard = self.0.write().map_err(map_poison_error_to_pyerr)?;
        dfa_guard.update_path_counter(string)?;
        Ok(())
    }

    fn update_local_state_counter(&self, state: u32, token_id: u64) -> PyResult<()> {
        let mut dfa_guard = self.0.write().map_err(map_poison_error_to_pyerr)?;
        dfa_guard.update_local_state_counter(state, token_id)?;
        Ok(())
    }

    fn reset_path_counter(&self) -> PyResult<()> {
        let mut dfa_guard = self.0.write().map_err(map_poison_error_to_pyerr)?;
        dfa_guard.reset_path_counter();
        Ok(())
    }

    fn reset_local_state_counter(&self) -> PyResult<()> {
        let mut dfa_guard = self.0.write().map_err(map_poison_error_to_pyerr)?;
        dfa_guard.reset_local_state_counter();
        Ok(())
    }

    fn compute_counts(&self, state: u32) -> PyResult<(Vec<u64>, Vec<u32>, Vec<u32>)> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        let counts = dfa_guard.compute_counts(state)?;
        Ok((counts.token_ids, counts.reward_counts, counts.penalty_counts))
    }

    fn __repr__(&self) -> PyResult<String> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(format!("{:#?}", *dfa_guard))
    }

    fn __str__(&self) -> PyResult<String> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(format!("{}", *dfa_guard))
    }

    fn __eq__(&self, other: &PyDiverseGuideDFA) -> PyResult<bool> {
        let self_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        let other_guard = other.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(*self_guard == *other_guard)
    }

    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> PyResult<Self> {
        let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
        Ok(PyDiverseGuideDFA(Arc::new(RwLock::new(dfa_guard.clone()))))
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        Python::attach(|py| {
            let cls = PyModule::import(py, "regex_dfa_guide.regex_dfa_guide_rs")?
                .getattr("DiverseGuideDFA")?;
            let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
            let binary_data: Vec<u8> =
                bincode::encode_to_vec(&*dfa_guard, config::standard()).map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "Serialization of DiverseGuideDFA failed: {}",
                        e
                    ))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (dfa, _): (DiverseGuideDFA, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Deserialization of DiverseGuideDFA failed: {}",
                    e
                ))
            })?;
        Ok(PyDiverseGuideDFA(Arc::new(RwLock::new(dfa))))
    }
}

#[pymodule]
fn regex_dfa_guide_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDiverseGuideDFA>()?;
    Ok(())
}
