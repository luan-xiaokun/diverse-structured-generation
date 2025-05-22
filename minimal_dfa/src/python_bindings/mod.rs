//! Provides tools and interfaces to integrate the crate's functionality with Python.

use std::sync::{Arc, PoisonError, RwLock};

use bincode::{config, Decode, Encode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::diverse_guide_dfa::DiverseGuideDFA;
use crate::mindiv_dfa::MinDivDFA;
use crate::minimal_dfa::MinimalDFA;

// macro_rules! type_name {
//     ($obj:expr) => {
//         // Safety: obj is always initialized and tp_name is a C-string
//         unsafe { std::ffi::CStr::from_ptr((&*(&*$obj.as_ptr()).ob_type).tp_name) }
//     };
// }
#[pyclass(name = "MinimalDFA", module = "minimal_dfa.minimal_dfa_rs")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyMinimalDFA(Arc<MinimalDFA>);

#[pyclass(name = "MinDivDFA", module = "minimal_dfa.minimal_dfa_rs")]
#[derive(Clone, Debug, PartialEq, Encode, Decode)]
pub struct PyMinDivDFA(Arc<MinDivDFA>);

#[pyclass(name = "DiverseGuideDFA", module = "minimal_dfa.minimal_dfa_rs")]
#[derive(Clone, Debug, Encode, Decode)]
pub struct PyDiverseGuideDFA(Arc<RwLock<DiverseGuideDFA>>);

#[pymethods]
impl PyMinimalDFA {
    #[new]
    fn __new__(py: Python<'_>, regex: &str) -> PyResult<Self> {
        py.allow_threads(|| {
            MinimalDFA::new(regex)
                .map(|x| PyMinimalDFA(Arc::new(x)))
                .map_err(Into::into)
        })
    }

    fn get_next_state(&self, state: u32, input: u8) -> Option<u32> {
        self.0.next_state(&state, &input)
    }

    fn is_final_state(&self, state: u32) -> bool {
        self.0.is_final_state(&state)
    }

    fn get_final_states(&self) -> HashSet<u32> {
        self.0.get_final_states().clone()
    }

    fn get_states(&self) -> HashSet<u32> {
        self.0.get_states().clone()
    }

    fn get_transitions(&self) -> HashMap<u32, HashMap<u8, u32>> {
        self.0.get_transitions().clone()
    }

    fn is_initial_state(&self, state: u32) -> bool {
        self.0.is_initial_state(state)
    }

    fn get_initial_state(&self) -> u32 {
        self.0.get_initial_state()
    }

    fn get_state_sequence(&self, inputs: String) -> Option<Vec<u32>> {
        self.0.get_state_sequence(&inputs[..])
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __eq__(&self, other: &PyMinimalDFA) -> bool {
        *self.0 == *other.0
    }

    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyMinimalDFA(Arc::new((*self.0).clone()))
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import(py, "minimal_dfa.minimal_dfa_rs")?.getattr("MinimalDFA")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "Serialization of MinimalDFA failed: {}",
                        e
                    ))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (dfa, _): (MinimalDFA, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!(
                    "Deserialization of MinimalDFA failed: {}",
                    e
                ))
            })?;
        Ok(PyMinimalDFA(Arc::new(dfa)))
    }
}

#[pymethods]
impl PyMinDivDFA {
    #[new]
    fn __new__(
        py: Python<'_>,
        regex: &str,
        eos_token_id: u64,
        vocabulary: HashMap<String, HashSet<u64>>,
    ) -> PyResult<Self> {
        py.allow_threads(|| {
            MinDivDFA::new(regex, eos_token_id, vocabulary)
                .map(|x| PyMinDivDFA(Arc::new(x)))
                .map_err(Into::into)
        })
    }

    fn get_next_state(&self, state: u32, input: u8) -> Option<u32> {
        self.0.next_state(&state, &input)
    }

    fn get_next_token_state(&self, state: u32, token_id: u64) -> Option<u32> {
        self.0.next_token_state(&state, &token_id)
    }

    fn is_initial_state(&self, state: u32) -> bool {
        self.0.is_initial_state(state)
    }

    fn get_initial_state(&self) -> u32 {
        self.0.get_initial_state()
    }

    fn is_final_state(&self, state: u32) -> bool {
        self.0.is_final_state(state)
    }

    fn get_final_states(&self) -> HashSet<u32> {
        self.0.get_final_states().clone()
    }

    fn get_states(&self) -> HashSet<u32> {
        self.0.get_states().clone()
    }

    fn get_transitions(&self) -> HashMap<u32, HashMap<u8, u32>> {
        self.0.get_transitions().clone()
    }

    fn get_token_transitions(&self) -> HashMap<u32, HashMap<u64, u32>> {
        self.0.get_token_transitions().clone()
    }

    fn get_allowed_token_ids(&self, state: u32) -> Option<Vec<u64>> {
        self.0.allowed_tokens(&state)
    }

    fn get_allowed_inputs(&self, state: u32) -> Option<Vec<u8>> {
        self.0.allowed_inputs(&state)
    }

    fn get_state_sequence_from_string(&self, state_id: u32, token: String) -> Option<Vec<u32>> {
        self.0.get_state_sequence_from_string(&state_id, token)
    }

    fn get_state_sequence(&self, token: String) -> Option<Vec<u32>> {
        self.0.get_state_sequence(token)
    }

    fn get_transition_sequence(&self, token: String) -> Option<Vec<(u8, u32)>> {
        self.0.get_transition_sequence(token)
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    fn __eq__(&self, other: &PyMinDivDFA) -> bool {
        *self.0 == *other.0
    }

    fn __deepcopy__(&self, _py: Python<'_>, _memo: Py<PyDict>) -> Self {
        PyMinDivDFA(Arc::new((*self.0).clone()))
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls = PyModule::import(py, "minimal_dfa.minimal_dfa_rs")?.getattr("MinDivDFA")?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&self.0, config::standard())
                .map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!(
                        "Serialization of MinDivDFA failed: {}",
                        e
                    ))
                })?;
            Ok((cls.getattr("from_binary")?.unbind(), (binary_data,)))
        })
    }

    #[staticmethod]
    fn from_binary(binary_data: Vec<u8>) -> PyResult<Self> {
        let (dfa, _): (MinDivDFA, usize) =
            bincode::decode_from_slice(&binary_data[..], config::standard()).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Deserialization of MinDivDFA failed: {}", e))
            })?;
        Ok(PyMinDivDFA(Arc::new(dfa)))
    }
}

fn map_poison_error_to_pyerr<G>(_: PoisonError<G>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("RwLock poisoned")
}

#[pymethods]
impl PyDiverseGuideDFA {
    #[new]
    fn __new__(
        py: Python<'_>,
        regex: &str,
        eos_token_id: u64,
        vocabulary: HashMap<u64, String>,
    ) -> PyResult<Self> {
        py.allow_threads(|| {
            DiverseGuideDFA::new(regex, eos_token_id, vocabulary)
                .map(|x| PyDiverseGuideDFA(Arc::new(RwLock::new(x))))
                .map_err(Into::into)
        })
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
        Ok(dfa_guard.get_final_states().clone())
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
        Ok(dfa_guard.compute_counts(state)?)
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

    fn __reduce__(&self) -> PyResult<(PyObject, (Vec<u8>,))> {
        Python::with_gil(|py| {
            let cls =
                PyModule::import(py, "minimal_dfa.minimal_dfa_rs")?.getattr("DiverseGuideDFA")?;
            let dfa_guard = self.0.read().map_err(map_poison_error_to_pyerr)?;
            let binary_data: Vec<u8> = bincode::encode_to_vec(&*dfa_guard, config::standard())
                .map_err(|e| {
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
fn minimal_dfa_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMinimalDFA>()?;
    m.add_class::<PyMinDivDFA>()?;
    m.add_class::<PyDiverseGuideDFA>()?;

    Ok(())
}
