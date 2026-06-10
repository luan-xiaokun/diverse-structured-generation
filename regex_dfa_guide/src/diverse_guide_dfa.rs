//! Build a DFA for guiding diverse structured generation

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::alphabet::Unit;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;
use std::sync::Arc;

use crate::{Error, Result};

// define type alias for state-id, token-id, and byte
pub type StateId = u32;
pub type TokenId = u64;
pub type Byte = u8;

/// Precomputed metadata for one token transition from a DFA state.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
struct AllowedTokenInfo {
    token_id: TokenId,
    next_state: StateId,
    edge_path: Vec<(StateId, StateId)>,
    inner_states: Vec<StateId>,
    edge_indices: Vec<usize>,
    state_indices: Vec<usize>,
}

/// Immutable DFA structure built from a regex and vocabulary.
/// All expensive construction is done once here; instances are Arc-shareable.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct DfaIndex {
    eos_token_id: TokenId,
    initial_state: StateId,
    final_states: HashSet<StateId>,
    byte_transitions: HashMap<StateId, HashMap<Byte, StateId>>,
    token_transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    token_state_paths: HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
    allowed_token_infos: HashMap<StateId, Vec<AllowedTokenInfo>>,
    allowed_token_ids: HashMap<StateId, Vec<TokenId>>,
    distance_to_accepting: HashMap<StateId, u32>,
    state_to_idx: HashMap<StateId, usize>,
    edge_to_idx: HashMap<(StateId, StateId), usize>,
    vocabulary: HashMap<TokenId, String>,
}

impl DfaIndex {
    pub fn new(
        regex: &str,
        eos_token_id: TokenId,
        vocabulary: HashMap<TokenId, String>,
    ) -> Result<Self> {
        let dfa = DFA::builder()
            .configure(DFA::config().minimize(true))
            .build(regex)
            .map_err(Box::new)?;
        // get initial state
        let start_state: AutomataStateId = match dfa.universal_start_state(Anchored::Yes) {
            Some(state) => state,
            None => return Err(Error::DfaHasNoStartState),
        };
        // construct byte-level transitions and collect final states
        let (final_states, transitions) = Self::build_byte_transitions(&dfa, start_state)?;
        // collect all byte-level states (local variable only)
        let mut states = final_states.clone();
        states.insert(start_state.as_u32());
        for (state, inputs) in &transitions {
            states.insert(*state);
            states.extend(inputs.values().cloned());
        }
        // iterate each state, construct token level transitions
        let mut token_transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut token_state_paths: HashMap<StateId, HashMap<TokenId, Vec<StateId>>> =
            HashMap::default();
        for &current_state in &states {
            'token_loop: for (token_id, token_str) in vocabulary.iter() {
                if *token_id == eos_token_id {
                    continue;
                }
                let mut next_state = current_state;
                let mut state_path = vec![current_state];
                for byte in token_str.as_bytes() {
                    next_state = match transitions.get(&next_state) {
                        Some(trans) => match trans.get(&byte) {
                            Some(state) => *state,
                            None => continue 'token_loop,
                        },
                        None => continue 'token_loop,
                    };
                    state_path.push(next_state);
                }
                token_transitions
                    .entry(current_state)
                    .or_default()
                    .insert(*token_id, next_state);
                token_state_paths
                    .entry(current_state)
                    .or_default()
                    .insert(*token_id, state_path);
            }
        }
        // add eos token transitions for final states
        for &final_state in &final_states {
            token_transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
            token_state_paths
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, vec![final_state, final_state]);
        }
        // compute live states (under token-transitions)
        let mut live_states: HashSet<StateId> = final_states.clone();
        let mut queue: VecDeque<StateId> = final_states.iter().copied().collect();
        let mut rev_transitions: HashMap<StateId, Vec<StateId>> = HashMap::default();
        for (from_state, token_map) in &token_transitions {
            for (_, to_state) in token_map {
                rev_transitions
                    .entry(*to_state)
                    .or_default()
                    .push(*from_state);
            }
        }
        while let Some(current_state) = queue.pop_front() {
            if let Some(predecessors) = rev_transitions.get(&current_state) {
                for &prev_state in predecessors {
                    if !live_states.contains(&prev_state) {
                        live_states.insert(prev_state);
                        queue.push_back(prev_state);
                    }
                }
            }
        }
        // Prune token_transitions to only transitions leading to live states
        for (from_state, token_map) in token_transitions.iter_mut() {
            let mut pruned_token_ids = Vec::new();
            token_map.retain(|token_id, to_state| {
                let keep = live_states.contains(to_state);
                if !keep {
                    pruned_token_ids.push(*token_id);
                }
                keep
            });
            if let Some(path_map) = token_state_paths.get_mut(from_state) {
                for token_id in pruned_token_ids {
                    path_map.remove(&token_id);
                }
            }
        }
        token_transitions.retain(|_, token_map| !token_map.is_empty());
        token_state_paths.retain(|_, path_map| !path_map.is_empty());

        let distance_to_accepting =
            Self::build_distance_to_accepting(&token_transitions, &final_states);
        let mut allowed_token_infos =
            Self::build_allowed_token_infos(&token_transitions, &token_state_paths);
        let (state_to_idx, edge_to_idx) = Self::build_dense_counter_indices(&allowed_token_infos);
        Self::attach_dense_counter_indices(&mut allowed_token_infos, &state_to_idx, &edge_to_idx);
        let allowed_token_ids: HashMap<StateId, Vec<TokenId>> = allowed_token_infos
            .iter()
            .map(|(state, infos)| {
                (
                    *state,
                    infos.iter().map(|info| info.token_id).collect::<Vec<_>>(),
                )
            })
            .collect();

        Ok(Self {
            eos_token_id,
            initial_state: start_state.as_u32(),
            final_states,
            byte_transitions: transitions,
            token_transitions,
            token_state_paths,
            allowed_token_infos,
            allowed_token_ids,
            distance_to_accepting,
            state_to_idx,
            edge_to_idx,
            vocabulary,
        })
    }

    fn build_allowed_token_infos(
        token_transitions: &HashMap<StateId, HashMap<TokenId, StateId>>,
        token_state_paths: &HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
    ) -> HashMap<StateId, Vec<AllowedTokenInfo>> {
        let mut allowed_token_infos: HashMap<StateId, Vec<AllowedTokenInfo>> = HashMap::default();
        for (state, token_map) in token_transitions {
            let mut infos = Vec::with_capacity(token_map.len());
            for (token_id, next_state) in token_map {
                if let Some(state_path) = token_state_paths
                    .get(state)
                    .and_then(|paths| paths.get(token_id))
                {
                    let edge_path = state_path.windows(2).map(|w| (w[0], w[1])).collect();
                    let inner_states = state_path[1..].to_vec();
                    infos.push(AllowedTokenInfo {
                        token_id: *token_id,
                        next_state: *next_state,
                        edge_path,
                        inner_states,
                        edge_indices: Vec::new(),
                        state_indices: Vec::new(),
                    });
                }
            }
            if !infos.is_empty() {
                allowed_token_infos.insert(*state, infos);
            }
        }
        allowed_token_infos
    }

    fn build_dense_counter_indices(
        allowed_token_infos: &HashMap<StateId, Vec<AllowedTokenInfo>>,
    ) -> (HashMap<StateId, usize>, HashMap<(StateId, StateId), usize>) {
        let mut state_to_idx: HashMap<StateId, usize> = HashMap::default();
        let mut edge_to_idx: HashMap<(StateId, StateId), usize> = HashMap::default();

        for infos in allowed_token_infos.values() {
            for info in infos {
                for &state in &info.inner_states {
                    if !state_to_idx.contains_key(&state) {
                        state_to_idx.insert(state, state_to_idx.len());
                    }
                }
                for &edge in &info.edge_path {
                    if !edge_to_idx.contains_key(&edge) {
                        edge_to_idx.insert(edge, edge_to_idx.len());
                    }
                }
            }
        }

        (state_to_idx, edge_to_idx)
    }

    fn build_distance_to_accepting(
        token_transitions: &HashMap<StateId, HashMap<TokenId, StateId>>,
        final_states: &HashSet<StateId>,
    ) -> HashMap<StateId, u32> {
        let mut rev_transitions: HashMap<StateId, Vec<StateId>> = HashMap::default();
        for (from_state, token_map) in token_transitions {
            for to_state in token_map.values() {
                rev_transitions.entry(*to_state).or_default().push(*from_state);
            }
        }

        let mut distance_to_accepting: HashMap<StateId, u32> = HashMap::default();
        let mut queue: VecDeque<StateId> = VecDeque::new();
        for &final_state in final_states {
            distance_to_accepting.insert(final_state, 0);
            queue.push_back(final_state);
        }

        while let Some(current_state) = queue.pop_front() {
            let current_distance = distance_to_accepting[&current_state];
            if let Some(predecessors) = rev_transitions.get(&current_state) {
                for &prev_state in predecessors {
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        distance_to_accepting.entry(prev_state)
                    {
                        entry.insert(current_distance + 1);
                        queue.push_back(prev_state);
                    }
                }
            }
        }

        distance_to_accepting
    }

    fn attach_dense_counter_indices(
        allowed_token_infos: &mut HashMap<StateId, Vec<AllowedTokenInfo>>,
        state_to_idx: &HashMap<StateId, usize>,
        edge_to_idx: &HashMap<(StateId, StateId), usize>,
    ) {
        for infos in allowed_token_infos.values_mut() {
            for info in infos {
                info.edge_indices = info
                    .edge_path
                    .iter()
                    .map(|edge| {
                        *edge_to_idx
                            .get(edge)
                            .expect("metadata edge path must have a dense edge index")
                    })
                    .collect();
                info.state_indices = info
                    .inner_states
                    .iter()
                    .map(|state| {
                        *state_to_idx
                            .get(state)
                            .expect("metadata inner state must have a dense state index")
                    })
                    .collect();
            }
        }
    }

    fn build_byte_transitions(
        dfa: &DFA<Vec<StateId>>,
        start_state: AutomataStateId,
    ) -> Result<(HashSet<StateId>, HashMap<StateId, HashMap<Byte, StateId>>)> {
        let mut transitions: HashMap<StateId, HashMap<Byte, StateId>> = HashMap::default();
        let mut final_states: HashSet<StateId> = HashSet::default();
        let mut seen: HashSet<StateId> = HashSet::from_iter([start_state.as_u32()]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];
        while let Some(current_state) = next_states.pop() {
            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
            }
            let classes = dfa.byte_classes();
            for representative in classes.representatives(0..255) {
                let input = representative.as_u8().unwrap();
                let next_state: AutomataStateId = dfa.next_state(current_state, input);
                if !dfa.is_dead_state(next_state)
                    && !dfa.is_quit_state(next_state)
                    && (!dfa.is_match_state(next_state)
                        || dfa.is_match_state(dfa.next_eoi_state(next_state)))
                {
                    for x in classes.elements(Unit::u8(classes.get(input))) {
                        transitions
                            .entry(current_state.as_u32())
                            .or_default()
                            .insert(x.as_u8().unwrap(), next_state.as_u32());
                    }
                    if !seen.contains(&next_state.as_u32()) {
                        seen.insert(next_state.as_u32());
                        next_states.push(next_state);
                    }
                }
            }
        }
        Ok((final_states, transitions))
    }

    // --- State queries ---

    pub fn is_initial_state(&self, state: StateId) -> bool {
        state == self.initial_state
    }

    pub fn is_final_state(&self, state: StateId) -> bool {
        self.final_states.contains(&state)
    }

    pub fn get_initial_state(&self) -> StateId {
        self.initial_state
    }

    pub fn get_final_states(&self) -> Vec<StateId> {
        self.final_states.iter().copied().collect()
    }

    pub fn get_states(&self) -> Vec<StateId> {
        let mut states: HashSet<StateId> = HashSet::default();
        states.insert(self.initial_state);
        states.extend(&self.final_states);
        for &state in self.byte_transitions.keys() {
            states.insert(state);
            if let Some(transitions) = self.byte_transitions.get(&state) {
                for &next_state in transitions.values() {
                    states.insert(next_state);
                }
            }
        }
        for &state in self.token_transitions.keys() {
            states.insert(state);
            if let Some(transitions) = self.token_transitions.get(&state) {
                for &next_state in transitions.values() {
                    states.insert(next_state);
                }
            }
        }
        states.into_iter().collect()
    }

    pub fn get_transitions(&self) -> HashMap<StateId, HashMap<Byte, StateId>> {
        self.byte_transitions.clone()
    }

    pub fn get_state_sequence(&self, string: &str) -> Result<Vec<StateId>> {
        let mut state_seq: Vec<StateId> = vec![self.initial_state];
        let mut current_state = self.initial_state;
        for byte in string.as_bytes() {
            current_state = self.get_next_byte_state(current_state, *byte)?;
            state_seq.push(current_state);
        }
        Ok(state_seq)
    }

    pub fn get_transition_sequence(&self, string: &str) -> Result<Vec<(Byte, StateId)>> {
        self.get_byte_transition_sequence(string)
    }

    pub fn get_allowed_bytes(&self, state: StateId) -> Result<Vec<Byte>> {
        self.byte_transitions
            .get(&state)
            .map(|transition_map| transition_map.keys().cloned().collect())
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    pub fn get_allowed_token_ids(&self, state: StateId) -> Result<Vec<TokenId>> {
        self.allowed_token_ids
            .get(&state)
            .cloned()
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    pub fn get_distance_to_accepting(&self, state: StateId) -> Result<u32> {
        self.distance_to_accepting
            .get(&state)
            .copied()
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    pub fn get_next_byte_state(&self, state: StateId, byte: Byte) -> Result<StateId> {
        self.byte_transitions
            .get(&state)
            .ok_or_else(|| Error::InvalidState(state as usize))
            .and_then(|transition_map| {
                transition_map
                    .get(&byte)
                    .copied()
                    .ok_or_else(|| Error::NoTransitionFound(state as usize, byte as usize))
            })
    }

    pub fn get_next_token_state(&self, state: StateId, token_id: TokenId) -> Result<StateId> {
        self.token_transitions
            .get(&state)
            .ok_or_else(|| Error::InvalidState(state as usize))
            .and_then(|token_transition_map| {
                token_transition_map
                    .get(&token_id)
                    .copied()
                    .ok_or_else(|| Error::NoTokenTransitionFound(state as usize, token_id as usize))
            })
    }

    fn get_cached_token_state_path(&self, state: StateId, token_id: TokenId) -> Option<&[StateId]> {
        self.token_state_paths
            .get(&state)
            .and_then(|paths| paths.get(&token_id))
            .map(Vec::as_slice)
    }

    fn get_allowed_token_infos(&self, state: StateId) -> Result<&[AllowedTokenInfo]> {
        self.allowed_token_infos
            .get(&state)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    fn find_allowed_token_info(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Option<&AllowedTokenInfo> {
        self.allowed_token_infos
            .get(&state)?
            .iter()
            .find(|info| info.token_id == token_id)
    }

    fn compute_byte_state_sequence(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<Vec<StateId>> {
        let mut state_seq: Vec<StateId> = vec![state];
        if token_id == self.eos_token_id {
            state_seq.push(self.get_next_token_state(state, token_id)?);
        } else {
            let token_str = self
                .vocabulary
                .get(&token_id)
                .ok_or_else(|| Error::InvalidTokenId(token_id as usize))?;
            let mut current_state = state;
            for byte_ref in token_str.as_bytes().iter() {
                let next_state = self.get_next_byte_state(current_state, *byte_ref)?;
                state_seq.push(next_state);
                current_state = next_state;
            }
        }
        Ok(state_seq)
    }

    pub fn get_byte_state_sequence(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<Vec<StateId>> {
        if let Some(path) = self.get_cached_token_state_path(state, token_id) {
            return Ok(path.to_vec());
        }
        self.compute_byte_state_sequence(state, token_id)
    }

    pub fn get_byte_transition_sequence(&self, string: &str) -> Result<Vec<(Byte, StateId)>> {
        let mut seq: Vec<(Byte, StateId)> = vec![(0, self.initial_state)];
        let mut current_state = self.initial_state;
        for byte_ref in string.as_bytes().iter() {
            let next_state = self.get_next_byte_state(current_state, *byte_ref)?;
            seq.push((*byte_ref, next_state));
            current_state = next_state;
        }
        Ok(seq)
    }
}

/// Per-token diversity counts returned by [`DiverseGuideDFA::compute_counts`].
/// All three vecs are parallel: index `i` refers to the same token.
#[derive(Debug)]
pub struct TokenCounts {
    /// Token IDs allowed from the queried state.
    pub token_ids: Vec<TokenId>,
    /// Minimum path-counter value along each token's byte-state path (global reward signal).
    /// Initialized to `u32::MAX` and saturated to 0 for tokens with no path history.
    pub reward_counts: Vec<u32>,
    /// Maximum local-state-counter value along each token's byte-state path (penalty signal).
    /// Baseline is 1 (not 0) to prevent division-by-zero in the logit adjustment formula.
    pub penalty_counts: Vec<u32>,
}

/// Per-token diversity adjustments returned by [`DiverseGuideDFA::compute_adjustments`].
/// Both vecs are parallel: index `i` refers to the same token.
#[derive(Debug)]
pub struct TokenAdjustments {
    /// Token IDs allowed from the queried state.
    pub token_ids: Vec<TokenId>,
    /// Precomputed diversity adjustment values for each allowed token.
    pub adjustments: Vec<f32>,
}

/// DFA guide for diverse structured generation.
/// Holds a shared immutable `DfaIndex` and mutable per-run counters.
#[derive(Debug, Clone)]
pub struct DiverseGuideDFA {
    pub(crate) index: Arc<DfaIndex>,
    path_counter: Vec<u32>,
    local_state_counter: Vec<u32>,
}

impl PartialEq for DiverseGuideDFA {
    fn eq(&self, other: &Self) -> bool {
        *self.index == *other.index
            && self.path_counter == other.path_counter
            && self.local_state_counter == other.local_state_counter
    }
}

impl bincode::Encode for DiverseGuideDFA {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        (*self.index).encode(encoder)?;
        self.path_counter.encode(encoder)?;
        self.local_state_counter.encode(encoder)?;
        Ok(())
    }
}

impl<Context> bincode::Decode<Context> for DiverseGuideDFA {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            index: Arc::new(DfaIndex::decode(decoder)?),
            path_counter: Vec::decode(decoder)?,
            local_state_counter: Vec::decode(decoder)?,
        })
    }
}

impl<'de, Context> bincode::BorrowDecode<'de, Context> for DiverseGuideDFA {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        <Self as bincode::Decode<Context>>::decode(decoder)
    }
}

impl DiverseGuideDFA {
    pub fn new(
        regex: &str,
        eos_token_id: TokenId,
        vocabulary: HashMap<TokenId, String>,
    ) -> Result<Self> {
        let index = Arc::new(DfaIndex::new(regex, eos_token_id, vocabulary)?);
        Ok(Self {
            path_counter: vec![0; index.edge_to_idx.len()],
            local_state_counter: vec![0; index.state_to_idx.len()],
            index,
        })
    }

    /// Create a new `DiverseGuideDFA` sharing the same index with fresh counters.
    /// O(1): only clones the Arc reference, not the underlying index data.
    pub fn fork(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            path_counter: vec![0; self.index.edge_to_idx.len()],
            local_state_counter: vec![0; self.index.state_to_idx.len()],
        }
    }

    // --- Delegates to DfaIndex ---

    pub fn is_initial_state(&self, state: StateId) -> bool {
        self.index.is_initial_state(state)
    }

    pub fn is_final_state(&self, state: StateId) -> bool {
        self.index.is_final_state(state)
    }

    pub fn get_initial_state(&self) -> StateId {
        self.index.get_initial_state()
    }

    pub fn get_final_states(&self) -> Vec<StateId> {
        self.index.get_final_states()
    }

    pub fn get_states(&self) -> Vec<StateId> {
        self.index.get_states()
    }

    pub fn get_transitions(&self) -> HashMap<StateId, HashMap<Byte, StateId>> {
        self.index.get_transitions()
    }

    pub fn get_state_sequence(&self, string: &str) -> Result<Vec<StateId>> {
        self.index.get_state_sequence(string)
    }

    pub fn get_transition_sequence(&self, string: &str) -> Result<Vec<(Byte, StateId)>> {
        self.index.get_transition_sequence(string)
    }

    pub fn get_allowed_bytes(&self, state: StateId) -> Result<Vec<Byte>> {
        self.index.get_allowed_bytes(state)
    }

    pub fn get_allowed_token_ids(&self, state: StateId) -> Result<Vec<TokenId>> {
        self.index.get_allowed_token_ids(state)
    }

    pub fn get_distance_to_accepting(&self, state: StateId) -> Result<u32> {
        self.index.get_distance_to_accepting(state)
    }

    pub fn get_next_byte_state(&self, state: StateId, byte: Byte) -> Result<StateId> {
        self.index.get_next_byte_state(state, byte)
    }

    pub fn get_next_token_state(&self, state: StateId, token_id: TokenId) -> Result<StateId> {
        self.index.get_next_token_state(state, token_id)
    }

    pub fn get_byte_state_sequence(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<Vec<StateId>> {
        self.index.get_byte_state_sequence(state, token_id)
    }

    pub fn get_byte_transition_sequence(&self, string: &str) -> Result<Vec<(Byte, StateId)>> {
        self.index.get_byte_transition_sequence(string)
    }

    // --- Counter methods (mutable) ---

    pub fn update_path_counter(&mut self, string: &str) -> Result<()> {
        let mut current_state = self.index.initial_state;
        for byte in string.as_bytes() {
            let next_state = self.index.get_next_byte_state(current_state, *byte)?;
            if let Some(&edge_idx) = self.index.edge_to_idx.get(&(current_state, next_state)) {
                self.path_counter[edge_idx] += 1;
            }
            current_state = next_state;
        }
        Ok(())
    }

    pub fn update_local_state_counter(&mut self, state: StateId, token_id: TokenId) -> Result<()> {
        if let Some(info) = self.index.find_allowed_token_info(state, token_id) {
            for &state_idx in &info.state_indices {
                self.local_state_counter[state_idx] += 1;
            }
            return Ok(());
        }

        for s in self.index.get_byte_state_sequence(state, token_id)?[1..].iter() {
            if let Some(&state_idx) = self.index.state_to_idx.get(s) {
                self.local_state_counter[state_idx] += 1;
            }
        }
        Ok(())
    }

    pub fn reset_path_counter(&mut self) {
        self.path_counter.fill(0);
    }

    pub fn reset_local_state_counter(&mut self) {
        self.local_state_counter.fill(0);
    }

    pub fn compute_counts(&self, state: StateId) -> Result<TokenCounts> {
        let infos = self.index.get_allowed_token_infos(state)?;
        let mut token_ids = Vec::with_capacity(infos.len());
        let mut reward_counts = Vec::with_capacity(infos.len());
        let mut penalty_counts = Vec::with_capacity(infos.len());

        for info in infos {
            token_ids.push(info.token_id);
            let mut minimal_path_count = u32::MAX;
            for &edge_idx in &info.edge_indices {
                minimal_path_count = minimal_path_count.min(self.path_counter[edge_idx]);
            }
            reward_counts.push(minimal_path_count);
            let mut maximal_local_state_count: u32 = 1;
            for &state_idx in &info.state_indices {
                maximal_local_state_count =
                    maximal_local_state_count.max(self.local_state_counter[state_idx]);
            }
            penalty_counts.push(maximal_local_state_count);
        }

        Ok(TokenCounts {
            token_ids,
            reward_counts,
            penalty_counts,
        })
    }

    pub fn compute_adjustments(
        &self,
        state: StateId,
        beta: f32,
        no_reward: bool,
        no_penalty: bool,
        completion_bias_eta: f32,
    ) -> Result<TokenAdjustments> {
        let counts = self.compute_counts(state)?;
        let current_distance = self.index.get_distance_to_accepting(state)? as f32;
        let reward_sum: f32 = counts.reward_counts.iter().map(|&count| count as f32).sum();
        let reward_numerator = (1.0 + reward_sum).ln();
        let mut adjustments = Vec::with_capacity(counts.token_ids.len());

        for ((&token_id, &reward_count), &penalty_count) in counts
            .token_ids
            .iter()
            .zip(&counts.reward_counts)
            .zip(&counts.penalty_counts)
        {
            let reward = if no_reward {
                1.0
            } else {
                reward_numerator / (1.0 + reward_count as f32)
            };
            let penalty = if no_penalty {
                1.0
            } else {
                beta * (penalty_count as f32).powi(2)
            };
            let base_adjustment = reward / penalty;
            let completion_bonus = if completion_bias_eta > 0.0 {
                let next_distance =
                    self.index.get_distance_to_accepting(self.index.get_next_token_state(state, token_id)?)?
                        as f32;
                let progress = (current_distance - next_distance).max(0.0);
                completion_bias_eta * progress / (1.0 + current_distance)
            } else {
                0.0
            };
            adjustments.push(base_adjustment + completion_bonus);
        }

        Ok(TokenAdjustments {
            token_ids: counts.token_ids,
            adjustments,
        })
    }
}

impl std::fmt::Display for DiverseGuideDFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DiverseGuideDFA with transitions:")?;
        for (state_id, input) in self.index.byte_transitions.iter() {
            writeln!(f, "{:?} -> {:#?},", state_id, input)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use bincode::config;
    use rustc_hash::FxHashMap as Map;

    use super::{DiverseGuideDFA, Error, TokenId};

    fn toy_vocab(eos: TokenId) -> Map<TokenId, String> {
        let mut vocab = Map::default();
        vocab.insert(eos, "<eos>".to_string());
        vocab.insert(1, "a".to_string());
        vocab.insert(2, "b".to_string());
        vocab.insert(3, "ab".to_string());
        vocab.insert(4, "x".to_string());
        vocab
    }

    fn toy_dfa() -> DiverseGuideDFA {
        let eos = 999;
        DiverseGuideDFA::new("ab", eos, toy_vocab(eos)).expect("toy DFA should build")
    }

    fn counts_by_token(dfa: &DiverseGuideDFA, state: u32) -> Map<TokenId, (u32, u32)> {
        let counts = dfa
            .compute_counts(state)
            .expect("compute_counts should succeed");
        let mut out = Map::default();
        for i in 0..counts.token_ids.len() {
            out.insert(
                counts.token_ids[i],
                (counts.reward_counts[i], counts.penalty_counts[i]),
            );
        }
        out
    }

    fn token_ids_from_counts(dfa: &DiverseGuideDFA, state: u32) -> Vec<TokenId> {
        dfa.compute_counts(state)
            .expect("compute_counts should succeed")
            .token_ids
    }

    fn assert_close(left: f32, right: f32) {
        assert!(
            (left - right).abs() < 1e-6,
            "left {left} should be close to right {right}"
        );
    }

    fn expected_adjustment(
        reward_sum: u32,
        reward_count: u32,
        penalty_count: u32,
        beta: f32,
        no_reward: bool,
        no_penalty: bool,
    ) -> f32 {
        let reward = if no_reward {
            1.0
        } else {
            (1.0 + reward_sum as f32).ln() / (1.0 + reward_count as f32)
        };
        let penalty = if no_penalty {
            1.0
        } else {
            beta * (penalty_count as f32).powi(2)
        };
        reward / penalty
    }

    #[test]
    fn new_rejects_invalid_regex() {
        let err = DiverseGuideDFA::new("[", 0, Map::default()).expect_err("must fail");
        assert!(matches!(err, Error::IndexDfaError(_)));
    }

    #[test]
    fn state_and_transition_queries_work_for_toy_regex() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        assert!(dfa.is_initial_state(initial));
        assert!(dfa.get_states().contains(&initial));

        let seq = dfa
            .get_state_sequence("ab")
            .expect("state sequence must be valid");
        assert_eq!(seq.len(), 3);
        let final_state = *seq.last().expect("non-empty sequence");
        assert!(dfa.is_final_state(final_state));
        assert!(dfa.get_final_states().contains(&final_state));

        let transition_seq = dfa
            .get_transition_sequence("ab")
            .expect("transition sequence must be valid");
        assert_eq!(transition_seq.len(), 3);
        assert_eq!(transition_seq[1].0, b'a');
        assert_eq!(transition_seq[2].0, b'b');

        let next_after_a = dfa
            .get_next_byte_state(initial, b'a')
            .expect("a transition must exist");
        assert_eq!(next_after_a, seq[1]);
        let next_after_b = dfa
            .get_next_byte_state(next_after_a, b'b')
            .expect("b transition must exist");
        assert_eq!(next_after_b, final_state);
    }

    #[test]
    fn invalid_state_and_invalid_token_paths_return_expected_errors() {
        let dfa = toy_dfa();
        let invalid_state_err = dfa
            .get_allowed_token_ids(u32::MAX)
            .expect_err("invalid state must fail");
        assert!(matches!(invalid_state_err, Error::InvalidState(_)));

        let initial = dfa.get_initial_state();
        let no_token_err = dfa
            .get_next_token_state(initial, 4)
            .expect_err("token id 4 ('x') should not be valid from initial");
        assert!(matches!(no_token_err, Error::NoTokenTransitionFound(_, _)));
    }

    #[test]
    fn eos_transition_only_available_at_final_state() {
        let dfa = toy_dfa();
        let eos = 999;
        let initial = dfa.get_initial_state();
        let final_state = *dfa
            .get_state_sequence("ab")
            .expect("state sequence must be valid")
            .last()
            .expect("non-empty sequence");

        let initial_tokens = dfa
            .get_allowed_token_ids(initial)
            .expect("allowed tokens from initial");
        assert!(!initial_tokens.contains(&eos));

        let final_tokens = dfa
            .get_allowed_token_ids(final_state)
            .expect("allowed tokens from final");
        assert!(final_tokens.contains(&eos));
    }

    #[test]
    fn counters_affect_compute_counts_and_can_be_reset() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();

        let before = counts_by_token(&dfa, initial);
        // Default: no path history -> reward 0, no local history -> penalty 1.
        assert_eq!(before.get(&1), Some(&(0, 1)));
        assert_eq!(before.get(&3), Some(&(0, 1)));

        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("second local counter update should succeed");

        let after = counts_by_token(&dfa, initial);
        assert_eq!(after.get(&1), Some(&(1, 2)));
        assert_eq!(after.get(&3), Some(&(1, 2)));

        dfa.reset_path_counter();
        dfa.reset_local_state_counter();
        let reset = counts_by_token(&dfa, initial);
        assert_eq!(reset.get(&1), Some(&(0, 1)));
        assert_eq!(reset.get(&3), Some(&(0, 1)));
    }

    #[test]
    fn compute_adjustments_zero_counters_match_formula() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let counts = dfa
            .compute_counts(initial)
            .expect("compute_counts should succeed");
        let adjustments = dfa
            .compute_adjustments(initial, 3.0, false, false, 0.0)
            .expect("compute_adjustments should succeed");

        assert_eq!(adjustments.token_ids, counts.token_ids);
        assert!(adjustments
            .adjustments
            .iter()
            .all(|&adjustment| adjustment == 0.0));
    }

    #[test]
    fn compute_adjustments_populated_counters_match_formula() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("second local counter update should succeed");

        let counts = dfa
            .compute_counts(initial)
            .expect("compute_counts should succeed");
        let adjustments = dfa
            .compute_adjustments(initial, 3.0, false, false, 0.0)
            .expect("compute_adjustments should succeed");
        let reward_sum: u32 = counts.reward_counts.iter().sum();

        assert_eq!(adjustments.token_ids, counts.token_ids);
        for ((&reward_count, &penalty_count), &adjustment) in counts
            .reward_counts
            .iter()
            .zip(&counts.penalty_counts)
            .zip(&adjustments.adjustments)
        {
            assert_close(
                adjustment,
                expected_adjustment(reward_sum, reward_count, penalty_count, 3.0, false, false),
            );
        }
    }

    #[test]
    fn compute_adjustments_ablation_flags_match_formula() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("second local counter update should succeed");

        let counts = dfa
            .compute_counts(initial)
            .expect("compute_counts should succeed");
        let reward_sum: u32 = counts.reward_counts.iter().sum();
        for (no_reward, no_penalty) in [(true, false), (false, true), (true, true)] {
            let adjustments = dfa
                .compute_adjustments(initial, 3.0, no_reward, no_penalty, 0.0)
                .expect("compute_adjustments should succeed");
            assert_eq!(adjustments.token_ids, counts.token_ids);
            for ((&reward_count, &penalty_count), &adjustment) in counts
                .reward_counts
                .iter()
                .zip(&counts.penalty_counts)
                .zip(&adjustments.adjustments)
            {
                assert_close(
                    adjustment,
                    expected_adjustment(
                        reward_sum,
                        reward_count,
                        penalty_count,
                        3.0,
                        no_reward,
                        no_penalty,
                    ),
                );
            }
        }
    }

    #[test]
    fn fork_shares_index_but_resets_counters() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("second local counter update should succeed");

        let forked = dfa.fork();
        let original_counts = counts_by_token(&dfa, initial);
        let forked_counts = counts_by_token(&forked, initial);
        assert_eq!(original_counts.get(&3), Some(&(1, 2)));
        assert_eq!(forked_counts.get(&3), Some(&(0, 1)));
    }

    #[test]
    fn bincode_roundtrip_preserves_state() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");

        let bytes =
            bincode::encode_to_vec(&dfa, config::standard()).expect("encoding must succeed");
        let (decoded, _): (DiverseGuideDFA, usize) =
            bincode::decode_from_slice(&bytes, config::standard()).expect("decoding must succeed");
        assert_eq!(decoded, dfa);
        assert_eq!(
            counts_by_token(&decoded, initial),
            counts_by_token(&dfa, initial)
        );
    }

    #[test]
    fn allowed_bytes_match_transition_table_keys() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let transitions = dfa.get_transitions();
        let mut allowed = dfa
            .get_allowed_bytes(initial)
            .expect("allowed bytes from initial state");
        allowed.sort_unstable();

        let mut from_transition_map: Vec<u8> = transitions
            .get(&initial)
            .expect("initial state must exist in transitions")
            .keys()
            .copied()
            .collect();
        from_transition_map.sort_unstable();
        assert_eq!(allowed, from_transition_map);
    }

    #[test]
    fn byte_state_sequence_handles_eos_and_invalid_token_id() {
        let dfa = toy_dfa();
        let eos = 999;
        let initial = dfa.get_initial_state();
        let final_state = *dfa
            .get_state_sequence("ab")
            .expect("state sequence must be valid")
            .last()
            .expect("non-empty sequence");

        let eos_seq = dfa
            .get_byte_state_sequence(final_state, eos)
            .expect("eos sequence should be valid at final state");
        assert_eq!(eos_seq, vec![final_state, final_state]);

        let err = dfa
            .get_byte_state_sequence(initial, 123456)
            .expect_err("unknown token id should fail");
        assert!(matches!(err, Error::InvalidTokenId(_)));
    }

    #[test]
    fn cached_token_state_paths_match_byte_state_walk() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let token_path = dfa
            .get_byte_state_sequence(initial, 3)
            .expect("token 'ab' path should be cached");
        let string_path = dfa
            .get_state_sequence("ab")
            .expect("string path should be valid");
        assert_eq!(token_path, string_path);
    }

    #[test]
    fn allowed_token_metadata_matches_transition_and_state_path_maps() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let infos = dfa
            .index
            .get_allowed_token_infos(initial)
            .expect("metadata for initial state");

        for info in infos {
            let next_state = dfa
                .index
                .token_transitions
                .get(&initial)
                .and_then(|tokens| tokens.get(&info.token_id))
                .copied()
                .expect("metadata token must have transition");
            assert_eq!(info.next_state, next_state);

            let state_path = dfa
                .index
                .token_state_paths
                .get(&initial)
                .and_then(|paths| paths.get(&info.token_id))
                .expect("metadata token must have state path");
            let expected_edges: Vec<(u32, u32)> =
                state_path.windows(2).map(|w| (w[0], w[1])).collect();
            assert_eq!(info.edge_path, expected_edges);
            assert_eq!(info.inner_states, state_path[1..].to_vec());
            assert_eq!(info.edge_indices.len(), info.edge_path.len());
            assert_eq!(info.state_indices.len(), info.inner_states.len());
        }
    }

    #[test]
    fn dense_metadata_indices_match_state_id_paths() {
        let dfa = toy_dfa();

        for infos in dfa.index.allowed_token_infos.values() {
            for info in infos {
                for (edge, edge_idx) in info.edge_path.iter().zip(&info.edge_indices) {
                    assert_eq!(dfa.index.edge_to_idx.get(edge), Some(edge_idx));
                }
                for (state, state_idx) in info.inner_states.iter().zip(&info.state_indices) {
                    assert_eq!(dfa.index.state_to_idx.get(state), Some(state_idx));
                }
            }
        }
    }

    #[test]
    fn dense_index_scope_is_metadata_referenced_states_and_edges() {
        let dfa = toy_dfa();
        let mut expected_states = Map::default();
        let mut expected_edges = Map::default();

        for infos in dfa.index.allowed_token_infos.values() {
            for info in infos {
                for &state in &info.inner_states {
                    expected_states.insert(state, ());
                }
                for &edge in &info.edge_path {
                    expected_edges.insert(edge, ());
                }
            }
        }

        assert_eq!(dfa.index.state_to_idx.len(), expected_states.len());
        assert_eq!(dfa.index.edge_to_idx.len(), expected_edges.len());
        for state in expected_states.keys() {
            assert!(dfa.index.state_to_idx.contains_key(state));
        }
        for edge in expected_edges.keys() {
            assert!(dfa.index.edge_to_idx.contains_key(edge));
        }
    }

    #[test]
    fn allowed_token_order_matches_compute_counts_order() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let allowed = dfa
            .get_allowed_token_ids(initial)
            .expect("allowed tokens from initial");
        assert_eq!(allowed, token_ids_from_counts(&dfa, initial));
    }

    #[test]
    fn eos_metadata_is_final_state_self_edge() {
        let dfa = toy_dfa();
        let eos = 999;
        let final_state = *dfa
            .get_state_sequence("ab")
            .expect("state sequence must be valid")
            .last()
            .expect("non-empty sequence");
        let eos_info = dfa
            .index
            .get_allowed_token_infos(final_state)
            .expect("metadata for final state")
            .iter()
            .find(|info| info.token_id == eos)
            .expect("EOS metadata must exist at final state");

        assert_eq!(eos_info.next_state, final_state);
        assert_eq!(eos_info.edge_path, vec![(final_state, final_state)]);
        assert_eq!(eos_info.inner_states, vec![final_state]);
        assert_eq!(
            dfa.index.edge_to_idx.get(&(final_state, final_state)),
            eos_info.edge_indices.first()
        );
        assert_eq!(
            dfa.index.state_to_idx.get(&final_state),
            eos_info.state_indices.first()
        );
    }

    #[test]
    fn update_methods_surface_transition_and_state_errors() {
        let mut dfa = toy_dfa();
        let invalid_path_err = dfa
            .update_path_counter("ax")
            .expect_err("x is not accepted after a in regex ab");
        assert!(matches!(invalid_path_err, Error::NoTransitionFound(_, _)));

        let invalid_state_err = dfa
            .compute_counts(u32::MAX)
            .expect_err("invalid state should fail");
        assert!(matches!(invalid_state_err, Error::InvalidState(_)));

        let invalid_adjustments_err = dfa
            .compute_adjustments(u32::MAX, 3.0, false, false, 0.0)
            .expect_err("invalid state should fail");
        assert!(matches!(invalid_adjustments_err, Error::InvalidState(_)));
    }

    #[test]
    fn distance_to_accepting_matches_token_level_shortest_path() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let final_state = *dfa
            .get_state_sequence("ab")
            .expect("state sequence must be valid")
            .last()
            .expect("non-empty sequence");

        assert_eq!(dfa.get_distance_to_accepting(initial).unwrap(), 1);
        assert_eq!(dfa.get_distance_to_accepting(final_state).unwrap(), 0);
    }

    #[test]
    fn completion_bias_bonus_prefers_tokens_closer_to_acceptance() {
        let mut vocab = Map::default();
        vocab.insert(999, "<eos>".to_string());
        vocab.insert(1, "a".to_string());
        vocab.insert(4, "b".to_string());
        vocab.insert(2, "ab".to_string());
        let dfa = DiverseGuideDFA::new("(?:ab)$", 999, vocab).expect("dfa must build");
        let initial = dfa.get_initial_state();
        let adjustments = dfa
            .compute_adjustments(initial, 3.0, false, false, 0.2)
            .expect("compute_adjustments should succeed");
        let by_token: Map<TokenId, f32> = adjustments
            .token_ids
            .iter()
            .copied()
            .zip(adjustments.adjustments.iter().copied())
            .collect();

        assert_close(*by_token.get(&1).expect("token a must exist"), 0.0);
        assert_close(*by_token.get(&2).expect("token ab must exist"), 0.1);
    }

    #[test]
    fn display_includes_expected_header() {
        let dfa = toy_dfa();
        let display = format!("{}", dfa);
        assert!(display.contains("DiverseGuideDFA with transitions:"));
    }
}
