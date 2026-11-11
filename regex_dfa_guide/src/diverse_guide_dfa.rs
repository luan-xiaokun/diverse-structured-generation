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
    edge_indices: Vec<usize>,
    state_indices: Vec<usize>,
}

/// Byte trie over the vocabulary, used to share prefix walks while building
/// token-level transitions. Children are kept as a short vector because most
/// nodes have very few outgoing bytes.
#[derive(Debug, Default)]
struct VocabTrieNode {
    children: Vec<(Byte, usize)>,
    token_ids: Vec<TokenId>,
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
        // construct token level transitions by walking the vocabulary trie from
        // every DFA state, so tokens sharing a byte prefix share the walk
        let vocab_trie = Self::build_vocabulary_trie(&vocabulary, eos_token_id);
        let (mut token_transitions, mut token_state_paths) =
            Self::build_token_level_transitions(&states, &transitions, &vocab_trie);
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

        let (state_to_idx, edge_to_idx) = Self::build_dense_counter_indices(&token_state_paths);
        let allowed_token_infos = Self::build_allowed_token_infos(
            &token_transitions,
            &token_state_paths,
            &state_to_idx,
            &edge_to_idx,
        );
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
            state_to_idx,
            edge_to_idx,
            vocabulary,
        })
    }

    fn build_vocabulary_trie(
        vocabulary: &HashMap<TokenId, String>,
        eos_token_id: TokenId,
    ) -> Vec<VocabTrieNode> {
        let mut trie = vec![VocabTrieNode::default()];
        let mut token_ids: Vec<TokenId> = vocabulary
            .keys()
            .copied()
            .filter(|token_id| *token_id != eos_token_id)
            .collect();
        token_ids.sort_unstable();

        for token_id in token_ids {
            let token_str = vocabulary
                .get(&token_id)
                .expect("token id collected from vocabulary must exist");
            let mut node_idx = 0;
            for &byte in token_str.as_bytes() {
                node_idx = if let Some(&(_, child_idx)) = trie[node_idx]
                    .children
                    .iter()
                    .find(|(child_byte, _)| *child_byte == byte)
                {
                    child_idx
                } else {
                    let child_idx = trie.len();
                    trie.push(VocabTrieNode::default());
                    trie[node_idx].children.push((byte, child_idx));
                    child_idx
                };
            }
            trie[node_idx].token_ids.push(token_id);
        }

        trie
    }

    #[allow(clippy::type_complexity)]
    fn build_token_level_transitions(
        states: &HashSet<StateId>,
        byte_transitions: &HashMap<StateId, HashMap<Byte, StateId>>,
        vocab_trie: &[VocabTrieNode],
    ) -> (
        HashMap<StateId, HashMap<TokenId, StateId>>,
        HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
    ) {
        let mut token_transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
        let mut token_state_paths: HashMap<StateId, HashMap<TokenId, Vec<StateId>>> =
            HashMap::default();

        for &current_state in states {
            let mut state_path = vec![current_state];
            Self::walk_vocabulary_trie_from_state(
                current_state,
                0,
                current_state,
                byte_transitions,
                vocab_trie,
                &mut state_path,
                &mut token_transitions,
                &mut token_state_paths,
            );
        }

        (token_transitions, token_state_paths)
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_vocabulary_trie_from_state(
        root_state: StateId,
        node_idx: usize,
        dfa_state: StateId,
        byte_transitions: &HashMap<StateId, HashMap<Byte, StateId>>,
        vocab_trie: &[VocabTrieNode],
        state_path: &mut Vec<StateId>,
        token_transitions: &mut HashMap<StateId, HashMap<TokenId, StateId>>,
        token_state_paths: &mut HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
    ) {
        // the root node holds tokens with an empty byte string, which (as in
        // the original per-token walk) stay in the current state
        for &token_id in &vocab_trie[node_idx].token_ids {
            token_transitions
                .entry(root_state)
                .or_default()
                .insert(token_id, dfa_state);
            token_state_paths
                .entry(root_state)
                .or_default()
                .insert(token_id, state_path.clone());
        }

        let Some(transitions) = byte_transitions.get(&dfa_state) else {
            return;
        };
        for &(byte, child_idx) in &vocab_trie[node_idx].children {
            if let Some(&next_state) = transitions.get(&byte) {
                state_path.push(next_state);
                Self::walk_vocabulary_trie_from_state(
                    root_state,
                    child_idx,
                    next_state,
                    byte_transitions,
                    vocab_trie,
                    state_path,
                    token_transitions,
                    token_state_paths,
                );
                state_path.pop();
            }
        }
    }

    fn build_allowed_token_infos(
        token_transitions: &HashMap<StateId, HashMap<TokenId, StateId>>,
        token_state_paths: &HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
        state_to_idx: &HashMap<StateId, usize>,
        edge_to_idx: &HashMap<(StateId, StateId), usize>,
    ) -> HashMap<StateId, Vec<AllowedTokenInfo>> {
        let mut allowed_token_infos: HashMap<StateId, Vec<AllowedTokenInfo>> = HashMap::default();
        for (state, token_map) in token_transitions {
            let mut infos = Vec::with_capacity(token_map.len());
            for (token_id, next_state) in token_map {
                if let Some(state_path) = token_state_paths
                    .get(state)
                    .and_then(|paths| paths.get(token_id))
                {
                    let edge_indices = state_path
                        .windows(2)
                        .map(|w| {
                            *edge_to_idx
                                .get(&(w[0], w[1]))
                                .expect("metadata edge path must have a dense edge index")
                        })
                        .collect();
                    let state_indices = state_path[1..]
                        .iter()
                        .map(|state| {
                            *state_to_idx
                                .get(state)
                                .expect("metadata inner state must have a dense state index")
                        })
                        .collect();
                    infos.push(AllowedTokenInfo {
                        token_id: *token_id,
                        next_state: *next_state,
                        edge_indices,
                        state_indices,
                    });
                }
            }
            if !infos.is_empty() {
                infos.sort_unstable_by_key(|info| info.token_id);
                allowed_token_infos.insert(*state, infos);
            }
        }
        allowed_token_infos
    }

    fn build_dense_counter_indices(
        token_state_paths: &HashMap<StateId, HashMap<TokenId, Vec<StateId>>>,
    ) -> (HashMap<StateId, usize>, HashMap<(StateId, StateId), usize>) {
        let mut state_to_idx: HashMap<StateId, usize> = HashMap::default();
        let mut edge_to_idx: HashMap<(StateId, StateId), usize> = HashMap::default();

        for path_map in token_state_paths.values() {
            for state_path in path_map.values() {
                for &state in &state_path[1..] {
                    if !state_to_idx.contains_key(&state) {
                        state_to_idx.insert(state, state_to_idx.len());
                    }
                }
                for edge in state_path.windows(2).map(|w| (w[0], w[1])) {
                    if !edge_to_idx.contains_key(&edge) {
                        edge_to_idx.insert(edge, edge_to_idx.len());
                    }
                }
            }
        }

        (state_to_idx, edge_to_idx)
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
            for representative in classes.representatives(0..=255) {
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
        // allowed token metadata is sorted by token id at construction time
        let infos = self.allowed_token_infos.get(&state)?;
        infos
            .binary_search_by_key(&token_id, |info| info.token_id)
            .ok()
            .map(|idx| &infos[idx])
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

    /// Commit one completed sample, given as generated token ids, to the global
    /// path counter.
    ///
    /// The sample is counted only if every token follows an allowed transition
    /// and the sequence ends in an accepting state. Tokens after the first EOS
    /// (e.g. batch padding) are ignored, and the EOS transition is counted
    /// exactly once whether or not it was generated. Returns `false` and
    /// leaves the counter untouched for invalid or unfinished samples.
    pub fn commit_valid_token_path(&mut self, token_ids: &[TokenId]) -> Result<bool> {
        let mut current_state = self.index.initial_state;
        let mut edge_indices = Vec::new();
        let mut saw_eos = false;

        for &token_id in token_ids {
            let Some(info) = self.index.find_allowed_token_info(current_state, token_id) else {
                return Ok(false);
            };
            edge_indices.extend_from_slice(&info.edge_indices);
            current_state = info.next_state;
            if token_id == self.index.eos_token_id {
                saw_eos = true;
                break;
            }
        }

        self.commit_completed_edges(current_state, edge_indices, saw_eos)
    }

    /// Commit one completed sample, given as text, to the global path counter.
    ///
    /// Same acceptance rule as [`Self::commit_valid_token_path`], but the path
    /// is recovered by replaying the string's bytes through the DFA.
    pub fn commit_valid_string(&mut self, string: &str) -> Result<bool> {
        let mut current_state = self.index.initial_state;
        let mut edge_indices = Vec::new();

        for byte in string.as_bytes() {
            let Ok(next_state) = self.index.get_next_byte_state(current_state, *byte) else {
                return Ok(false);
            };
            if let Some(&edge_idx) = self.index.edge_to_idx.get(&(current_state, next_state)) {
                edge_indices.push(edge_idx);
            }
            current_state = next_state;
        }

        self.commit_completed_edges(current_state, edge_indices, false)
    }

    fn commit_completed_edges(
        &mut self,
        end_state: StateId,
        mut edge_indices: Vec<usize>,
        saw_eos: bool,
    ) -> Result<bool> {
        if !self.index.is_final_state(end_state) {
            return Ok(false);
        }
        if !saw_eos {
            let eos_token_id = self.index.eos_token_id;
            let eos_info = self
                .index
                .find_allowed_token_info(end_state, eos_token_id)
                .ok_or(Error::NoTokenTransitionFound(
                    end_state as usize,
                    eos_token_id as usize,
                ))?;
            edge_indices.extend_from_slice(&eos_info.edge_indices);
        }
        for edge_idx in edge_indices {
            self.path_counter[edge_idx] += 1;
        }
        Ok(true)
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
    ) -> Result<TokenAdjustments> {
        let counts = self.compute_counts(state)?;
        // Sum the integer counts exactly before converting: a running f32 sum
        // loses precision once it exceeds 2^24, which large allowed-token sets
        // (e.g. negative constraints) reach after a few hundred samples.
        let reward_sum = counts
            .reward_counts
            .iter()
            .map(|&count| u64::from(count))
            .sum::<u64>() as f32;
        let reward_numerator = (1.0 + reward_sum).ln();
        let mut adjustments = Vec::with_capacity(counts.token_ids.len());

        for (&reward_count, &penalty_count) in
            counts.reward_counts.iter().zip(&counts.penalty_counts)
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
            adjustments.push(reward / penalty);
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
    use regex_automata::dfa::dense::DFA;
    use regex_automata::dfa::Automaton;
    use regex_automata::util::syntax;
    use regex_automata::Anchored;
    use rustc_hash::FxHashMap as Map;

    use super::{DfaIndex, DiverseGuideDFA, Error, TokenId};

    const EOS: TokenId = 999;

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
        DiverseGuideDFA::new("ab", EOS, toy_vocab(EOS)).expect("toy DFA should build")
    }

    fn toy_final_state(dfa: &DiverseGuideDFA) -> u32 {
        *dfa.get_state_sequence("ab")
            .expect("state sequence must be valid")
            .last()
            .expect("non-empty sequence")
    }

    /// Toy DFA with one full path count on "ab" and two local visits after token "ab".
    fn toy_dfa_with_counters() -> DiverseGuideDFA {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        dfa.update_path_counter("ab")
            .expect("path counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        dfa.update_local_state_counter(initial, 3)
            .expect("second local counter update should succeed");
        dfa
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

    fn reward_count(dfa: &DiverseGuideDFA, state: u32, token_id: TokenId) -> u32 {
        counts_by_token(dfa, state)
            .get(&token_id)
            .expect("token must be allowed from state")
            .0
    }

    fn all_reward_counts_zero(dfa: &DiverseGuideDFA, state: u32) -> bool {
        counts_by_token(dfa, state)
            .values()
            .all(|(reward, _)| *reward == 0)
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

        let mut allowed = dfa
            .get_allowed_bytes(initial)
            .expect("allowed bytes from initial state");
        allowed.sort_unstable();
        let mut from_transition_map: Vec<u8> = dfa
            .get_transitions()
            .get(&initial)
            .expect("initial state must exist in transitions")
            .keys()
            .copied()
            .collect();
        from_transition_map.sort_unstable();
        assert_eq!(allowed, from_transition_map);

        assert!(format!("{}", dfa).contains("DiverseGuideDFA with transitions:"));
    }

    #[test]
    fn error_paths_return_expected_errors() {
        let err = DiverseGuideDFA::new("[", 0, Map::default()).expect_err("must fail");
        assert!(matches!(err, Error::IndexDfaError(_)));

        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        assert!(matches!(
            dfa.get_allowed_token_ids(u32::MAX),
            Err(Error::InvalidState(_))
        ));
        assert!(matches!(
            dfa.compute_counts(u32::MAX),
            Err(Error::InvalidState(_))
        ));
        assert!(matches!(
            dfa.compute_adjustments(u32::MAX, 3.0, false, false),
            Err(Error::InvalidState(_))
        ));
        // Token 4 ('x') is not valid from the initial state of regex "ab".
        assert!(matches!(
            dfa.get_next_token_state(initial, 4),
            Err(Error::NoTokenTransitionFound(_, _))
        ));
        assert!(matches!(
            dfa.get_byte_state_sequence(initial, 123456),
            Err(Error::InvalidTokenId(_))
        ));
        assert!(matches!(
            dfa.update_path_counter("ax"),
            Err(Error::NoTransitionFound(_, _))
        ));
    }

    #[test]
    fn eos_is_a_final_state_self_edge() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let final_state = toy_final_state(&dfa);

        let initial_tokens = dfa
            .get_allowed_token_ids(initial)
            .expect("allowed tokens from initial");
        assert!(!initial_tokens.contains(&EOS));
        let final_tokens = dfa
            .get_allowed_token_ids(final_state)
            .expect("allowed tokens from final");
        assert!(final_tokens.contains(&EOS));

        let eos_seq = dfa
            .get_byte_state_sequence(final_state, EOS)
            .expect("eos sequence should be valid at final state");
        assert_eq!(eos_seq, vec![final_state, final_state]);

        let eos_info = dfa
            .index
            .get_allowed_token_infos(final_state)
            .expect("metadata for final state")
            .iter()
            .find(|info| info.token_id == EOS)
            .expect("EOS metadata must exist at final state");
        assert_eq!(eos_info.next_state, final_state);
        assert_eq!(eos_info.edge_indices.len(), 1);
        assert_eq!(eos_info.state_indices.len(), 1);
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
    fn allowed_token_metadata_is_consistent_with_index_maps() {
        let dfa = toy_dfa();
        let initial = dfa.get_initial_state();

        // Token "ab" follows the same state path as the string "ab".
        assert_eq!(
            dfa.get_byte_state_sequence(initial, 3)
                .expect("token 'ab' path should be cached"),
            dfa.get_state_sequence("ab")
                .expect("string path should be valid")
        );

        let mut expected_states = Map::default();
        let mut expected_edges = Map::default();
        for (state, infos) in &dfa.index.allowed_token_infos {
            for info in infos {
                let next_state = dfa
                    .index
                    .token_transitions
                    .get(state)
                    .and_then(|tokens| tokens.get(&info.token_id))
                    .copied()
                    .expect("metadata token must have transition");
                assert_eq!(info.next_state, next_state);

                let state_path = dfa
                    .index
                    .token_state_paths
                    .get(state)
                    .and_then(|paths| paths.get(&info.token_id))
                    .expect("metadata token must have state path");
                let edges: Vec<(u32, u32)> = state_path.windows(2).map(|w| (w[0], w[1])).collect();
                assert_eq!(info.edge_indices.len(), edges.len());
                assert_eq!(info.state_indices.len(), state_path[1..].len());
                for (edge, edge_idx) in edges.iter().zip(&info.edge_indices) {
                    assert_eq!(dfa.index.edge_to_idx.get(edge), Some(edge_idx));
                    expected_edges.insert(*edge, ());
                }
                for (path_state, state_idx) in state_path[1..].iter().zip(&info.state_indices) {
                    assert_eq!(dfa.index.state_to_idx.get(path_state), Some(state_idx));
                    expected_states.insert(*path_state, ());
                }
            }
        }

        // Dense indices cover exactly the states and edges referenced by metadata.
        assert_eq!(dfa.index.state_to_idx.len(), expected_states.len());
        assert_eq!(dfa.index.edge_to_idx.len(), expected_edges.len());
    }

    #[test]
    fn allowed_token_order_is_sorted_by_token_id() {
        let mut vocab = Map::default();
        vocab.insert(99, "<eos>".to_string());
        vocab.insert(20, "b".to_string());
        vocab.insert(10, "a".to_string());
        vocab.insert(30, "ab".to_string());
        let dfa = DiverseGuideDFA::new("(?:[ab]|ab)$", 99, vocab).expect("dfa must build");
        let initial = dfa.get_initial_state();

        let allowed = dfa
            .get_allowed_token_ids(initial)
            .expect("allowed tokens from initial");
        assert_eq!(allowed, vec![10, 20, 30]);
        let count_tokens = dfa
            .compute_counts(initial)
            .expect("compute_counts should succeed")
            .token_ids;
        assert_eq!(count_tokens, allowed);
        let adjustment_tokens = dfa
            .compute_adjustments(initial, 3.0, false, false)
            .expect("compute_adjustments should succeed")
            .token_ids;
        assert_eq!(adjustment_tokens, allowed);
    }

    #[test]
    fn token_transition_builder_handles_shared_prefixes_and_duplicate_strings() {
        let mut vocab = Map::default();
        vocab.insert(99, "<eos>".to_string());
        vocab.insert(10, "a".to_string());
        vocab.insert(20, "ab".to_string());
        vocab.insert(30, "abc".to_string());
        vocab.insert(40, "ab".to_string());
        vocab.insert(50, "abc".to_string());
        let dfa = DiverseGuideDFA::new("(?:abc)$", 99, vocab).expect("dfa must build");
        let initial = dfa.get_initial_state();

        let allowed = dfa
            .get_allowed_token_ids(initial)
            .expect("allowed tokens from initial");
        assert_eq!(allowed, vec![30, 50]);
        assert_eq!(
            dfa.get_byte_state_sequence(initial, 30)
                .expect("abc token path should exist")
                .len(),
            4
        );
    }

    #[test]
    fn byte_transition_builder_includes_255() {
        let dfa = DFA::builder()
            .configure(DFA::config().minimize(true))
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build(r"(?-u:\xFF)")
            .expect("byte-oriented DFA must build");
        let start_state = dfa
            .universal_start_state(Anchored::Yes)
            .expect("DFA must have start state");
        let (_, transitions) =
            DfaIndex::build_byte_transitions(&dfa, start_state).expect("transitions must build");

        assert!(transitions
            .get(&start_state.as_u32())
            .expect("start state must have transitions")
            .contains_key(&255));
    }

    #[test]
    fn counters_affect_compute_counts_and_can_be_reset() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();

        let before = counts_by_token(&dfa, initial);
        // Default: no path history -> reward 0, no local history -> penalty 1.
        assert_eq!(before.get(&1), Some(&(0, 1)));
        assert_eq!(before.get(&3), Some(&(0, 1)));

        let mut dfa_with_counts = toy_dfa_with_counters();
        let after = counts_by_token(&dfa_with_counts, initial);
        assert_eq!(after.get(&1), Some(&(1, 2)));
        assert_eq!(after.get(&3), Some(&(1, 2)));

        dfa_with_counts.reset_path_counter();
        dfa_with_counts.reset_local_state_counter();
        assert_eq!(counts_by_token(&dfa_with_counts, initial), before);

        dfa.update_local_state_counter(initial, 3)
            .expect("local counter update should succeed");
        assert_eq!(counts_by_token(&dfa, initial).get(&3), Some(&(0, 1)));
    }

    #[test]
    fn compute_adjustments_match_formula() {
        let fresh = toy_dfa();
        let initial = fresh.get_initial_state();
        let zero = fresh
            .compute_adjustments(initial, 3.0, false, false)
            .expect("compute_adjustments should succeed");
        // Without path history the reward numerator ln(1 + 0) is zero.
        assert!(zero.adjustments.iter().all(|&adjustment| adjustment == 0.0));

        let dfa = toy_dfa_with_counters();
        let counts = dfa
            .compute_counts(initial)
            .expect("compute_counts should succeed");
        let reward_sum: u32 = counts.reward_counts.iter().sum();
        for (no_reward, no_penalty) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let adjustments = dfa
                .compute_adjustments(initial, 3.0, no_reward, no_penalty)
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
        let dfa = toy_dfa_with_counters();
        let initial = dfa.get_initial_state();

        let forked = dfa.fork();
        assert_eq!(counts_by_token(&dfa, initial).get(&3), Some(&(1, 2)));
        assert_eq!(counts_by_token(&forked, initial).get(&3), Some(&(0, 1)));
    }

    #[test]
    fn bincode_roundtrip_preserves_state() {
        let dfa = toy_dfa_with_counters();
        let initial = dfa.get_initial_state();

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
    fn commit_valid_token_path_commits_only_complete_valid_sequences() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();

        // An unfinished prefix and a sequence with an invalid token leave the
        // counter untouched (the commit is atomic).
        assert!(!dfa
            .commit_valid_token_path(&[1])
            .expect("valid prefix should not fail"));
        assert!(!dfa
            .commit_valid_token_path(&[1, 4])
            .expect("invalid token sequence should return false"));
        assert!(all_reward_counts_zero(&dfa, initial));

        assert!(dfa
            .commit_valid_token_path(&[3])
            .expect("complete token path should commit"));
        assert_eq!(reward_count(&dfa, initial, 3), 1);
        assert_eq!(reward_count(&dfa, initial, 1), 1);
    }

    #[test]
    fn commit_valid_token_path_counts_eos_once_and_ignores_padding() {
        let mut implicit_eos = toy_dfa();
        let mut explicit_eos = toy_dfa();
        let mut padded = toy_dfa();
        let initial = implicit_eos.get_initial_state();
        let final_state = implicit_eos
            .get_next_token_state(initial, 3)
            .expect("ab token should be valid");

        assert!(implicit_eos
            .commit_valid_token_path(&[3])
            .expect("implicit EOS path should commit"));
        assert!(explicit_eos
            .commit_valid_token_path(&[3, EOS])
            .expect("explicit EOS path should commit"));
        assert!(padded
            .commit_valid_token_path(&[1, 2, EOS, 4, 4])
            .expect("tokens after EOS should be ignored"));

        let implicit_counts = counts_by_token(&implicit_eos, final_state);
        assert_eq!(implicit_counts.get(&EOS).map(|c| c.0), Some(1));
        assert_eq!(implicit_counts, counts_by_token(&explicit_eos, final_state));
        assert_eq!(implicit_counts, counts_by_token(&padded, final_state));
    }

    #[test]
    fn commit_valid_string_matches_token_commit_and_skips_invalid_strings() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        for rejected in ["a", "ax", "abb"] {
            assert!(!dfa
                .commit_valid_string(rejected)
                .expect("prefix, invalid, or overlong string should not fail"));
        }
        assert!(all_reward_counts_zero(&dfa, initial));

        let mut from_tokens = toy_dfa();
        let final_state = dfa
            .get_next_token_state(initial, 3)
            .expect("ab token should be valid");
        assert!(dfa
            .commit_valid_string("ab")
            .expect("complete string should commit"));
        assert!(from_tokens
            .commit_valid_token_path(&[1, 2])
            .expect("complete token path should commit"));
        for state in [initial, final_state] {
            assert_eq!(
                counts_by_token(&dfa, state),
                counts_by_token(&from_tokens, state)
            );
        }
        assert_eq!(reward_count(&dfa, final_state, EOS), 1);
    }

    #[test]
    fn update_path_counter_keeps_raw_replay_semantics() {
        let mut dfa = toy_dfa();
        let initial = dfa.get_initial_state();
        let final_state = dfa
            .get_next_token_state(initial, 3)
            .expect("ab token should be valid");

        dfa.update_path_counter("a")
            .expect("prefix replay should succeed");
        assert_eq!(reward_count(&dfa, initial, 1), 1);

        dfa.update_path_counter("ab")
            .expect("full replay should succeed");
        assert_eq!(reward_count(&dfa, final_state, EOS), 0);
    }
}
