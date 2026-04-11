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

/// Immutable DFA structure built from a regex and vocabulary.
/// All expensive construction is done once here; instances are Arc-shareable.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct DfaIndex {
    eos_token_id: TokenId,
    initial_state: StateId,
    final_states: HashSet<StateId>,
    byte_transitions: HashMap<StateId, HashMap<Byte, StateId>>,
    token_transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    allowed_token_ids: HashMap<StateId, Vec<TokenId>>,
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
        let mut token_transitions: HashMap<StateId, HashMap<TokenId, StateId>> =
            HashMap::default();
        for &current_state in &states {
            'token_loop: for (token_id, token_str) in vocabulary.iter() {
                if *token_id == eos_token_id {
                    continue;
                }
                let mut next_state = current_state;
                for byte in token_str.as_bytes() {
                    next_state = match transitions.get(&next_state) {
                        Some(trans) => match trans.get(&byte) {
                            Some(state) => *state,
                            None => continue 'token_loop,
                        },
                        None => continue 'token_loop,
                    }
                }
                token_transitions
                    .entry(current_state)
                    .or_default()
                    .insert(*token_id, next_state);
            }
        }
        // add eos token transitions for final states
        for &final_state in &final_states {
            token_transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
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
        // compute allowed token ids for each state
        let mut allowed_token_ids: HashMap<StateId, HashSet<TokenId>> = HashMap::default();
        for (from_state, token_map) in &token_transitions {
            for (token_id, to_state) in token_map {
                if live_states.contains(to_state) {
                    allowed_token_ids
                        .entry(*from_state)
                        .or_default()
                        .insert(*token_id);
                }
            }
        }
        // Prune token_transitions to only transitions leading to live states
        for token_map in token_transitions.values_mut() {
            token_map.retain(|_, to_state| live_states.contains(to_state));
        }
        token_transitions.retain(|_, token_map| !token_map.is_empty());

        Ok(Self {
            eos_token_id,
            initial_state: start_state.as_u32(),
            final_states,
            byte_transitions: transitions,
            token_transitions,
            allowed_token_ids: allowed_token_ids
                .into_iter()
                .map(|(state, tokens)| (state, tokens.into_iter().collect()))
                .collect(),
            vocabulary,
        })
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
                    .ok_or_else(|| {
                        Error::NoTokenTransitionFound(state as usize, token_id as usize)
                    })
            })
    }

    pub fn get_byte_state_sequence(
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

/// DFA guide for diverse structured generation.
/// Holds a shared immutable `DfaIndex` and mutable per-run counters.
#[derive(Debug, Clone)]
pub struct DiverseGuideDFA {
    pub(crate) index: Arc<DfaIndex>,
    path_counter: HashMap<(StateId, StateId), u32>,
    local_state_counter: HashMap<StateId, u32>,
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
            path_counter: HashMap::decode(decoder)?,
            local_state_counter: HashMap::decode(decoder)?,
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
        Ok(Self {
            index: Arc::new(DfaIndex::new(regex, eos_token_id, vocabulary)?),
            path_counter: HashMap::default(),
            local_state_counter: HashMap::default(),
        })
    }

    /// Create a new `DiverseGuideDFA` sharing the same index with fresh counters.
    /// O(1): only clones the Arc reference, not the underlying index data.
    pub fn fork(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            path_counter: HashMap::default(),
            local_state_counter: HashMap::default(),
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
            *self
                .path_counter
                .entry((current_state, next_state))
                .or_insert(0) += 1;
            current_state = next_state;
        }
        Ok(())
    }

    pub fn update_local_state_counter(
        &mut self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<()> {
        let byte_state_seq = self.index.get_byte_state_sequence(state, token_id)?;
        for s in byte_state_seq[1..].iter() {
            *self.local_state_counter.entry(*s).or_insert(0) += 1;
        }
        Ok(())
    }

    pub fn reset_path_counter(&mut self) {
        self.path_counter.clear();
    }

    pub fn reset_local_state_counter(&mut self) {
        self.local_state_counter.clear();
    }

    pub fn compute_counts(&self, state: StateId) -> Result<TokenCounts> {
        let token_ids: Vec<TokenId> = self.index.get_allowed_token_ids(state)?;
        let num_tokens = token_ids.len();
        let mut reward_counts: Vec<u32> = vec![0; num_tokens];
        let mut penalty_counts: Vec<u32> = vec![0; num_tokens];

        for (i, token_id) in token_ids.iter().enumerate() {
            let byte_state_seq = self.index.get_byte_state_sequence(state, *token_id)?;
            let mut minimal_path_count = u32::MAX;
            for (state1, state2) in byte_state_seq.windows(2).map(|w| (w[0], w[1])) {
                minimal_path_count = minimal_path_count
                    .min(*self.path_counter.get(&(state1, state2)).unwrap_or(&0));
            }
            reward_counts[i] = minimal_path_count;
            let mut maximal_local_state_count: u32 = 1;
            for s in byte_state_seq[1..].iter() {
                if let Some(count) = self.local_state_counter.get(s) {
                    maximal_local_state_count = maximal_local_state_count.max(*count);
                }
            }
            penalty_counts[i] = maximal_local_state_count;
        }

        Ok(TokenCounts { token_ids, reward_counts, penalty_counts })
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

    fn counts_by_token(
        dfa: &DiverseGuideDFA,
        state: u32,
    ) -> Map<TokenId, (u32, u32)> {
        let counts = dfa.compute_counts(state).expect("compute_counts should succeed");
        let mut out = Map::default();
        for i in 0..counts.token_ids.len() {
            out.insert(
                counts.token_ids[i],
                (counts.reward_counts[i], counts.penalty_counts[i]),
            );
        }
        out
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

        let bytes = bincode::encode_to_vec(&dfa, config::standard())
            .expect("encoding must succeed");
        let (decoded, _): (DiverseGuideDFA, usize) =
            bincode::decode_from_slice(&bytes, config::standard())
                .expect("decoding must succeed");
        assert_eq!(decoded, dfa);
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
    }

    #[test]
    fn display_includes_expected_header() {
        let dfa = toy_dfa();
        let display = format!("{}", dfa);
        assert!(display.contains("DiverseGuideDFA with transitions:"));
    }
}
