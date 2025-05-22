//! Build a DFA for guiding diverse structured generation

use bincode::{Decode, Encode};
use rayon::prelude::*;
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::alphabet::Unit;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;

use crate::{Error, Result};

// define type alias for state-id, token-id, and byte
pub type StateId = u32;
pub type TokenId = u64;
pub type Byte = u8;

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct DiverseGuideDFA {
    eos_token_id: TokenId,
    initial_state: StateId,
    final_states: HashSet<StateId>,
    states: HashSet<StateId>,
    transitions: HashMap<StateId, HashMap<Byte, StateId>>,
    token_transitions: HashMap<StateId, HashMap<TokenId, StateId>>,
    allowed_token_ids: HashMap<StateId, Vec<TokenId>>,
    path_counter: HashMap<(StateId, StateId), u32>,
    local_state_counter: HashMap<StateId, u32>,
    vocabulary: HashMap<TokenId, String>,
}

impl DiverseGuideDFA {
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
        // collect all byte-level states
        let mut states = final_states.clone();
        states.insert(start_state.as_u32());
        for (state, inputs) in &transitions {
            states.insert(*state);
            states.extend(inputs.values().cloned());
        }
        // iterate each state, construct token level transitions
        let mut token_transitions: HashMap<StateId, HashMap<TokenId, StateId>> = HashMap::default();
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

        Ok(Self {
            eos_token_id,
            initial_state: start_state.as_u32(),
            final_states,
            states,
            transitions,
            token_transitions,
            allowed_token_ids: allowed_token_ids
                .into_iter()
                .map(|(state, tokens)| (state, tokens.into_iter().collect()))
                .collect(),
            path_counter: HashMap::default(),
            local_state_counter: HashMap::default(),
            vocabulary,
        })
    }

    fn build_byte_transitions(
        dfa: &DFA<Vec<StateId>>,
        start_state: AutomataStateId,
    ) -> Result<(HashSet<StateId>, HashMap<StateId, HashMap<Byte, StateId>>), Error> {
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

    // initial state and final states
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

    // get allowed bytes/tokens for current state
    pub fn get_allowed_bytes(&self, state: StateId) -> Result<Vec<Byte>, Error> {
        self.transitions
            .get(&state)
            .map(|transition_map| transition_map.keys().cloned().collect())
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    pub fn get_allowed_token_ids(&self, state: StateId) -> Result<Vec<TokenId>, Error> {
        self.allowed_token_ids
            .get(&state)
            .map(|token_ids| token_ids.clone())
            .ok_or_else(|| Error::InvalidState(state as usize))
    }

    // get next byte-state/token-state for current state
    pub fn get_next_byte_state(&self, state: StateId, byte: Byte) -> Result<StateId, Error> {
        self.transitions
            .get(&state)
            .ok_or_else(|| Error::InvalidState(state as usize))
            .and_then(|transition_map| {
                transition_map
                    .get(&byte)
                    .copied()
                    .ok_or_else(|| Error::NoTransitionFound(state as usize, byte as usize))
            })
    }

    pub fn get_next_token_state(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<StateId, Error> {
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

    // get byte-state sequence starting from the current state when reading a token
    pub fn get_byte_state_sequence(
        &self,
        state: StateId,
        token_id: TokenId,
    ) -> Result<Vec<StateId>, Error> {
        let mut state_seq: Vec<StateId> = vec![state];
        if token_id == self.eos_token_id {
            // this won't panic if we ensure that transitions for eos_token_id
            // are always present
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

    // get byte-transition sequence starting from the initial state when reading a string
    pub fn get_byte_transition_sequence(
        &self,
        string: &str,
    ) -> Result<Vec<(Byte, StateId)>, Error> {
        let mut seq: Vec<(Byte, StateId)> = vec![(0, self.initial_state)];
        let mut current_state = self.initial_state;
        for byte_ref in string.as_bytes().iter() {
            let next_state = self.get_next_byte_state(current_state, *byte_ref)?;
            seq.push((*byte_ref, next_state));
            current_state = next_state;
        }
        Ok(seq)
    }

    // update and reset the counter
    pub fn update_path_counter(&mut self, string: &str) -> Result<(), Error> {
        let mut current_state = self.initial_state;
        for byte in string.as_bytes() {
            let next_state = self.get_next_byte_state(current_state, *byte)?;
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
    ) -> Result<(), Error> {
        let byte_state_seq = self.get_byte_state_sequence(state, token_id)?;
        for state in byte_state_seq[1..].iter() {
            *self.local_state_counter.entry(*state).or_insert(0) += 1;
        }
        Ok(())
    }

    pub fn reset_path_counter(&mut self) {
        self.path_counter.clear();
    }

    pub fn reset_local_state_counter(&mut self) {
        self.local_state_counter.clear();
    }

    // compute counts for rewards and penalties
    pub fn compute_counts(&self, state: StateId) -> Result<(Vec<u64>, Vec<u32>, Vec<u32>), Error> {
        let allowed_tokens: Vec<u64> = self.get_allowed_token_ids(state)?;
        let num_tokens = allowed_tokens.len();
        let mut reward_counts: Vec<u32> = vec![0; num_tokens];
        let mut penalty_counts: Vec<u32> = vec![0; num_tokens];

        for (i, token_id) in allowed_tokens.iter().enumerate() {
            let byte_state_seq = self.get_byte_state_sequence(state, *token_id)?;
            let mut minimal_path_count = u32::MAX;
            for (state1, state2) in byte_state_seq.windows(2).map(|w| (w[0], w[1])) {
                minimal_path_count =
                    minimal_path_count.min(*self.path_counter.get(&(state1, state2)).unwrap_or(&0));
            }
            reward_counts[i] = minimal_path_count;
            let mut maximal_local_state_count: u32 = 1;
            for state in byte_state_seq[1..].iter() {
                if let Some(count) = self.local_state_counter.get(state) {
                    maximal_local_state_count = maximal_local_state_count.max(*count);
                }
            }
            penalty_counts[i] = maximal_local_state_count;
        }

        // let computation_results: Result<Vec<(u32, u32)>, Error> = allowed_tokens
        //     .par_iter()
        //     .map(|token_id| {
        //         let byte_state_seq = self.get_byte_state_sequence(state, *token_id)?;
        //         let mut minimal_path_count = u32::MAX;
        //         for (state1, state2) in byte_state_seq.windows(2).map(|w| (w[0], w[1])) {
        //             minimal_path_count = minimal_path_count
        //                 .min(*self.path_counter.get(&(state1, state2)).unwrap_or(&0));
        //         }
        //         let mut maximal_local_state_count: u32 = 1;
        //         for state in byte_state_seq[1..].iter() {
        //             if let Some(count) = self.local_state_counter.get(state) {
        //                 maximal_local_state_count = maximal_local_state_count.max(*count);
        //             }
        //         }
        //         Ok((minimal_path_count, maximal_local_state_count))
        //     })
        //     .collect();

        // match computation_results {
        //     Ok(pairs) => {
        //         for (i, (reward_count, penalty_count)) in pairs.iter().enumerate() {
        //             reward_counts[i] = *reward_count;
        //             penalty_counts[i] = *penalty_count;
        //         }
        //     }
        //     Err(e) => {
        //         return Err(e);
        //     }
        // }

        Ok((allowed_tokens, reward_counts, penalty_counts))
    }
}

impl std::fmt::Display for DiverseGuideDFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DiverseGuideDFA with transitions:")?;
        for (state_id, input) in self.transitions.iter() {
            writeln!(f, "{:?} -> {:#?},", state_id, input)?;
        }
        Ok(())
    }
}
