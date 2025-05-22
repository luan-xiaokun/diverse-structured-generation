//! Building a `SimpleIndex` to represent the minimal DFA of a regex.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::alphabet::Unit;
use regex_automata::util::primitives::StateID as AutomataStateId;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct MinDivDFA {
    eos_token_id: u64,
    initial_state: u32,
    final_states: HashSet<u32>,
    states: HashSet<u32>,
    transitions: HashMap<u32, HashMap<u8, u32>>,
    token_transitions: HashMap<u32, HashMap<u64, u32>>,
    path_counter: HashMap<(u32, u32), u32>,
}

impl MinDivDFA {
    pub fn new(
        regex: &str,
        eos_token_id: u64,
        vocabulary: HashMap<String, HashSet<u64>>,
    ) -> Result<Self> {
        let dfa = DFA::builder()
            .configure(DFA::config().minimize(true))
            .build(regex)
            .map_err(Box::new)?;
        // let dfa = DFA::new(regex).map_err(Box::new)?;
        let start_state = match dfa.universal_start_state(Anchored::Yes) {
            Some(s) => s,
            None => return Err(Error::DfaHasNoStartState),
        };
        let mut transitions: HashMap<u32, HashMap<u8, u32>> = HashMap::default();
        let mut token_transitions: HashMap<u32, HashMap<u64, u32>> = HashMap::default();
        let mut final_states: HashSet<u32> = HashSet::default();
        let mut seen: HashSet<u32> = HashSet::from_iter([start_state.as_u32()]);
        let mut next_states: Vec<AutomataStateId> = vec![start_state];
        // construct byte level transitions
        while let Some(current_state) = next_states.pop() {
            if dfa.is_match_state(dfa.next_eoi_state(current_state)) {
                final_states.insert(current_state.as_u32());
            }
            let classes = dfa.byte_classes();
            for representative in classes.representatives(0..255) {
                let input = representative.as_u8().unwrap();
                let next_state = dfa.next_state(current_state, input);
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
        // collect all states
        let mut states = final_states.clone();
        states.insert(start_state.as_u32());
        for (state, inputs) in &transitions {
            states.insert(*state);
            states.extend(inputs.values().cloned());
        }
        // iterate each state, construct token level transitions
        for current_state in states.clone() {
            'token_loop: for (token, ids) in vocabulary.iter() {
                if ids.contains(&eos_token_id) {
                    continue;
                }
                let mut next_state = current_state;
                for byte in token.bytes() {
                    next_state = match transitions.get(&next_state) {
                        Some(trans) => match trans.get(&byte) {
                            Some(state) => *state,
                            None => continue 'token_loop,
                        },
                        None => continue 'token_loop,
                    }
                }
                for token_id in ids {
                    token_transitions
                        .entry(current_state)
                        .or_default()
                        .insert(*token_id, next_state);
                }
            }
        }
        // add eos token transitions for final states
        for &final_state in &final_states {
            token_transitions
                .entry(final_state)
                .or_default()
                .insert(eos_token_id, final_state);
        }

        let path_counter: HashMap<(u32, u32), u32> = HashMap::default();

        Ok(Self {
            eos_token_id,
            initial_state: start_state.as_u32(),
            final_states,
            states,
            transitions,
            token_transitions,
            path_counter,
        })
    }

    pub fn is_initial_state(&self, state: u32) -> bool {
        state == self.initial_state
    }

    pub fn get_initial_state(&self) -> u32 {
        self.initial_state
    }

    pub fn is_final_state(&self, state: u32) -> bool {
        self.final_states.contains(&state)
    }

    pub fn get_final_states(&self) -> &HashSet<u32> {
        &self.final_states
    }

    pub fn get_states(&self) -> HashSet<u32> {
        let mut states = self.final_states.clone();
        states.insert(self.initial_state);
        for (state, inputs) in &self.transitions {
            states.insert(*state);
            states.extend(inputs.values().cloned());
        }
        states
    }

    pub fn get_transitions(&self) -> &HashMap<u32, HashMap<u8, u32>> {
        &self.transitions
    }

    pub fn get_token_transitions(&self) -> &HashMap<u32, HashMap<u64, u32>> {
        &self.token_transitions
    }

    pub fn allowed_inputs(&self, state: &u32) -> Option<Vec<u8>> {
        self.transitions
            .get(state)
            .map(|res| res.keys().cloned().collect())
    }

    pub fn allowed_tokens(&self, state: &u32) -> Option<Vec<u64>> {
        self.token_transitions
            .get(state)
            .map(|res| res.keys().cloned().collect())
    }

    pub fn next_state(&self, state: &u32, input: &u8) -> Option<u32> {
        Some(*self.transitions.get(state)?.get(input)?)
    }

    pub fn next_token_state(&self, state: &u32, token_id: &u64) -> Option<u32> {
        Some(*self.token_transitions.get(state)?.get(token_id)?)
    }

    pub fn get_state_sequence_from_string(&self, state: &u32, string: String) -> Option<Vec<u32>> {
        let mut seq = vec![];
        let mut current_state = *state;
        for byte in string.bytes() {
            current_state = match self.transitions.get(&current_state) {
                Some(trans) => match trans.get(&byte) {
                    Some(s) => {
                        seq.push(*s);
                        *s
                    }
                    None => return None,
                },
                None => return None,
            }
        }
        Some(seq)
    }

    pub fn get_state_sequence(&self, string: String) -> Option<Vec<u32>> {
        self.get_state_sequence_from_string(&self.initial_state, string)
    }

    pub fn get_transition_sequence(&self, string: String) -> Option<Vec<(u8, u32)>> {
        let mut seq = vec![(0, self.initial_state)];
        let mut current_state = self.initial_state;
        for byte in string.bytes() {
            current_state = match self.transitions.get(&current_state) {
                Some(trans) => match trans.get(&byte) {
                    Some(s) => {
                        seq.push((byte, *s));
                        *s
                    }
                    None => return None,
                },
                None => return None,
            }
        }
        Some(seq)
    }
}

impl std::fmt::Display for MinDivDFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MinDivDFA with transitions:")?;
        for (state_id, input) in self.transitions.iter() {
            writeln!(f, "{:?} -> {:#?}", state_id, input)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_dfa_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let vocab: HashMap<String, HashSet<u64>> = HashMap::default();
        let dfa = MinDivDFA::new(regex, 888, vocab).unwrap();
        println!("{:?}", dfa);
        let initial_state = dfa.get_initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(dfa.get_final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!dfa.is_final_state(initial_state));

        let trans1 = HashMap::from_iter([
            (48, 24),
            (49, 24),
            (50, 24),
            (51, 24),
            (52, 24),
            (53, 24),
            (54, 24),
            (55, 24),
            (56, 24),
            (57, 24),
        ]);
        let trans2 = HashMap::from_iter([
            (48, 48),
            (49, 56),
            (50, 56),
            (51, 56),
            (52, 56),
            (53, 56),
            (54, 56),
            (55, 56),
            (56, 56),
            (57, 56),
        ]);
        let expected = HashMap::from_iter([
            (24, trans1.clone()),
            (40, trans2.clone()),
            (56, trans1.clone()),
        ]);
        println!("{:?}", dfa.get_transitions());
        assert_eq!(dfa.get_transitions(), &expected);

        let string: String = "1998".parse().unwrap();
        let seq = dfa
            .get_state_sequence_from_string(&dfa.get_initial_state(), string)
            .unwrap();
        assert_eq!(seq, vec![56, 24, 24, 24]);
    }

    #[test]
    fn simple_dfa_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let dfa = MinDivDFA::new(regex, 888, HashMap::default()).unwrap();
        println!("{:?}", dfa);
        let initial_state = dfa.get_initial_state();
        assert_eq!(initial_state, 80);
        assert_eq!(dfa.get_final_states(), &HashSet::from_iter([128, 208]));
        assert!(!dfa.is_final_state(initial_state));
    }
}

