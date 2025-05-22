//! Building a `SimpleIndex` to represent the minimal DFA of a regex.

use bincode::{Decode, Encode};
use regex_automata::dfa::dense::DFA;
use regex_automata::dfa::Automaton;
use regex_automata::util::alphabet::Unit;
use regex_automata::util::primitives::StateID;
use regex_automata::Anchored;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct MinimalDFA {
    initial_state: u32,
    final_states: HashSet<u32>,
    transitions: HashMap<u32, HashMap<u8, u32>>,
}

impl MinimalDFA {
    pub fn new(regex: &str) -> Result<Self> {
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
        let mut final_states: HashSet<u32> = HashSet::default();
        let mut seen = HashSet::from_iter([start_state]);
        let mut next_states: Vec<StateID> = vec![start_state];

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
                    if !seen.contains(&next_state) {
                        seen.insert(next_state);
                        next_states.push(next_state);
                    }
                }
            }
        }
        Ok(Self {
            initial_state: start_state.as_u32(),
            final_states,
            transitions,
        })
    }

    pub fn get_initial_state(&self) -> u32 {
        self.initial_state
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

    pub fn is_initial_state(&self, state: u32) -> bool {
        state == self.initial_state
    }

    pub fn is_final_state(&self, state: &u32) -> bool {
        self.final_states.contains(state)
    }

    pub fn next_state(&self, state: &u32, input: &u8) -> Option<u32> {
        Some(*self.transitions.get(state)?.get(input)?)
    }

    pub fn get_state_sequence(&self, inputs: &str) -> Option<Vec<u32>> {
        let mut state = self.initial_state;
        let mut seq = vec![state];
        for input in inputs.bytes() {
            if let Some(s) = self.next_state(&state, &input) {
                seq.push(s);
                state = s;
            } else {
                return None;
            }
        }
        Some(seq)
    }
}

impl std::fmt::Display for MinimalDFA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Minimal DFA with transitions:")?;
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
    fn threefold_test() {
        let regex = "-?(0|[369]|([147][0369]*[258]|[258][0369]*[147]|[0369])([0369]*([0369]|[147][0369]*[258]|[258][0369]*[147]))*?)$";
        let dfa = MinimalDFA::new(regex).unwrap();
        println!("{:?}", dfa.get_transitions());
        let seq = dfa.get_state_sequence("103989");
        println!("{:?}", seq);
    }
    #[test]
    fn simple_dfa_from_regex() {
        let regex = "0|[1-9][0-9]*";
        let dfa = MinimalDFA::new(regex).unwrap();
        println!("{:?}", dfa);
        let initial_state = dfa.get_initial_state();
        assert_eq!(initial_state, 40);
        assert_eq!(dfa.get_final_states(), &HashSet::from_iter([24, 48, 56]));
        assert!(!dfa.is_final_state(&initial_state));

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

        let seq = dfa.get_state_sequence("1998").unwrap();
        assert_eq!(seq, vec![40, 56, 24, 24, 24]);
    }

    #[test]
    fn simple_dfa_from_regex_multibyte() {
        let regex = "😇| [😈-😍][😇-😎]*";
        let dfa = MinimalDFA::new(regex).unwrap();
        println!("{:?}", dfa);
        let initial_state = dfa.get_initial_state();
        assert_eq!(initial_state, 80);
        assert_eq!(dfa.get_final_states(), &HashSet::from_iter([128, 208]));
        assert!(!dfa.is_final_state(&initial_state));
    }
}
