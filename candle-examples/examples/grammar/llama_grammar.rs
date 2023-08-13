use crate::grammar::{GrammarElement, Rules};
use std::vec::Vec;

#[derive(Clone, Debug)]
pub struct LlamaGrammarCandidate {
    pub index: usize,
    pub code_points: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LlamaGrammar {
    pub rules: Vec<Vec<GrammarElement>>,
    pub stacks: Vec<Vec<GrammarElement>>,
}

#[allow(dead_code)]
impl LlamaGrammar {
    fn is_end_of_sequence(pos: &GrammarElement) -> bool {
        matches!(pos.0, Rules::End | Rules::Alt)
    }

    // transforms a grammar pushdown stack into N possible stacks, all ending
    // at a character range (terminal element)
    pub fn advance_stack(
        (last_index, last_pos_index): (usize, usize),
        rules: &Vec<Vec<GrammarElement>>,
        stack: &mut Vec<GrammarElement>,
        new_stacks: &mut Vec<Vec<GrammarElement>>,
    ) {
        if stack.is_empty() {
            new_stacks.push(stack.clone());
            return;
        }

        let pos_index: usize = stack.len() - 1;
        let pos: (Rules, u8) = stack[pos_index].clone();

        match pos.0 {
            Rules::RuleRef => {
                let rule_id = pos.1 as usize;
                let subpos = &rules[rule_id];
                let mut idx = 0;
                loop {
                    let mut new_stack = stack.clone();
                    new_stack.pop();

                    if let Some(next_pos) = rules[last_index].get(last_pos_index + 1) {
                        if !LlamaGrammar::is_end_of_sequence(next_pos) {
                            new_stack.push(next_pos.clone());
                        }
                    }

                    if let Some(next_subpos) = subpos.get(idx) {
                        if !LlamaGrammar::is_end_of_sequence(next_subpos) {
                            new_stack.push(next_subpos.clone());
                        }
                    }

                    LlamaGrammar::advance_stack((rule_id, idx), rules, &mut new_stack, new_stacks);

                    while !LlamaGrammar::is_end_of_sequence(&subpos[idx]) {
                        idx += 1;
                    }

                    if subpos[idx].0 == Rules::Alt {
                        idx += 1;
                    } else {
                        break;
                    }
                }
            }
            Rules::Char | Rules::CharNot => {
                new_stacks.push(stack.clone());
            }
            _ => {
                panic!("Invalid grammar element type");
            }
        }
    }

    pub fn new(rules: &Vec<Vec<GrammarElement>>, start_rule_index: usize) -> Box<LlamaGrammar> {
        let mut vec_rules = Vec::with_capacity(rules.len());

        for rule in rules.iter() {
            let mut inner_vec = Vec::new();
            for pos in rule.iter().take_while(|&el| el.0 != Rules::End) {
                inner_vec.push(pos.clone());
            }
            inner_vec.push((Rules::End, 0));
            vec_rules.push(inner_vec);
        }

        let mut stacks: Vec<Vec<(Rules, u8)>> = Vec::new();
        let first_index = &rules[start_rule_index];
        let mut count = 0;
        let mut pos: (Rules, u8) = first_index[count].clone();

        while count <= first_index.len() {
            let mut stack = Vec::new();

            if !LlamaGrammar::is_end_of_sequence(&pos) {
                stack.push(pos.clone());
            }

            LlamaGrammar::advance_stack((0, 0), &vec_rules, &mut stack, &mut stacks);

            while !LlamaGrammar::is_end_of_sequence(&pos) {
                count += 1;

                if count < first_index.len() {
                    pos = first_index[count].clone();
                } else {
                    break;
                }
            }

            if count >= first_index.len() {
                break;
            }

            if first_index[count].0 == Rules::Alt {
                count += 1;
                pos = first_index[count].clone();
            } else {
                break;
            }
        }

        Box::new(LlamaGrammar {
            rules: vec_rules,
            stacks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::Parser;
    use crate::grammar::Rules::*;

    #[test]
    fn test_fixed_output() {
        let rules: Vec<Vec<GrammarElement>> = vec![
            vec![(RuleRef, 4), (End, 0)],
            vec![(RuleRef, 2), (Char, 61), (RuleRef, 3), (Char, 10), (End, 0)],
            vec![(RuleRef, 3), (RuleRef, 6), (End, 0)],
            vec![(RuleRef, 7), (End, 0)],
            vec![(RuleRef, 1), (RuleRef, 4), (Alt, 0), (RuleRef, 1), (End, 0)],
            vec![
                (Char, 45),
                // TODO: fix parse
                (CharAlt, 43),
                (CharAlt, 42),
                (CharAlt, 47),
                (RuleRef, 3),
                (End, 0),
            ],
            vec![(RuleRef, 5), (RuleRef, 6), (Alt, 0), (End, 0)],
            vec![
                (Char, 48),
                // TODO: fix parse
                (CharRngUpper, 57),
                (RuleRef, 7),
                (Alt, 0),
                (Char, 48),
                // TODO: fix parse
                (CharRngUpper, 57),
                (End, 0),
            ],
        ];

        let llama_grammar = LlamaGrammar::new(&rules, 0);
        assert_eq!(llama_grammar.stacks.len(), 4);
        assert_eq!(
            llama_grammar.stacks,
            vec![
                vec![(RuleRef, 4), (Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(RuleRef, 4), (Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(Char, 61), (RuleRef, 6), (Char, 48)],
            ]
        );
    }

    #[test]
    fn test_parser_output() {
        let data = "root  ::= (expr \"=\" term \"\n\")+
        expr  ::= term ([-+*/] term)*
        term  ::= [0-9]+";

        let mut parser = Parser::new(data);

        while let Some(_) = parser.input.chars().next() {
            parser.parse_rule();
        }

        let llama_grammar = LlamaGrammar::new(&parser.rules, 0);

        assert_eq!(llama_grammar.stacks.len(), 4);

        assert_eq!(
            llama_grammar.stacks,
            vec![
                vec![(RuleRef, 4), (Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(RuleRef, 4), (Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(Char, 61), (RuleRef, 6), (Char, 48)],
                vec![(Char, 61), (RuleRef, 6), (Char, 48)],
            ]
        );
    }

    #[test]
    fn test_fixed_full_grammar_output() {
        let rules: Vec<Vec<GrammarElement>> = vec![
            vec![(RuleRef, 5), (End, 0)],
            vec![
                (RuleRef, 2),
                (Char, 61),
                (RuleRef, 3),
                (RuleRef, 4),
                (Char, 10),
                (End, 0),
            ],
            vec![(RuleRef, 4), (RuleRef, 7), (End, 0)],
            vec![(RuleRef, 12), (End, 0)],
            vec![
                (RuleRef, 8),
                (Alt, 0),
                (RuleRef, 9),
                (Alt, 0),
                (Char, 40),
                (RuleRef, 3),
                (RuleRef, 2),
                (Char, 41),
                (RuleRef, 3),
                (End, 0),
            ],
            vec![(RuleRef, 1), (RuleRef, 5), (Alt, 0), (RuleRef, 1), (End, 0)],
            vec![
                (Char, 45),
                (CharAlt, 43),
                (CharAlt, 42),
                (CharAlt, 47),
                (RuleRef, 4),
                (End, 0),
            ],
            vec![(RuleRef, 6), (RuleRef, 7), (Alt, 0), (End, 0)],
            vec![
                (Char, 97),
                (CharRngUpper, 122),
                (RuleRef, 10),
                (RuleRef, 3),
                (End, 0),
            ],
            vec![(RuleRef, 11), (RuleRef, 3), (End, 0)],
            vec![
                (Char, 97),
                (CharRngUpper, 122),
                (CharAlt, 48),
                (CharRngUpper, 57),
                (CharAlt, 95),
                (RuleRef, 10),
                (Alt, 0),
                (End, 0),
            ],
            vec![
                (Char, 48),
                (CharRngUpper, 57),
                (RuleRef, 11),
                (Alt, 0),
                (Char, 48),
                (CharRngUpper, 57),
                (End, 0),
            ],
            vec![
                (Char, 32),
                (CharAlt, 9),
                (CharAlt, 10),
                (RuleRef, 12),
                (Alt, 0),
                (End, 0),
            ],
        ];

        let llama_grammar = LlamaGrammar::new(&rules, 0);

        assert_eq!(llama_grammar.stacks.len(), 8);
        assert_eq!(
            llama_grammar.stacks,
            vec![
                vec![(RuleRef, 5), (Char, 61), (RuleRef, 7), (Char, 97)],
                vec![
                    (RuleRef, 5),
                    (Char, 61),
                    (RuleRef, 7),
                    (RuleRef, 3),
                    (Char, 48),
                ],
                vec![
                    (RuleRef, 5),
                    (Char, 61),
                    (RuleRef, 7),
                    (RuleRef, 3),
                    (Char, 48),
                ],
                vec![(RuleRef, 5), (Char, 61), (RuleRef, 7), (Char, 40)],
                vec![(Char, 61), (RuleRef, 7), (Char, 97)],
                vec![(Char, 61), (RuleRef, 7), (RuleRef, 3), (Char, 48)],
                vec![(Char, 61), (RuleRef, 7), (RuleRef, 3), (Char, 48)],
                vec![(Char, 61), (RuleRef, 7), (Char, 40)],
            ]
        );
    }
}
