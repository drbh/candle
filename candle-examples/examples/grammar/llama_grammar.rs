use crate::grammar::Rules;
use std::vec::{self, Vec};
pub type LlamaGrammarElement = (u8, Rules);

#[derive(Clone, Debug)]
pub struct LlamaGrammarCandidate {
    pub index: usize,
    pub code_points: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LlamaGrammar {
    pub rules: Vec<Vec<LlamaGrammarElement>>,
    pub stacks: Vec<Vec<LlamaGrammarElement>>,
}

impl LlamaGrammar {
    fn is_end_of_sequence(pos: &LlamaGrammarElement) -> bool {
        match pos.1 {
            Rules::End | Rules::Alt => true,
            _ => false,
        }
    }

    // transforms a grammar pushdown stack into N possible stacks, all ending
    // at a character range (terminal element)
    pub fn advance_stack(
        (index, last_pos_index): (usize, usize),
        rules: &Vec<Vec<LlamaGrammarElement>>,
        stack: &mut Vec<LlamaGrammarElement>,
        new_stacks: &mut Vec<Vec<LlamaGrammarElement>>,
    ) {
        if stack.is_empty() {
            new_stacks.push(stack.clone());
            return; // return new_stacks;
        }

        let mut pos_index = stack.len() - 1;
        let mut pos: (u8, Rules) = stack[pos_index].clone();

        match pos.1 {
            Rules::RuleRef => {
                let rule_id = pos.0 as usize;
                let subpos = &rules[rule_id];
                let mut idx = 0;
                loop {
                    // make a new stack with the current stack but without the last element
                    // use drop_last() instead of clone() to avoid copying the whole stack
                    let num = 0;
                    let mut new_stack = stack.clone();
                    new_stack.pop();

                    if let Some(next_pos) = rules[index].get(1) {
                        if !LlamaGrammar::is_end_of_sequence(&next_pos) {
                            new_stack.push(next_pos.clone());
                        }
                    }

                    if let Some(next_subpos) = subpos.get(idx) {
                        if !LlamaGrammar::is_end_of_sequence(&next_subpos) {
                            new_stack.push(next_subpos.clone());
                        }
                    }

                    LlamaGrammar::advance_stack(
                        (rule_id, pos_index),
                        rules,
                        &mut new_stack,
                        new_stacks,
                    );

                    while !LlamaGrammar::is_end_of_sequence(&subpos[idx]) {
                        idx += 1;
                    }

                    if subpos[idx].1 == Rules::Alt {
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
                // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
                // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
                // those
                panic!("Invalid grammar element type");
            }
        }
    }

    pub fn new(
        rules: &Vec<Vec<LlamaGrammarElement>>,
        start_rule_index: usize,
    ) -> Box<LlamaGrammar> {
        let mut vec_rules = Vec::with_capacity(rules.len());

        for rule in rules.iter() {
            let mut inner_vec = Vec::new();
            for pos in rule.iter().take_while(|&el| el.1 != Rules::End) {
                inner_vec.push(pos.clone());
            }
            inner_vec.push((0, Rules::End));
            vec_rules.push(inner_vec);
        }

        let mut stacks: Vec<Vec<(u8, Rules)>> = Vec::new();
        let first_index = &rules[start_rule_index];
        let mut count = 0;
        let mut pos: (u8, Rules) = first_index[count].clone();

        while count <= first_index.len() {
            let mut stack = Vec::new();

            if !LlamaGrammar::is_end_of_sequence(&pos) {
                stack.push(pos.clone());
            }

            LlamaGrammar::advance_stack((0, 0), &vec_rules, &mut stack, &mut stacks);

            while !LlamaGrammar::is_end_of_sequence(&pos) {
                count += 1;
                pos = first_index[count].clone();
            }
            if first_index[count].1 == Rules::Alt {
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
    fn test_parser_output() {
        let data = "root  ::= (expr \"=\" term \"\n\")+
expr  ::= term ([-+*/] term)*
term  ::= [0-9]+";

        let mut parser = Parser::new(data);

        while let Some(ch) = parser.input.chars().next() {
            parser.parse_rule();
        }

        let llama_grammar = LlamaGrammar::new(&parser.rules, 0);

        assert_eq!(llama_grammar.stacks.len(), 4);

        // WRONG

        assert_eq!(
            llama_grammar.stacks,
            vec![
                vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)],
                vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)],
                vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)],
                vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)]
            ]
        );

        // CORRECT

        // assert_eq!(
        //     llama_grammar.stacks,
        //     vec![
        //         vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)],
        //         vec![(4, RuleRef), (61, Char), (6, RuleRef), (48, Char)],
        //         vec![(61, Char), (6, RuleRef), (48, Char)],
        //         vec![(61, Char), (6, RuleRef), (48, Char)],
        //     ]
        // );
    }
}
