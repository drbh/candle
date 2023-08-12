use std::collections::{btree_map::Entry, BTreeMap};
use std::vec::Vec;

#[derive(Clone, Debug)]
pub struct Parser<'a> {
    pub input: &'a str,
    pub symbol_ids: BTreeMap<String, u32>,
    pub rules: Vec<Vec<(u8, Rules)>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rules {
    End = 0,
    Alt = 1,
    RuleRef = 2,
    Char = 3,
    CharNot = 4,
    CharRngUpper = 5,
    CharAlt = 6,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Parser {
            input,
            symbol_ids: BTreeMap::new(),
            rules: Vec::new(),
        }
    }

    fn add_rule(&mut self, rule: Vec<(u8, Rules)>, rule_id: usize) {
        // check if len is less then or equal to rule_id
        if self.rules.len() <= rule_id {
            self.rules.resize(rule_id + 1, vec![]);
        }
        self.rules[rule_id] = rule;
    }

    fn is_word_char(ch: char) -> bool {
        // TODO 'A'..='Z' | 'a'..='z' | '-' | '0'..='9'
        ch.is_ascii_alphanumeric() || ch == '-'
    }

    fn get_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id = self.symbol_ids.len() as u32;
        match self.symbol_ids.entry(base_name.to_string()) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(next_id);
                next_id
            }
        }
    }

    fn generate_symbol_id(&mut self, base_name: &str) -> u32 {
        let next_id: u32 = self.symbol_ids.len() as u32;
        self.symbol_ids
            .insert(format!("{}_{}", base_name, next_id), next_id);
        next_id
    }

    // MUTATING PARSE FUNCTIONS
    fn get_name(&mut self) -> &'a str {
        let name_len = self
            .input
            .chars()
            .take_while(|&c| c.is_alphanumeric())
            .count();
        if name_len == 0 {
            panic!("expecting name at {}", self.input);
        }
        let name = &self.input[..name_len];
        self.input = &self.input[name_len..];
        name
    }

    fn move_past_substring(&mut self, sub_str: &'a str) {
        if let Some(pos) = self.input.find(sub_str) {
            self.input = &self.input[pos + sub_str.len()..];
        } else {
            panic!("Failed to find substring: {}", sub_str);
        }
    }

    fn parse_space(&mut self, newline_ok: bool) {
        while let Some(ch) = self.input.chars().next() {
            match ch {
                ' ' | '\t' | '#' => {
                    if ch == '#' {
                        // Move past the entire comment
                        while let Some(ch) = self.input.chars().next() {
                            if ch == '\r' || ch == '\n' {
                                break;
                            } else {
                                self.input = &self.input[1..];
                            }
                        }
                    } else {
                        self.input = &self.input[1..];
                    }
                }
                '\r' | '\n' if newline_ok => {
                    self.input = &self.input[1..];
                }
                _ => break,
            }
        }
    }

    fn parse_alternative(&mut self, rule_name: &str, rule_id: usize, is_nested: bool) {
        // println!("======= parse_alternative =======");
        let mut rule = vec![];
        self.parse_sequence(rule_name, &mut rule, is_nested);
        let end_pos = 0;
        while let Some(ch) = self.input.chars().next() {
            if ch == '|' {
                rule.push((0, Rules::Alt));
                // move input forward 1 char
                self.input = &self.input[1..];
                self.parse_space(true);
                self.parse_sequence(rule_name, &mut rule, is_nested);
            } else {
                break;
            }
        }
        rule.push((0, Rules::End));
        self.add_rule(rule, rule_id);
        self.input = &self.input[end_pos..];
    }

    fn parse_sequence(
        &mut self,
        rule_name: &str,
        out_element: &mut Vec<(u8, Rules)>,
        is_nested: bool,
    ) {
        // println!("======= parse_sequence =======");
        let mut last_sym_start = out_element.len();
        while let Some(current_char) = self.input.chars().next() {
            match current_char {
                '"' => {
                    // literal string
                    // println!("======= literal string =======");
                    self.input = &self.input[1..];
                    while let Some(ch) = self.input.chars().next() {
                        if ch != '"' {
                            let mut utf_char = String::new();
                            utf_char.push(ch);
                            let byte_len = utf_char.len();
                            self.input = &self.input[byte_len..];
                            out_element.push((ch as u8, Rules::Char));
                        } else {
                            break;
                        }
                    }
                    // self.move_past_substring("\"");
                    self.input = &self.input[1..];
                    self.parse_space(is_nested);
                }
                '[' => {
                    // println!("======= char range(s) =======");
                    self.input = &self.input[1..];
                    if let Some(new_current_char) = self.input.chars().next() {
                        if new_current_char == '^' {
                            self.input = &self.input[1..];
                        }
                    }

                    while let Some(ch) = self.input.chars().next() {
                        if ch != ']' {
                            let mut utf_char = String::new();
                            utf_char.push(ch);
                            let mut byte_len = utf_char.len();
                            if ch as u8 == 48 {
                                byte_len += 1; // TODO: handle special cases in one place 
                            }
                            self.input = &self.input[byte_len..];
                            out_element.push((ch as u8, Rules::Char));
                        } else {
                            break;
                        }
                    }
                    self.input = &self.input[1..];
                    self.parse_space(is_nested)
                }
                'A'..='Z' | 'a'..='z' | '-' | '0'..='9' => {
                    // rule reference
                    // println!("===== is_word_char(*pos) =====");
                    let new_name_end = self.get_name();
                    let ref_rule_id: u32 = self.get_symbol_id(new_name_end);
                    self.parse_space(is_nested);
                    last_sym_start = out_element.len();
                    out_element.push((ref_rule_id.try_into().unwrap(), Rules::RuleRef));
                }
                '(' => {
                    // grouping
                    // println!("===== (*pos == '(') =====");
                    self.move_past_substring("(");
                    self.parse_space(true);
                    let sub_rule_id = self.generate_symbol_id(rule_name);
                    self.parse_alternative(rule_name, sub_rule_id as usize, true);
                    last_sym_start = out_element.len();
                    out_element.push((sub_rule_id.try_into().unwrap(), Rules::RuleRef));

                    let new_current_char = self.input.chars().next();
                    if let Some(')') = new_current_char {
                        self.move_past_substring(")");
                    } else {
                        panic!("expecting ) at {}", self.input);
                    }
                    self.parse_space(is_nested);
                }
                '*' | '+' | '?' => {
                    if last_sym_start == out_element.len() {
                        panic!("expecting rule at {}", self.input);
                    }

                    let sub_rule_id = self.generate_symbol_id(rule_name);
                    let mut sub_rules = vec![];

                    // add preceding symbol to generated rule
                    sub_rules.extend(out_element[last_sym_start..].to_vec());

                    let new_current_char = self.input.chars().next().unwrap();
                    if new_current_char == '*' || new_current_char == '+' {
                        // cause generated rule to recurse
                        sub_rules.push((sub_rule_id as u8, Rules::RuleRef));
                    }
                    // mark start of alternate def
                    sub_rules.push((0, Rules::Alt));

                    if new_current_char == '+' || new_current_char == '?' {
                        // add empty string to generated rule
                        sub_rules.extend(out_element[last_sym_start..].to_vec());
                    }
                    sub_rules.push((0, Rules::End));
                    // update self.rules
                    self.add_rule(sub_rules, sub_rule_id as usize);

                    // in original rule, replace previous symbol with reference to generated rule
                    out_element.drain(last_sym_start..);
                    out_element.push((sub_rule_id.try_into().unwrap(), Rules::RuleRef));

                    // move past the repetition character
                    self.input = &self.input[1..];
                    self.parse_space(is_nested);
                }
                _ => break,
            }
        }
    }

    pub fn parse_rule(&mut self) {
        // get the name and move past it
        let name = self.get_name();
        // move past the space
        self.parse_space(false);
        // get first rule
        let rule_id: u32 = self.get_symbol_id(name);
        // skip over ::=
        self.move_past_substring("::=");
        // move past the space
        self.parse_space(true);
        self.parse_alternative(name, rule_id as usize, false);
        // TODO: skip whitespace
        let new_current_char = self.input.chars().next();
        if let Some('\r') = new_current_char {
            if self.input.chars().nth(1) == Some('\n') {
                self.input = &self.input[2..];
            } else {
                self.input = &self.input[1..];
            }
        } else if let Some('\n') = new_current_char {
            self.input = &self.input[1..];
        }

        self.parse_space(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::Rules::*;

    #[test]
    fn test_parser_output() {
        let data = "root  ::= (expr \"=\" ws term \"\n\")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | \"(\" ws expr \")\" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*";

        let mut parser = Parser::new(data);

        while let Some(ch) = parser.input.chars().next() {
            parser.parse_rule();
        }

        let mut expected_symbol_ids = vec![
            ("expr", 2),
            ("expr_6", 6),
            ("expr_7", 7),
            ("ident", 8),
            ("ident_10", 10),
            ("num", 9),
            ("num_11", 11),
            ("root", 0),
            ("root_1", 1),
            ("root_5", 5),
            ("term", 4),
            ("ws", 3),
            ("ws_12", 12),
        ];

        let mut expected_rules = vec![
            (5, RuleRef),
            (0, End),
            (2, RuleRef),
            (61, Char),
            (3, RuleRef),
            (4, RuleRef),
            (10, Char),
            (0, End),
            (4, RuleRef),
            (7, RuleRef),
            (0, End),
            (12, RuleRef),
            (0, End),
            (8, RuleRef),
            (0, Alt),
            (9, RuleRef),
            (0, Alt),
            (40, Char),
            (3, RuleRef),
            (2, RuleRef),
            (41, Char),
            (3, RuleRef),
            (0, End),
            (1, RuleRef),
            (5, RuleRef),
            (0, Alt),
            (1, RuleRef),
            (0, End),
            (45, Char),
            (43, Char),
            (42, Char),
            (47, Char),
            (4, RuleRef),
            (0, End),
            (6, RuleRef),
            (7, RuleRef),
            (0, Alt),
            (0, End),
            (10, RuleRef),
            (3, RuleRef),
            (0, End),
            (11, RuleRef),
            (3, RuleRef),
            (0, End),
            (97, Char),
            (45, Char),
            (122, Char),
            (97, Char),
            (45, Char),
            (122, Char),
            (48, Char),
            // (45, Char),
            (57, Char),
            (95, Char),
            (10, RuleRef),
            (0, Alt),
            (0, End),
            (48, Char),
            // (45, Char),
            (57, Char),
            (11, RuleRef),
            (0, Alt),
            (48, Char),
            // (45, Char),
            (57, Char),
            (0, End),
            (32, Char),
            (9, Char),
            (10, Char),
            (12, RuleRef),
            (0, Alt),
            (0, End),
        ];

        for (k, v) in &parser.symbol_ids {
            assert_eq!(
                *v,
                expected_symbol_ids.remove(0).1,
                "Mismatched id for {}",
                k
            );
        }

        for v in &parser.rules {
            for x in v {
                assert_eq!(*x, expected_rules.remove(0), "Mismatched rule");
            }
        }

        // Ensure we matched and checked all expected rules
        assert!(expected_rules.is_empty());
    }
}
