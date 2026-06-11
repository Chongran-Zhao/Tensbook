use crate::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    True,
    False,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Eq,
    Dot,
    Colon,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

pub fn lex(src: &str) -> Result<Vec<Token>, Error> {
    let mut tokens = Vec::new();
    let mut chars = src.chars().peekable();
    let mut line = 1usize;

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' => {
                chars.next();
            }
            '\n' => {
                chars.next();
                tokens.push(Token {
                    tok: Tok::Newline,
                    line,
                });
                line += 1;
            }
            '#' => {
                // comment to end of line
                while let Some(&c) = chars.peek() {
                    if c == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\n') | None => {
                            return Err(Error::new("unterminated string literal", Some(line)))
                        }
                        // Strings hold LaTeX like "\bm F"; keep backslashes verbatim,
                        // only \" escapes a quote.
                        Some('\\') => {
                            if let Some(&'"') = chars.peek() {
                                chars.next();
                                s.push('"');
                            } else {
                                s.push('\\');
                            }
                        }
                        Some(c) => s.push(c),
                    }
                }
                tokens.push(Token {
                    tok: Tok::Str(s),
                    line,
                });
            }
            '0'..='9' => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        s.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let n: f64 = s
                    .parse()
                    .map_err(|_| Error::new(format!("invalid number `{s}`"), Some(line)))?;
                tokens.push(Token {
                    tok: Tok::Num(n),
                    line,
                });
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        s.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let tok = match s.as_str() {
                    "true" => Tok::True,
                    "false" => Tok::False,
                    _ => Tok::Ident(s),
                };
                tokens.push(Token { tok, line });
            }
            _ => {
                chars.next();
                let tok = match c {
                    '+' => Tok::Plus,
                    '-' => Tok::Minus,
                    '*' => Tok::Star,
                    '/' => Tok::Slash,
                    '^' => Tok::Caret,
                    '=' => Tok::Eq,
                    '.' => Tok::Dot,
                    ':' => Tok::Colon,
                    ',' => Tok::Comma,
                    '(' => Tok::LParen,
                    ')' => Tok::RParen,
                    '[' => Tok::LBracket,
                    ']' => Tok::RBracket,
                    _ => {
                        return Err(Error::new(
                            format!("unexpected character `{c}`"),
                            Some(line),
                        ))
                    }
                };
                tokens.push(Token { tok, line });
            }
        }
    }
    tokens.push(Token {
        tok: Tok::Eof,
        line,
    });
    Ok(tokens)
}
