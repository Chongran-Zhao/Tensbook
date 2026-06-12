//! Hand-written Pratt (precedence-climbing) parser for the `.tens` DSL.
//!
//! Grammar (MVP subset):
//! ```text
//! program   := statement (NEWLINE statement)*
//! statement := IDENT "=" expr | expr
//! expr      := additive
//! additive  := multiplicative (("+" | "-") multiplicative)*
//! multiplicative := unary (("*" | "/") unary)*
//! unary     := "-" unary | power
//! power     := postfix ("^" unary)?          # right associative
//! postfix   := primary ("." IDENT | "[" expr "]")*
//! primary   := NUM | STR | true | false | IDENT | IDENT "(" callargs ")" | "(" expr ")"
//! callargs  := (expr | IDENT "=" expr) ("," ...)*
//! ```

pub mod lexer;

use crate::ast::{BinOp, Expr, Stmt, UnOp};
use crate::error::Error;
use lexer::{lex, Tok, Token};

/// Positional and keyword arguments of a call.
type CallArgs = (Vec<Expr>, Vec<(String, Expr)>);

pub fn parse(src: &str) -> Result<Vec<Stmt>, Error> {
    let stripped = strip_note_blocks(src)?;
    let tokens = lex(&stripped)?;
    Parser { tokens, pos: 0 }.parse_program()
}

fn strip_note_blocks(src: &str) -> Result<String, Error> {
    let mut out = String::with_capacity(src.len());
    let mut in_note = false;
    // Legacy ```notes blocks can contain fenced code only when the inner
    // fence carries an info string (```rust … ```); new note blocks use
    // HTML-comment sentinels so ordinary Markdown fences never conflict.
    // Mirrors noteBlocksFromSource in ui/main.js — keep the two in sync.
    let mut in_inner_fence = false;
    let mut note_kind = NoteFence::Html;
    let mut start_line = None;

    for (idx, line) in src.split_inclusive('\n').enumerate() {
        let line_no = idx + 1;
        let body = line.trim_end_matches(['\r', '\n']);
        let trimmed = body.trim();

        if in_note {
            out.push_str(blank_like(line));
            if note_kind == NoteFence::Backtick {
                if in_inner_fence {
                    if is_legacy_note_close(trimmed) {
                        in_inner_fence = false;
                    }
                } else if is_inner_fence_open(trimmed) {
                    in_inner_fence = true;
                } else if is_note_close(trimmed, note_kind) {
                    in_note = false;
                    start_line = None;
                }
            } else if is_note_close(trimmed, note_kind) {
                in_note = false;
                start_line = None;
            }
            continue;
        }

        if let Some(kind) = note_open_kind(trimmed) {
            in_note = true;
            in_inner_fence = false;
            note_kind = kind;
            start_line = Some(line_no);
            out.push_str(blank_like(line));
            continue;
        }

        out.push_str(line);
    }

    if in_note {
        return Err(Error::new("unterminated notes block", start_line));
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NoteFence {
    Html,
    Backtick,
}

fn note_open_kind(trimmed: &str) -> Option<NoteFence> {
    if trimmed.eq_ignore_ascii_case("<!-- tensorforge:note -->") {
        Some(NoteFence::Html)
    } else if trimmed.eq_ignore_ascii_case("```notes") {
        Some(NoteFence::Backtick)
    } else {
        None
    }
}

fn is_note_close(trimmed: &str, kind: NoteFence) -> bool {
    match kind {
        NoteFence::Html => trimmed.eq_ignore_ascii_case("<!-- /tensorforge:note -->"),
        NoteFence::Backtick => is_legacy_note_close(trimmed),
    }
}

fn is_legacy_note_close(trimmed: &str) -> bool {
    trimmed == "```"
}

/// An opening fence with an info string (```rust, ```text, …) starts a code
/// block inside the note; the bare ``` that ends it is consumed by the
/// inner-fence state instead of closing the note.
fn is_inner_fence_open(trimmed: &str) -> bool {
    trimmed.starts_with("```") && trimmed != "```"
}

fn blank_like(line: &str) -> &'static str {
    if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos].tok
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn next(&mut self) -> Tok {
        let tok = self.tokens[self.pos].tok.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), Error> {
        if self.peek() == want {
            self.next();
            Ok(())
        } else {
            Err(Error::new(
                format!("expected {what}, found {:?}", self.peek()),
                Some(self.line()),
            ))
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.next();
        }
    }

    fn parse_program(&mut self) -> Result<Vec<Stmt>, Error> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Tok::Eof) {
                break;
            }
            stmts.push(self.parse_statement()?);
            match self.peek() {
                Tok::Newline | Tok::Eof => {}
                _ => {
                    return Err(Error::new(
                        format!("unexpected token after statement: {:?}", self.peek()),
                        Some(self.line()),
                    ))
                }
            }
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> Result<Stmt, Error> {
        let line = self.line();
        // `[a, b] = expr` — destructuring assignment.
        if matches!(self.peek(), Tok::LBracket) {
            return self.parse_pair_assignment(line);
        }
        if let Tok::Ident(name) = self.peek().clone() {
            // Lookahead: IDENT "=" starts an assignment (but IDENT "==" would
            // not; the MVP grammar has no "==", so a single Eq is unambiguous).
            if self.tokens[self.pos + 1].tok == Tok::Eq {
                self.next(); // ident
                self.next(); // =
                let expr = self.parse_expr()?;
                return Ok(Stmt::Assign { name, expr, line });
            }
            // Tentative: IDENT ("[" expr "]")+ "=" is a component assignment;
            // anything else falls back to an expression statement.
            if self.tokens[self.pos + 1].tok == Tok::LBracket {
                let saved = self.pos;
                if let Some(stmt) = self.try_parse_component_assignment(name, line)? {
                    return Ok(stmt);
                }
                self.pos = saved;
            }
        }
        Ok(Stmt::Expr(self.parse_expr()?, line))
    }

    /// `[a, b] = expr`
    fn parse_pair_assignment(&mut self, line: usize) -> Result<Stmt, Error> {
        self.next(); // [
        let first = match self.next() {
            Tok::Ident(name) => name,
            tok => {
                return Err(Error::new(
                    format!("expected a set name in `[a, b] = ...`, found {tok:?}"),
                    Some(line),
                ))
            }
        };
        self.expect(&Tok::Comma, "`,`")?;
        let second = match self.next() {
            Tok::Ident(name) => name,
            tok => {
                return Err(Error::new(
                    format!("expected a set name in `[a, b] = ...`, found {tok:?}"),
                    Some(line),
                ))
            }
        };
        self.expect(&Tok::RBracket, "`]`")?;
        self.expect(&Tok::Eq, "`=`")?;
        let expr = self.parse_expr()?;
        Ok(Stmt::AssignPair {
            first,
            second,
            expr,
            line,
        })
    }

    /// `F[1][1] = expr` — returns `None` (caller rewinds) when the indexed
    /// expression is not followed by `=`.
    fn try_parse_component_assignment(
        &mut self,
        name: String,
        line: usize,
    ) -> Result<Option<Stmt>, Error> {
        self.next(); // ident
        let mut indices = Vec::new();
        while matches!(self.peek(), Tok::LBracket) {
            self.next();
            indices.push(self.parse_expr()?);
            self.expect(&Tok::RBracket, "`]`")?;
        }
        if self.peek() != &Tok::Eq {
            return Ok(None);
        }
        self.next(); // =
        let expr = self.parse_expr()?;
        Ok(Some(Stmt::AssignComponent {
            name,
            indices,
            expr,
            line,
        }))
    }

    fn parse_expr(&mut self) -> Result<Expr, Error> {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            self.next();
            let rhs = self.parse_multiplicative()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.parse_outer()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Colon => BinOp::Ddot,
                _ => break,
            };
            self.next();
            let rhs = self.parse_outer()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_outer(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek(), Tok::Amp) {
            self.next();
            let rhs = self.parse_unary()?;
            lhs = Expr::Binary {
                op: BinOp::Outer,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<Expr, Error> {
        if matches!(self.peek(), Tok::Minus) {
            self.next();
            let expr = self.parse_unary()?;
            return Ok(Expr::Unary {
                op: UnOp::Neg,
                expr: Box::new(expr),
            });
        }
        self.parse_power()
    }

    fn parse_power(&mut self) -> Result<Expr, Error> {
        let base = self.parse_postfix()?;
        if matches!(self.peek(), Tok::Caret) {
            self.next();
            // Right associative: 2^3^2 == 2^(3^2). Exponent goes through
            // unary so `x^-2` also parses.
            let exp = self.parse_unary()?;
            return Ok(Expr::Binary {
                op: BinOp::Pow,
                lhs: Box::new(base),
                rhs: Box::new(exp),
            });
        }
        Ok(base)
    }

    fn parse_postfix(&mut self) -> Result<Expr, Error> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek() {
                Tok::Dot => {
                    self.next();
                    match self.next() {
                        Tok::Ident(name) => {
                            expr = Expr::Field {
                                target: Box::new(expr),
                                name,
                            };
                        }
                        tok => {
                            return Err(Error::new(
                                format!("expected property name after `.`, found {tok:?}"),
                                Some(self.line()),
                            ))
                        }
                    }
                }
                Tok::LBracket => {
                    self.next();
                    let index = self.parse_expr()?;
                    self.expect(&Tok::RBracket, "`]`")?;
                    expr = Expr::Index {
                        target: Box::new(expr),
                        index: Box::new(index),
                    };
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, Error> {
        let line = self.line();
        match self.next() {
            Tok::Num(n) => Ok(Expr::Num(n)),
            Tok::Str(s) => Ok(Expr::Str(s)),
            Tok::True => Ok(Expr::Bool(true)),
            Tok::False => Ok(Expr::Bool(false)),
            Tok::Ident(name) => {
                if matches!(self.peek(), Tok::LParen) {
                    self.next();
                    let (args, kwargs) = self.parse_call_args()?;
                    Ok(Expr::Call {
                        callee: name,
                        args,
                        kwargs,
                    })
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            Tok::LParen => {
                let expr = self.parse_expr()?;
                self.expect(&Tok::RParen, "`)`")?;
                Ok(expr)
            }
            tok => Err(Error::new(format!("unexpected token {tok:?}"), Some(line))),
        }
    }

    fn parse_call_args(&mut self) -> Result<CallArgs, Error> {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        if matches!(self.peek(), Tok::RParen) {
            self.next();
            return Ok((args, kwargs));
        }
        loop {
            // kwarg: IDENT "=" expr
            let is_kwarg =
                matches!(self.peek(), Tok::Ident(_)) && self.tokens[self.pos + 1].tok == Tok::Eq;
            if is_kwarg {
                let name = match self.next() {
                    Tok::Ident(name) => name,
                    _ => unreachable!(),
                };
                self.next(); // =
                let value = self.parse_expr()?;
                kwargs.push((name, value));
            } else {
                if !kwargs.is_empty() {
                    let hint = match self.peek() {
                        Tok::Ident(name) => {
                            format!(" (flags are written `{name}=true`, not bare `{name}`)")
                        }
                        _ => String::new(),
                    };
                    return Err(Error::new(
                        format!("positional argument after keyword argument{hint}"),
                        Some(self.line()),
                    ));
                }
                args.push(self.parse_expr()?);
            }
            match self.next() {
                Tok::Comma => continue,
                Tok::RParen => break,
                tok => {
                    return Err(Error::new(
                        format!("expected `,` or `)` in argument list, found {tok:?}"),
                        Some(self.line()),
                    ))
                }
            }
        }
        Ok((args, kwargs))
    }
}
