//! TensorForge — a symbolic tensor algebra system for continuum mechanics.
//!
//! MVP pipeline:
//! ```text
//! .tens source --parser--> syntactic AST --interpreter--> semantic values
//!                                             |
//!                              display/export +--renderer--> LaTeX
//! ```
//!
//! Module layout:
//! - [`parser`]      — lexer + Pratt parser for the `.tens` DSL
//! - [`ast`]         — syntactic AST (no mathematical meaning)
//! - [`symbolic`]    — symbolic scalar expressions (the symbolic engine)
//! - [`tensor`]      — tensor object system + conservative property inference
//! - [`interpreter`] — evaluation, environment, type checking
//! - [`renderer`]    — LaTeX symbol-mode and component-mode rendering

pub mod ast;
pub mod differentiation;
pub mod error;
pub mod indices;
pub mod integration;
pub mod interpreter;
pub mod metadata;
pub mod numeric;
pub mod ode;
pub mod parser;
pub mod renderer;
pub mod simplifier;
pub mod substitute;
pub mod symbolic;
pub mod tensor;

use error::Error;
use interpreter::{Interpreter, Output};

/// Parse and run a `.tens` program, returning the display/export outputs.
pub fn run_source(src: &str) -> Result<Vec<Output>, Error> {
    let stmts = parser::parse(src)?;
    Interpreter::new().run(&stmts)
}

/// Parse and run a program, returning both the outputs and the interpreter
/// (so callers can inspect the environment, e.g. in tests).
pub fn run_source_with_env(src: &str) -> Result<(Vec<Output>, Interpreter), Error> {
    let stmts = parser::parse(src)?;
    let mut interp = Interpreter::new();
    let outputs = interp.run(&stmts)?;
    Ok((outputs, interp))
}
