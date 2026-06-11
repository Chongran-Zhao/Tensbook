//! Component-mode expansion: turn a second-order tensor expression into a
//! dim × dim matrix of scalar component expressions, rendered as a LaTeX
//! `bmatrix`.
//!
//! The MVP expands sums explicitly for a fixed dim (no abstract index
//! system yet): `(F^T F)_{ij} = Σ_k F_{ki} F_{kj}`.

use crate::error::Error;
use crate::renderer::latex::scalar_to_latex;
use crate::symbolic::ScalarExpr;
use crate::tensor::TensorExpr;
use std::rc::Rc;

/// Expand a second-order tensor expression into its components and render
/// them as a LaTeX `bmatrix`.
pub fn tensor_to_component_matrix(expr: &TensorExpr) -> Result<String, Error> {
    let entries = expand(expr)?;
    let mut rows = Vec::with_capacity(entries.len());
    for row_entries in &entries {
        let row: Vec<String> = row_entries.iter().map(scalar_to_latex).collect();
        rows.push(row.join(" & "));
    }
    Ok(format!(
        "\\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
        rows.join(" \\\\\n")
    ))
}

/// Component expression for entry (i, j) of a second-order tensor expression.
pub fn component(expr: &TensorExpr, i: usize, j: usize) -> Result<ScalarExpr, Error> {
    if expr.order() != 2 {
        return Err(Error::msg(format!(
            "component expansion requires a second-order tensor, got order {}",
            expr.order()
        )));
    }
    expandable(expr)?;
    Ok(entry(expr, i, j))
}

fn expand(expr: &TensorExpr) -> Result<Vec<Vec<ScalarExpr>>, Error> {
    if expr.order() != 2 {
        return Err(Error::msg(format!(
            "component expansion requires a second-order tensor, got order {}; \
             higher-order display (block_components) is not part of the MVP",
            expr.order()
        )));
    }
    expandable(expr)?;
    let dim = expr.dim();
    let mut entries = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut row = Vec::with_capacity(dim);
        for j in 0..dim {
            row.push(entry(expr, i, j));
        }
        entries.push(row);
    }
    Ok(entries)
}

/// Entry (i, j), 0-based, of a second-order tensor expression.
/// Inverse/Diff nodes cannot be expanded entrywise; callers route Diff
/// through the abstract-index engine instead, and `expand` rejects them
/// up front via [`expandable`].
fn entry(expr: &TensorExpr, i: usize, j: usize) -> ScalarExpr {
    match expr {
        TensorExpr::Var { latex, props, .. } => {
            if props.identity {
                ScalarExpr::Num(if i == j { 1.0 } else { 0.0 })
            } else {
                indexed_sym(latex, i, j)
            }
        }
        TensorExpr::Transpose(t) => entry(t, j, i),
        TensorExpr::MatMul(a, b) => {
            let dim = expr.dim();
            let mut sum: Option<ScalarExpr> = None;
            for k in 0..dim {
                let term = mul(entry(a, i, k), entry(b, k, j));
                sum = Some(match sum {
                    None => term,
                    Some(acc) => add(acc, term),
                });
            }
            sum.unwrap_or(ScalarExpr::Num(0.0))
        }
        TensorExpr::Add(a, b) => add(entry(a, i, j), entry(b, i, j)),
        TensorExpr::Sub(a, b) => {
            let lhs = entry(a, i, j);
            let rhs = entry(b, i, j);
            match (&lhs, &rhs) {
                (_, ScalarExpr::Num(n)) if *n == 0.0 => lhs,
                (ScalarExpr::Num(n), _) if *n == 0.0 => ScalarExpr::Neg(Rc::new(rhs)),
                _ => ScalarExpr::Sub(Rc::new(lhs), Rc::new(rhs)),
            }
        }
        TensorExpr::ScalarMul(s, t) => mul((**s).clone(), entry(t, i, j)),
        TensorExpr::Neg(t) => {
            let inner = entry(t, i, j);
            match inner {
                ScalarExpr::Num(n) => ScalarExpr::Num(-n),
                _ => ScalarExpr::Neg(Rc::new(inner)),
            }
        }
        // Rejected by `expandable` before this point.
        TensorExpr::Inverse(_)
        | TensorExpr::InverseTranspose(_)
        | TensorExpr::Diff { .. }
        | TensorExpr::Outer(..)
        | TensorExpr::Spectral { .. } => {
            unreachable!("non-expandable node reached entry(); expandable() must screen first")
        }
    }
}

/// Check the whole tree for nodes that have no entrywise expansion.
fn expandable(expr: &TensorExpr) -> Result<(), Error> {
    match expr {
        TensorExpr::Inverse(_) | TensorExpr::InverseTranspose(_) => Err(Error::msg(
            "component expansion of tensor inverses is not supported in this phase; \
             use mode=symbol",
        )),
        TensorExpr::Diff { .. } => Err(Error::msg(
            "use display(diff(...), mode=components) on a derivative variable; \
             nested derivative nodes inside expressions cannot be expanded",
        )),
        TensorExpr::Outer(..) => Err(Error::msg(
            "component expansion of outer products is not supported yet; use mode=symbol",
        )),
        TensorExpr::Spectral { .. } => Err(Error::msg(
            "spectral decompositions are display-only; use mode=symbol",
        )),
        TensorExpr::Var { .. } => Ok(()),
        TensorExpr::Transpose(t) | TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => {
            expandable(t)
        }
        TensorExpr::MatMul(a, b) | TensorExpr::Add(a, b) | TensorExpr::Sub(a, b) => {
            expandable(a).and_then(|_| expandable(b))
        }
    }
}

/// Build the component symbol `X_{i+1,j+1}` from the tensor's LaTeX name,
/// stripping a leading `\bm ` so `\bm F` yields `F_{11}` rather than
/// `\bm F_{11}`.
fn indexed_sym(latex: &str, i: usize, j: usize) -> ScalarExpr {
    let base = latex
        .trim()
        .strip_prefix("\\bm ")
        .or_else(|| latex.trim().strip_prefix("\\bm"))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| latex.trim());
    ScalarExpr::Sym {
        name: format!("{base}_{}{}", i + 1, j + 1),
        latex: format!("{base}_{{{}{}}}", i + 1, j + 1),
    }
}

// Light constant folding so identity factors don't litter the output with
// `0 \, F_{11}` or `1 \, F_{11}` terms.
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    match (&a, &b) {
        (ScalarExpr::Num(x), _) if *x == 0.0 => ScalarExpr::Num(0.0),
        (_, ScalarExpr::Num(x)) if *x == 0.0 => ScalarExpr::Num(0.0),
        (ScalarExpr::Num(x), _) if *x == 1.0 => b,
        (_, ScalarExpr::Num(x)) if *x == 1.0 => a,
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => ScalarExpr::Num(x * y),
        _ => ScalarExpr::Mul(Rc::new(a), Rc::new(b)),
    }
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    match (&a, &b) {
        (ScalarExpr::Num(x), _) if *x == 0.0 => b,
        (_, ScalarExpr::Num(x)) if *x == 0.0 => a,
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => ScalarExpr::Num(x + y),
        _ => ScalarExpr::Add(Rc::new(a), Rc::new(b)),
    }
}
