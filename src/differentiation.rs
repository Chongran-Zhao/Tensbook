//! Symbolic differentiation.
//!
//! Phase-2 scope:
//! - scalar-by-tensor: `diff(J, F)`, `diff(W, F)` — evaluated immediately
//!   into a second-order tensor expression using continuum-mechanics rules
//!   (`∂(det F)/∂F = det(F) F^{-T}`, `∂tr(FᵀF)/∂F = 2F`, chain rule, ...);
//! - tensor-by-tensor: `diff(C, F)` — kept as an opaque order-4
//!   [`TensorExpr::Diff`] node; its component formula is produced on demand
//!   via the abstract-index engine ([`crate::indices`]).
//!
//! Index convention: `(∂P/∂F)_{iJkL} = ∂P_{iJ}/∂F_{kL}` — result indices are
//! the numerator's followed by the denominator's.

use crate::error::Error;
use crate::indices::{
    abstract_component, component_base, contract, differentiate, instantiate,
    render_terms,
};
use crate::symbolic::{fold_mul, fold_pow, with_coeff, ScalarExpr};
use crate::tensor::{TensorExpr, TensorProperties};
use std::rc::Rc;

/// `∂s/∂X` for a scalar `s` and a declared second-order tensor variable `X`.
/// Returns a second-order tensor expression (zero if `s` does not depend on `X`).
pub fn diff_scalar_by_tensor(
    s: &Rc<ScalarExpr>,
    x: &Rc<TensorExpr>,
) -> Result<Rc<TensorExpr>, Error> {
    match d_scalar(s, x)? {
        Some(t) => Ok(t),
        None => Ok(Rc::new(TensorExpr::ScalarMul(
            Rc::new(ScalarExpr::Num(0.0)),
            x.clone(),
        ))),
    }
}

/// `None` means the derivative is identically zero.
fn d_scalar(
    s: &Rc<ScalarExpr>,
    x: &Rc<TensorExpr>,
) -> Result<Option<Rc<TensorExpr>>, Error> {
    if !s_contains(s, x) {
        return Ok(None);
    }
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => Ok(None),
        ScalarExpr::Add(a, b) => Ok(tadd(d_scalar(a, x)?, d_scalar(b, x)?)),
        ScalarExpr::Sub(a, b) => Ok(tsub(d_scalar(a, x)?, d_scalar(b, x)?)),
        ScalarExpr::Neg(a) => Ok(d_scalar(a, x)?.map(tneg)),
        ScalarExpr::Mul(a, b) => {
            // d(ab) = b·da + a·db (a, b scalars; derivative is a tensor)
            let da = d_scalar(a, x)?.map(|t| smul(b.clone(), t));
            let db = d_scalar(b, x)?.map(|t| smul(a.clone(), t));
            Ok(tadd(da, db))
        }
        ScalarExpr::Div(a, b) => {
            // d(a/b) = da/b − (a/b²)·db
            let da = d_scalar(a, x)?.map(|t| {
                smul(
                    Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), b.clone())),
                    t,
                )
            });
            let db = d_scalar(b, x)?.map(|t| {
                let coeff = Rc::new(ScalarExpr::Div(
                    a.clone(),
                    Rc::new(ScalarExpr::Pow(b.clone(), Rc::new(ScalarExpr::Num(2.0)))),
                ));
                tneg(smul(coeff, t))
            });
            Ok(tsub_opt(da, db))
        }
        ScalarExpr::Pow(base, exp) => {
            let n = match &**exp {
                ScalarExpr::Num(n) => *n,
                _ if !s_contains(exp, x) => {
                    return Err(Error::msg(
                        "d(a^b)/dX with a non-numeric exponent is not supported yet",
                    ))
                }
                _ => {
                    return Err(Error::msg(
                        "d(a^b)/dX with X-dependent exponent is not supported yet",
                    ))
                }
            };
            // d(a^n) = n a^{n-1} da
            let coeff = with_coeff(n, Some(fold_pow(base.clone(), n - 1.0)));
            Ok(d_scalar(base, x)?.map(|t| smul(coeff.clone(), t)))
        }
        ScalarExpr::Log(a) => {
            // ∂ log(det X)/∂X = X^{-T} (the J/J cancellation done exactly).
            if let ScalarExpr::Det(t) = &**a {
                if t == x {
                    return Ok(Some(Rc::new(TensorExpr::InverseTranspose(x.clone()))));
                }
            }
            let inv = Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), a.clone()));
            Ok(d_scalar(a, x)?.map(|t| smul(inv.clone(), t)))
        }
        ScalarExpr::Det(t) => {
            if t == x {
                // Jacobi: ∂(det X)/∂X = det(X) X^{-T}
                Ok(Some(Rc::new(TensorExpr::ScalarMul(
                    Rc::new(ScalarExpr::Det(x.clone())),
                    Rc::new(TensorExpr::InverseTranspose(x.clone())),
                ))))
            } else {
                Err(Error::msg(
                    "∂det(A)/∂X for A ≠ X requires the general chain rule (not in this phase)",
                ))
            }
        }
        ScalarExpr::Tr(t) => d_trace(t, x),
        ScalarExpr::Ddot(_, _) => Err(Error::msg(
            "differentiating a double contraction is not supported yet",
        )),
    }
}

/// `∂ tr(T)/∂X` using linearity of the trace.
fn d_trace(
    t: &Rc<TensorExpr>,
    x: &Rc<TensorExpr>,
) -> Result<Option<Rc<TensorExpr>>, Error> {
    if !t_contains(t, x) {
        return Ok(None);
    }
    match &**t {
        // ∂tr(X)/∂X = ∂tr(Xᵀ)/∂X = I
        _ if t == x => Ok(Some(identity_like(x))),
        TensorExpr::Transpose(inner) if inner == x => Ok(Some(identity_like(x))),
        TensorExpr::MatMul(a, b) => {
            // ∂tr(XᵀX)/∂X = 2X and ∂tr(XXᵀ)/∂X = 2X
            if let TensorExpr::Transpose(inner) = &**a {
                if inner == b && b == x {
                    return Ok(Some(smul(Rc::new(ScalarExpr::Num(2.0)), x.clone())));
                }
            }
            if let TensorExpr::Transpose(inner) = &**b {
                if inner == a && a == x {
                    return Ok(Some(smul(Rc::new(ScalarExpr::Num(2.0)), x.clone())));
                }
            }
            // ∂tr(AX)/∂X = ∂tr(XA)/∂X = Aᵀ for X-independent A
            if a == x && !t_contains(b, x) {
                return Ok(Some(Rc::new(TensorExpr::Transpose(b.clone()))));
            }
            if b == x && !t_contains(a, x) {
                return Ok(Some(Rc::new(TensorExpr::Transpose(a.clone()))));
            }
            Err(Error::msg(
                "∂tr(AB)/∂X is only supported for AᵀA, AAᵀ, AX, XA forms in this phase",
            ))
        }
        TensorExpr::Add(a, b) => Ok(tadd(d_trace(a, x)?, d_trace(b, x)?)),
        TensorExpr::Sub(a, b) => Ok(tsub(d_trace(a, x)?, d_trace(b, x)?)),
        TensorExpr::ScalarMul(s, inner) if !s_contains(s, x) => {
            Ok(d_trace(inner, x)?.map(|t| smul(s.clone(), t)))
        }
        TensorExpr::Neg(inner) => Ok(d_trace(inner, x)?.map(tneg)),
        _ => Err(Error::msg(
            "this trace derivative form is not supported in this phase",
        )),
    }
}

/// LaTeX equation for the components of `∂num/∂den`, e.g.
/// `\frac{\partial C_{ij}}{\partial F_{mn}} = \delta_{in} F_{mj} + \delta_{jn} F_{mi}`.
pub fn diff_component_equation(
    num: &Rc<TensorExpr>,
    num_latex: &str,
    den: &Rc<TensorExpr>,
) -> Result<String, Error> {
    let den_base = match &**den {
        TensorExpr::Var { latex, .. } => component_base(latex).to_string(),
        _ => {
            return Err(Error::msg(
                "diff is only supported with respect to a declared tensor variable",
            ))
        }
    };
    if num.order() != 2 || den.order() != 2 {
        return Err(Error::msg(
            "component display of derivatives currently requires second-order/second-order",
        ));
    }
    let (i, j, m, n) = (
        "i".to_string(),
        "j".to_string(),
        "m".to_string(),
        "n".to_string(),
    );
    let mut counter = 0usize;
    let terms = abstract_component(num, &i, &j, &mut counter)?;
    let differentiated = differentiate(&terms, &den_base, &m, &n)?;
    let contracted = contract(differentiated, num.dim());
    Ok(format!(
        "\\frac{{\\partial {}_{{ij}}}}{{\\partial {}_{{mn}}}} = {}",
        component_base(num_latex),
        den_base,
        render_terms(&contracted)
    ))
}

/// Block-component display of a fourth-order derivative `A = ∂P/∂F`:
/// expand over the *last two* indices `(k, L)` into a dim × dim grid of
/// second-order blocks, `(\bm A^{kL})_{ij} = A_{ijkL}`, each block rendered
/// as the index formula for the corresponding concrete `(k, L)`.
pub fn diff_block_components(
    num: &Rc<TensorExpr>,
    num_latex: &str,
    den: &Rc<TensorExpr>,
) -> Result<String, Error> {
    let den_base = match &**den {
        TensorExpr::Var { latex, .. } => component_base(latex).to_string(),
        _ => {
            return Err(Error::msg(
                "diff is only supported with respect to a declared tensor variable",
            ))
        }
    };
    if num.order() != 2 || den.order() != 2 {
        return Err(Error::msg(
            "block_components requires a fourth-order derivative ∂(2nd)/∂(2nd)",
        ));
    }
    let dim = num.dim();
    let (i, j, m, n) = (
        "i".to_string(),
        "j".to_string(),
        "m".to_string(),
        "n".to_string(),
    );
    let mut counter = 0usize;
    let terms = abstract_component(num, &i, &j, &mut counter)?;
    let differentiated = differentiate(&terms, &den_base, &m, &n)?;
    let general = contract(differentiated, dim);

    let mut blocks = Vec::with_capacity(dim);
    for k in 1..=dim {
        let mut row = Vec::with_capacity(dim);
        for l in 1..=dim {
            // Fix the denominator indices (m, n) = (k, L); leave (i, j) free.
            let inst = instantiate(&general, &[(m.clone(), k), (n.clone(), l)], dim);
            row.push(format!(
                "\\left[ {} \\right]_{{ij}}",
                render_terms(&inst)
            ));
        }
        blocks.push(row.join(" & "));
    }
    Ok(format!(
        "\\frac{{\\partial {}_{{ij}}}}{{\\partial {}_{{kL}}}} = \n\
         \\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
        component_base(num_latex),
        den_base,
        blocks.join(" \\\\\n")
    ))
}

// ---- structural dependence ------------------------------------------------

pub fn t_contains(t: &TensorExpr, x: &TensorExpr) -> bool {
    if t == x {
        return true;
    }
    match t {
        TensorExpr::Var { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => t_contains(a, x),
        TensorExpr::Diff { num, den, .. } => t_contains(num, x) || t_contains(den, x),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b) => t_contains(a, x) || t_contains(b, x),
        TensorExpr::Spectral { base, .. } | TensorExpr::SpectralFn { base, .. } => {
            t_contains(base, x)
        }
        TensorExpr::ScalarMul(s, a) => s_contains(s, x) || t_contains(a, x),
    }
}

pub fn s_contains(s: &ScalarExpr, x: &TensorExpr) -> bool {
    match s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => s_contains(a, x) || s_contains(b, x),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) => s_contains(a, x),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => t_contains(t, x),
        ScalarExpr::Ddot(a, b) => t_contains(a, x) || t_contains(b, x),
    }
}

// ---- tensor combination helpers --------------------------------------------

/// Scalar·tensor with coefficient folding (`mu/2 · 2F → mu F`).
fn smul(s: Rc<ScalarExpr>, t: Rc<TensorExpr>) -> Rc<TensorExpr> {
    if let TensorExpr::ScalarMul(s2, t2) = &*t {
        return smul(fold_mul(&s, s2), t2.clone());
    }
    if let ScalarExpr::Num(n) = &*s {
        if *n == 1.0 {
            return t;
        }
        if *n == -1.0 {
            return tneg(t);
        }
    }
    Rc::new(TensorExpr::ScalarMul(s, t))
}

fn tneg(t: Rc<TensorExpr>) -> Rc<TensorExpr> {
    match &*t {
        TensorExpr::Neg(inner) => inner.clone(),
        _ => Rc::new(TensorExpr::Neg(t)),
    }
}

fn tadd(
    a: Option<Rc<TensorExpr>>,
    b: Option<Rc<TensorExpr>>,
) -> Option<Rc<TensorExpr>> {
    match (a, b) {
        (None, None) => None,
        (Some(t), None) | (None, Some(t)) => Some(t),
        (Some(x), Some(y)) => Some(Rc::new(TensorExpr::Add(x, y))),
    }
}

fn tsub(
    a: Option<Rc<TensorExpr>>,
    b: Option<Rc<TensorExpr>>,
) -> Option<Rc<TensorExpr>> {
    match (a, b) {
        (None, None) => None,
        (Some(t), None) => Some(t),
        (None, Some(t)) => Some(tneg(t)),
        (Some(x), Some(y)) => Some(Rc::new(TensorExpr::Sub(x, y))),
    }
}

/// Like [`tsub`] but the second operand is already negated (used by the
/// quotient rule where the db term carries its own minus sign).
fn tsub_opt(
    a: Option<Rc<TensorExpr>>,
    neg_b: Option<Rc<TensorExpr>>,
) -> Option<Rc<TensorExpr>> {
    tadd(a, neg_b)
}

fn identity_like(x: &TensorExpr) -> Rc<TensorExpr> {
    Rc::new(TensorExpr::Var {
        name: "I".to_string(),
        latex: "\\bm I".to_string(),
        order: 2,
        dim: x.dim(),
        props: TensorProperties {
            identity: true,
            ..Default::default()
        },
    })
}
