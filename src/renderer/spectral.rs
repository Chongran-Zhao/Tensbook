//! Spectral-mode rendering (`display(X, mode=spectral)`).
//!
//! Renders order-2 expressions built around a generalized strain as
//! principal-axis sums:
//! - `E(C)`            → `Σ_a E(λ_a) M_a`
//! - `T = 2μE + λtr(E)I` → `Σ_a (2μ E(λ_a) + λ Σ_b E(λ_b)) M_a`
//! - `S = T : Q`       → `Σ_a (E'(λ_a)/λ_a) T_a M_a`
//!
//! The last form uses the coaxiality of `T` with `C` (guaranteed because
//! `T` is restricted to isotropic functions of `E`: built from `E`, `I`,
//! `tr E`, `E : E` and scalars), under which the off-diagonal part of
//! `Q = 2∂E/∂C` does not contribute.

use crate::error::Error;
use crate::renderer::latex::scalar_to_latex;
use crate::symbolic::{fold_add, fold_mul, fold_sub, with_coeff, ScalarExpr};
use crate::tensor::{Scale, TensorExpr};
use std::rc::Rc;

pub fn tensor_to_spectral(t: &Rc<TensorExpr>) -> Result<String, Error> {
    match &**t {
        // pull numeric/scalar prefactors into the summand
        TensorExpr::ScalarMul(c, inner) => spectral_sum_scaled(inner, Some(c.clone())),
        _ => spectral_sum_scaled(t, None),
    }
}

fn spectral_sum_scaled(t: &Rc<TensorExpr>, coeff: Option<Rc<ScalarExpr>>) -> Result<String, Error> {
    let (body, dim) = spectral_summand(t)?;
    let body = match coeff {
        Some(c) => fold_mul(&c, &body),
        None => body,
    };
    Ok(format!(
        "\\sum_{{a=1}}^{{{dim}}} \\left[ {} \\right] \\bm M_a",
        scalar_to_latex(&body)
    ))
}

/// The principal-value summand of an order-2 spectral expression.
fn spectral_summand(t: &Rc<TensorExpr>) -> Result<(Rc<ScalarExpr>, usize), Error> {
    match &**t {
        // S = T : Q  →  S_a = (E'(λ_a)/λ_a) T_a
        TensorExpr::DdotTQ { second, fourth } => {
            let TensorExpr::QTensor { strain } = &**fourth else {
                return Err(Error::msg(
                    "spectral display of T : A is only available when A is a \
                     strain projection Q = 2∂E/∂C",
                ));
            };
            let (g, scale, lam) = strain_parts(strain)?;
            let t_a = principal(second, &g, scale, &lam)?;
            let factor = Rc::new(ScalarExpr::Div(scale.dapply(&lam), lam.clone()));
            Ok((fold_mul(&factor, &t_a), strain.dim()))
        }
        _ => {
            let g = find_strain(t)?;
            let (gnode, scale, lam) = strain_parts(&g)?;
            Ok((principal(t, &gnode, scale, &lam)?, t.dim()))
        }
    }
}

/// Destructure a GenStrain into (node, scale, λ_a symbol).
fn strain_parts(
    strain: &Rc<TensorExpr>,
) -> Result<(Rc<TensorExpr>, &Scale, Rc<ScalarExpr>), Error> {
    let TensorExpr::GenStrain { base, scale, .. } = &**strain else {
        return Err(Error::msg("expected a generalized strain"));
    };
    let lam = Rc::new(ScalarExpr::Eig {
        base: base.clone(),
        symbol: "\\lambda".to_string(),
        index: "a".to_string(),
    });
    Ok((strain.clone(), scale, lam))
}

/// Find the unique generalized strain inside an order-2 expression.
fn find_strain(t: &Rc<TensorExpr>) -> Result<Rc<TensorExpr>, Error> {
    let mut found: Vec<Rc<TensorExpr>> = Vec::new();
    collect(t, &mut found);
    found.dedup();
    if found.is_empty() {
        return Err(Error::msg(
            "spectral display requires a generalized strain (gstrain) in the \
             expression; use mode=symbol otherwise",
        ));
    }
    if found.iter().any(|g| g != &found[0]) {
        return Err(Error::msg(
            "spectral display with multiple distinct strains is not supported",
        ));
    }
    Ok(found.remove(0))
}

fn collect(t: &Rc<TensorExpr>, out: &mut Vec<Rc<TensorExpr>>) {
    if matches!(&**t, TensorExpr::GenStrain { .. }) {
        out.push(t.clone());
        return;
    }
    match &**t {
        TensorExpr::ScalarMul(_, a) | TensorExpr::Neg(a) | TensorExpr::Transpose(a) => {
            collect(a, out)
        }
        TensorExpr::Add(a, b) | TensorExpr::Sub(a, b) => {
            collect(a, out);
            collect(b, out);
        }
        _ => {}
    }
}

/// Principal value of an order-2 expression coaxial with the strain:
/// `E → E(λ_a)`, `I → 1`, linear combinations map through.
fn principal(
    t: &Rc<TensorExpr>,
    g: &Rc<TensorExpr>,
    scale: &Scale,
    lam: &Rc<ScalarExpr>,
) -> Result<Rc<ScalarExpr>, Error> {
    if t == g {
        return Ok(scale.apply(lam));
    }
    match &**t {
        TensorExpr::Var { props, .. } if props.identity => Ok(Rc::new(ScalarExpr::Num(1.0))),
        TensorExpr::Add(a, b) => Ok(fold_add(
            principal(a, g, scale, lam)?,
            principal(b, g, scale, lam)?,
        )),
        TensorExpr::Sub(a, b) => Ok(fold_sub(
            principal(a, g, scale, lam)?,
            principal(b, g, scale, lam)?,
        )),
        TensorExpr::Neg(a) => Ok(with_coeff(-1.0, Some(principal(a, g, scale, lam)?))),
        TensorExpr::ScalarMul(s, a) => Ok(fold_mul(
            &principal_scalar(s, g, scale)?,
            &principal(a, g, scale, lam)?,
        )),
        TensorExpr::Transpose(a) => principal(a, g, scale, lam), // coaxial ⇒ symmetric
        _ => Err(Error::msg(
            "spectral display supports expressions built from the strain E, the \
             identity, and scalar coefficients (isotropic tensor functions of E)",
        )),
    }
}

/// Map scalar coefficients: invariants of E become spectral sums over a
/// fresh index b (`tr E → Σ_b E(λ_b)`, `E : E → Σ_b E(λ_b)²`).
fn principal_scalar(
    s: &Rc<ScalarExpr>,
    g: &Rc<TensorExpr>,
    scale: &Scale,
) -> Result<Rc<ScalarExpr>, Error> {
    let lam_b = |base: &Rc<TensorExpr>| {
        Rc::new(ScalarExpr::Eig {
            base: base.clone(),
            symbol: "\\lambda".to_string(),
            index: "b".to_string(),
        })
    };
    let TensorExpr::GenStrain { base, .. } = &**g else {
        return Err(Error::msg("expected a generalized strain"));
    };
    let dim = g.dim();
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::Eig { .. } => Ok(s.clone()),
        ScalarExpr::Tr(t) if t == g => Ok(Rc::new(ScalarExpr::SpecSum {
            body: scale.apply(&lam_b(base)),
            index: "b".to_string(),
            dim,
        })),
        ScalarExpr::Ddot(a, b) if a == g && b == g => Ok(Rc::new(ScalarExpr::SpecSum {
            body: Rc::new(ScalarExpr::Pow(
                scale.apply(&lam_b(base)),
                Rc::new(ScalarExpr::Num(2.0)),
            )),
            index: "b".to_string(),
            dim,
        })),
        ScalarExpr::Add(a, b) => Ok(fold_add(
            principal_scalar(a, g, scale)?,
            principal_scalar(b, g, scale)?,
        )),
        ScalarExpr::Sub(a, b) => Ok(fold_sub(
            principal_scalar(a, g, scale)?,
            principal_scalar(b, g, scale)?,
        )),
        ScalarExpr::Mul(a, b) => Ok(fold_mul(
            &principal_scalar(a, g, scale)?,
            &principal_scalar(b, g, scale)?,
        )),
        ScalarExpr::Div(a, b) => Ok(Rc::new(ScalarExpr::Div(
            principal_scalar(a, g, scale)?,
            principal_scalar(b, g, scale)?,
        ))),
        ScalarExpr::Pow(a, b) => Ok(Rc::new(ScalarExpr::Pow(
            principal_scalar(a, g, scale)?,
            principal_scalar(b, g, scale)?,
        ))),
        ScalarExpr::Neg(a) => Ok(with_coeff(-1.0, Some(principal_scalar(a, g, scale)?))),
        _ => Err(Error::msg(
            "this scalar coefficient is not supported in spectral display yet",
        )),
    }
}
