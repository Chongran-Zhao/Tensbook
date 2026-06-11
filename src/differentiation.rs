//! Symbolic differentiation.
//!
//! Phase-2 scope:
//! - scalar-by-tensor: `diff(J, F)`, `diff(W, F)` — evaluated immediately
//!   into a second-order tensor expression using continuum-mechanics rules
//!   (`∂(det F)/∂F = det(F) F^{-T}`, `∂tr(FᵀF)/∂F = 2F`, chain rule, ...);
//! - tensor-by-tensor: `diff(C, F)` — kept as an opaque order-4
//!   [`TensorExpr::Diff`] node; compound denominators are first treated as a
//!   synthetic independent variable, then their component formula is produced
//!   on demand via the abstract-index engine ([`crate::indices`]).
//!
//! Index convention: `(∂P/∂F)_{iJkL} = ∂P_{iJ}/∂F_{kL}` — result indices are
//! the numerator's followed by the denominator's.

use crate::error::Error;
use crate::indices::{
    abstract_component, component_base, contract, differentiate, instantiate, render_terms,
};
use crate::simplifier::{simplify_scalar, RuleSet};
use crate::symbolic::{fold_mul, fold_pow, with_coeff, ScalarExpr, SetIndex};
use crate::tensor::{TensorExpr, TensorProperties};
use std::rc::Rc;

/// `∂s/∂X` for a scalar `s` and a second-order tensor expression `X`.
///
/// `X` may be a compound expression (e.g. `C = FᵀF`): it is then treated as
/// an independent variable, matched *structurally* inside `s`.
///
/// When `X = AᵀA` (or `A Aᵀ`), occurrences of `det A` in `s` are first
/// rewritten through the exact identity `det X = (det A)²`:
/// `log(det A) → ½ log(det X)` and `det A → (det X)^{1/2}` (using the
/// standard orientation-preserving assumption `det A > 0`). This makes the
/// ubiquitous continuum-mechanics pattern `W(J)` with `J = det F`
/// differentiable by `C`.
///
/// Any remaining dependence on `X`'s underlying variables outside
/// occurrences of `X` itself is rejected by [`check_independence_scalar`] —
/// e.g. `diff(tr(F), C)` errors rather than silently returning 0.
pub fn diff_scalar_by_tensor(
    s: &Rc<ScalarExpr>,
    x: &Rc<TensorExpr>,
) -> Result<Rc<TensorExpr>, Error> {
    // Chain rule through a generalized strain: if s depends on x only
    // through E = GenStrain(x), then ∂s/∂x = (∂s/∂E) : ∂E/∂C = ½ T : Q
    // with T = ∂s/∂E and Q = 2 ∂E/∂C kept opaque.
    if let Some(strain) = find_gen_strain_of(s, x)? {
        let t = diff_scalar_by_tensor(s, &strain)?;
        let q = Rc::new(TensorExpr::QTensor {
            strain: strain.clone(),
        });
        return Ok(Rc::new(TensorExpr::ScalarMul(
            Rc::new(ScalarExpr::Num(0.5)),
            Rc::new(TensorExpr::ddot_tq(t, q)?),
        )));
    }
    let s_eff = if matches!(&**x, TensorExpr::Var { .. }) {
        s.clone()
    } else {
        let rewritten = match cauchy_green_factor(x) {
            Some(a) => rewrite_scalar_for_compound(s, x, &a),
            None => s.clone(),
        };
        check_independence_scalar(&rewritten, x)?;
        rewritten
    };
    match d_scalar(&s_eff, x)? {
        Some(t) => Ok(t),
        None => Ok(Rc::new(TensorExpr::ScalarMul(
            Rc::new(ScalarExpr::Num(0.0)),
            x.clone(),
        ))),
    }
}

/// `∂T/∂X` for a second-order tensor `T` and a second-order tensor expression
/// `X`.
///
/// Declared tensor denominators are kept directly. Compound denominators are
/// treated as synthetic independent variables, mirroring
/// [`diff_scalar_by_tensor`]: occurrences of the compound `X` are replaced by
/// a fresh tensor variable for component differentiation, while hidden
/// dependence on `X`'s underlying variables is rejected.
pub fn diff_tensor_by_tensor(
    t: &Rc<TensorExpr>,
    x: &Rc<TensorExpr>,
    num_label: Option<String>,
    den_label: Option<String>,
) -> Result<Rc<TensorExpr>, Error> {
    if t.order() != 2 || x.order() != 2 {
        return Err(Error::msg(
            "tensor-by-tensor diff requires second-order numerator and denominator",
        ));
    }
    if let Some(q) = stretch_spectral_derivative(t, x)? {
        return Ok(Rc::new(TensorExpr::ScalarMul(
            Rc::new(ScalarExpr::Num(0.5)),
            q,
        )));
    }
    if is_manual_spectral_tensor(t) {
        return Err(Error::msg(
            "manual spectral tensor functions must be differentiated with \
             respect to a matching stretch spectral decomposition \
             C = sum(lambda[a]^2 * N[a] & N[a], a)",
        ));
    }
    if let Some(strain) = strain_projection_for(t, x, num_label.as_deref())? {
        return Ok(Rc::new(TensorExpr::ScalarMul(
            Rc::new(ScalarExpr::Num(0.5)),
            Rc::new(TensorExpr::QTensor { strain }),
        )));
    }
    if matches!(&**x, TensorExpr::Var { .. }) {
        return Ok(Rc::new(TensorExpr::Diff {
            num: t.clone(),
            den: x.clone(),
            num_label,
        }));
    }

    let rewritten = match cauchy_green_factor(x) {
        Some(a) => rewrite_tensor_for_compound(t, x, &a),
        None => t.clone(),
    };
    check_independence_tensor(&rewritten, x)?;

    let latex = den_label.unwrap_or_else(|| "\\bm X".to_string());
    let x_var = Rc::new(TensorExpr::Var {
        name: latex.clone(),
        latex,
        order: x.order(),
        dim: x.dim(),
        props: TensorProperties {
            symmetric: x.is_symmetric(),
            ..Default::default()
        },
    });
    let rewritten = replace_tensor(&rewritten, x, &x_var);
    let d = d_tensor(&rewritten, &x_var)?.unwrap_or_else(|| zero_fourth_like(&x_var));
    // Map the synthetic variable back to the compound expression so the
    // result composes structurally with other expressions (e.g. `C : ℂ`
    // contractions can cancel C⁻¹ C); display still substitutes the label.
    Ok(replace_tensor(&d, &x_var, x))
}

#[derive(Clone)]
struct StretchFrame {
    lambda_latex: String,
    vector_latex: String,
    dim: usize,
    index: String,
}

struct SpectralTensorFunction {
    frame: StretchFrame,
    coeff: Rc<ScalarExpr>,
    lambda: ScalarSetElemOwned,
}

#[derive(Clone)]
struct ScalarSetElemOwned {
    latex: String,
    index: SetIndex,
    set_dim: usize,
}

fn stretch_spectral_derivative(
    t: &Rc<TensorExpr>,
    x: &Rc<TensorExpr>,
) -> Result<Option<Rc<TensorExpr>>, Error> {
    let Some(frame) = match_stretch_spectral_decomposition(x)? else {
        return Ok(None);
    };
    let Some(func) = match_spectral_tensor_function(t, &frame)? else {
        return Ok(None);
    };
    Ok(Some(build_stretch_q(&func)?))
}

fn is_manual_spectral_tensor(t: &Rc<TensorExpr>) -> bool {
    let TensorExpr::SumIdx { index, body, .. } = &**t else {
        return false;
    };
    let Some((_coeff, projector)) = scalar_times_tensor(body) else {
        return false;
    };
    projector_vector_set(projector, index).is_some()
}

fn match_stretch_spectral_decomposition(x: &Rc<TensorExpr>) -> Result<Option<StretchFrame>, Error> {
    let TensorExpr::SumIdx { index, body, .. } = &**x else {
        return Ok(None);
    };
    let Some((coeff, projector)) = scalar_times_tensor(body) else {
        return Ok(None);
    };
    let Some((vector_latex, dim)) = projector_vector_set(projector, index) else {
        return Ok(None);
    };
    let Some(lambda) = squared_scalar_set_elem(coeff, index)? else {
        return Ok(None);
    };
    if lambda.set_dim != dim {
        return Err(Error::msg(
            "stretch spectral C uses scalar and vector sets with different dimensions",
        ));
    }
    Ok(Some(StretchFrame {
        lambda_latex: lambda.latex,
        vector_latex,
        dim,
        index: index.clone(),
    }))
}

fn match_spectral_tensor_function(
    t: &Rc<TensorExpr>,
    frame: &StretchFrame,
) -> Result<Option<SpectralTensorFunction>, Error> {
    let TensorExpr::SumIdx { index, body, .. } = &**t else {
        return Ok(None);
    };
    let Some((coeff, projector)) = scalar_times_tensor(body) else {
        return Ok(None);
    };
    if index != &frame.index {
        return Err(Error::msg(
            "spectral tensor function must use the same summation index as C",
        ));
    }
    let Some((vector_latex, dim)) = projector_vector_set(projector, index) else {
        return Ok(None);
    };
    if vector_latex != frame.vector_latex || dim != frame.dim {
        return Err(Error::msg(
            "spectral tensor function must use the same vector set and dimension as C",
        ));
    }
    let Some(lambda_ref) = unique_scalar_set_elem(coeff, index)? else {
        return Err(Error::msg(
            "spectral tensor function coefficient must depend on the stretch set",
        ));
    };
    if lambda_ref.latex != frame.lambda_latex || lambda_ref.set_dim != frame.dim {
        return Err(Error::msg(
            "spectral tensor function must use the same stretch set as C",
        ));
    }
    Ok(Some(SpectralTensorFunction {
        frame: frame.clone(),
        coeff: coeff.clone(),
        lambda: ScalarSetElemOwned {
            latex: lambda_ref.latex.to_string(),
            index: lambda_ref.index.clone(),
            set_dim: lambda_ref.set_dim,
        },
    }))
}

fn build_stretch_q(func: &SpectralTensorFunction) -> Result<Rc<TensorExpr>, Error> {
    let frame = &func.frame;
    let var = func.lambda.latex.as_str();
    let body = replace_scalar_set_elem_with_var(&func.coeff, func.lambda.as_ref(), var);
    let dbody = simplify_scalar(&diff_scalar_by_scalar(&body, var)?, RuleSet::Continuum);

    let lam_a = scalar_set_elem(
        &frame.lambda_latex,
        &SetIndex::Sym("a".to_string()),
        frame.dim,
    );
    let lam_b = scalar_set_elem(
        &frame.lambda_latex,
        &SetIndex::Sym("b".to_string()),
        frame.dim,
    );
    let n_a = vector_set_elem(
        &frame.vector_latex,
        &SetIndex::Sym("a".to_string()),
        frame.dim,
    );
    let n_b = vector_set_elem(
        &frame.vector_latex,
        &SetIndex::Sym("b".to_string()),
        frame.dim,
    );
    let p_a = Rc::new(TensorExpr::outer(n_a.clone(), n_a)?);
    let p_b = Rc::new(TensorExpr::outer(n_b.clone(), n_b)?);

    let phi_a = crate::substitute::subst_scalar_sym(&body, var, &lam_a);
    let phi_b = crate::substitute::subst_scalar_sym(&body, var, &lam_b);
    let dphi_a = crate::substitute::subst_scalar_sym(&dbody, var, &lam_a);

    let diag_coeff = simplify_scalar(&divide_flat(dphi_a, lam_a.clone()), RuleSet::Continuum);
    let diag_body = Rc::new(TensorExpr::ScalarMul(
        diag_coeff,
        Rc::new(TensorExpr::outer(p_a.clone(), p_a.clone())?),
    ));
    let diag = Rc::new(TensorExpr::SumIdx {
        index: "a".to_string(),
        range: frame.dim,
        exclude: None,
        body: diag_body,
    });

    let numerator = fold_mul(&Rc::new(ScalarExpr::Num(2.0)), &subtract_flat(phi_a, phi_b));
    let denominator = Rc::new(ScalarExpr::Sub(
        Rc::new(ScalarExpr::Pow(lam_a, Rc::new(ScalarExpr::Num(2.0)))),
        Rc::new(ScalarExpr::Pow(lam_b, Rc::new(ScalarExpr::Num(2.0)))),
    ));
    let off_coeff = simplify_scalar(&divide_flat(numerator, denominator), RuleSet::Continuum);
    let off_body = Rc::new(TensorExpr::ScalarMul(
        off_coeff,
        Rc::new(TensorExpr::box_times(p_a, p_b)?),
    ));
    let off_inner = Rc::new(TensorExpr::SumIdx {
        index: "b".to_string(),
        range: frame.dim,
        exclude: Some("a".to_string()),
        body: off_body,
    });
    let off = Rc::new(TensorExpr::SumIdx {
        index: "a".to_string(),
        range: frame.dim,
        exclude: None,
        body: off_inner,
    });

    Ok(Rc::new(TensorExpr::Add(diag, off)))
}

impl ScalarSetElemOwned {
    fn as_ref(&self) -> ScalarSetElemRef<'_> {
        ScalarSetElemRef {
            latex: &self.latex,
            index: &self.index,
            set_dim: self.set_dim,
        }
    }
}

fn scalar_times_tensor(t: &Rc<TensorExpr>) -> Option<(&Rc<ScalarExpr>, &Rc<TensorExpr>)> {
    match &**t {
        TensorExpr::ScalarMul(s, body) => Some((s, body)),
        _ => None,
    }
}

fn projector_vector_set(t: &Rc<TensorExpr>, index: &str) -> Option<(String, usize)> {
    let TensorExpr::Outer(a, b) = &**t else {
        return None;
    };
    let TensorExpr::SetElem {
        latex: la,
        order: 1,
        dim: da,
        index: ia,
        set_dim: sda,
        ..
    } = &**a
    else {
        return None;
    };
    let TensorExpr::SetElem {
        latex: lb,
        order: 1,
        dim: db,
        index: ib,
        set_dim: sdb,
        ..
    } = &**b
    else {
        return None;
    };
    if la == lb && da == db && sda == sdb && ia == ib && ia.sym() == Some(index) {
        Some((la.clone(), *sda))
    } else {
        None
    }
}

fn squared_scalar_set_elem(
    s: &Rc<ScalarExpr>,
    index: &str,
) -> Result<Option<ScalarSetElemOwned>, Error> {
    let ScalarExpr::Pow(base, exp) = &**s else {
        return Ok(None);
    };
    if !matches!(&**exp, ScalarExpr::Num(n) if *n == 2.0) {
        return Ok(None);
    }
    let Some(lambda) = unique_scalar_set_elem(base, index)? else {
        return Ok(None);
    };
    Ok(Some(ScalarSetElemOwned {
        latex: lambda.latex.to_string(),
        index: lambda.index.clone(),
        set_dim: lambda.set_dim,
    }))
}

fn scalar_set_elem(latex: &str, index: &SetIndex, dim: usize) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::SetElem {
        latex: latex.to_string(),
        index: index.clone(),
        set_dim: dim,
        eig: None,
    })
}

fn vector_set_elem(latex: &str, index: &SetIndex, dim: usize) -> Rc<TensorExpr> {
    Rc::new(TensorExpr::SetElem {
        latex: latex.to_string(),
        order: 1,
        dim,
        index: index.clone(),
        set_dim: dim,
        base: None,
    })
}

fn divide_flat(num: Rc<ScalarExpr>, den: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    let (flat_num, flat_den) = fraction_parts(num);
    Rc::new(ScalarExpr::Div(flat_num, fold_mul(&flat_den, &den)))
}

fn subtract_flat(a: Rc<ScalarExpr>, b: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    let (an, ad) = fraction_parts(a);
    let (bn, bd) = fraction_parts(b);
    if ad == bd {
        divide_flat(Rc::new(ScalarExpr::Sub(an, bn)), ad)
    } else {
        divide_flat(
            Rc::new(ScalarExpr::Sub(fold_mul(&an, &bd), fold_mul(&bn, &ad))),
            fold_mul(&ad, &bd),
        )
    }
}

fn fraction_parts(s: Rc<ScalarExpr>) -> (Rc<ScalarExpr>, Rc<ScalarExpr>) {
    match &*s {
        ScalarExpr::Div(a, b) => {
            let (an, ad) = fraction_parts(a.clone());
            let (bn, bd) = fraction_parts(b.clone());
            (fold_mul(&an, &bd), fold_mul(&ad, &bn))
        }
        ScalarExpr::Mul(a, b) => {
            let (an, ad) = fraction_parts(a.clone());
            let (bn, bd) = fraction_parts(b.clone());
            (fold_mul(&an, &bn), fold_mul(&ad, &bd))
        }
        _ => (s, Rc::new(ScalarExpr::Num(1.0))),
    }
}

/// Recognize a native `gstrain(C, ...)` as a generalized strain of `x`, so
/// legacy `gstrain` programs can still use the opaque Q path.
fn strain_projection_for(
    t: &Rc<TensorExpr>,
    x: &Rc<TensorExpr>,
    label: Option<&str>,
) -> Result<Option<Rc<TensorExpr>>, Error> {
    if let TensorExpr::GenStrain { base, .. } = &**t {
        if base == x {
            return Ok(Some(t.clone()));
        }
        return Ok(None);
    }
    let _ = label;
    Ok(None)
}

#[derive(Clone, Copy)]
struct ScalarSetElemRef<'a> {
    latex: &'a str,
    index: &'a SetIndex,
    set_dim: usize,
}

fn unique_scalar_set_elem<'a>(
    s: &'a Rc<ScalarExpr>,
    index: &str,
) -> Result<Option<ScalarSetElemRef<'a>>, Error> {
    let mut found = Vec::new();
    collect_scalar_set_elems(s, index, &mut found);
    let Some(first) = found.first().copied() else {
        return Ok(None);
    };
    if found
        .iter()
        .any(|e| e.latex != first.latex || e.index != first.index || e.set_dim != first.set_dim)
    {
        return Err(Error::msg(
            "manual spectral strain scale must use one eigenvalue set with the \
             summation index",
        ));
    }
    Ok(Some(first))
}

fn collect_scalar_set_elems<'a>(
    s: &'a Rc<ScalarExpr>,
    index: &str,
    out: &mut Vec<ScalarSetElemRef<'a>>,
) {
    match &**s {
        ScalarExpr::SetElem {
            latex,
            index: set_index,
            set_dim,
            eig: _,
        } if set_index.sym() == Some(index) => out.push(ScalarSetElemRef {
            latex,
            index: set_index,
            set_dim: *set_dim,
        }),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            collect_scalar_set_elems(a, index, out);
            collect_scalar_set_elems(b, index, out);
        }
        ScalarExpr::Ddot(a, b) => {
            collect_scalar_set_elems_tensor(a, index, out);
            collect_scalar_set_elems_tensor(b, index, out);
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            collect_scalar_set_elems(a, index, out)
        }
        ScalarExpr::SpecSum { body, .. } => collect_scalar_set_elems(body, index, out),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => collect_scalar_set_elems_tensor(t, index, out),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Eig { .. }
        | ScalarExpr::SetElem { .. } => {}
    }
}

fn collect_scalar_set_elems_tensor<'a>(
    t: &'a Rc<TensorExpr>,
    index: &str,
    out: &mut Vec<ScalarSetElemRef<'a>>,
) {
    match &**t {
        TensorExpr::ScalarMul(s, a) => {
            collect_scalar_set_elems(s, index, out);
            collect_scalar_set_elems_tensor(a, index, out);
        }
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => collect_scalar_set_elems_tensor(a, index, out),
        TensorExpr::Diff { num, den, .. } => {
            collect_scalar_set_elems_tensor(num, index, out);
            collect_scalar_set_elems_tensor(den, index, out);
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b)
        | TensorExpr::DdotTQ {
            second: a,
            fourth: b,
        } => {
            collect_scalar_set_elems_tensor(a, index, out);
            collect_scalar_set_elems_tensor(b, index, out);
        }
        TensorExpr::Spectral { base, .. }
        | TensorExpr::SpectralFn { base, .. }
        | TensorExpr::GenStrain { base, .. } => collect_scalar_set_elems_tensor(base, index, out),
        TensorExpr::QTensor { strain } => collect_scalar_set_elems_tensor(strain, index, out),
        TensorExpr::Var { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. }
        | TensorExpr::SumIdx { .. } => {}
    }
}

fn replace_scalar_set_elem_with_var(
    s: &Rc<ScalarExpr>,
    target: ScalarSetElemRef<'_>,
    var: &str,
) -> Rc<ScalarExpr> {
    let rs = |e: &Rc<ScalarExpr>| replace_scalar_set_elem_with_var(e, target, var);
    match &**s {
        ScalarExpr::SetElem {
            latex,
            index,
            set_dim,
            ..
        } if latex == target.latex && index == target.index && *set_dim == target.set_dim => {
            Rc::new(ScalarExpr::Sym {
                name: var.to_string(),
                latex: var.to_string(),
            })
        }
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(rs(a), rs(b))),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(rs(a), rs(b))),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(rs(a), rs(b))),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(rs(a), rs(b))),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(rs(a), rs(b))),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(rs(a))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(rs(a))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: rs(arg),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: rs(body),
            index: index.clone(),
            dim: *dim,
        }),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Eig { .. }
        | ScalarExpr::SetElem { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _) => s.clone(),
    }
}

/// If `x` is structurally `AᵀA` or `A Aᵀ`, return `A`.
fn cauchy_green_factor(x: &TensorExpr) -> Option<Rc<TensorExpr>> {
    if let TensorExpr::MatMul(a, b) = x {
        if let TensorExpr::Transpose(inner) = &**a {
            if inner == b {
                return Some(b.clone());
            }
        }
        if let TensorExpr::Transpose(inner) = &**b {
            if inner == a {
                return Some(a.clone());
            }
        }
    }
    None
}

/// Rewrite `det A` in terms of `X = AᵀA` (exact, assuming `det A > 0`):
/// `log(det A) → log(det X)/2`, `det A → (det X)^{1/2}`. Occurrences of `X`
/// itself are left opaque so structural matching still sees them.
fn rewrite_scalar_for_compound(
    s: &Rc<ScalarExpr>,
    x: &Rc<TensorExpr>,
    a: &Rc<TensorExpr>,
) -> Rc<ScalarExpr> {
    let rw = |e: &Rc<ScalarExpr>| rewrite_scalar_for_compound(e, x, a);
    let rwt = |t: &Rc<TensorExpr>| rewrite_tensor_for_compound(t, x, a);
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::Eig { .. } => s.clone(),
        ScalarExpr::SetElem {
            latex,
            index,
            set_dim,
            eig,
        } => Rc::new(ScalarExpr::SetElem {
            latex: latex.clone(),
            index: index.clone(),
            set_dim: *set_dim,
            eig: eig.as_ref().map(|l| crate::symbolic::EigLink {
                base: rwt(&l.base),
                vec_latex: l.vec_latex.clone(),
            }),
        }),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: rw(arg),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: rw(body),
            index: index.clone(),
            dim: *dim,
        }),
        // log(det A) → log(det X)/2 — cleaner than going through the
        // power rule on (det X)^{1/2}.
        ScalarExpr::Log(inner) => {
            if let ScalarExpr::Det(t) = &**inner {
                if t == a {
                    return Rc::new(ScalarExpr::Div(
                        Rc::new(ScalarExpr::Log(Rc::new(ScalarExpr::Det(x.clone())))),
                        Rc::new(ScalarExpr::Num(2.0)),
                    ));
                }
            }
            Rc::new(ScalarExpr::Log(rw(inner)))
        }
        ScalarExpr::Det(t) => {
            if t == a {
                return Rc::new(ScalarExpr::Pow(
                    Rc::new(ScalarExpr::Det(x.clone())),
                    Rc::new(ScalarExpr::Num(0.5)),
                ));
            }
            Rc::new(ScalarExpr::Det(rwt(t)))
        }
        ScalarExpr::Tr(t) => Rc::new(ScalarExpr::Tr(rwt(t))),
        ScalarExpr::Ddot(p, q) => Rc::new(ScalarExpr::Ddot(rwt(p), rwt(q))),
        ScalarExpr::Add(p, q) => Rc::new(ScalarExpr::Add(rw(p), rw(q))),
        ScalarExpr::Sub(p, q) => Rc::new(ScalarExpr::Sub(rw(p), rw(q))),
        ScalarExpr::Mul(p, q) => Rc::new(ScalarExpr::Mul(rw(p), rw(q))),
        ScalarExpr::Div(p, q) => Rc::new(ScalarExpr::Div(rw(p), rw(q))),
        ScalarExpr::Pow(p, q) => {
            if let ScalarExpr::Det(t) = &**p {
                if t == a {
                    if let Some(exp) = half_rational_expr(q) {
                        return Rc::new(ScalarExpr::Pow(Rc::new(ScalarExpr::Det(x.clone())), exp));
                    }
                    if let Some(n) = numeric_value(q) {
                        return Rc::new(ScalarExpr::Pow(
                            Rc::new(ScalarExpr::Det(x.clone())),
                            Rc::new(ScalarExpr::Num(n / 2.0)),
                        ));
                    }
                }
            }
            Rc::new(ScalarExpr::Pow(rw(p), rw(q)))
        }
        ScalarExpr::Neg(p) => Rc::new(ScalarExpr::Neg(rw(p))),
    }
}

fn rewrite_tensor_for_compound(
    t: &Rc<TensorExpr>,
    x: &Rc<TensorExpr>,
    a: &Rc<TensorExpr>,
) -> Rc<TensorExpr> {
    if t == x {
        return t.clone(); // occurrence of X: opaque
    }
    let rw = |e: &Rc<ScalarExpr>| rewrite_scalar_for_compound(e, x, a);
    let rwt = |e: &Rc<TensorExpr>| rewrite_tensor_for_compound(e, x, a);
    match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => t.clone(),
        TensorExpr::SetElem {
            latex,
            order,
            dim,
            index,
            set_dim,
            base,
        } => Rc::new(TensorExpr::SetElem {
            latex: latex.clone(),
            order: *order,
            dim: *dim,
            index: index.clone(),
            set_dim: *set_dim,
            base: base.as_ref().map(&rwt),
        }),
        TensorExpr::GenStrain { base, scale, latex } => Rc::new(TensorExpr::GenStrain {
            base: rwt(base),
            scale: scale.clone(),
            latex: latex.clone(),
        }),
        TensorExpr::QTensor { strain } => Rc::new(TensorExpr::QTensor {
            strain: rwt(strain),
        }),
        TensorExpr::DdotTQ { second, fourth } => Rc::new(TensorExpr::DdotTQ {
            second: rwt(second),
            fourth: rwt(fourth),
        }),
        TensorExpr::Transpose(p) => Rc::new(TensorExpr::Transpose(rwt(p))),
        TensorExpr::Inverse(p) => Rc::new(TensorExpr::Inverse(rwt(p))),
        TensorExpr::InverseTranspose(p) => Rc::new(TensorExpr::InverseTranspose(rwt(p))),
        TensorExpr::Diff {
            num,
            den,
            num_label,
        } => Rc::new(TensorExpr::Diff {
            num: rwt(num),
            den: rwt(den),
            num_label: num_label.clone(),
        }),
        TensorExpr::MatMul(p, q) => Rc::new(TensorExpr::MatMul(rwt(p), rwt(q))),
        TensorExpr::Add(p, q) => Rc::new(TensorExpr::Add(rwt(p), rwt(q))),
        TensorExpr::Sub(p, q) => Rc::new(TensorExpr::Sub(rwt(p), rwt(q))),
        TensorExpr::Outer(p, q) => Rc::new(TensorExpr::Outer(rwt(p), rwt(q))),
        TensorExpr::BoxTimes(p, q) => Rc::new(TensorExpr::BoxTimes(rwt(p), rwt(q))),
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => Rc::new(TensorExpr::SumIdx {
            index: index.clone(),
            range: *range,
            exclude: exclude.clone(),
            body: rwt(body),
        }),
        TensorExpr::Spectral { base, base_latex } => Rc::new(TensorExpr::Spectral {
            base: rwt(base),
            base_latex: base_latex.clone(),
        }),
        TensorExpr::SpectralFn {
            func,
            base,
            base_latex,
        } => Rc::new(TensorExpr::SpectralFn {
            func: func.clone(),
            base: rwt(base),
            base_latex: base_latex.clone(),
        }),
        TensorExpr::ScalarMul(s, p) => Rc::new(TensorExpr::ScalarMul(rw(s), rwt(p))),
        TensorExpr::Neg(p) => Rc::new(TensorExpr::Neg(rwt(p))),
    }
}

fn replace_scalar(
    s: &Rc<ScalarExpr>,
    target: &Rc<TensorExpr>,
    replacement: &Rc<TensorExpr>,
) -> Rc<ScalarExpr> {
    let rs = |e: &Rc<ScalarExpr>| replace_scalar(e, target, replacement);
    let rt = |t: &Rc<TensorExpr>| replace_tensor(t, target, replacement);
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => s.clone(),
        ScalarExpr::SetElem {
            latex,
            index,
            set_dim,
            eig,
        } => Rc::new(ScalarExpr::SetElem {
            latex: latex.clone(),
            index: index.clone(),
            set_dim: *set_dim,
            eig: eig.as_ref().map(|l| crate::symbolic::EigLink {
                base: rt(&l.base),
                vec_latex: l.vec_latex.clone(),
            }),
        }),
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(rs(a), rs(b))),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(rs(a), rs(b))),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(rs(a), rs(b))),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(rs(a), rs(b))),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(rs(a), rs(b))),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(rs(a))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(rs(a))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: rs(arg),
        }),
        ScalarExpr::Det(t) => Rc::new(ScalarExpr::Det(rt(t))),
        ScalarExpr::Tr(t) => Rc::new(ScalarExpr::Tr(rt(t))),
        ScalarExpr::Ddot(a, b) => Rc::new(ScalarExpr::Ddot(rt(a), rt(b))),
        ScalarExpr::Eig {
            base,
            symbol,
            index,
        } => Rc::new(ScalarExpr::Eig {
            base: rt(base),
            symbol: symbol.clone(),
            index: index.clone(),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: rs(body),
            index: index.clone(),
            dim: *dim,
        }),
    }
}

fn replace_tensor(
    t: &Rc<TensorExpr>,
    target: &Rc<TensorExpr>,
    replacement: &Rc<TensorExpr>,
) -> Rc<TensorExpr> {
    if t == target {
        return replacement.clone();
    }
    let rs = |s: &Rc<ScalarExpr>| replace_scalar(s, target, replacement);
    let rt = |e: &Rc<TensorExpr>| replace_tensor(e, target, replacement);
    match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => t.clone(),
        TensorExpr::SetElem {
            latex,
            order,
            dim,
            index,
            set_dim,
            base,
        } => Rc::new(TensorExpr::SetElem {
            latex: latex.clone(),
            order: *order,
            dim: *dim,
            index: index.clone(),
            set_dim: *set_dim,
            base: base.as_ref().map(&rt),
        }),
        TensorExpr::Transpose(a) => Rc::new(TensorExpr::Transpose(rt(a))),
        TensorExpr::Inverse(a) => Rc::new(TensorExpr::Inverse(rt(a))),
        TensorExpr::InverseTranspose(a) => Rc::new(TensorExpr::InverseTranspose(rt(a))),
        TensorExpr::Diff {
            num,
            den,
            num_label,
        } => Rc::new(TensorExpr::Diff {
            num: rt(num),
            den: rt(den),
            num_label: num_label.clone(),
        }),
        TensorExpr::MatMul(a, b) => Rc::new(TensorExpr::MatMul(rt(a), rt(b))),
        TensorExpr::Add(a, b) => Rc::new(TensorExpr::Add(rt(a), rt(b))),
        TensorExpr::Sub(a, b) => Rc::new(TensorExpr::Sub(rt(a), rt(b))),
        TensorExpr::Outer(a, b) => Rc::new(TensorExpr::Outer(rt(a), rt(b))),
        TensorExpr::BoxTimes(a, b) => Rc::new(TensorExpr::BoxTimes(rt(a), rt(b))),
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => Rc::new(TensorExpr::SumIdx {
            index: index.clone(),
            range: *range,
            exclude: exclude.clone(),
            body: rt(body),
        }),
        TensorExpr::Spectral { base, base_latex } => Rc::new(TensorExpr::Spectral {
            base: rt(base),
            base_latex: base_latex.clone(),
        }),
        TensorExpr::SpectralFn {
            func,
            base,
            base_latex,
        } => Rc::new(TensorExpr::SpectralFn {
            func: func.clone(),
            base: rt(base),
            base_latex: base_latex.clone(),
        }),
        TensorExpr::GenStrain { base, scale, latex } => Rc::new(TensorExpr::GenStrain {
            base: rt(base),
            scale: scale.clone(),
            latex: latex.clone(),
        }),
        TensorExpr::QTensor { strain } => Rc::new(TensorExpr::QTensor { strain: rt(strain) }),
        TensorExpr::DdotTQ { second, fourth } => Rc::new(TensorExpr::DdotTQ {
            second: rt(second),
            fourth: rt(fourth),
        }),
        TensorExpr::ScalarMul(s, a) => Rc::new(TensorExpr::ScalarMul(rs(s), rt(a))),
        TensorExpr::Neg(a) => Rc::new(TensorExpr::Neg(rt(a))),
    }
}

fn d_tensor(t: &Rc<TensorExpr>, x: &Rc<TensorExpr>) -> Result<Option<Rc<TensorExpr>>, Error> {
    if !t_contains(t, x) {
        return Ok(None);
    }
    if t == x {
        return Ok(Some(Rc::new(TensorExpr::Identity4 { dim: x.dim() })));
    }
    match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => Ok(None),
        TensorExpr::SetElem { .. } => Err(Error::msg(
            "the derivative of eigenvectors is not supported yet; write the \
             energy in terms of the eigenvalues",
        )),
        TensorExpr::Add(a, b) => Ok(tadd(d_tensor(a, x)?, d_tensor(b, x)?)),
        TensorExpr::Sub(a, b) => Ok(tsub(d_tensor(a, x)?, d_tensor(b, x)?)),
        TensorExpr::Neg(a) => Ok(d_tensor(a, x)?.map(tneg)),
        TensorExpr::ScalarMul(s, a) => {
            let da = d_tensor(a, x)?.map(|dt| smul(s.clone(), dt));
            let ds = if s_contains(s, x) {
                Some(outer_scaled(a.clone(), diff_scalar_by_tensor(s, x)?)?)
            } else {
                None
            };
            Ok(tadd(da, ds))
        }
        // ∂(Σ_a body)/∂X = Σ_a ∂body/∂X
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => Ok(d_tensor(body, x)?.map(|d| {
            Rc::new(TensorExpr::SumIdx {
                index: index.clone(),
                range: *range,
                exclude: exclude.clone(),
                body: d,
            })
        })),
        // Aᵀ = A for symmetric A: differentiate through the transpose.
        TensorExpr::Transpose(a) if a.is_symmetric() => d_tensor(a, x),
        // ∂(X⁻¹)_{ij}/∂X_{mn} = −(X⁻¹)_{im}(X⁻¹)_{nj}, i.e. −X⁻¹ ⊠ X⁻ᵀ
        // (matching the component rule in [`crate::indices`]). For symmetric
        // X the second factor is displayed as X⁻¹.
        TensorExpr::Inverse(a) if a == x => Ok(Some(tneg(Rc::new(TensorExpr::box_times(
            Rc::new(TensorExpr::Inverse(x.clone())),
            inverse_factor(x),
        )?)))),
        // X⁻ᵀ = X⁻¹ for symmetric X.
        TensorExpr::InverseTranspose(a) if a == x && x.is_symmetric() => {
            Ok(Some(tneg(Rc::new(TensorExpr::box_times(
                Rc::new(TensorExpr::Inverse(x.clone())),
                Rc::new(TensorExpr::Inverse(x.clone())),
            )?))))
        }
        TensorExpr::Transpose(_)
        | TensorExpr::Inverse(_)
        | TensorExpr::InverseTranspose(_)
        | TensorExpr::Diff { .. }
        | TensorExpr::MatMul(..)
        | TensorExpr::Outer(..)
        | TensorExpr::BoxTimes(..)
        | TensorExpr::Spectral { .. }
        | TensorExpr::SpectralFn { .. }
        | TensorExpr::GenStrain { .. }
        | TensorExpr::QTensor { .. }
        | TensorExpr::DdotTQ { .. } => Err(Error::msg(
            "this tensor-by-tensor derivative form is not supported yet",
        )),
    }
}

/// The second `⊠` factor of `∂(X⁻¹)/∂X`: `X⁻ᵀ`, displayed as `X⁻¹` when
/// `X` is provably symmetric.
fn inverse_factor(x: &Rc<TensorExpr>) -> Rc<TensorExpr> {
    if x.is_symmetric() {
        Rc::new(TensorExpr::Inverse(x.clone()))
    } else {
        Rc::new(TensorExpr::InverseTranspose(x.clone()))
    }
}

fn outer_scaled(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<Rc<TensorExpr>, Error> {
    if let TensorExpr::ScalarMul(s, inner) = &*a {
        return Ok(smul(
            s.clone(),
            Rc::new(TensorExpr::outer(inner.clone(), b)?),
        ));
    }
    if let TensorExpr::ScalarMul(s, inner) = &*b {
        return Ok(smul(
            s.clone(),
            Rc::new(TensorExpr::outer(a, inner.clone())?),
        ));
    }
    Ok(Rc::new(TensorExpr::outer(a, b)?))
}

/// `f'(arg)` for the named univariate functions.
fn func_derivative(name: &str, arg: &Rc<ScalarExpr>) -> Result<Rc<ScalarExpr>, Error> {
    let f = |n: &str| {
        Rc::new(ScalarExpr::Func {
            name: n.to_string(),
            arg: arg.clone(),
        })
    };
    Ok(match name {
        "exp" => f("exp"),
        "sinh" => f("cosh"),
        "cosh" => f("sinh"),
        "sin" => f("cos"),
        // d/dx √x = 1/(2√x)
        "sqrt" => Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Num(1.0)),
            Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(2.0)), f("sqrt"))),
        )),
        other => {
            return Err(Error::msg(format!(
                "the derivative of `{other}` is not in the rule table yet"
            )))
        }
    })
}

/// Find the (unique) generalized strain `E(x)` inside `s`, if any.
/// Multiple distinct strains of the same tensor are rejected.
fn find_gen_strain_of(s: &ScalarExpr, x: &Rc<TensorExpr>) -> Result<Option<Rc<TensorExpr>>, Error> {
    let mut found: Vec<Rc<TensorExpr>> = Vec::new();
    collect_strains_scalar(s, x, &mut found);
    found.dedup();
    match found.len() {
        0 => Ok(None),
        1 => Ok(Some(found.remove(0))),
        _ => {
            if found.iter().all(|g| g == &found[0]) {
                Ok(Some(found.remove(0)))
            } else {
                Err(Error::msg(
                    "multiple distinct generalized strains of the same tensor in one \
                     expression are not supported yet",
                ))
            }
        }
    }
}

fn collect_strains_scalar(s: &ScalarExpr, x: &Rc<TensorExpr>, out: &mut Vec<Rc<TensorExpr>>) {
    match s {
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Eig { .. }
        | ScalarExpr::SetElem { .. } => {}
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            collect_strains_scalar(a, x, out);
            collect_strains_scalar(b, x, out);
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            collect_strains_scalar(a, x, out)
        }
        ScalarExpr::SpecSum { body, .. } => collect_strains_scalar(body, x, out),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => collect_strains_tensor(t, x, out),
        ScalarExpr::Ddot(a, b) => {
            collect_strains_tensor(a, x, out);
            collect_strains_tensor(b, x, out);
        }
    }
}

fn collect_strains_tensor(t: &Rc<TensorExpr>, x: &Rc<TensorExpr>, out: &mut Vec<Rc<TensorExpr>>) {
    if let TensorExpr::GenStrain { base, .. } = &**t {
        if base == x {
            out.push(t.clone());
            return;
        }
    }
    match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => {}
        TensorExpr::SumIdx { body, .. } => collect_strains_tensor(body, x, out),
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => collect_strains_tensor(a, x, out),
        TensorExpr::Diff { num, den, .. } => {
            collect_strains_tensor(num, x, out);
            collect_strains_tensor(den, x, out);
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            collect_strains_tensor(a, x, out);
            collect_strains_tensor(b, x, out);
        }
        TensorExpr::Spectral { base, .. }
        | TensorExpr::SpectralFn { base, .. }
        | TensorExpr::GenStrain { base, .. } => collect_strains_tensor(base, x, out),
        TensorExpr::QTensor { strain } => collect_strains_tensor(strain, x, out),
        TensorExpr::DdotTQ { second, fourth } => {
            collect_strains_tensor(second, x, out);
            collect_strains_tensor(fourth, x, out);
        }
        TensorExpr::ScalarMul(s, a) => {
            collect_strains_scalar(s, x, out);
            collect_strains_tensor(a, x, out);
        }
    }
}

/// `∂s/∂a` for a declared scalar symbol `a` (matched by name).
/// Returns `None`-as-zero folded into `Num(0)`.
pub fn diff_scalar_by_scalar(s: &Rc<ScalarExpr>, name: &str) -> Result<Rc<ScalarExpr>, Error> {
    Ok(d_scalar_scalar(s, name)?.unwrap_or_else(|| Rc::new(ScalarExpr::Num(0.0))))
}

/// `None` means the derivative is identically zero.
fn d_scalar_scalar(s: &Rc<ScalarExpr>, name: &str) -> Result<Option<Rc<ScalarExpr>>, Error> {
    use crate::symbolic::{fold_add, fold_sub};
    let one = || Rc::new(ScalarExpr::Num(1.0));
    match &**s {
        ScalarExpr::Sym { name: n, .. } => Ok((n == name).then(one)),
        ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => Ok(None),
        ScalarExpr::Add(a, b) => Ok(
            match (d_scalar_scalar(a, name)?, d_scalar_scalar(b, name)?) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(x), Some(y)) => Some(fold_add(x, y)),
            },
        ),
        ScalarExpr::Sub(a, b) => Ok(
            match (d_scalar_scalar(a, name)?, d_scalar_scalar(b, name)?) {
                (None, None) => None,
                (Some(x), None) => Some(x),
                (None, Some(y)) => Some(with_coeff(-1.0, Some(y))),
                (Some(x), Some(y)) => Some(fold_sub(x, y)),
            },
        ),
        ScalarExpr::Neg(a) => Ok(d_scalar_scalar(a, name)?.map(|d| with_coeff(-1.0, Some(d)))),
        ScalarExpr::Mul(a, b) => {
            let da = d_scalar_scalar(a, name)?.map(|d| fold_mul(&d, b));
            let db = d_scalar_scalar(b, name)?.map(|d| fold_mul(a, &d));
            Ok(match (da, db) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(x), Some(y)) => Some(crate::symbolic::fold_add(x, y)),
            })
        }
        ScalarExpr::Div(a, b) => {
            // d(a/b) = da/b − a db / b²
            let da = d_scalar_scalar(a, name)?.map(|d| Rc::new(ScalarExpr::Div(d, b.clone())));
            let db = d_scalar_scalar(b, name)?.map(|d| {
                with_coeff(
                    -1.0,
                    Some(Rc::new(ScalarExpr::Div(
                        fold_mul(a, &d),
                        Rc::new(ScalarExpr::Pow(b.clone(), Rc::new(ScalarExpr::Num(2.0)))),
                    ))),
                )
            });
            Ok(match (da, db) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(x), Some(y)) => Some(crate::symbolic::fold_add(x, y)),
            })
        }
        ScalarExpr::Pow(base, exp) => {
            let db = d_scalar_scalar(exp, name)?;
            if db.is_some() {
                return Err(Error::msg(
                    "d(a^b) with the differentiation variable in the exponent \
                     is not supported yet",
                ));
            }
            let Some(da) = d_scalar_scalar(base, name)? else {
                return Ok(None);
            };
            // d(a^n) = n a^{n-1} da for numeric n, else b a^{b-1} da
            let coeff = match &**exp {
                ScalarExpr::Num(n) => with_coeff(*n, Some(fold_pow(base.clone(), n - 1.0))),
                _ => fold_mul(
                    exp,
                    &Rc::new(ScalarExpr::Pow(
                        base.clone(),
                        Rc::new(ScalarExpr::Sub(exp.clone(), Rc::new(ScalarExpr::Num(1.0)))),
                    )),
                ),
            };
            Ok(Some(fold_mul(&coeff, &da)))
        }
        ScalarExpr::Log(a) => {
            let Some(da) = d_scalar_scalar(a, name)? else {
                return Ok(None);
            };
            Ok(Some(Rc::new(ScalarExpr::Div(da, a.clone()))))
        }
        ScalarExpr::Func { name: f, arg } => {
            let Some(da) = d_scalar_scalar(arg, name)? else {
                return Ok(None);
            };
            Ok(Some(fold_mul(&func_derivative(f, arg)?, &da)))
        }
        ScalarExpr::Eig { .. } => Ok(None),
        ScalarExpr::SpecSum { body, index, dim } => Ok(d_scalar_scalar(body, name)?.map(|d| {
            Rc::new(ScalarExpr::SpecSum {
                body: d,
                index: index.clone(),
                dim: *dim,
            })
        })),
        // det/tr/ddot of tensors: zero unless the scalar hides inside a
        // tensor (ScalarMul coefficient), which is not supported yet.
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => {
            if tensor_mentions_scalar(t, name) {
                Err(Error::msg(
                    "differentiating det/tr of a tensor whose coefficients contain \
                     the scalar variable is not supported yet",
                ))
            } else {
                Ok(None)
            }
        }
        ScalarExpr::Ddot(a, b) => {
            if tensor_mentions_scalar(a, name) || tensor_mentions_scalar(b, name) {
                Err(Error::msg(
                    "differentiating a double contraction whose factors contain \
                     the scalar variable is not supported yet",
                ))
            } else {
                Ok(None)
            }
        }
    }
}

/// Does a scalar symbol (by name) occur anywhere inside a tensor expression?
fn tensor_mentions_scalar(t: &TensorExpr, name: &str) -> bool {
    match t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => false,
        TensorExpr::SetElem { base, .. } => base
            .as_ref()
            .is_some_and(|b| tensor_mentions_scalar(b, name)),
        TensorExpr::SumIdx { body, .. } => tensor_mentions_scalar(body, name),
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => tensor_mentions_scalar(a, name),
        TensorExpr::Diff { num, den, .. } => {
            tensor_mentions_scalar(num, name) || tensor_mentions_scalar(den, name)
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            tensor_mentions_scalar(a, name) || tensor_mentions_scalar(b, name)
        }
        TensorExpr::Spectral { base, .. } | TensorExpr::SpectralFn { base, .. } => {
            tensor_mentions_scalar(base, name)
        }
        TensorExpr::GenStrain { base, scale, .. } => {
            tensor_mentions_scalar(base, name) || scale_mentions_scalar(scale, name)
        }
        TensorExpr::QTensor { strain } => tensor_mentions_scalar(strain, name),
        TensorExpr::DdotTQ { second, fourth } => {
            tensor_mentions_scalar(second, name) || tensor_mentions_scalar(fourth, name)
        }
        TensorExpr::ScalarMul(s, a) => {
            scalar_mentions_scalar(s, name) || tensor_mentions_scalar(a, name)
        }
    }
}

fn scale_mentions_scalar(scale: &crate::tensor::Scale, name: &str) -> bool {
    match scale {
        crate::tensor::Scale::SethHill { m } => scalar_mentions_scalar(m, name),
        crate::tensor::Scale::CR { m, n } => {
            scalar_mentions_scalar(m, name) || scalar_mentions_scalar(n, name)
        }
        crate::tensor::Scale::Hencky => false,
        // The scale's own bound variable is not a free mention.
        crate::tensor::Scale::Custom { var, body, .. } => {
            name != var && scalar_mentions_scalar(body, name)
        }
    }
}

pub(crate) fn scalar_mentions_scalar(s: &ScalarExpr, name: &str) -> bool {
    match s {
        ScalarExpr::Sym { name: n, .. } => n == name,
        ScalarExpr::Num(_) | ScalarExpr::Eig { .. } => false,
        ScalarExpr::SetElem { eig, .. } => eig
            .as_ref()
            .is_some_and(|l| tensor_mentions_scalar(&l.base, name)),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            scalar_mentions_scalar(a, name) || scalar_mentions_scalar(b, name)
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            scalar_mentions_scalar(a, name)
        }
        ScalarExpr::SpecSum { body, .. } => scalar_mentions_scalar(body, name),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => tensor_mentions_scalar(t, name),
        ScalarExpr::Ddot(a, b) => {
            tensor_mentions_scalar(a, name) || tensor_mentions_scalar(b, name)
        }
    }
}

// ---- independence guard for compound denominators ---------------------------

/// Collect the names of tensor variables appearing in `x`.
fn tensor_var_names(t: &TensorExpr, out: &mut Vec<String>) {
    match t {
        TensorExpr::SetElem { .. } => {}
        TensorExpr::SumIdx { body, .. } => tensor_var_names(body, out),
        TensorExpr::Var { name, .. } => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        TensorExpr::Identity4 { .. } => {}
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => tensor_var_names(a, out),
        TensorExpr::Diff { num, den, .. } => {
            tensor_var_names(num, out);
            tensor_var_names(den, out);
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            tensor_var_names(a, out);
            tensor_var_names(b, out);
        }
        TensorExpr::Spectral { base, .. }
        | TensorExpr::SpectralFn { base, .. }
        | TensorExpr::GenStrain { base, .. } => tensor_var_names(base, out),
        TensorExpr::QTensor { strain } => tensor_var_names(strain, out),
        TensorExpr::DdotTQ { second, fourth } => {
            tensor_var_names(second, out);
            tensor_var_names(fourth, out);
        }
        TensorExpr::ScalarMul(_, a) => tensor_var_names(a, out),
    }
}

/// Reject `diff(s, X)` for compound `X` when `s` depends on `X`'s underlying
/// variables outside occurrences of `X` itself: differentiating would then
/// silently produce a wrong result (the hidden dependence would read as 0).
fn check_independence_scalar(s: &ScalarExpr, x: &Rc<TensorExpr>) -> Result<(), Error> {
    let mut vars = Vec::new();
    tensor_var_names(x, &mut vars);
    walk_scalar(s, x, &vars)
}

fn check_independence_tensor(t: &TensorExpr, x: &Rc<TensorExpr>) -> Result<(), Error> {
    let mut vars = Vec::new();
    tensor_var_names(x, &mut vars);
    walk_tensor(t, x, &vars)
}

fn walk_scalar(s: &ScalarExpr, x: &Rc<TensorExpr>, vars: &[String]) -> Result<(), Error> {
    match s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::Eig { .. } => Ok(()),
        // λ_a(base): an eigenvalue of X itself is an opaque occurrence of X;
        // any other base is checked for hidden dependence.
        ScalarExpr::SetElem { eig, .. } => match eig {
            Some(link) => walk_tensor(&link.base, x, vars),
            None => Ok(()),
        },
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            walk_scalar(a, x, vars)?;
            walk_scalar(b, x, vars)
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            walk_scalar(a, x, vars)
        }
        ScalarExpr::SpecSum { body, .. } => walk_scalar(body, x, vars),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => walk_tensor(t, x, vars),
        ScalarExpr::Ddot(a, b) => {
            walk_tensor(a, x, vars)?;
            walk_tensor(b, x, vars)
        }
    }
}

fn walk_tensor(t: &TensorExpr, x: &Rc<TensorExpr>, vars: &[String]) -> Result<(), Error> {
    if *t == **x {
        return Ok(()); // an occurrence of X itself: opaque, don't descend
    }
    match t {
        TensorExpr::Identity4 { .. } => Ok(()),
        TensorExpr::SetElem { base, .. } => match base {
            Some(b) => walk_tensor(b, x, vars),
            None => Ok(()),
        },
        TensorExpr::SumIdx { body, .. } => walk_tensor(body, x, vars),
        TensorExpr::Var { name, .. } => {
            if vars.contains(name) {
                Err(Error::msg(format!(
                    "cannot differentiate with respect to this compound expression: \
                     the expression also depends on `{name}` outside of it; \
                     rewrite the energy in terms of the new variable \
                     (e.g. use det(C) instead of det(F)) or differentiate \
                     with respect to `{name}` directly"
                )))
            } else {
                Ok(())
            }
        }
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => walk_tensor(a, x, vars),
        TensorExpr::Diff { num, den, .. } => {
            walk_tensor(num, x, vars)?;
            walk_tensor(den, x, vars)
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            walk_tensor(a, x, vars)?;
            walk_tensor(b, x, vars)
        }
        TensorExpr::Spectral { base, .. }
        | TensorExpr::SpectralFn { base, .. }
        | TensorExpr::GenStrain { base, .. } => walk_tensor(base, x, vars),
        TensorExpr::QTensor { strain } => walk_tensor(strain, x, vars),
        TensorExpr::DdotTQ { second, fourth } => {
            walk_tensor(second, x, vars)?;
            walk_tensor(fourth, x, vars)
        }
        TensorExpr::ScalarMul(s, a) => {
            walk_scalar(s, x, vars)?;
            walk_tensor(a, x, vars)
        }
    }
}

/// `None` means the derivative is identically zero.
fn d_scalar(s: &Rc<ScalarExpr>, x: &Rc<TensorExpr>) -> Result<Option<Rc<TensorExpr>>, Error> {
    if !s_contains(s, x) {
        return Ok(None);
    }
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => Ok(None),
        // ∂λ_a/∂X = N_a ⊗ N_a for an eigenvalue of X (exact for simple
        // eigenvalues; the standard spectral-calculus rule).
        ScalarExpr::SetElem {
            eig,
            index,
            set_dim,
            ..
        } => {
            let Some(link) = eig else { return Ok(None) };
            if link.base != *x {
                return Err(Error::msg(
                    "∂λ_a/∂X is only supported when X is the decomposed tensor \
                     itself (declare the eigenvalues with eigvals(X, ...))",
                ));
            }
            let n_a = || {
                Rc::new(TensorExpr::SetElem {
                    latex: link.vec_latex.clone(),
                    order: 1,
                    dim: x.dim(),
                    index: index.clone(),
                    set_dim: *set_dim,
                    base: Some(link.base.clone()),
                })
            };
            Ok(Some(Rc::new(TensorExpr::Outer(n_a(), n_a()))))
        }
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
            if s_contains(exp, x) {
                return Err(Error::msg(
                    "d(a^b)/dX with X-dependent exponent is not supported yet",
                ));
            }
            if let ScalarExpr::Det(t) = &**base {
                if t == x {
                    // d((det X)^q)/dX = q (det X)^q X^{-T}
                    let coeff = fold_mul(exp, &Rc::new(ScalarExpr::Pow(base.clone(), exp.clone())));
                    return Ok(Some(smul(
                        coeff,
                        Rc::new(TensorExpr::InverseTranspose(x.clone())),
                    )));
                }
            }
            // d(a^q) = q a^{q-1} da — exact for the positive bases (dets,
            // eigenvalues, stretches) that symbolic powers assume here.
            let coeff = match numeric_value(exp) {
                Some(n) => with_coeff(n, Some(fold_pow(base.clone(), n - 1.0))),
                None => fold_mul(
                    exp,
                    &Rc::new(ScalarExpr::Pow(
                        base.clone(),
                        Rc::new(ScalarExpr::Sub(exp.clone(), Rc::new(ScalarExpr::Num(1.0)))),
                    )),
                ),
            };
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
        // ∂(A : B)/∂X for symmetric X-matching: X:X → 2X, A:X → A (A indep).
        ScalarExpr::Ddot(a, b) => {
            let a_is_x = a == x;
            let b_is_x = b == x;
            let a_dep = t_contains(a, x);
            let b_dep = t_contains(b, x);
            match (a_is_x, b_is_x) {
                (true, true) => Ok(Some(smul(Rc::new(ScalarExpr::Num(2.0)), x.clone()))),
                (true, false) if !b_dep => Ok(Some(b.clone())),
                (false, true) if !a_dep => Ok(Some(a.clone())),
                (false, false) if !a_dep && !b_dep => Ok(None),
                _ => Err(Error::msg(
                    "∂(A:B)/∂X is only supported for X:X and A:X with X-independent A \
                     in this phase",
                )),
            }
        }
        ScalarExpr::Func { name, arg } => {
            let Some(da) = d_scalar(arg, x)? else {
                return Ok(None);
            };
            Ok(Some(smul(func_derivative(name, arg)?, da)))
        }
        // ∂(Σ_a body)/∂X = Σ_a ∂body/∂X — the scalar sum becomes a tensor sum.
        ScalarExpr::SpecSum { body, index, dim } => Ok(d_scalar(body, x)?.map(|t| {
            Rc::new(TensorExpr::SumIdx {
                index: index.clone(),
                range: *dim,
                exclude: None,
                body: t,
            })
        })),
        // Display-level eigenvalue symbols of opaque spectral objects.
        ScalarExpr::Eig { .. } => Err(Error::msg(
            "differentiating eigenvalue symbols is not supported; differentiate the \
             tensor expression they came from instead",
        )),
    }
}

/// `∂ tr(T)/∂X` using linearity of the trace.
fn d_trace(t: &Rc<TensorExpr>, x: &Rc<TensorExpr>) -> Result<Option<Rc<TensorExpr>>, Error> {
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
        TensorExpr::ScalarMul(s, inner) => {
            let d_inner = d_trace(inner, x)?.map(|t| smul(s.clone(), t));
            let d_coeff = if s_contains(s, x) {
                d_scalar(s, x)?.map(|t| smul(Rc::new(ScalarExpr::Tr(inner.clone())), t))
            } else {
                None
            };
            Ok(tadd(d_inner, d_coeff))
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
            row.push(format!("\\left[ {} \\right]_{{ij}}", render_terms(&inst)));
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
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => false,
        // An eigenvector element depends on its decomposed tensor.
        TensorExpr::SetElem { base, .. } => base.as_ref().is_some_and(|b| t_contains(b, x)),
        TensorExpr::SumIdx { body, .. } => t_contains(body, x),
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => t_contains(a, x),
        TensorExpr::Diff { num, den, .. } => t_contains(num, x) || t_contains(den, x),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => t_contains(a, x) || t_contains(b, x),
        TensorExpr::Spectral { base, .. }
        | TensorExpr::SpectralFn { base, .. }
        | TensorExpr::GenStrain { base, .. } => t_contains(base, x),
        TensorExpr::QTensor { strain } => t_contains(strain, x),
        TensorExpr::DdotTQ { second, fourth } => t_contains(second, x) || t_contains(fourth, x),
        TensorExpr::ScalarMul(s, a) => s_contains(s, x) || t_contains(a, x),
    }
}

pub fn s_contains(s: &ScalarExpr, x: &TensorExpr) -> bool {
    match s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => false,
        // An eigenvalue element depends on its decomposed tensor.
        ScalarExpr::SetElem { eig, .. } => eig.as_ref().is_some_and(|l| t_contains(&l.base, x)),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => s_contains(a, x) || s_contains(b, x),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            s_contains(a, x)
        }
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => t_contains(t, x),
        ScalarExpr::Ddot(a, b) => t_contains(a, x) || t_contains(b, x),
        ScalarExpr::Eig { base, .. } => t_contains(base, x),
        ScalarExpr::SpecSum { body, .. } => s_contains(body, x),
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

fn tadd(a: Option<Rc<TensorExpr>>, b: Option<Rc<TensorExpr>>) -> Option<Rc<TensorExpr>> {
    match (a, b) {
        (None, None) => None,
        (Some(t), None) | (None, Some(t)) => Some(t),
        (Some(x), Some(y)) => Some(Rc::new(TensorExpr::Add(x, y))),
    }
}

fn tsub(a: Option<Rc<TensorExpr>>, b: Option<Rc<TensorExpr>>) -> Option<Rc<TensorExpr>> {
    match (a, b) {
        (None, None) => None,
        (Some(t), None) => Some(t),
        (None, Some(t)) => Some(tneg(t)),
        (Some(x), Some(y)) => Some(Rc::new(TensorExpr::Sub(x, y))),
    }
}

/// Like [`tsub`] but the second operand is already negated (used by the
/// quotient rule where the db term carries its own minus sign).
fn tsub_opt(a: Option<Rc<TensorExpr>>, neg_b: Option<Rc<TensorExpr>>) -> Option<Rc<TensorExpr>> {
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

fn zero_fourth_like(x: &TensorExpr) -> Rc<TensorExpr> {
    smul(
        Rc::new(ScalarExpr::Num(0.0)),
        Rc::new(TensorExpr::Identity4 { dim: x.dim() }),
    )
}

fn numeric_value(s: &ScalarExpr) -> Option<f64> {
    match s {
        ScalarExpr::Num(n) => Some(*n),
        ScalarExpr::Add(a, b) => Some(numeric_value(a)? + numeric_value(b)?),
        ScalarExpr::Sub(a, b) => Some(numeric_value(a)? - numeric_value(b)?),
        ScalarExpr::Mul(a, b) => Some(numeric_value(a)? * numeric_value(b)?),
        ScalarExpr::Div(a, b) => Some(numeric_value(a)? / numeric_value(b)?),
        ScalarExpr::Pow(a, b) => Some(numeric_value(a)?.powf(numeric_value(b)?)),
        ScalarExpr::Neg(a) => Some(-numeric_value(a)?),
        _ => None,
    }
}

fn half_rational_expr(s: &ScalarExpr) -> Option<Rc<ScalarExpr>> {
    let (num, den) = rational_value(s)?;
    Some(rational_expr(num, den * 2))
}

fn rational_value(s: &ScalarExpr) -> Option<(i64, i64)> {
    match s {
        ScalarExpr::Num(n) if n.fract() == 0.0 && n.abs() < 1e12 => Some((*n as i64, 1)),
        ScalarExpr::Add(a, b) => {
            let (an, ad) = rational_value(a)?;
            let (bn, bd) = rational_value(b)?;
            Some(reduce(an * bd + bn * ad, ad * bd))
        }
        ScalarExpr::Sub(a, b) => {
            let (an, ad) = rational_value(a)?;
            let (bn, bd) = rational_value(b)?;
            Some(reduce(an * bd - bn * ad, ad * bd))
        }
        ScalarExpr::Mul(a, b) => {
            let (an, ad) = rational_value(a)?;
            let (bn, bd) = rational_value(b)?;
            Some(reduce(an * bn, ad * bd))
        }
        ScalarExpr::Div(a, b) => {
            let (an, ad) = rational_value(a)?;
            let (bn, bd) = rational_value(b)?;
            if bn == 0 {
                None
            } else {
                Some(reduce(an * bd, ad * bn))
            }
        }
        ScalarExpr::Neg(a) => {
            let (an, ad) = rational_value(a)?;
            Some((-an, ad))
        }
        _ => None,
    }
}

fn rational_expr(num: i64, den: i64) -> Rc<ScalarExpr> {
    let (num, den) = reduce(num, den);
    if den == 1 {
        Rc::new(ScalarExpr::Num(num as f64))
    } else if num < 0 {
        Rc::new(ScalarExpr::Neg(Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Num((-num) as f64)),
            Rc::new(ScalarExpr::Num(den as f64)),
        ))))
    } else {
        Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Num(num as f64)),
            Rc::new(ScalarExpr::Num(den as f64)),
        ))
    }
}

fn reduce(num: i64, den: i64) -> (i64, i64) {
    let sign = if den < 0 { -1 } else { 1 };
    let num = num * sign;
    let den = den.abs();
    let g = gcd(num.abs(), den);
    (num / g, den / g)
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}
