//! Definition substitution for display.
//!
//! When the user writes `C = F.T * F` and later displays an expression
//! containing the subtree `FᵀF`, the display should read `\bm C`, not the
//! expanded form. This module substitutes registered definitions (most
//! recent first, so derived definitions like `I1 = tr(C)` win over their
//! ingredients) into a value *at display time only* — internal values stay
//! fully expanded so differentiation keeps working.

use crate::interpreter::Value;
use crate::symbolic::ScalarExpr;
use crate::tensor::{TensorExpr, TensorProperties};
use std::rc::Rc;

/// One substitutable definition: replace structural occurrences of `value`
/// by a symbolic leaf labeled `latex`.
pub struct Def {
    pub latex: String,
    pub value: Value,
}

/// Substitute all definitions into `v`. `defs` is in insertion order; they
/// are applied most-recent-first so derived definitions like `I1 = tr(C)`
/// win over their ingredients. Definitions whose value equals the whole
/// expression are skipped (so `display(C)` still shows `C = FᵀF`).
pub fn substitute(v: &Value, defs: &[Def]) -> Value {
    let mut cur = v.clone();
    for def in defs.iter().rev() {
        // Skip self: substituting the whole expression would display `C = C`.
        let whole = match (&cur, &def.value) {
            (Value::Tensor(t), Value::Tensor(d)) => t == d,
            (Value::Scalar(s), Value::Scalar(d)) => s == d,
            _ => false,
        };
        if whole {
            continue;
        }
        cur = match &def.value {
            Value::Tensor(target) => {
                let replacement = Rc::new(TensorExpr::Var {
                    name: def.latex.clone(),
                    latex: def.latex.clone(),
                    order: target.order(),
                    dim: target.dim(),
                    props: TensorProperties {
                        symmetric: target.is_symmetric(),
                        ..Default::default()
                    },
                });
                match &cur {
                    Value::Tensor(t) => Value::Tensor(subst_t(t, target, &replacement)),
                    Value::Scalar(s) => Value::Scalar(subst_s_tensor(s, target, &replacement)),
                }
            }
            Value::Scalar(target) => {
                let replacement = Rc::new(ScalarExpr::Sym {
                    name: def.latex.clone(),
                    latex: def.latex.clone(),
                });
                match &cur {
                    Value::Tensor(t) => Value::Tensor(subst_t_scalar(t, target, &replacement)),
                    Value::Scalar(s) => Value::Scalar(subst_s(s, target, &replacement)),
                }
            }
        };
    }
    substitute_derived_invariants(&cur, defs)
}

#[derive(Clone)]
struct DetAlias {
    tensor_latex: String,
    tensor_value: Rc<TensorExpr>,
    jacobian_latex: String,
}

fn substitute_derived_invariants(v: &Value, defs: &[Def]) -> Value {
    let aliases = det_aliases(defs);
    if aliases.is_empty() {
        return v.clone();
    }
    match v {
        Value::Tensor(t) => Value::Tensor(rewrite_derived_tensor(t, &aliases)),
        Value::Scalar(s) => Value::Scalar(rewrite_derived_scalar(s, &aliases)),
    }
}

fn det_aliases(defs: &[Def]) -> Vec<DetAlias> {
    let mut jacobians: Vec<(Rc<TensorExpr>, String)> = Vec::new();
    for def in defs {
        if let Value::Scalar(s) = &def.value {
            if let ScalarExpr::Det(base) = &**s {
                jacobians.push((base.clone(), def.latex.clone()));
            }
        }
    }

    let mut out = Vec::new();
    for def in defs {
        let Value::Tensor(t) = &def.value else {
            continue;
        };
        let Some(base) = right_or_left_cauchy_green_base(t) else {
            continue;
        };
        for (j_base, j_latex) in &jacobians {
            if base == *j_base {
                out.push(DetAlias {
                    tensor_latex: def.latex.clone(),
                    tensor_value: t.clone(),
                    jacobian_latex: j_latex.clone(),
                });
            }
        }
    }
    out
}

fn right_or_left_cauchy_green_base(t: &Rc<TensorExpr>) -> Option<Rc<TensorExpr>> {
    match &**t {
        TensorExpr::MatMul(a, b) => match (&**a, &**b) {
            (TensorExpr::Transpose(base), _) if base == b => Some(base.clone()),
            (_, TensorExpr::Transpose(base)) if a == base => Some(a.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn rewrite_derived_tensor(t: &Rc<TensorExpr>, aliases: &[DetAlias]) -> Rc<TensorExpr> {
    let rt = |x: &Rc<TensorExpr>| rewrite_derived_tensor(x, aliases);
    let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
    rebuild_tensor(t, &rt, &rs)
}

fn rewrite_derived_scalar(s: &Rc<ScalarExpr>, aliases: &[DetAlias]) -> Rc<ScalarExpr> {
    match &**s {
        ScalarExpr::Det(t) => {
            if let Some(alias) = det_target_alias(t, aliases) {
                return jacobian_pow(alias, Rc::new(ScalarExpr::Num(2.0)));
            }
            let rt = |x: &Rc<TensorExpr>| rewrite_derived_tensor(x, aliases);
            Rc::new(ScalarExpr::Det(rt(t)))
        }
        ScalarExpr::Pow(base, exp) => {
            if let ScalarExpr::Det(t) = &**base {
                if let Some(alias) = det_target_alias(t, aliases) {
                    return jacobian_pow(alias, double_exponent(exp));
                }
            }
            if let Some(alias) = jacobian_square_alias(base, aliases) {
                return jacobian_pow(alias, double_exponent(exp));
            }
            let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
            Rc::new(ScalarExpr::Pow(rs(base), rs(exp)))
        }
        ScalarExpr::Mul(a, b) => {
            if let Some(alias) = jacobian_pair_alias(a, b, aliases) {
                return jacobian_pow(alias, Rc::new(ScalarExpr::Num(2.0)));
            }
            let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
            Rc::new(ScalarExpr::Mul(rs(a), rs(b)))
        }
        ScalarExpr::Log(inner) => {
            if let ScalarExpr::Det(t) = &**inner {
                if let Some(alias) = det_target_alias(t, aliases) {
                    return Rc::new(ScalarExpr::Mul(
                        Rc::new(ScalarExpr::Num(2.0)),
                        Rc::new(ScalarExpr::Log(jacobian_sym(alias))),
                    ));
                }
            }
            let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
            Rc::new(ScalarExpr::Log(rs(inner)))
        }
        ScalarExpr::Func { name, arg } if name == "sqrt" => {
            if let ScalarExpr::Det(t) = &**arg {
                if let Some(alias) = det_target_alias(t, aliases) {
                    return jacobian_sym(alias);
                }
            }
            let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
            Rc::new(ScalarExpr::Func {
                name: name.clone(),
                arg: rs(arg),
            })
        }
        _ => {
            let rt = |x: &Rc<TensorExpr>| rewrite_derived_tensor(x, aliases);
            let rs = |x: &Rc<ScalarExpr>| rewrite_derived_scalar(x, aliases);
            rebuild_scalar(s, &rt, &rs)
        }
    }
}

fn det_target_alias<'a>(t: &Rc<TensorExpr>, aliases: &'a [DetAlias]) -> Option<&'a DetAlias> {
    aliases.iter().find(|alias| {
        if t == &alias.tensor_value {
            return true;
        }
        matches!(&**t, TensorExpr::Var { latex, .. } if latex == &alias.tensor_latex)
    })
}

fn jacobian_square_alias<'a>(s: &Rc<ScalarExpr>, aliases: &'a [DetAlias]) -> Option<&'a DetAlias> {
    match &**s {
        ScalarExpr::Mul(a, b) => jacobian_pair_alias(a, b, aliases),
        _ => None,
    }
}

fn jacobian_pair_alias<'a>(
    a: &Rc<ScalarExpr>,
    b: &Rc<ScalarExpr>,
    aliases: &'a [DetAlias],
) -> Option<&'a DetAlias> {
    aliases
        .iter()
        .find(|alias| is_jacobian_sym(a, alias) && is_jacobian_sym(b, alias))
}

fn is_jacobian_sym(s: &Rc<ScalarExpr>, alias: &DetAlias) -> bool {
    matches!(&**s, ScalarExpr::Sym { latex, .. } if latex == &alias.jacobian_latex)
}

fn jacobian_sym(alias: &DetAlias) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Sym {
        name: alias.jacobian_latex.clone(),
        latex: alias.jacobian_latex.clone(),
    })
}

fn jacobian_pow(alias: &DetAlias, exp: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    match &*exp {
        ScalarExpr::Num(n) if *n == 0.0 => Rc::new(ScalarExpr::Num(1.0)),
        ScalarExpr::Num(n) if *n == 1.0 => jacobian_sym(alias),
        _ => Rc::new(ScalarExpr::Pow(jacobian_sym(alias), exp)),
    }
}

fn double_exponent(exp: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    match &**exp {
        ScalarExpr::Num(n) => Rc::new(ScalarExpr::Num(2.0 * n)),
        ScalarExpr::Neg(inner) => Rc::new(ScalarExpr::Neg(double_exponent(inner))),
        ScalarExpr::Div(num, den) => {
            if let Some(n) = signed_number(num) {
                return match &**den {
                    ScalarExpr::Num(d) => numeric_fraction(2.0 * n, *d),
                    _ => Rc::new(ScalarExpr::Div(
                        Rc::new(ScalarExpr::Num(2.0 * n)),
                        den.clone(),
                    )),
                };
            }
            Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(2.0)), exp.clone()))
        }
        ScalarExpr::Mul(a, b) => match &**a {
            ScalarExpr::Num(n) => Rc::new(ScalarExpr::Mul(
                Rc::new(ScalarExpr::Num(2.0 * n)),
                b.clone(),
            )),
            _ => Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(2.0)), exp.clone())),
        },
        _ => Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(2.0)), exp.clone())),
    }
}

fn signed_number(expr: &Rc<ScalarExpr>) -> Option<f64> {
    match &**expr {
        ScalarExpr::Num(n) => Some(*n),
        ScalarExpr::Neg(inner) => signed_number(inner).map(|n| -n),
        _ => None,
    }
}

fn numeric_fraction(num: f64, den: f64) -> Rc<ScalarExpr> {
    if den != 0.0 {
        let quotient = num / den;
        if quotient.fract() == 0.0 {
            return Rc::new(ScalarExpr::Num(quotient));
        }
    }
    Rc::new(ScalarExpr::Div(
        Rc::new(ScalarExpr::Num(num)),
        Rc::new(ScalarExpr::Num(den)),
    ))
}

// ---- tensor-target substitution ---------------------------------------------

fn subst_t(t: &Rc<TensorExpr>, target: &Rc<TensorExpr>, rep: &Rc<TensorExpr>) -> Rc<TensorExpr> {
    if t == target {
        return rep.clone();
    }
    let rt = |x: &Rc<TensorExpr>| subst_t(x, target, rep);
    let rs = |x: &Rc<ScalarExpr>| subst_s_tensor(x, target, rep);
    rebuild_tensor(t, &rt, &rs)
}

fn subst_s_tensor(
    s: &Rc<ScalarExpr>,
    target: &Rc<TensorExpr>,
    rep: &Rc<TensorExpr>,
) -> Rc<ScalarExpr> {
    let rt = |x: &Rc<TensorExpr>| subst_t(x, target, rep);
    let rs = |x: &Rc<ScalarExpr>| subst_s_tensor(x, target, rep);
    rebuild_scalar(s, &rt, &rs)
}

// ---- scalar-target substitution ---------------------------------------------

fn subst_s(s: &Rc<ScalarExpr>, target: &Rc<ScalarExpr>, rep: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    if s == target {
        return rep.clone();
    }
    let rt = |x: &Rc<TensorExpr>| subst_t_scalar(x, target, rep);
    let rs = |x: &Rc<ScalarExpr>| subst_s(x, target, rep);
    rebuild_scalar(s, &rt, &rs)
}

fn subst_t_scalar(
    t: &Rc<TensorExpr>,
    target: &Rc<ScalarExpr>,
    rep: &Rc<ScalarExpr>,
) -> Rc<TensorExpr> {
    let rt = |x: &Rc<TensorExpr>| subst_t_scalar(x, target, rep);
    let rs = |x: &Rc<ScalarExpr>| subst_s(x, target, rep);
    rebuild_tensor(t, &rt, &rs)
}

// ---- structural rebuilders ----------------------------------------------------

fn rebuild_tensor(
    t: &Rc<TensorExpr>,
    rt: &impl Fn(&Rc<TensorExpr>) -> Rc<TensorExpr>,
    rs: &impl Fn(&Rc<ScalarExpr>) -> Rc<ScalarExpr>,
) -> Rc<TensorExpr> {
    match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => t.clone(),
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

fn rebuild_scalar(
    s: &Rc<ScalarExpr>,
    rt: &impl Fn(&Rc<TensorExpr>) -> Rc<TensorExpr>,
    rs: &impl Fn(&Rc<ScalarExpr>) -> Rc<ScalarExpr>,
) -> Rc<ScalarExpr> {
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => s.clone(),
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(rs(a), rs(b))),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(rs(a), rs(b))),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(rs(a), rs(b))),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(rs(a), rs(b))),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(rs(a), rs(b))),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(rs(a))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(rs(a))),
        ScalarExpr::Det(t) => Rc::new(ScalarExpr::Det(rt(t))),
        ScalarExpr::Tr(t) => Rc::new(ScalarExpr::Tr(rt(t))),
        ScalarExpr::Ddot(a, b) => Rc::new(ScalarExpr::Ddot(rt(a), rt(b))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: rs(arg),
        }),
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
