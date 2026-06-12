//! Abstract-index component engine.
//!
//! Used for component display of tensor-by-tensor derivatives: a
//! second-order tensor expression is expanded into a sum of terms over
//! abstract indices (`C_{ij} = Σ_k F_{ki} F_{kj}`), differentiated
//! component-wise (`∂F_{ab}/∂F_{mn} = δ_{am} δ_{bn}`), and Kronecker deltas
//! contracted against bound (summed) indices.

use crate::error::Error;
use crate::renderer::latex::scalar_to_latex;
use crate::symbolic::ScalarExpr;
use crate::tensor::TensorExpr;
use std::rc::Rc;

pub type Idx = String;

/// A single tensor component with abstract indices, e.g. `F_{mj}` or
/// `F^{-1}_{mj}` (`inv = true`).
#[derive(Debug, Clone, PartialEq)]
pub struct Comp {
    pub base: String,
    pub inv: bool,
    pub indices: Vec<Idx>,
}

/// One additive term: `coeff · scalars · deltas · comps`, summed over
/// `bound` indices.
#[derive(Debug, Clone)]
pub struct Term {
    pub coeff: f64,
    pub scalars: Vec<Rc<ScalarExpr>>,
    pub deltas: Vec<(Idx, Idx)>,
    pub comps: Vec<Comp>,
    pub bound: Vec<Idx>,
}

impl Term {
    fn one() -> Term {
        Term {
            coeff: 1.0,
            scalars: vec![],
            deltas: vec![],
            comps: vec![],
            bound: vec![],
        }
    }

    fn mul(mut self, other: &Term) -> Term {
        self.coeff *= other.coeff;
        self.scalars.extend(other.scalars.iter().cloned());
        self.deltas.extend(other.deltas.iter().cloned());
        self.comps.extend(other.comps.iter().cloned());
        self.bound.extend(other.bound.iter().cloned());
        self
    }
}

/// Strip a leading `\bm ` from a tensor's LaTeX name so components render
/// as `F_{ij}` rather than `\bm F_{ij}`.
pub fn component_base(latex: &str) -> &str {
    latex
        .trim()
        .strip_prefix("\\bm ")
        .or_else(|| latex.trim().strip_prefix("\\bm"))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| latex.trim())
}

fn fresh_bound(counter: &mut usize) -> Idx {
    const NAMES: [&str; 6] = ["k", "l", "p", "q", "r", "s"];
    let name = NAMES
        .get(*counter)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("k_{{{}}}", *counter));
    *counter += 1;
    name
}

/// Expand the `(i, j)` component of a second-order tensor expression into a
/// sum of abstract-index terms.
pub fn abstract_component(
    expr: &TensorExpr,
    i: &Idx,
    j: &Idx,
    counter: &mut usize,
) -> Result<Vec<Term>, Error> {
    match expr {
        TensorExpr::Filled { .. } => Err(Error::msg(
            "component-filled tensors have no abstract component form; use \
             mode=matrix",
        )),
        TensorExpr::Var { latex, props, .. } => {
            let mut t = Term::one();
            if props.identity {
                t.deltas.push((i.clone(), j.clone()));
            } else {
                t.comps.push(Comp {
                    base: component_base(latex).to_string(),
                    inv: false,
                    indices: vec![i.clone(), j.clone()],
                });
            }
            Ok(vec![t])
        }
        TensorExpr::Identity4 { .. } => Err(Error::msg(
            "component expansion of fourth-order identity tensors is not \
             supported in second-order component mode",
        )),
        TensorExpr::Transpose(t) => abstract_component(t, j, i, counter),
        // (F^{-1})_{ij} and (F^{-T})_{ij} = (F^{-1})_{ji}: opaque inverse
        // components of a declared variable.
        TensorExpr::Inverse(inner) | TensorExpr::InverseTranspose(inner) => {
            let TensorExpr::Var { latex, .. } = &**inner else {
                return Err(Error::msg(
                    "component expansion of the inverse of a compound expression \
                     is not supported in this phase",
                ));
            };
            let (a, b) = if matches!(expr, TensorExpr::Inverse(_)) {
                (i.clone(), j.clone())
            } else {
                (j.clone(), i.clone())
            };
            let mut t = Term::one();
            t.comps.push(Comp {
                base: component_base(latex).to_string(),
                inv: true,
                indices: vec![a, b],
            });
            Ok(vec![t])
        }
        TensorExpr::MatMul(a, b) => {
            let k = fresh_bound(counter);
            let lhs = abstract_component(a, i, &k, counter)?;
            let rhs = abstract_component(b, &k, j, counter)?;
            let mut out = Vec::new();
            for ta in &lhs {
                for tb in &rhs {
                    let mut t = ta.clone().mul(tb);
                    t.bound.push(k.clone());
                    out.push(t);
                }
            }
            Ok(out)
        }
        TensorExpr::Add(a, b) => {
            let mut out = abstract_component(a, i, j, counter)?;
            out.extend(abstract_component(b, i, j, counter)?);
            Ok(out)
        }
        TensorExpr::Sub(a, b) => {
            let mut out = abstract_component(a, i, j, counter)?;
            for mut t in abstract_component(b, i, j, counter)? {
                t.coeff = -t.coeff;
                out.push(t);
            }
            Ok(out)
        }
        TensorExpr::ScalarMul(s, t) => {
            let mut out = abstract_component(t, i, j, counter)?;
            for term in &mut out {
                if let ScalarExpr::Num(n) = &**s {
                    term.coeff *= n;
                } else {
                    term.scalars.push(s.clone());
                }
            }
            Ok(out)
        }
        TensorExpr::Neg(t) => {
            let mut out = abstract_component(t, i, j, counter)?;
            for term in &mut out {
                term.coeff = -term.coeff;
            }
            Ok(out)
        }
        TensorExpr::Diff { .. } => Err(Error::msg(
            "nested derivatives cannot be expanded into components in this phase",
        )),
        // u ⊗ v: (u⊗v)_{ij} = u_i v_j — but the MVP only declares order-2
        // variables, so order-2 Outer means 1⊗1, which has no Var form yet.
        TensorExpr::Outer(..) => Err(Error::msg(
            "component expansion of outer products is not supported yet",
        )),
        TensorExpr::BoxTimes(..) => Err(Error::msg(
            "fourth-order ⊠ products have no second-order component form",
        )),
        TensorExpr::SetElem { .. } | TensorExpr::SumIdx { .. } => Err(Error::msg(
            "set elements and spectral sums have no component expansion in this phase",
        )),
    }
}

/// Differentiate a term sum with respect to component `base_{mn}` using the
/// full product rule:
/// - direct components: `∂X_{ab}/∂X_{mn} = δ_{am} δ_{bn}`;
/// - inverse components: `∂(X⁻¹)_{ab}/∂X_{mn} = −(X⁻¹)_{am} (X⁻¹)_{nb}`;
/// - scalar coefficients (det, log det, products, powers) via
///   [`d_scalar_comp`].
pub fn differentiate(terms: &[Term], base: &str, m: &Idx, n: &Idx) -> Result<Vec<Term>, Error> {
    let mut out = Vec::new();
    for term in terms {
        // component factors
        for (pos, comp) in term.comps.iter().enumerate() {
            if comp.base == base && comp.indices.len() == 2 {
                let mut t = term.clone();
                t.comps.remove(pos);
                if comp.inv {
                    // d(X⁻¹)_{ab} = −(X⁻¹)_{am} (X⁻¹)_{nb} dX_{mn}
                    t.coeff = -t.coeff;
                    t.comps.push(Comp {
                        base: base.to_string(),
                        inv: true,
                        indices: vec![comp.indices[0].clone(), m.clone()],
                    });
                    t.comps.push(Comp {
                        base: base.to_string(),
                        inv: true,
                        indices: vec![n.clone(), comp.indices[1].clone()],
                    });
                } else {
                    t.deltas.push((comp.indices[0].clone(), m.clone()));
                    t.deltas.push((comp.indices[1].clone(), n.clone()));
                }
                out.push(t);
            }
        }
        // scalar coefficient factors
        for (pos, s) in term.scalars.iter().enumerate() {
            if !scalar_depends(s, base) {
                continue;
            }
            for dterm in d_scalar_comp(s, base, m, n)? {
                let mut t = term.clone();
                t.scalars.remove(pos);
                out.push(t.mul(&dterm));
            }
        }
    }
    Ok(out)
}

/// Does a scalar expression depend on the tensor variable named `base`?
fn scalar_depends(s: &ScalarExpr, base: &str) -> bool {
    match s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => scalar_depends(a, base) || scalar_depends(b, base),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) => scalar_depends(a, base),
        ScalarExpr::Func { arg, .. } => scalar_depends(arg, base),
        ScalarExpr::SpecSum { body, .. } => scalar_depends(body, base),
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => tensor_mentions(t, base),
        ScalarExpr::Ddot(a, b) => tensor_mentions(a, base) || tensor_mentions(b, base),
    }
}

fn tensor_mentions(t: &TensorExpr, base: &str) -> bool {
    match t {
        TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => false,
        TensorExpr::Filled { entries, .. } => entries.iter().any(|e| scalar_depends(e, base)),
        TensorExpr::SumIdx { body, .. } => tensor_mentions(body, base),
        TensorExpr::Var { latex, .. } => component_base(latex) == base,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => tensor_mentions(a, base),
        TensorExpr::Diff { num, den, .. } => {
            tensor_mentions(num, base) || tensor_mentions(den, base)
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => tensor_mentions(a, base) || tensor_mentions(b, base),
        TensorExpr::ScalarMul(s, a) => scalar_depends(s, base) || tensor_mentions(a, base),
    }
}

/// `∂s/∂X_{mn}` for a scalar coefficient, as index terms.
fn d_scalar_comp(s: &Rc<ScalarExpr>, base: &str, m: &Idx, n: &Idx) -> Result<Vec<Term>, Error> {
    if !scalar_depends(s, base) {
        return Ok(vec![]);
    }
    let inv_comp = |a: &Idx, b: &Idx| Comp {
        base: base.to_string(),
        inv: true,
        indices: vec![a.clone(), b.clone()],
    };
    match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => Ok(vec![]),
        ScalarExpr::Add(a, b) => {
            let mut out = d_scalar_comp(a, base, m, n)?;
            out.extend(d_scalar_comp(b, base, m, n)?);
            Ok(out)
        }
        ScalarExpr::Sub(a, b) => {
            let mut out = d_scalar_comp(a, base, m, n)?;
            for mut t in d_scalar_comp(b, base, m, n)? {
                t.coeff = -t.coeff;
                out.push(t);
            }
            Ok(out)
        }
        ScalarExpr::Neg(a) => {
            let mut out = d_scalar_comp(a, base, m, n)?;
            for t in &mut out {
                t.coeff = -t.coeff;
            }
            Ok(out)
        }
        ScalarExpr::Mul(a, b) => {
            // d(ab) = da·b + a·db
            let mut out = Vec::new();
            for mut t in d_scalar_comp(a, base, m, n)? {
                t.scalars.push(b.clone());
                out.push(t);
            }
            for mut t in d_scalar_comp(b, base, m, n)? {
                t.scalars.push(a.clone());
                out.push(t);
            }
            Ok(out)
        }
        ScalarExpr::Pow(a, e) => {
            let Some(p) = numeric_value(e) else {
                return Err(Error::msg(
                    "component derivative of a^b with symbolic exponent is unsupported",
                ));
            };
            // d(a^p) = p a^{p-1} da
            let mut out = Vec::new();
            for mut t in d_scalar_comp(a, base, m, n)? {
                t.coeff *= p;
                t.scalars
                    .push(crate::symbolic::fold_pow(a.clone(), p - 1.0));
                out.push(t);
            }
            Ok(out)
        }
        ScalarExpr::Div(a, b) => {
            // d(a/b) = da/b − (a/b²) db
            let mut out = Vec::new();
            for mut t in d_scalar_comp(a, base, m, n)? {
                t.scalars.push(Rc::new(ScalarExpr::Div(
                    Rc::new(ScalarExpr::Num(1.0)),
                    b.clone(),
                )));
                out.push(t);
            }
            for mut t in d_scalar_comp(b, base, m, n)? {
                t.coeff = -t.coeff;
                t.scalars.push(Rc::new(ScalarExpr::Div(
                    a.clone(),
                    Rc::new(ScalarExpr::Pow(b.clone(), Rc::new(ScalarExpr::Num(2.0)))),
                )));
                out.push(t);
            }
            Ok(out)
        }
        ScalarExpr::Log(a) => {
            // Exact form for log(det X): d = (X⁻¹)_{nm} (no det/det quotient).
            if let ScalarExpr::Det(t) = &**a {
                if matches!(&**t, TensorExpr::Var { latex, .. } if component_base(latex) == base) {
                    let mut term = Term::one();
                    term.comps.push(inv_comp(n, m));
                    return Ok(vec![term]);
                }
            }
            let mut out = Vec::new();
            for mut t in d_scalar_comp(a, base, m, n)? {
                t.scalars.push(Rc::new(ScalarExpr::Div(
                    Rc::new(ScalarExpr::Num(1.0)),
                    a.clone(),
                )));
                out.push(t);
            }
            Ok(out)
        }
        ScalarExpr::Det(t) => {
            // Jacobi componentwise: ∂(det X)/∂X_{mn} = det(X) (X⁻¹)_{nm}
            if matches!(&**t, TensorExpr::Var { latex, .. } if component_base(latex) == base) {
                let mut term = Term::one();
                term.scalars.push(s.clone());
                term.comps.push(inv_comp(n, m));
                Ok(vec![term])
            } else {
                Err(Error::msg(
                    "component derivative of det(A) for compound A is unsupported \
                     in this phase",
                ))
            }
        }
        ScalarExpr::Tr(t) => {
            // tr(T) = T_kk: expand with a fresh bound index, then recurse.
            let mut counter = 100; // distinct namespace from outer expansion
            let k: Idx = "t".to_string();
            let inner = abstract_component(t, &k, &k, &mut counter)?;
            let summed: Vec<Term> = inner
                .into_iter()
                .map(|mut term| {
                    term.bound.push(k.clone());
                    term
                })
                .collect();
            differentiate(&summed, base, m, n)
        }
        ScalarExpr::Ddot(_, _) => Err(Error::msg(
            "component derivative of a double contraction is not supported yet",
        )),
        ScalarExpr::Func { .. } | ScalarExpr::SpecSum { .. } => Err(Error::msg(
            "component derivatives of spectral/function coefficients are not \
                 supported yet",
        )),
    }
}

/// Contract Kronecker deltas against bound indices: `Σ_k δ_{km} X_k = X_m`.
/// A surviving `δ_{kk}` over a bound index contributes a factor `dim`.
pub fn contract(terms: Vec<Term>, dim: usize) -> Vec<Term> {
    terms
        .into_iter()
        .map(|t| contract_term(t, dim))
        .filter(|t| t.coeff != 0.0)
        .collect()
}

fn contract_term(mut term: Term, dim: usize) -> Term {
    loop {
        let mut changed = false;
        let mut idx = 0;
        while idx < term.deltas.len() {
            let (p, q) = term.deltas[idx].clone();
            if p == q {
                // δ_{pp}: full trace if p is bound, plain 1 otherwise.
                term.deltas.remove(idx);
                if let Some(pos) = term.bound.iter().position(|b| *b == p) {
                    term.bound.remove(pos);
                    term.coeff *= dim as f64;
                }
                changed = true;
                continue;
            }
            let bound_side = if term.bound.contains(&q) {
                Some((q.clone(), p.clone()))
            } else if term.bound.contains(&p) {
                Some((p.clone(), q.clone()))
            } else {
                None
            };
            if let Some((from, to)) = bound_side {
                term.deltas.remove(idx);
                term.bound.retain(|b| *b != from);
                rename_index(&mut term, &from, &to);
                changed = true;
                continue;
            }
            idx += 1;
        }
        if !changed {
            break;
        }
    }
    term
}

fn rename_index(term: &mut Term, from: &Idx, to: &Idx) {
    for comp in &mut term.comps {
        for ix in &mut comp.indices {
            if ix == from {
                *ix = to.clone();
            }
        }
    }
    for (p, q) in &mut term.deltas {
        if p == from {
            *p = to.clone();
        }
        if q == from {
            *q = to.clone();
        }
    }
}

/// Substitute concrete values (1-based, as strings) for the free indices of
/// each term, expanding bound sums over `1..=dim` and evaluating Kronecker
/// deltas numerically. Used by block-component display of fourth-order
/// derivatives.
pub fn instantiate(terms: &[Term], assign: &[(Idx, usize)], dim: usize) -> Vec<Term> {
    let mut out = Vec::new();
    for term in terms {
        let mut t = term.clone();
        for (name, value) in assign {
            rename_index(&mut t, name, &value.to_string());
        }
        expand_bound(t, dim, &mut out);
    }
    // Evaluate deltas where both indices are concrete numbers; keep deltas
    // that still involve a free symbolic index (e.g. δ_{i1}).
    out.retain_mut(|t| {
        let mut kept = Vec::new();
        for (p, q) in t.deltas.drain(..) {
            let concrete = p.parse::<usize>().is_ok() && q.parse::<usize>().is_ok();
            if concrete {
                if p != q {
                    return false; // δ with distinct concrete indices = 0
                }
                // δ_{vv} = 1: drop the factor
            } else {
                kept.push((p, q));
            }
        }
        t.deltas = kept;
        t.coeff != 0.0
    });
    out
}

fn expand_bound(term: Term, dim: usize, out: &mut Vec<Term>) {
    match term.bound.first().cloned() {
        None => out.push(term),
        Some(k) => {
            for v in 1..=dim {
                let mut t = term.clone();
                t.bound.remove(0);
                rename_index(&mut t, &k, &v.to_string());
                expand_bound(t, dim, out);
            }
        }
    }
}

/// Render a term sum to LaTeX, e.g. `\delta_{in} F_{mj} + \delta_{jn} F_{mi}`.
pub fn render_terms(terms: &[Term]) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    let mut out = String::new();
    for (idx, term) in terms.iter().enumerate() {
        let negative = term.coeff < 0.0;
        if idx == 0 {
            if negative {
                out.push('-');
            }
        } else {
            out.push_str(if negative { " - " } else { " + " });
        }
        out.push_str(&render_term(term));
    }
    out
}

fn render_term(term: &Term) -> String {
    let mut parts = Vec::new();
    let mag = term.coeff.abs();
    if mag != 1.0 {
        parts.push(if mag.fract() == 0.0 {
            format!("{}", mag as i64)
        } else {
            format!("{mag}")
        });
    }
    for s in &term.scalars {
        parts.push(scalar_to_latex(s));
    }
    for (p, q) in &term.deltas {
        parts.push(format!("\\delta_{{{p}{q}}}"));
    }
    for c in &term.comps {
        if c.inv {
            parts.push(format!("{}^{{-1}}_{{{}}}", c.base, c.indices.join("")));
        } else {
            parts.push(format!("{}_{{{}}}", c.base, c.indices.join("")));
        }
    }
    if parts.is_empty() {
        parts.push("1".to_string());
    }
    let body = parts.join(" ");
    if term.bound.is_empty() {
        body
    } else {
        format!("\\sum_{{{}}} {body}", term.bound.join(","))
    }
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
