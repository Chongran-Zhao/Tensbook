//! Expression simplification.
//!
//! `simplify(expr)` / `simplify(expr, rules=algebra|tensor|continuum)`.
//! Rule sets are cumulative: `tensor` includes `algebra`, `continuum`
//! includes both. Rules are applied bottom-up to a fixed point; every rule
//! is an exact mathematical identity (no guessing), e.g.:
//!
//! - algebra: `x + 0 → x`, `1·x → x`, `x^1 → x`, `log(x^n) → n log x`
//! - tensor: `(Aᵀ)ᵀ → A`, `A I → A`, `F⁻¹F → I`, `F^{-T}Fᵀ → I`,
//!   `(Aᵀ)⁻¹ → A^{-T}`, `Aᵀ → A` when A is provably symmetric
//! - continuum: `tr(AᵀB) ↔` cyclic canonicalization, `det(Fᵀ) → det F`,
//!   `tr(I) → dim`, `det(I) → 1`

use crate::error::Error;
use crate::symbolic::{extract_coeff, fold_add, fold_mul, with_coeff, ScalarExpr};
use crate::tensor::TensorExpr;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RuleSet {
    Algebra,
    Tensor,
    Continuum,
}

impl RuleSet {
    pub fn parse(name: &str) -> Result<RuleSet, Error> {
        match name {
            "algebra" => Ok(RuleSet::Algebra),
            "tensor" => Ok(RuleSet::Tensor),
            "continuum" => Ok(RuleSet::Continuum),
            other => Err(Error::msg(format!(
                "unknown rule set `{other}` (expected algebra, tensor, or continuum)"
            ))),
        }
    }

    fn includes(self, other: RuleSet) -> bool {
        self >= other
    }
}

const MAX_PASSES: usize = 32;

pub fn simplify_tensor(t: &Rc<TensorExpr>, rules: RuleSet) -> Rc<TensorExpr> {
    let mut cur = t.clone();
    for _ in 0..MAX_PASSES {
        let next = tensor_pass(&cur, rules);
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

pub fn simplify_scalar(s: &Rc<ScalarExpr>, rules: RuleSet) -> Rc<ScalarExpr> {
    let mut cur = s.clone();
    for _ in 0..MAX_PASSES {
        let next = scalar_pass(&cur, rules);
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

// ---- one bottom-up pass ----------------------------------------------------

fn tensor_pass(t: &Rc<TensorExpr>, rules: RuleSet) -> Rc<TensorExpr> {
    // Rebuild children first, then rewrite this node.
    let rebuilt: Rc<TensorExpr> = match &**t {
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => {
            t.clone()
        }
        TensorExpr::Transpose(a) => Rc::new(TensorExpr::Transpose(tensor_pass(a, rules))),
        TensorExpr::Inverse(a) => Rc::new(TensorExpr::Inverse(tensor_pass(a, rules))),
        TensorExpr::InverseTranspose(a) => {
            Rc::new(TensorExpr::InverseTranspose(tensor_pass(a, rules)))
        }
        TensorExpr::Diff {
            num,
            den,
            num_label,
        } => Rc::new(TensorExpr::Diff {
            num: tensor_pass(num, rules),
            den: tensor_pass(den, rules),
            num_label: num_label.clone(),
        }),
        TensorExpr::MatMul(a, b) => Rc::new(TensorExpr::MatMul(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        TensorExpr::Add(a, b) => Rc::new(TensorExpr::Add(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        TensorExpr::Sub(a, b) => Rc::new(TensorExpr::Sub(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        TensorExpr::ScalarMul(s, a) => Rc::new(TensorExpr::ScalarMul(
            scalar_pass(s, rules),
            tensor_pass(a, rules),
        )),
        TensorExpr::Outer(a, b) => Rc::new(TensorExpr::Outer(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        TensorExpr::BoxTimes(a, b) => Rc::new(TensorExpr::BoxTimes(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => Rc::new(TensorExpr::SumIdx {
            index: index.clone(),
            range: *range,
            exclude: exclude.clone(),
            body: tensor_pass(body, rules),
        }),
        TensorExpr::Neg(a) => Rc::new(TensorExpr::Neg(tensor_pass(a, rules))),
    };
    rewrite_tensor(rebuilt, rules)
}

fn rewrite_tensor(t: Rc<TensorExpr>, rules: RuleSet) -> Rc<TensorExpr> {
    if !rules.includes(RuleSet::Tensor) {
        return t;
    }
    match &*t {
        // (Aᵀ)ᵀ → A; Iᵀ → I; Aᵀ → A for provably symmetric A
        TensorExpr::Transpose(a) => match &**a {
            TensorExpr::Transpose(inner) => inner.clone(),
            _ if a.is_symmetric() => a.clone(),
            // (A⁻¹)ᵀ → A^{-T}
            TensorExpr::Inverse(inner) => Rc::new(TensorExpr::InverseTranspose(inner.clone())),
            _ => t,
        },
        TensorExpr::Inverse(a) => match &**a {
            // (A⁻¹)⁻¹ → A; I⁻¹ → I
            TensorExpr::Inverse(inner) => inner.clone(),
            _ if a.is_identity() => a.clone(),
            // (Aᵀ)⁻¹ → A^{-T}
            TensorExpr::Transpose(inner) => Rc::new(TensorExpr::InverseTranspose(inner.clone())),
            // A⁻¹ → A^{-1}; for symmetric A, A^{-T} = A^{-1}: keep Inverse form
            _ => t,
        },
        TensorExpr::InverseTranspose(a) => {
            if a.is_identity() {
                a.clone()
            } else {
                t
            }
        }
        TensorExpr::MatMul(a, b) => {
            // I B → B, A I → A
            if a.is_identity() {
                return b.clone();
            }
            if b.is_identity() {
                return a.clone();
            }
            // Cancellations to I (all exact):
            //   F⁻¹ F → I,  F F⁻¹ → I,  F^{-T} Fᵀ → I,  Fᵀ F^{-T} → I
            let cancels = match (&**a, &**b) {
                (TensorExpr::Inverse(x), y) | (y, TensorExpr::Inverse(x)) => **x == *y,
                (TensorExpr::InverseTranspose(x), TensorExpr::Transpose(y))
                | (TensorExpr::Transpose(y), TensorExpr::InverseTranspose(x)) => x == y,
                _ => false,
            };
            if cancels {
                return identity(a.dim());
            }
            t
        }
        TensorExpr::Add(a, b) => factor_common_tensor_scalar(a, b, false).unwrap_or(t),
        TensorExpr::Sub(a, b) => factor_common_tensor_scalar(a, b, true).unwrap_or(t),
        // 0 · A handled at ScalarMul; -(-A) → A
        TensorExpr::Neg(a) => match &**a {
            TensorExpr::Neg(inner) => inner.clone(),
            // −(−x + y) → x − y
            TensorExpr::Add(x, y) => match &**x {
                TensorExpr::Neg(inner) => Rc::new(TensorExpr::Sub(inner.clone(), y.clone())),
                _ => t,
            },
            // −(a − b) → b − a
            TensorExpr::Sub(x, y) => Rc::new(TensorExpr::Sub(y.clone(), x.clone())),
            _ => t,
        },
        TensorExpr::ScalarMul(s, a) => match &**s {
            ScalarExpr::Num(n) if *n == 1.0 => a.clone(),
            ScalarExpr::Num(n) if *n == -1.0 => Rc::new(TensorExpr::Neg(a.clone())),
            _ if matches!(&**a, TensorExpr::ScalarMul(..)) => {
                if let TensorExpr::ScalarMul(s2, inner) = &**a {
                    Rc::new(TensorExpr::ScalarMul(fold_mul(s, s2), inner.clone()))
                } else {
                    t
                }
            }
            // s · (−A) → −(s · A)
            _ if matches!(&**a, TensorExpr::Neg(..)) => {
                if let TensorExpr::Neg(inner) = &**a {
                    Rc::new(TensorExpr::Neg(Rc::new(TensorExpr::ScalarMul(
                        s.clone(),
                        inner.clone(),
                    ))))
                } else {
                    t
                }
            }
            // (−c · x) A → −(c x A): hoist negative numeric coefficients so
            // signs live on the tensor sum, not inside coefficients.
            _ => {
                let (c, rest) = extract_coeff(s);
                if c < 0.0 {
                    Rc::new(TensorExpr::Neg(Rc::new(TensorExpr::ScalarMul(
                        with_coeff(-c, rest),
                        a.clone(),
                    ))))
                } else {
                    t
                }
            }
        },
        _ => t,
    }
}

fn factor_common_tensor_scalar(
    a: &Rc<TensorExpr>,
    b: &Rc<TensorExpr>,
    subtract: bool,
) -> Option<Rc<TensorExpr>> {
    let (sa, ta) = tensor_scalar_factor(a);
    let (sb, tb) = tensor_scalar_factor(b);
    let (common, rem_a, rem_b) = common_scalar_factor(&sa, &sb)?;

    let left = scale_tensor_by_scalar(rem_a, ta);
    let right = scale_tensor_by_scalar(rem_b, tb);
    let inner = if subtract {
        Rc::new(TensorExpr::Sub(left, right))
    } else {
        Rc::new(TensorExpr::Add(left, right))
    };
    Some(scale_tensor_by_scalar(common, inner))
}

fn common_scalar_factor(
    a: &Rc<ScalarExpr>,
    b: &Rc<ScalarExpr>,
) -> Option<(Rc<ScalarExpr>, Rc<ScalarExpr>, Rc<ScalarExpr>)> {
    let (ca, fa) = scalar_product_factors(a);
    let (cb, fb) = scalar_product_factors(b);
    let mut used_b = vec![false; fb.len()];
    let mut common = Vec::new();
    let mut rem_a = Vec::new();

    for factor in fa {
        if let Some(pos) = fb
            .iter()
            .enumerate()
            .find_map(|(i, candidate)| (!used_b[i] && candidate == &factor).then_some(i))
        {
            used_b[pos] = true;
            common.push(factor);
        } else {
            rem_a.push(factor);
        }
    }
    let rem_b: Vec<Rc<ScalarExpr>> = fb
        .into_iter()
        .enumerate()
        .filter_map(|(i, factor)| (!used_b[i]).then_some(factor))
        .collect();

    if common.is_empty() {
        if same_nontrivial_numeric_coeff(ca, cb) {
            return Some((
                numeric_coeff_expr(ca),
                scalar_from_coeff_and_factors(1.0, rem_a),
                scalar_from_coeff_and_factors(1.0, rem_b),
            ));
        }
        return None;
    }

    Some((
        scalar_product(common),
        scalar_from_coeff_and_factors(ca, rem_a),
        scalar_from_coeff_and_factors(cb, rem_b),
    ))
}

fn scalar_product_factors(s: &Rc<ScalarExpr>) -> (f64, Vec<Rc<ScalarExpr>>) {
    let (coeff, rest) = extract_coeff(s);
    let mut factors = Vec::new();
    if let Some(rest) = rest {
        flatten_scalar_product(&rest, &mut factors);
    }
    (coeff, factors)
}

fn flatten_scalar_product(s: &Rc<ScalarExpr>, out: &mut Vec<Rc<ScalarExpr>>) {
    match &**s {
        ScalarExpr::Mul(a, b) => {
            flatten_scalar_product(a, out);
            flatten_scalar_product(b, out);
        }
        _ => out.push(s.clone()),
    }
}

fn scalar_from_coeff_and_factors(coeff: f64, factors: Vec<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    let body = scalar_product(factors);
    if coeff == 1.0 {
        body
    } else if coeff == -1.0 {
        Rc::new(ScalarExpr::Neg(body))
    } else if matches!(&*body, ScalarExpr::Num(n) if *n == 1.0) {
        numeric_coeff_expr(coeff)
    } else {
        Rc::new(ScalarExpr::Mul(numeric_coeff_expr(coeff), body))
    }
}

fn scalar_product(factors: Vec<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    let mut iter = factors.into_iter();
    let Some(first) = iter.next() else {
        return Rc::new(ScalarExpr::Num(1.0));
    };
    iter.fold(first, |acc, factor| Rc::new(ScalarExpr::Mul(acc, factor)))
}

fn same_nontrivial_numeric_coeff(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12 && (a - 1.0).abs() >= 1e-12 && (a + 1.0).abs() >= 1e-12
}

fn tensor_scalar_factor(t: &Rc<TensorExpr>) -> (Rc<ScalarExpr>, Rc<TensorExpr>) {
    match &**t {
        TensorExpr::ScalarMul(s, inner) => (s.clone(), inner.clone()),
        TensorExpr::Neg(inner) => {
            let (s, body) = tensor_scalar_factor(inner);
            (Rc::new(ScalarExpr::Neg(s)), body)
        }
        _ => (Rc::new(ScalarExpr::Num(1.0)), t.clone()),
    }
}

fn scale_tensor_by_scalar(s: Rc<ScalarExpr>, t: Rc<TensorExpr>) -> Rc<TensorExpr> {
    match &*s {
        ScalarExpr::Num(n) if *n == 1.0 => t,
        ScalarExpr::Num(n) if *n == -1.0 => Rc::new(TensorExpr::Neg(t)),
        _ => Rc::new(TensorExpr::ScalarMul(s, t)),
    }
}

fn numeric_coeff_expr(coeff: f64) -> Rc<ScalarExpr> {
    for den in 1..=12 {
        let num = (coeff * den as f64).round();
        if (coeff - num / den as f64).abs() < 1e-12 {
            if den == 1 {
                return Rc::new(ScalarExpr::Num(num));
            }
            let frac = Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Num(num.abs())),
                Rc::new(ScalarExpr::Num(den as f64)),
            ));
            return if num < 0.0 {
                Rc::new(ScalarExpr::Neg(frac))
            } else {
                frac
            };
        }
    }
    Rc::new(ScalarExpr::Num(coeff))
}

fn scalar_pass(s: &Rc<ScalarExpr>, rules: RuleSet) -> Rc<ScalarExpr> {
    let rebuilt: Rc<ScalarExpr> = match &**s {
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => s.clone(),
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(
            scalar_pass(a, rules),
            scalar_pass(b, rules),
        )),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(
            scalar_pass(a, rules),
            scalar_pass(b, rules),
        )),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(
            scalar_pass(a, rules),
            scalar_pass(b, rules),
        )),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(
            scalar_pass(a, rules),
            scalar_pass(b, rules),
        )),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(
            scalar_pass(a, rules),
            scalar_pass(b, rules),
        )),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(scalar_pass(a, rules))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(scalar_pass(a, rules))),
        ScalarExpr::Det(t) => Rc::new(ScalarExpr::Det(tensor_pass(t, rules))),
        ScalarExpr::Tr(t) => Rc::new(ScalarExpr::Tr(tensor_pass(t, rules))),
        ScalarExpr::Ddot(a, b) => Rc::new(ScalarExpr::Ddot(
            tensor_pass(a, rules),
            tensor_pass(b, rules),
        )),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: scalar_pass(arg, rules),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: scalar_pass(body, rules),
            index: index.clone(),
            dim: *dim,
        }),
    };
    rewrite_scalar(rebuilt, rules)
}

fn rewrite_scalar(s: Rc<ScalarExpr>, rules: RuleSet) -> Rc<ScalarExpr> {
    // algebra rules (always on: every set includes algebra)
    let s = match &*s {
        ScalarExpr::Add(a, b) => match (&**a, &**b) {
            (ScalarExpr::Num(x), _) if *x == 0.0 => b.clone(),
            (_, ScalarExpr::Num(x)) if *x == 0.0 => a.clone(),
            (ScalarExpr::Num(x), ScalarExpr::Num(y)) => Rc::new(ScalarExpr::Num(x + y)),
            // a + (−b) → a − b
            (_, ScalarExpr::Neg(y)) => Rc::new(ScalarExpr::Sub(a.clone(), y.clone())),
            _ => s,
        },
        ScalarExpr::Sub(a, b) => match (&**a, &**b) {
            (_, ScalarExpr::Num(x)) if *x == 0.0 => a.clone(),
            (ScalarExpr::Num(x), ScalarExpr::Num(y)) => Rc::new(ScalarExpr::Num(x - y)),
            _ if a == b => Rc::new(ScalarExpr::Num(0.0)),
            // a − (−b) → a + b
            (_, ScalarExpr::Neg(y)) => Rc::new(ScalarExpr::Add(a.clone(), y.clone())),
            _ => s,
        },
        ScalarExpr::Mul(a, b) => match (&**a, &**b) {
            (ScalarExpr::Num(x), _) if *x == 0.0 => Rc::new(ScalarExpr::Num(0.0)),
            (_, ScalarExpr::Num(x)) if *x == 0.0 => Rc::new(ScalarExpr::Num(0.0)),
            (ScalarExpr::Num(x), _) if *x == 1.0 => b.clone(),
            (_, ScalarExpr::Num(x)) if *x == 1.0 => a.clone(),
            (ScalarExpr::Num(x), ScalarExpr::Num(y)) => Rc::new(ScalarExpr::Num(x * y)),
            // (−x) · y → −(x y), x · (−y) → −(x y)
            (ScalarExpr::Neg(x), _) => Rc::new(ScalarExpr::Neg(Rc::new(ScalarExpr::Mul(
                x.clone(),
                b.clone(),
            )))),
            (_, ScalarExpr::Neg(y)) => Rc::new(ScalarExpr::Neg(Rc::new(ScalarExpr::Mul(
                a.clone(),
                y.clone(),
            )))),
            // x · x → x²
            _ if a == b => Rc::new(ScalarExpr::Pow(a.clone(), Rc::new(ScalarExpr::Num(2.0)))),
            // x^p · x^q → x^{p+q}
            (ScalarExpr::Pow(xa, p), ScalarExpr::Pow(xb, q)) if xa == xb => {
                Rc::new(ScalarExpr::Pow(xa.clone(), fold_add(p.clone(), q.clone())))
            }
            _ => s,
        },
        ScalarExpr::Div(a, b) => match (&**a, &**b) {
            (ScalarExpr::Num(x), _) if *x == 0.0 => Rc::new(ScalarExpr::Num(0.0)),
            (_, ScalarExpr::Num(x)) if *x == 1.0 => a.clone(),
            _ if a == b => Rc::new(ScalarExpr::Num(1.0)),
            _ => s,
        },
        ScalarExpr::Pow(a, b) => match (&**a, &**b) {
            (_, ScalarExpr::Num(n)) if *n == 1.0 => a.clone(),
            (_, ScalarExpr::Num(n)) if *n == 0.0 => Rc::new(ScalarExpr::Num(1.0)),
            // (x^p)^q → x^{pq} (exact for the positive bases — dets,
            // stretches — that symbolic powers assume here)
            (ScalarExpr::Pow(base, p), _) => Rc::new(ScalarExpr::Pow(base.clone(), fold_mul(p, b))),
            _ => s,
        },
        ScalarExpr::Neg(a) => match &**a {
            ScalarExpr::Neg(inner) => inner.clone(),
            ScalarExpr::Num(n) => Rc::new(ScalarExpr::Num(-n)),
            _ => s,
        },
        // log(x^n) → n log x (valid for x > 0, which symbolic dets/J assume)
        ScalarExpr::Log(a) => match &**a {
            ScalarExpr::Pow(base, exp) => Rc::new(ScalarExpr::Mul(
                exp.clone(),
                Rc::new(ScalarExpr::Log(base.clone())),
            )),
            _ => s,
        },
        _ => s,
    };

    if !rules.includes(RuleSet::Continuum) {
        return s;
    }
    // continuum rules
    match &*s {
        ScalarExpr::Det(t) => match &**t {
            _ if t.is_identity() => Rc::new(ScalarExpr::Num(1.0)),
            // det(Aᵀ) → det A; det(A⁻¹) → 1/det A
            TensorExpr::Transpose(a) => Rc::new(ScalarExpr::Det(a.clone())),
            TensorExpr::Inverse(a) | TensorExpr::InverseTranspose(a) => Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Num(1.0)),
                Rc::new(ScalarExpr::Det(a.clone())),
            )),
            // det(AB) → det A · det B
            TensorExpr::MatMul(a, b) => Rc::new(ScalarExpr::Mul(
                Rc::new(ScalarExpr::Det(a.clone())),
                Rc::new(ScalarExpr::Det(b.clone())),
            )),
            _ => s,
        },
        ScalarExpr::Tr(t) => match &**t {
            // tr(I) → dim
            _ if t.is_identity() => Rc::new(ScalarExpr::Num(t.dim() as f64)),
            // tr(Aᵀ) → tr A
            TensorExpr::Transpose(a) => Rc::new(ScalarExpr::Tr(a.clone())),
            // tr(A + B) → tr A + tr B
            TensorExpr::Add(a, b) => Rc::new(ScalarExpr::Add(
                Rc::new(ScalarExpr::Tr(a.clone())),
                Rc::new(ScalarExpr::Tr(b.clone())),
            )),
            // tr(sA) → s tr A
            TensorExpr::ScalarMul(c, a) => Rc::new(ScalarExpr::Mul(
                c.clone(),
                Rc::new(ScalarExpr::Tr(a.clone())),
            )),
            // tr(AB): canonicalize cyclic order so tr(AB) and tr(BA) compare
            // equal. Rotation picks the lexicographically smallest factor
            // sequence — purely a canonical form, mathematically exact.
            TensorExpr::MatMul(..) => {
                let factors = matmul_chain(t);
                if factors.len() >= 2 {
                    let canon = canonical_rotation(&factors);
                    if canon != factors {
                        return Rc::new(ScalarExpr::Tr(rebuild_chain(&canon)));
                    }
                }
                s
            }
            _ => s,
        },
        _ => s,
    }
}

// ---- trace cyclic canonicalization helpers ---------------------------------

fn matmul_chain(t: &TensorExpr) -> Vec<Rc<TensorExpr>> {
    match t {
        TensorExpr::MatMul(a, b) => {
            let mut out = matmul_chain(a);
            out.extend(matmul_chain(b));
            out
        }
        _ => vec![Rc::new(t.clone())],
    }
}

fn canonical_rotation(factors: &[Rc<TensorExpr>]) -> Vec<Rc<TensorExpr>> {
    let n = factors.len();
    let mut best: Vec<Rc<TensorExpr>> = factors.to_vec();
    let mut best_key = chain_key(&best);
    for r in 1..n {
        let rotated: Vec<Rc<TensorExpr>> = (0..n).map(|i| factors[(i + r) % n].clone()).collect();
        let key = chain_key(&rotated);
        if key < best_key {
            best_key = key;
            best = rotated;
        }
    }
    best
}

fn chain_key(factors: &[Rc<TensorExpr>]) -> String {
    factors
        .iter()
        .map(|f| format!("{f:?}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn rebuild_chain(factors: &[Rc<TensorExpr>]) -> Rc<TensorExpr> {
    let mut iter = factors.iter().cloned();
    let first = iter.next().expect("non-empty chain");
    iter.fold(first, |acc, f| Rc::new(TensorExpr::MatMul(acc, f)))
}

fn identity(dim: usize) -> Rc<TensorExpr> {
    Rc::new(TensorExpr::Var {
        name: "I".to_string(),
        latex: "\\bm I".to_string(),
        order: 2,
        dim,
        props: crate::tensor::TensorProperties {
            identity: true,
            ..Default::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorProperties;

    fn var(name: &str) -> Rc<TensorExpr> {
        Rc::new(TensorExpr::Var {
            name: name.into(),
            latex: format!("\\bm {name}"),
            order: 2,
            dim: 3,
            props: TensorProperties::default(),
        })
    }

    #[test]
    fn double_transpose_cancels() {
        let f = var("F");
        let ftt = Rc::new(TensorExpr::Transpose(Rc::new(TensorExpr::Transpose(
            f.clone(),
        ))));
        assert_eq!(simplify_tensor(&ftt, RuleSet::Tensor), f);
    }

    #[test]
    fn inverse_cancellation_gives_identity() {
        let f = var("F");
        // F^{-T} Fᵀ → I
        let expr = Rc::new(TensorExpr::MatMul(
            Rc::new(TensorExpr::InverseTranspose(f.clone())),
            Rc::new(TensorExpr::Transpose(f.clone())),
        ));
        let out = simplify_tensor(&expr, RuleSet::Tensor);
        assert!(out.is_identity(), "got {out:?}");
        // F⁻¹ F → I
        let expr = Rc::new(TensorExpr::MatMul(
            Rc::new(TensorExpr::Inverse(f.clone())),
            f.clone(),
        ));
        assert!(simplify_tensor(&expr, RuleSet::Tensor).is_identity());
    }

    #[test]
    fn trace_cyclic_canonicalization() {
        let (a, b) = (var("A"), var("B"));
        let tr_ab = Rc::new(ScalarExpr::Tr(Rc::new(TensorExpr::MatMul(
            a.clone(),
            b.clone(),
        ))));
        let tr_ba = Rc::new(ScalarExpr::Tr(Rc::new(TensorExpr::MatMul(b, a))));
        assert_eq!(
            simplify_scalar(&tr_ab, RuleSet::Continuum),
            simplify_scalar(&tr_ba, RuleSet::Continuum),
            "tr(AB) and tr(BA) must canonicalize identically"
        );
    }

    #[test]
    fn det_identities() {
        let f = var("F");
        // det(Fᵀ) → det F
        let d = Rc::new(ScalarExpr::Det(Rc::new(TensorExpr::Transpose(f.clone()))));
        assert_eq!(
            simplify_scalar(&d, RuleSet::Continuum),
            Rc::new(ScalarExpr::Det(f.clone()))
        );
        // tr(I) → 3
        let i = identity(3);
        let t = Rc::new(ScalarExpr::Tr(i));
        assert_eq!(
            simplify_scalar(&t, RuleSet::Continuum),
            Rc::new(ScalarExpr::Num(3.0))
        );
    }
}
