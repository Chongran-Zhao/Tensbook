//! Symbolic scalar expressions.
//!
//! Scalars stay fully symbolic: `det`, `tr` and `log` are opaque nodes, not
//! evaluated. Tensor-valued subexpressions appear inside scalar expressions
//! through [`ScalarExpr::Det`] and [`ScalarExpr::Tr`].

use crate::tensor::TensorExpr;
use std::rc::Rc;

/// Index of a set element: an abstract index name (`lambda[a]`) or a
/// concrete 1-based position (`lambda[1]`).
#[derive(Debug, Clone, PartialEq)]
pub enum SetIndex {
    Sym(String),
    Num(usize),
}

impl SetIndex {
    pub fn latex(&self) -> String {
        match self {
            SetIndex::Sym(s) => s.clone(),
            SetIndex::Num(n) => n.to_string(),
        }
    }

    /// The abstract index name, if any.
    pub fn sym(&self) -> Option<&str> {
        match self {
            SetIndex::Sym(s) => Some(s),
            SetIndex::Num(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarExpr {
    /// A named symbolic scalar with its LaTeX display, e.g. `Sym("mu", "\mu")`.
    Sym {
        name: String,
        latex: String,
    },
    Num(f64),
    Add(Rc<ScalarExpr>, Rc<ScalarExpr>),
    Sub(Rc<ScalarExpr>, Rc<ScalarExpr>),
    Mul(Rc<ScalarExpr>, Rc<ScalarExpr>),
    Div(Rc<ScalarExpr>, Rc<ScalarExpr>),
    Pow(Rc<ScalarExpr>, Rc<ScalarExpr>),
    Neg(Rc<ScalarExpr>),
    Log(Rc<ScalarExpr>),
    /// Named univariate function applied symbolically: exp, sqrt, sinh,
    /// cosh, tanh, or any user-named function (kept opaque).
    Func {
        name: String,
        arg: Rc<ScalarExpr>,
    },
    Det(Rc<TensorExpr>),
    Tr(Rc<TensorExpr>),
    /// Double contraction of two second-order tensors, `A : B` (a scalar).
    Ddot(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Principal value of a generalized strain or deformation tensor:
    /// `Eig { base, index }` is λ_index(base) — e.g. the principal stretch
    /// λ_a of C. `index` is the abstract spectral index name (usually "a").
    Eig {
        base: Rc<TensorExpr>,
        /// LaTeX of the eigenvalue symbol, e.g. `\lambda`.
        symbol: String,
        index: String,
    },
    /// Spectral sum `Σ_{index=1}^{dim} body`, where `body` may reference
    /// `Eig` values with the same index.
    SpecSum {
        body: Rc<ScalarExpr>,
        index: String,
        dim: usize,
    },
    /// Element of a user-declared scalar set: `lambda[a]` for
    /// `lambda = ScalarSet("\lambda", dim=3)`. `set_dim` is the family size.
    SetElem {
        latex: String,
        index: SetIndex,
        set_dim: usize,
    },
}

impl ScalarExpr {
    pub fn sym(name: impl Into<String>, latex: impl Into<String>) -> Self {
        ScalarExpr::Sym {
            name: name.into(),
            latex: latex.into(),
        }
    }

    pub fn num(n: f64) -> Self {
        ScalarExpr::Num(n)
    }
}

// ---- light normalization helpers ------------------------------------------
//
// These keep derivative results readable (e.g. `mu/2 * 2F` folds to `mu F`)
// without being a general-purpose simplifier. A scalar expression is viewed
// as `coefficient * rest` where the coefficient is the numeric part that can
// be extracted structurally.

/// Split an expression into `(numeric coefficient, symbolic rest)`.
/// `rest == None` means the expression is the pure number `coefficient`.
pub fn extract_coeff(s: &Rc<ScalarExpr>) -> (f64, Option<Rc<ScalarExpr>>) {
    match &**s {
        ScalarExpr::Num(n) => (*n, None),
        ScalarExpr::Neg(x) => {
            let (c, r) = extract_coeff(x);
            (-c, r)
        }
        ScalarExpr::Mul(a, b) => {
            let (ca, ra) = extract_coeff(a);
            let (cb, rb) = extract_coeff(b);
            (ca * cb, mul_opt(ra, rb))
        }
        ScalarExpr::Div(a, b) => {
            let (ca, ra) = extract_coeff(a);
            if let ScalarExpr::Num(n) = &**b {
                (ca / n, ra)
            } else {
                let num = ra.unwrap_or_else(|| Rc::new(ScalarExpr::Num(1.0)));
                (ca, Some(Rc::new(ScalarExpr::Div(num, b.clone()))))
            }
        }
        _ => (1.0, Some(s.clone())),
    }
}

/// Rebuild `coefficient * rest`, choosing the most readable form
/// (`1 * x -> x`, `0.5 * x -> \frac{x}{2}`, `-1 * x -> -x`).
pub fn with_coeff(c: f64, rest: Option<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    match rest {
        None => Rc::new(ScalarExpr::Num(c)),
        Some(r) => {
            if c == 0.0 {
                Rc::new(ScalarExpr::Num(0.0))
            } else if c == 1.0 {
                r
            } else if c == -1.0 {
                Rc::new(ScalarExpr::Neg(r))
            } else if c.fract() == 0.0 {
                Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(c)), r))
            } else if (1.0 / c).fract() == 0.0 {
                let frac = Rc::new(ScalarExpr::Div(
                    r,
                    Rc::new(ScalarExpr::Num((1.0 / c).abs())),
                ));
                if c < 0.0 {
                    Rc::new(ScalarExpr::Neg(frac))
                } else {
                    frac
                }
            } else {
                Rc::new(ScalarExpr::Mul(Rc::new(ScalarExpr::Num(c)), r))
            }
        }
    }
}

fn mul_opt(a: Option<Rc<ScalarExpr>>, b: Option<Rc<ScalarExpr>>) -> Option<Rc<ScalarExpr>> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(x), Some(y)) => Some(Rc::new(ScalarExpr::Mul(x, y))),
    }
}

/// Multiply two scalars with coefficient folding.
pub fn fold_mul(a: &Rc<ScalarExpr>, b: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    let (ca, ra) = extract_coeff(a);
    let (cb, rb) = extract_coeff(b);
    with_coeff(ca * cb, mul_opt(ra, rb))
}

/// Add two scalars, folding additive zeros.
pub fn fold_add(a: Rc<ScalarExpr>, b: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    match (&*a, &*b) {
        (ScalarExpr::Num(x), _) if *x == 0.0 => b,
        (_, ScalarExpr::Num(x)) if *x == 0.0 => a,
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => Rc::new(ScalarExpr::Num(x + y)),
        _ => Rc::new(ScalarExpr::Add(a, b)),
    }
}

/// Subtract two scalars, folding zeros.
pub fn fold_sub(a: Rc<ScalarExpr>, b: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    match (&*a, &*b) {
        (_, ScalarExpr::Num(x)) if *x == 0.0 => a,
        (ScalarExpr::Num(x), _) if *x == 0.0 => with_coeff(-1.0, Some(b)),
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => Rc::new(ScalarExpr::Num(x - y)),
        _ => Rc::new(ScalarExpr::Sub(a, b)),
    }
}

/// `base^exp` with `x^0 -> 1`, `x^1 -> x` folding (numeric exponents only).
pub fn fold_pow(base: Rc<ScalarExpr>, exp: f64) -> Rc<ScalarExpr> {
    if exp == 0.0 {
        Rc::new(ScalarExpr::Num(1.0))
    } else if exp == 1.0 {
        base
    } else {
        Rc::new(ScalarExpr::Pow(base, Rc::new(ScalarExpr::Num(exp))))
    }
}
