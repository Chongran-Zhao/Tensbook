//! The tensor object system: semantic tensor-valued expressions with
//! order/dim metadata and conservative property inference.

pub mod inference;

use crate::error::Error;
use crate::symbolic::{fold_mul, ScalarExpr};
use std::rc::Rc;

/// Scale function of a generalized strain, mapping principal stretches to
/// principal strain values (Hill's generalized strain family).
#[derive(Debug, Clone, PartialEq)]
pub enum Scale {
    /// Seth–Hill: `E(λ) = (λ^m − 1)/m`. `m = 2` is Green–Lagrange,
    /// `m = -2` Almansi (referential), `m = 1` Biot.
    SethHill { m: Rc<ScalarExpr> },
    /// Curnier–Rakotomanana (coercive): `E(λ) = (λ^m − λ^{−n})/(m + n)`.
    CR {
        m: Rc<ScalarExpr>,
        n: Rc<ScalarExpr>,
    },
    /// Hencky (logarithmic, coercive): `E(λ) = ln λ`.
    Hencky,
    /// A user-defined scale function: `body` is an ordinary scalar
    /// expression in the declared variable `var` (e.g. `lam = Var("\lambda")`
    /// then `E = (lam^m - lam^(-n))/(m+n)`). `dbody` is `d body/d var`,
    /// computed once at declaration by the scalar differentiation engine.
    Custom {
        var: String,
        body: Rc<ScalarExpr>,
        dbody: Rc<ScalarExpr>,
    },
}

impl Scale {
    /// `E(λ)` applied to a symbolic principal value.
    pub fn apply(&self, lambda: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
        match self {
            Scale::SethHill { m } => Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Sub(
                    Rc::new(ScalarExpr::Pow(lambda.clone(), m.clone())),
                    Rc::new(ScalarExpr::Num(1.0)),
                )),
                m.clone(),
            )),
            Scale::CR { m, n } => Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Sub(
                    Rc::new(ScalarExpr::Pow(lambda.clone(), m.clone())),
                    Rc::new(ScalarExpr::Pow(
                        lambda.clone(),
                        Rc::new(ScalarExpr::Neg(n.clone())),
                    )),
                )),
                Rc::new(ScalarExpr::Add(m.clone(), n.clone())),
            )),
            Scale::Hencky => Rc::new(ScalarExpr::Log(lambda.clone())),
            Scale::Custom { var, body, .. } => {
                crate::substitute::subst_scalar_sym(body, var, lambda)
            }
        }
    }

    /// `E'(λ)`, the derivative of the scale function.
    pub fn dapply(&self, lambda: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
        let pow = |b: &Rc<ScalarExpr>, e: Rc<ScalarExpr>| Rc::new(ScalarExpr::Pow(b.clone(), e));
        let num = |v: f64| Rc::new(ScalarExpr::Num(v));
        match self {
            // d/dλ (λ^m − 1)/m = λ^{m−1}
            Scale::SethHill { m } => pow(lambda, Rc::new(ScalarExpr::Sub(m.clone(), num(1.0)))),
            // d/dλ (λ^m − λ^{−n})/(m+n) = (m λ^{m−1} + n λ^{−n−1})/(m+n)
            Scale::CR { m, n } => Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Add(
                    Rc::new(ScalarExpr::Mul(
                        m.clone(),
                        pow(lambda, Rc::new(ScalarExpr::Sub(m.clone(), num(1.0)))),
                    )),
                    Rc::new(ScalarExpr::Mul(
                        n.clone(),
                        pow(
                            lambda,
                            Rc::new(ScalarExpr::Sub(
                                Rc::new(ScalarExpr::Neg(n.clone())),
                                num(1.0),
                            )),
                        ),
                    )),
                )),
                Rc::new(ScalarExpr::Add(m.clone(), n.clone())),
            )),
            // d/dλ ln λ = 1/λ
            Scale::Hencky => Rc::new(ScalarExpr::Div(num(1.0), lambda.clone())),
            Scale::Custom { var, dbody, .. } => {
                crate::substitute::subst_scalar_sym(dbody, var, lambda)
            }
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Scale::SethHill { .. } => "SethHill",
            Scale::CR { .. } => "CR",
            Scale::Hencky => "Hencky",
            Scale::Custom { .. } => "Custom",
        }
    }
}

/// Declared properties of a tensor variable.
///
/// Mirrors the full design (identity / symmetric / antisymmetric /
/// orthogonal / isotropic); the MVP only acts on `identity` and `symmetric`,
/// the rest are carried as metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorProperties {
    pub identity: bool,
    pub symmetric: bool,
    pub antisymmetric: bool,
    pub orthogonal: bool,
    pub isotropic: bool,
}

/// A semantic tensor-valued expression. Every node knows its order and dim;
/// construction enforces shape compatibility, so an ill-shaped tree cannot
/// be built.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorExpr {
    /// A declared tensor variable, e.g. `Tensor("\bm F", order=2, dim=3)`.
    Var {
        name: String,
        latex: String,
        order: usize,
        dim: usize,
        props: TensorProperties,
    },
    /// Transpose of a second-order tensor.
    Transpose(Rc<TensorExpr>),
    /// Inverse of a second-order tensor, `F^{-1}`. Kept opaque (no component
    /// expansion in this phase); produced by differentiation rules.
    Inverse(Rc<TensorExpr>),
    /// Inverse transpose, `F^{-T}`. A dedicated node because it is the
    /// canonical display form in continuum mechanics (`∂J/∂F = J F^{-T}`).
    InverseTranspose(Rc<TensorExpr>),
    /// Unevaluated tensor-by-tensor derivative `∂ num / ∂ den`.
    /// Order = num.order() + den.order() (e.g. ∂C/∂F is order 4).
    /// Component formulas are produced by the differentiation engine.
    /// `num_label` is an optional LaTeX label for the numerator (e.g.
    /// `\bm C` when the user wrote `diff(C, F)`), used only for display.
    Diff {
        num: Rc<TensorExpr>,
        den: Rc<TensorExpr>,
        num_label: Option<String>,
    },
    /// Second-order tensor product (single contraction), `A * B` -> `AB`.
    MatMul(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Tensor (outer) product `A ⊗ B`; order = a.order() + b.order().
    Outer(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Non-standard fourth-order product `A ⊠ B` of two second-order
    /// tensors, with components `(A ⊠ B)_{ijmn} = A_{im} B_{jn}`
    /// (Itskov's `⊗̄`). Produced by the derivative of the tensor inverse:
    /// `∂(X⁻¹)/∂X = −X⁻¹ ⊠ X⁻ᵀ`, matching the component rule
    /// `∂(X⁻¹)_{ij}/∂X_{mn} = −(X⁻¹)_{im} (X⁻¹)_{nj}`.
    BoxTimes(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Fourth-order identity tensor `I4`, with components δ_im δ_jn.
    Identity4 {
        dim: usize,
    },
    /// Spectral decomposition of a provably symmetric second-order tensor:
    /// `T = Σ_a λ_a N_a ⊗ N_a`. `base_latex` names the decomposed tensor
    /// (used to derive eigenvalue/eigenvector symbols).
    Spectral {
        base: Rc<TensorExpr>,
        base_latex: String,
    },
    /// An isotropic tensor function applied through the spectral form:
    /// `f(T) = Σ_a f(λ_a) N_a ⊗ N_a` for symmetric `T`.
    /// `func` is the scalar function name: "sqrt" | "log" | "exp".
    SpectralFn {
        func: String,
        base: Rc<TensorExpr>,
        base_latex: String,
    },
    /// Generalized strain of a symmetric deformation tensor:
    /// `E(C) = Σ_a E(λ_a) M_a` with scale function `scale` (order 2).
    GenStrain {
        base: Rc<TensorExpr>,
        scale: Scale,
        /// LaTeX of the strain symbol, e.g. `\bm E`.
        latex: String,
    },
    /// The rank-four strain projection `Q = 2 ∂E/∂C` induced by a
    /// generalized strain (order 4). Kept opaque; spectral component
    /// formulas (Miehe–Lambrecht) are display-only in this phase.
    QTensor {
        strain: Rc<TensorExpr>, // the GenStrain this Q belongs to
    },
    /// Double contraction of a second-order tensor with a fourth-order
    /// tensor over its first two indices: `(T : Q)_{kl} = T_{ij} Q_{ijkl}`.
    /// Result is order 2. This is the paper's `S = T : Q` structure.
    DdotTQ {
        second: Rc<TensorExpr>,
        fourth: Rc<TensorExpr>,
    },
    Add(Rc<TensorExpr>, Rc<TensorExpr>),
    Sub(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Scalar times tensor.
    ScalarMul(Rc<ScalarExpr>, Rc<TensorExpr>),
    Neg(Rc<TensorExpr>),
    /// Element of a user-declared tensor set: `N[a]` for
    /// `N = VectorSet("\bm N", dim=3)`. `set_dim` is the family size.
    /// When declared via `eigvecs(C, ...)`, `base` records the decomposed
    /// tensor (so dependence on `C` is tracked).
    SetElem {
        latex: String,
        order: usize,
        dim: usize,
        index: crate::symbolic::SetIndex,
        set_dim: usize,
        base: Option<Rc<TensorExpr>>,
    },
    /// Sum over an abstract set index: `Σ_{index=1}^{range} body`, written
    /// `sum(body, index)` in the DSL. Order/dim are the body's.
    SumIdx {
        index: String,
        range: usize,
        exclude: Option<String>,
        body: Rc<TensorExpr>,
    },
}

impl TensorExpr {
    pub fn order(&self) -> usize {
        match self {
            TensorExpr::Var { order, .. } => *order,
            TensorExpr::Transpose(t) | TensorExpr::Inverse(t) | TensorExpr::InverseTranspose(t) => {
                t.order()
            }
            TensorExpr::Diff { num, den, .. } => num.order() + den.order(),
            TensorExpr::Outer(a, b) => a.order() + b.order(),
            TensorExpr::BoxTimes(..) | TensorExpr::Identity4 { .. } => 4,
            TensorExpr::Spectral { base, .. } | TensorExpr::SpectralFn { base, .. } => base.order(),
            TensorExpr::GenStrain { .. } => 2,
            TensorExpr::QTensor { .. } => 4,
            TensorExpr::DdotTQ { .. } => 2,
            TensorExpr::MatMul(a, _) => a.order(), // 2 in MVP
            TensorExpr::Add(a, _) | TensorExpr::Sub(a, _) => a.order(),
            TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => t.order(),
            TensorExpr::SetElem { order, .. } => *order,
            TensorExpr::SumIdx { body, .. } => body.order(),
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            TensorExpr::Var { dim, .. } => *dim,
            TensorExpr::Transpose(t) | TensorExpr::Inverse(t) | TensorExpr::InverseTranspose(t) => {
                t.dim()
            }
            TensorExpr::Diff { num, .. } => num.dim(),
            TensorExpr::Outer(a, _) | TensorExpr::BoxTimes(a, _) => a.dim(),
            TensorExpr::Identity4 { dim } => *dim,
            TensorExpr::Spectral { base, .. } | TensorExpr::SpectralFn { base, .. } => base.dim(),
            TensorExpr::GenStrain { base, .. } => base.dim(),
            TensorExpr::QTensor { strain } => strain.dim(),
            TensorExpr::DdotTQ { second, .. } => second.dim(),
            TensorExpr::MatMul(a, _) => a.dim(),
            TensorExpr::Add(a, _) | TensorExpr::Sub(a, _) => a.dim(),
            TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => t.dim(),
            TensorExpr::SetElem { dim, .. } => *dim,
            TensorExpr::SumIdx { body, .. } => body.dim(),
        }
    }

    pub fn is_identity(&self) -> bool {
        matches!(self, TensorExpr::Var { props, .. } if props.identity)
    }

    /// `true` iff symmetry is strictly provable. See [`inference`].
    pub fn is_symmetric(&self) -> bool {
        inference::is_symmetric(self)
    }

    // ---- checked constructors -------------------------------------------

    pub fn transpose(t: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if t.order() != 2 {
            return Err(Error::msg(format!(
                "`.T` requires a second-order tensor, got order {}",
                t.order()
            )));
        }
        Ok(TensorExpr::Transpose(t))
    }

    pub fn inverse(t: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if t.order() != 2 {
            return Err(Error::msg(format!(
                "`inv` requires a second-order tensor, got order {}",
                t.order()
            )));
        }
        Ok(TensorExpr::Inverse(t))
    }

    pub fn inverse_transpose(t: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if t.order() != 2 {
            return Err(Error::msg(format!(
                "inverse transpose requires a second-order tensor, got order {}",
                t.order()
            )));
        }
        Ok(TensorExpr::InverseTranspose(t))
    }

    pub fn outer(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if a.dim() != b.dim() {
            return Err(Error::msg(format!(
                "dimension mismatch in outer product: {} vs {}",
                a.dim(),
                b.dim()
            )));
        }
        Ok(TensorExpr::Outer(a, b))
    }

    /// `A ⊠ B` — both factors must be second-order and of equal dimension.
    pub fn box_times(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if a.order() != 2 || b.order() != 2 {
            return Err(Error::msg(format!(
                "`⊠` requires second-order tensors, got orders {} and {}",
                a.order(),
                b.order()
            )));
        }
        if a.dim() != b.dim() {
            return Err(Error::msg(format!(
                "dimension mismatch in `⊠` product: {} vs {}",
                a.dim(),
                b.dim()
            )));
        }
        Ok(TensorExpr::BoxTimes(a, b))
    }

    /// Generalized strain `E(C) = Σ E(λ_a) M_a` — requires a provably
    /// symmetric base (the spectral form must exist).
    pub fn gen_strain(
        base: Rc<TensorExpr>,
        scale: Scale,
        latex: String,
    ) -> Result<TensorExpr, Error> {
        if base.order() != 2 {
            return Err(Error::msg(format!(
                "a generalized strain requires a second-order tensor, got order {}",
                base.order()
            )));
        }
        if !base.is_symmetric() {
            return Err(Error::msg(
                "a generalized strain requires a provably symmetric tensor \
                 (e.g. C = F.T * F)",
            ));
        }
        Ok(TensorExpr::GenStrain { base, scale, latex })
    }

    /// `T : Q` — contract a second-order tensor with the first two indices
    /// of a fourth-order tensor.
    pub fn ddot_tq(second: Rc<TensorExpr>, fourth: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if second.order() != 2 || fourth.order() != 4 {
            return Err(Error::msg(format!(
                "T : Q requires a second-order and a fourth-order tensor, got \
                 orders {} and {}",
                second.order(),
                fourth.order()
            )));
        }
        if second.dim() != fourth.dim() {
            return Err(Error::msg("dimension mismatch in T : Q"));
        }
        Ok(contract_second_fourth(second, fourth))
    }

    /// `spectral(T)` requires symmetry to be strictly provable — the
    /// decomposition `T = Σ λ_a N_a ⊗ N_a` only exists for symmetric tensors.
    pub fn spectral(base: Rc<TensorExpr>, base_latex: String) -> Result<TensorExpr, Error> {
        if base.order() != 2 {
            return Err(Error::msg(format!(
                "`spectral` requires a second-order tensor, got order {}",
                base.order()
            )));
        }
        if !base.is_symmetric() {
            return Err(Error::msg(
                "`spectral` requires a provably symmetric tensor; declare it with \
                 symmetric=true or build it from a symmetric form like F.T * F",
            ));
        }
        Ok(TensorExpr::Spectral { base, base_latex })
    }

    /// Isotropic tensor function `f(T) = Σ f(λ_a) N_a ⊗ N_a` — same
    /// symmetry requirement as [`Self::spectral`].
    pub fn spectral_fn(
        func: &str,
        base: Rc<TensorExpr>,
        base_latex: String,
    ) -> Result<TensorExpr, Error> {
        if base.order() != 2 {
            return Err(Error::msg(format!(
                "`{func}` of a tensor requires a second-order tensor, got order {}",
                base.order()
            )));
        }
        if !base.is_symmetric() {
            return Err(Error::msg(format!(
                "`{func}` of a tensor is defined through the spectral decomposition \
                 and requires a provably symmetric tensor",
            )));
        }
        Ok(TensorExpr::SpectralFn {
            func: func.to_string(),
            base,
            base_latex,
        })
    }

    pub fn matmul(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        if a.order() != 2 || b.order() != 2 {
            return Err(Error::msg(format!(
                "tensor product `*` requires second-order tensors, got orders {} and {}",
                a.order(),
                b.order()
            )));
        }
        if a.dim() != b.dim() {
            return Err(Error::msg(format!(
                "dimension mismatch in tensor product: {} vs {}",
                a.dim(),
                b.dim()
            )));
        }
        Ok(TensorExpr::MatMul(a, b))
    }

    // Not std::ops::Add/Sub: these are fallible (shape-checked) constructors.
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        Self::check_same_shape(&a, &b, "+")?;
        Ok(TensorExpr::Add(a, b))
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(a: Rc<TensorExpr>, b: Rc<TensorExpr>) -> Result<TensorExpr, Error> {
        Self::check_same_shape(&a, &b, "-")?;
        Ok(TensorExpr::Sub(a, b))
    }

    fn check_same_shape(a: &TensorExpr, b: &TensorExpr, op: &str) -> Result<(), Error> {
        if a.order() != b.order() || a.dim() != b.dim() {
            return Err(Error::msg(format!(
                "shape mismatch in tensor `{op}`: order {}/dim {} vs order {}/dim {}",
                a.order(),
                a.dim(),
                b.order(),
                b.dim()
            )));
        }
        Ok(())
    }
}

fn contract_second_fourth(second: Rc<TensorExpr>, fourth: Rc<TensorExpr>) -> TensorExpr {
    if let TensorExpr::ScalarMul(s, inner) = &*second {
        return scale_tensor(
            s.clone(),
            Rc::new(contract_second_fourth(inner.clone(), fourth)),
        );
    }

    match &*fourth {
        TensorExpr::Identity4 { .. } => (*second).clone(),
        TensorExpr::ScalarMul(s, inner) => scale_tensor(
            s.clone(),
            Rc::new(contract_second_fourth(second, inner.clone())),
        ),
        TensorExpr::Add(a, b) => TensorExpr::Add(
            Rc::new(contract_second_fourth(second.clone(), a.clone())),
            Rc::new(contract_second_fourth(second, b.clone())),
        ),
        TensorExpr::Sub(a, b) => TensorExpr::Sub(
            Rc::new(contract_second_fourth(second.clone(), a.clone())),
            Rc::new(contract_second_fourth(second, b.clone())),
        ),
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => TensorExpr::SumIdx {
            index: index.clone(),
            range: *range,
            exclude: exclude.clone(),
            body: Rc::new(contract_second_fourth(second, body.clone())),
        },
        TensorExpr::Neg(inner) => Rc::new(contract_second_fourth(second, inner.clone()))
            .as_ref()
            .clone()
            .negated(),
        TensorExpr::Outer(a, b) if a.order() == 2 && b.order() == 2 => {
            scale_tensor(contract_coeff(second, a.clone()), b.clone())
        }
        // T_{ij} (A ⊠ B)_{ijmn} = T_{ij} A_{im} B_{jn} = (Aᵀ T B)_{mn}
        TensorExpr::BoxTimes(a, b) => {
            let at = if a.is_symmetric() {
                a.clone()
            } else {
                Rc::new(TensorExpr::Transpose(a.clone()))
            };
            TensorExpr::MatMul(Rc::new(TensorExpr::MatMul(at, second)), b.clone())
        }
        _ => TensorExpr::DdotTQ { second, fourth },
    }
}

fn contract_coeff(second: Rc<TensorExpr>, first: Rc<TensorExpr>) -> Rc<ScalarExpr> {
    if second.is_identity() {
        Rc::new(ScalarExpr::Tr(first))
    } else if first.is_identity() {
        Rc::new(ScalarExpr::Tr(second))
    } else {
        Rc::new(ScalarExpr::Ddot(second, first))
    }
}

fn scale_tensor(s: Rc<ScalarExpr>, t: Rc<TensorExpr>) -> TensorExpr {
    if let TensorExpr::ScalarMul(s2, inner) = &*t {
        return scale_tensor(fold_mul(&s, s2), inner.clone());
    }
    if let ScalarExpr::Num(n) = &*s {
        if *n == 1.0 {
            return (*t).clone();
        }
        if *n == -1.0 {
            return TensorExpr::Neg(t);
        }
    }
    TensorExpr::ScalarMul(s, t)
}

trait NegTensor {
    fn negated(self) -> TensorExpr;
}

impl NegTensor for TensorExpr {
    fn negated(self) -> TensorExpr {
        match self {
            TensorExpr::Neg(inner) => (*inner).clone(),
            other => TensorExpr::Neg(Rc::new(other)),
        }
    }
}
