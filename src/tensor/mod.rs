//! The tensor object system: semantic tensor-valued expressions with
//! order/dim metadata and conservative property inference.

pub mod inference;

use crate::error::Error;
use crate::symbolic::{fold_mul, ScalarExpr};
use std::rc::Rc;

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
    /// `\bm C` when the user wrote `Diff(C, F)`), used only for display.
    Diff {
        num: Rc<TensorExpr>,
        den: Rc<TensorExpr>,
        num_label: Option<String>,
    },
    /// Second-order tensor product (single contraction), `A * B` -> `AB`.
    MatMul(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Integer matrix power `A^n` (n ≥ 2) of a second-order tensor. `A^1`
    /// folds to `A`; `A^0` is not represented (use the identity). Produced by
    /// `A^2` syntax and by collapsing `A * A` in the simplifier.
    Power {
        base: Rc<TensorExpr>,
        exp: u32,
    },
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
    Add(Rc<TensorExpr>, Rc<TensorExpr>),
    Sub(Rc<TensorExpr>, Rc<TensorExpr>),
    /// Scalar times tensor.
    ScalarMul(Rc<ScalarExpr>, Rc<TensorExpr>),
    Neg(Rc<TensorExpr>),
    /// Element of a user-declared tensor set: `N[a]` for
    /// `N = VectorSet("\bm N", dim=3)`. `set_dim` is the family size.
    /// When declared via `[lambda, N] = Spectral(C, ...)`, `base` records the decomposed
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
    /// `Sum(body, index)` in the DSL. Order/dim are the body's.
    SumIdx {
        index: String,
        range: usize,
        exclude: Option<String>,
        body: Rc<TensorExpr>,
    },
    /// A component-filled tensor (a "tensor function" of the scalars in its
    /// entries): built by `F[1][1] = ...` assignments. Entries are row-major
    /// (`order` 1 or 2); unassigned components are zero. Displays as `latex`
    /// in symbol mode and as its entries in matrix mode.
    Filled {
        latex: String,
        order: usize,
        dim: usize,
        entries: Vec<Rc<ScalarExpr>>,
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
            TensorExpr::MatMul(a, _) => a.order(), // 2 in MVP
            TensorExpr::Power { base, .. } => base.order(),
            TensorExpr::Add(a, _) | TensorExpr::Sub(a, _) => a.order(),
            TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => t.order(),
            TensorExpr::SetElem { order, .. } => *order,
            TensorExpr::SumIdx { body, .. } => body.order(),
            TensorExpr::Filled { order, .. } => *order,
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
            TensorExpr::MatMul(a, _) => a.dim(),
            TensorExpr::Power { base, .. } => base.dim(),
            TensorExpr::Add(a, _) | TensorExpr::Sub(a, _) => a.dim(),
            TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => t.dim(),
            TensorExpr::SetElem { dim, .. } => *dim,
            TensorExpr::SumIdx { body, .. } => body.dim(),
            TensorExpr::Filled { dim, .. } => *dim,
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

    /// `T : Q` — contract a second-order tensor with the first two indices
    /// of a fourth-order tensor.
    pub fn double_contract_second_fourth(
        second: Rc<TensorExpr>,
        fourth: Rc<TensorExpr>,
    ) -> Result<TensorExpr, Error> {
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
        contract_second_fourth(second, fourth)
    }

    /// `A^n` for a second-order tensor and integer `n ≥ 1`. `n == 1` returns
    /// `A` unchanged; `n == 0` is rejected (write the identity instead).
    pub fn power(base: Rc<TensorExpr>, exp: u32) -> Result<TensorExpr, Error> {
        if base.order() != 2 {
            return Err(Error::msg(format!(
                "`^` (matrix power) requires a second-order tensor, got order {}",
                base.order()
            )));
        }
        if exp == 0 {
            return Err(Error::msg(
                "tensor power 0 is not supported; use the identity tensor",
            ));
        }
        if exp == 1 {
            return Ok((*base).clone());
        }
        Ok(TensorExpr::Power { base, exp })
    }

    /// Expand `A^n` into the left-folded product `((A A) A) …` (n factors).
    /// Used by component expansion and differentiation, which only know
    /// `MatMul`.
    pub fn expand_power(base: &Rc<TensorExpr>, exp: u32) -> Rc<TensorExpr> {
        let mut acc = base.clone();
        for _ in 1..exp {
            acc = Rc::new(TensorExpr::MatMul(acc, base.clone()));
        }
        acc
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

fn contract_second_fourth(
    second: Rc<TensorExpr>,
    fourth: Rc<TensorExpr>,
) -> Result<TensorExpr, Error> {
    if let TensorExpr::ScalarMul(s, inner) = &*second {
        return Ok(scale_tensor(
            s.clone(),
            Rc::new(contract_second_fourth(inner.clone(), fourth)?),
        ));
    }

    Ok(match &*fourth {
        TensorExpr::Identity4 { .. } => (*second).clone(),
        TensorExpr::ScalarMul(s, inner) => scale_tensor(
            s.clone(),
            Rc::new(contract_second_fourth(second, inner.clone())?),
        ),
        TensorExpr::Add(a, b) => TensorExpr::Add(
            Rc::new(contract_second_fourth(second.clone(), a.clone())?),
            Rc::new(contract_second_fourth(second, b.clone())?),
        ),
        TensorExpr::Sub(a, b) => TensorExpr::Sub(
            Rc::new(contract_second_fourth(second.clone(), a.clone())?),
            Rc::new(contract_second_fourth(second, b.clone())?),
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
            body: Rc::new(contract_second_fourth(second, body.clone())?),
        },
        TensorExpr::Neg(inner) => Rc::new(contract_second_fourth(second, inner.clone())?)
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
        _ => {
            return Err(Error::msg(
                "this 2:4 contraction has no closed form; contract with sums of \
                 outer/boxtimes products of second-order tensors",
            ))
        }
    })
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
