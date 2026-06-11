//! Conservative property inference.
//!
//! A property is inferred only when it is strictly provable from the
//! expression structure and declared properties. Anything not provable is
//! reported as `false` ("not known to hold") — never guessed.

use super::TensorExpr;

/// Strict symmetry proof for second-order tensor expressions.
///
/// Provable cases:
/// - a variable declared `symmetric=true` or `identity=true`;
/// - `A^T A` and `A A^T` for *structurally identical* `A` (so `F.T * F` ✓);
/// - `A^T` where `A` is symmetric;
/// - `A + B`, `A - B` where both are symmetric;
/// - `s * A` and `-A` where `A` is symmetric.
///
/// Deliberately NOT inferred: `A * B` with `A`, `B` both symmetric — the
/// product of symmetric tensors is not symmetric in general.
pub fn is_symmetric(expr: &TensorExpr) -> bool {
    if expr.order() != 2 {
        return false;
    }
    match expr {
        TensorExpr::Var { props, .. } => props.symmetric || props.identity,
        TensorExpr::Transpose(t) => is_symmetric(t),
        // A symmetric ⇒ A^{-1} symmetric ⇒ A^{-T} = A^{-1} symmetric.
        TensorExpr::Inverse(t) | TensorExpr::InverseTranspose(t) => is_symmetric(t),
        // A derivative node is at least order 3; never reaches here (order
        // check above), but listed for exhaustiveness.
        TensorExpr::Diff { .. } => false,
        TensorExpr::MatMul(a, b) => {
            // (A^T A)^T = A^T A and (A A^T)^T = A A^T, by structural equality.
            match (&**a, &**b) {
                (TensorExpr::Transpose(inner), rhs) if **inner == *rhs => true,
                (lhs, TensorExpr::Transpose(inner)) if **inner == *lhs => true,
                // I B = B and A I = A: symmetry passes through identity factors.
                (lhs, rhs) if lhs.is_identity() => is_symmetric(rhs),
                (lhs, rhs) if rhs.is_identity() => is_symmetric(lhs),
                _ => false,
            }
        }
        TensorExpr::Add(a, b) | TensorExpr::Sub(a, b) => is_symmetric(a) && is_symmetric(b),
        TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => is_symmetric(t),
        // u ⊗ u is symmetric only for structurally identical factors.
        TensorExpr::Outer(a, b) => a == b,
        // Σ λ_a N_a ⊗ N_a is symmetric by construction.
        TensorExpr::Spectral { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorProperties;
    use std::rc::Rc;

    fn var(name: &str, symmetric: bool) -> Rc<TensorExpr> {
        Rc::new(TensorExpr::Var {
            name: name.into(),
            latex: format!("\\bm {name}"),
            order: 2,
            dim: 3,
            props: TensorProperties {
                symmetric,
                ..Default::default()
            },
        })
    }

    #[test]
    fn ftf_is_symmetric() {
        let f = var("F", false);
        let ft = Rc::new(TensorExpr::Transpose(f.clone()));
        let c = TensorExpr::MatMul(ft, f);
        assert!(is_symmetric(&c));
    }

    #[test]
    fn product_of_symmetric_tensors_is_not_inferred_symmetric() {
        let a = var("A", true);
        let b = var("B", true);
        let ab = TensorExpr::MatMul(a, b);
        assert!(!is_symmetric(&ab));
    }

    #[test]
    fn plain_product_is_not_symmetric() {
        let f = var("F", false);
        let g = var("G", false);
        let fg = TensorExpr::MatMul(f, g);
        assert!(!is_symmetric(&fg));
    }
}
