//! Tests for the fourth-order tangent: ∂(X⁻¹)/∂X via the ⊠ product and the
//! full material tangent ℂ = 2 ∂S/∂C of an isochoric neo-Hookean model.

use tensorforge::run_source;

const PRELUDE: &str = r#"
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
"#;

#[test]
fn d_inverse_by_compound_symmetric_tensor() {
    // ∂(C⁻¹)/∂C = −C⁻¹ ⊠ C⁻¹ for symmetric C.
    let src = format!("{PRELUDE}\nA = diff(inv(C), C)\nexport(A, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "-\\bm C^{-1} \\boxtimes \\bm C^{-1}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn d_inverse_transpose_matches_inverse_for_symmetric() {
    // C^{-T} = C⁻¹ for symmetric C, so the derivatives agree.
    let src = format!(
        "{PRELUDE}\nA = diff(inv(C), C)\nB = diff(inv(C).T, C)\n\
         export(A, format=latex)\nexport(B, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, outputs[1].latex);
}

#[test]
fn product_rule_keeps_scalar_factors() {
    // ∂(tr(C) C⁻¹)/∂C = −(tr C) C⁻¹ ⊠ C⁻¹ + C⁻¹ ⊗ I.
    // Regression: the simplifier used to drop the (tr C) factor when
    // factoring terms with equal numeric coefficients.
    let src = format!("{PRELUDE}\nB = diff(tr(C) * inv(C), C)\nexport(B, format=latex)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\left( \\operatorname{tr} \\bm C \\right) \\, \\bm C^{-1} \\boxtimes"),
        "tr C factor must survive on the boxtimes term: {latex}"
    );
    assert!(latex.contains("\\otimes \\bm I"), "got: {latex}");
}

#[test]
fn isochoric_neo_hookean_material_tangent() {
    // The screenshot pipeline: S = 2 ∂Ψ/∂C, then ℂ = 2 ∂S/∂C.
    let src = format!(
        "{PRELUDE}\nJ = det(F)\nC_bar = J^(-2/3) * C\nI_1_bar = tr(C_bar)\n\
         Psi = 1/2 * mu * I_1_bar\nS = 2 * diff(Psi, C)\nCC = 2 * diff(S, C)\n\
         display(CC, mode=symbol)"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    // The inverse-derivative term and both outer-product terms must appear.
    assert!(latex.contains("\\boxtimes"), "got: {latex}");
    assert!(latex.contains("\\otimes"), "got: {latex}");
    // The tr C factor on the ⊠ term must survive factoring.
    assert!(
        latex.contains("\\left( \\operatorname{tr} \\bm C \\right) \\, \\bm C^{-1} \\boxtimes"),
        "got: {latex}"
    );
    // J^{-2/3} must be factored exactly once and stay rational.
    assert_eq!(
        latex.matches("{J}^{-\\frac{2}{3}}").count(),
        1,
        "got: {latex}"
    );
    assert!(!latex.contains("0.66"), "no decimal exponents: {latex}");
    // No double negation or J·J leftovers.
    assert!(!latex.contains("--"), "got: {latex}");
    assert!(!latex.contains("J \\, J"), "got: {latex}");
}

#[test]
fn boxtimes_contracts_with_second_order_tensor() {
    // T : (A ⊠ B) = Aᵀ T B; with A = B = C⁻¹ and T = C this collapses to
    // C⁻¹ C C⁻¹ = C⁻¹ under the tensor cancellation rules.
    let src = format!("{PRELUDE}\nA = diff(inv(C), C)\nT = C : A\nexport(T, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "-\\bm C^{-1}",
        "C : (-C^-1 boxtimes C^-1) must collapse to -C^-1, got: {}",
        outputs[0].latex
    );
}
