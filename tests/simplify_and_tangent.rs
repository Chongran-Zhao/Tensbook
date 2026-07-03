//! Phase 3–4 tests: simplify rule sets, fourth-order material tangent,
//! block_components display.

use tensbook::run_source;

const PRELUDE: &str = r#"
mu = Scalar("\mu")
lambda = Scalar("\lambda")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
J = Det(F)
I1 = Tr(C)
W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2
"#;

// ---- simplify ---------------------------------------------------------------

#[test]
fn simplify_inverse_cancellation() {
    // F^{-T} Fᵀ → I  (continuum rule from the spec)
    let src = format!("{PRELUDE}\nX = Simplify(Inv(F.T) * F.T, rules=tensor)\nX.show(symbol)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm X = \\bm I",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn simplify_double_transpose() {
    let src = format!("{PRELUDE}\nX = Simplify((F.T).T)\nX.show(symbol)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\bm X = \\bm F");
}

#[test]
fn simplify_transpose_of_symmetric() {
    // Cᵀ → C because C = FᵀF is provably symmetric
    let src = format!("{PRELUDE}\nX = Simplify(C.T)\nX.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\bm X = \\bm F^{\\mathsf{T}} \\, \\bm F");
}

#[test]
fn simplify_det_transpose_and_tr_identity() {
    let src = format!(
        "{PRELUDE}\nI = Tensor(\"\\bm I\", order=2, dim=3, identity=true)\n\
         a = Simplify(Det(F.T), rules=continuum)\n\
         b = Simplify(Tr(I), rules=continuum)\n\
         a.show(symbol)\nb.show(symbol)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "a = \\det \\bm F");
    assert_eq!(outputs[1].latex, "b = 3");
}

#[test]
fn simplify_trace_cyclic() {
    // Tr(AB) and Tr(BA) canonicalize to the same expression
    let src_ab = format!(
        "{PRELUDE}\nG = Tensor(\"\\bm G\", order=2, dim=3)\n\
         x = Simplify(Tr(F * G), rules=continuum)\nx.show()"
    );
    let src_ba = format!(
        "{PRELUDE}\nG = Tensor(\"\\bm G\", order=2, dim=3)\n\
         x = Simplify(Tr(G * F), rules=continuum)\nx.show()"
    );
    let a = run_source(&src_ab).unwrap();
    let b = run_source(&src_ba).unwrap();
    assert_eq!(a[0].latex, b[0].latex);
}

#[test]
fn simplify_scalar_power_quotient() {
    let src = "x = Var(\"x\")\nq = Simplify(x/x^2)\nq.show()";
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("{x}^{-1}") || latex.contains("\\frac{1}{x}"),
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\frac{x}{{x}^{2}}"),
        "quotient should be simplified: {latex}"
    );
}

// ---- material tangent A = ∂P/∂F --------------------------------------------

#[test]
fn tangent_component_formula() {
    let src = format!("{PRELUDE}\nP = Diff(W, F)\nA = Diff(P, F)\nA.show(components)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.starts_with("\\frac{\\partial P_{ij}}{\\partial F_{mn}} ="),
        "got: {latex}"
    );
    // A_{iJmN} = μ δ_im δ_jn + μ F⁻¹_jm F⁻¹_ni − λ log(det F) F⁻¹_jm F⁻¹_ni
    //            + λ F⁻¹_ji F⁻¹_nm
    assert!(
        latex.contains("\\mu \\delta_{im} \\delta_{jn}"),
        "missing mu delta delta in: {latex}"
    );
    assert!(
        latex.contains("\\mu F^{-1}_{jm} F^{-1}_{ni}"),
        "missing mu F^-1 F^-1 in: {latex}"
    );
    assert!(
        latex.contains("\\lambda F^{-1}_{ji} F^{-1}_{nm}"),
        "missing lambda F^-1_ji F^-1_nm in: {latex}"
    );
}

#[test]
fn tangent_symbol_mode() {
    let src = format!("{PRELUDE}\nP = Diff(W, F)\nA = Diff(P, F)\nA.show(symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0]
            .latex
            .contains("\\frac{\\partial \\bm P}{\\partial \\bm F}"),
        "got: {}",
        outputs[0].latex
    );
}

// ---- block_components --------------------------------------------------------

#[test]
fn dcdf_block_components_is_3x3_of_blocks() {
    let src = format!("{PRELUDE}\ndCdF = Diff(C, F)\ndCdF.show(block_components)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.starts_with("\\frac{\\partial C_{ij}}{\\partial F_{kL}}"),
        "got: {latex}"
    );
    assert!(latex.contains("\\begin{bmatrix}"));
    // 3x3 grid: 2 row separators, 6 column separators, 9 blocks
    assert_eq!(latex.matches("\\\\").count(), 2, "got: {latex}");
    assert_eq!(latex.matches('&').count(), 6, "got: {latex}");
    assert_eq!(latex.matches("\\right]_{ij}").count(), 9, "got: {latex}");
    // block (k=1, L=1): δ_i1 F_1j + δ_j1 F_1i
    assert!(
        latex.contains("\\delta_{i1} F_{1j} + \\delta_{j1} F_{1i}"),
        "got: {latex}"
    );
    // block (k=3, L=2): δ_i2 F_3j + δ_j2 F_3i
    assert!(
        latex.contains("\\delta_{i2} F_{3j} + \\delta_{j2} F_{3i}"),
        "got: {latex}"
    );
}

#[test]
fn block_components_accepts_scaled_derivative_tensor() {
    let src = format!("{PRELUDE}\nA = 2 * Diff(C, F)\nA.show(block_components)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\mathbb A = \\begin{bmatrix}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("2 \\, \\frac{\\partial"),
        "scaled derivative should not be rejected as non-diff: {latex}"
    );
}

#[test]
fn block_components_rejects_non_derivative() {
    let src = format!("{PRELUDE}\nC.show(block_components)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message
            .contains("only available for order-4 tensors; this tensor has order 2"),
        "got: {}",
        err.message
    );
}
