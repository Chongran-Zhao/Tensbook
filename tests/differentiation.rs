//! Phase-2 tests: symbolic differentiation.
//!
//! Covers: dJ/dF = J F^{-T}; P = ∂W/∂F for neo-Hookean W;
//! ∂C_ij/∂F_mn component formula with Kronecker deltas; ∂tr(C)/∂F = 2F;
//! order/shape bookkeeping of Diff nodes; unsupported-form errors.

use tensorforge::interpreter::Value;
use tensorforge::tensor::TensorExpr;
use tensorforge::{run_source, run_source_with_env};

const PRELUDE: &str = r#"
mu = Scalar("\mu")
lambda = Scalar("\lambda")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
J = det(F)
I1 = tr(C)
"#;

#[test]
fn djdf_is_j_f_inverse_transpose() {
    let src = format!("{PRELUDE}\ndJdF = diff(J, F)\ndisplay(dJdF, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    // det F is back-substituted to J in the display.
    assert!(
        outputs[0].latex.contains("J \\, \\bm F^{-\\mathsf{T}}"),
        "expected J F^-T, got: {}",
        outputs[0].latex
    );
}

#[test]
fn d_trace_c_by_f_is_2f() {
    let src = format!("{PRELUDE}\ndI1dF = diff(I1, F)\ndisplay(dI1dF, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("2 \\, \\bm F"),
        "expected 2F, got: {}",
        outputs[0].latex
    );
}

#[test]
fn first_piola_stress_neo_hookean() {
    // P = ∂W/∂F = mu F − mu F^{-T} + lambda log(det F) F^{-T}
    let src = format!(
        "{PRELUDE}\nW = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2\n\
         P = diff(W, F)\nexport(P, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\mu \\, \\left( \\bm F - \\bm F^{-\\mathsf{T}} \\right)"),
        "missing factored mu(F - F^-T) in: {latex}"
    );
    assert!(
        latex.contains("\\lambda \\, \\log J \\, \\bm F^{-\\mathsf{T}}"),
        "missing lambda log(J) F^-T in: {latex}"
    );
}

#[test]
fn p_is_a_second_order_tensor() {
    let src = format!(
        "{PRELUDE}\nW = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2\nP = diff(W, F)"
    );
    let (_, interp) = run_source_with_env(&src).unwrap();
    match interp.get("P") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 2);
            assert_eq!(t.dim(), 3);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn dcdf_is_order_4() {
    let src = format!("{PRELUDE}\ndCdF = diff(C, F)");
    let (_, interp) = run_source_with_env(&src).unwrap();
    match interp.get("dCdF") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 4, "∂C/∂F must be a fourth-order tensor");
            assert_eq!(t.dim(), 3);
            assert!(matches!(&**t, TensorExpr::Diff { .. }));
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn dcdf_component_formula_has_deltas() {
    let src = format!("{PRELUDE}\ndCdF = diff(C, F)\ndisplay(dCdF, mode=components)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.starts_with("\\frac{\\partial C_{ij}}{\\partial F_{mn}} ="),
        "got: {latex}"
    );
    // ∂C_ij/∂F_mn = δ_in F_mj + δ_jn F_mi  (i.e. δ F + δ F, two delta terms)
    assert_eq!(latex.matches("\\delta_").count(), 2, "got: {latex}");
    assert!(
        latex.contains("\\delta_{in} F_{mj}") && latex.contains("\\delta_{jn} F_{mi}"),
        "got: {latex}"
    );
}

#[test]
fn dcdf_symbol_mode_renders_partial_fraction() {
    let src = format!("{PRELUDE}\ndCdF = diff(C, F)\ndisplay(dCdF, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0]
            .latex
            .contains("\\frac{\\partial \\bm C}{\\partial \\bm F}"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn diff_of_independent_scalar_is_zero() {
    let src = format!("{PRELUDE}\nZ = diff(mu, F)\ndisplay(Z, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(outputs[0].latex.contains('0'), "got: {}", outputs[0].latex);
}

#[test]
fn diff_errors_are_reported() {
    // det F is rewritable through C2 = FᵀF (det C = (det F)²), so
    // diff(J, C2) is now legal: ∂(det C)^{1/2}/∂C = J/2 C^{-T}.
    let src = format!("{PRELUDE}\nC2 = F.T * F\nX = diff(J, C2)\nexport(X, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("^{-\\mathsf{T}}"),
        "got: {}",
        outputs[0].latex
    );

    // but tr(F) has no rewriting through C: hidden dependence is an error
    let src = format!("{PRELUDE}\nC2 = F.T * F\nX = diff(tr(F), C2)");
    assert!(run_source(&src).is_err());

    // scalar denominator must be a declared symbol, not a compound scalar
    let src = format!("{PRELUDE}\nX = diff(J, mu * mu)");
    assert!(run_source(&src).is_err());

    // diff(J, mu) is now legal and is identically zero
    let src = format!("{PRELUDE}\nX = diff(J, mu)\nexport(X, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "0");
}
