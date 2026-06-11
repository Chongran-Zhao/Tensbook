//! Tests for user-reported gaps: compound-denominator diff (diff(W, C)),
//! the `:` operator, otimes alias, scalar-by-scalar diff.

use tensorforge::interpreter::Value;
use tensorforge::{run_source, run_source_with_env};

const PRELUDE: &str = r#"
mu = Scalar("\mu")
lambda = Scalar("\lambda")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
I1 = tr(C)
"#;

#[test]
fn diff_by_compound_tensor() {
    // S = 2 ∂W/∂C with W written in terms of C: ∂tr(C)/∂C = I.
    let src =
        format!("{PRELUDE}\nW = mu/2 * (I1 - 3)\nS = 2 * diff(W, C)\ndisplay(S, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm I"),
        "dtr(C)/dC must give I, got: {}",
        outputs[0].latex
    );
}

#[test]
fn diff_by_compound_log_det() {
    // ∂log(det C)/∂C = C^{-T}
    let src = format!("{PRELUDE}\nW = log(det(C))\nS = diff(W, C)\nexport(S, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("^{-\\mathsf{T}}"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn diff_by_compound_rejects_hidden_dependence() {
    // W = tr(F) depends on F in a way not expressible through C: must be
    // rejected rather than silently returning 0.
    let src = format!("{PRELUDE}\nW = tr(F)\nS = diff(W, C)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("compound"),
        "expected hidden-dependence error, got: {}",
        err.message
    );
}

#[test]
fn det_f_rewritten_through_c() {
    // The ubiquitous pattern: W(J) with J = det F, differentiated by
    // C = FᵀF. det F is rewritten via det C = (det F)²:
    // ∂(-mu log J)/∂C = ∂(-mu/2 log det C)/∂C = -mu/2 C^{-T}.
    let src = format!(
        "{PRELUDE}\nJ = det(F)\nW = mu/2 * (I1 - 3) - mu * log(J)\n\
         S = 2 * diff(W, C)\nexport(S, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\bm I"), "tr term must give I: {latex}");
    assert!(
        latex.contains("\\mu \\, \\left( \\bm I - \\bm C^{-\\mathsf{T}} \\right)"),
        "S should simplify to mu(I - C^-T): {latex}"
    );
    // The original variable F must not appear bare via det(F):
    assert!(
        !latex.contains("\\det \\bm F"),
        "det F must be rewritten: {latex}"
    );
}

#[test]
fn full_neo_hookean_second_piola() {
    // S = 2 ∂W/∂C for the full neo-Hookean energy written with J = det F.
    let src = format!(
        "{PRELUDE}\nJ = det(F)\n\
         W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2\n\
         S = 2 * diff(W, C)\nexport(S, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    // λ/2 (log J)² = λ/8 (log det C)² → 2·∂/∂C gives λ/4 log(det C) C^{-1}
    assert!(
        outputs[0].latex.contains("}{4}"),
        "lambda term coefficient must be /4: {}",
        outputs[0].latex
    );
}

#[test]
fn colon_operator_is_double_contraction() {
    let src = format!("{PRELUDE}\ng = C : C\ndisplay(g, mode=symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains(" : "),
        "got: {}",
        outputs[0].latex
    );
    assert!(matches!(interp.get("g"), Some(Value::Scalar(_))));
}

#[test]
fn colon_with_diff_result() {
    // G = C : ∂W/∂C — the user's reported case.
    let src =
        format!("{PRELUDE}\nW = mu/2 * (I1 - 3)\nG = C : diff(W, C)\ndisplay(G, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains(" : "),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn tensor_by_compound_denominator_uses_synthetic_base() {
    let src = format!("{PRELUDE}\nD = diff(C, C)\ndisplay(D, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\mathbb{I}"), "got: {latex}");
    assert!(
        !latex.contains("F_"),
        "C must be treated as an independent base: {latex}"
    );
}

#[test]
fn tensor_by_compound_rejects_hidden_dependence() {
    let src = format!("{PRELUDE}\nD = diff(F, C)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("compound expression"),
        "expected hidden-dependence error, got: {}",
        err.message
    );
}

#[test]
fn isochoric_cbar_chain_rule_runs() {
    let src = r#"
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
J = det(F)
C = Tensor("\bm C", order=2, dim=3)
C = F.T * F
C_bar = Tensor("\bar{\bm C}", order=2, dim=3)
C_bar = J^(-(2/3)) * C
I_1_bar = Scalar("\bar{I}_1")
I_1_bar = tr(C_bar)
Psi = Scalar("\Psi")
Psi = 1/2 * mu * I_1_bar
S = Tensor("\bm S", order=2, dim=3)
S = 2 * diff(Psi, C_bar) : diff(C_bar, C)
display(S, mode=symbol)
"#;
    let (outputs, interp) = run_source_with_env(src).unwrap();
    match interp.get("S") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 2);
            assert_eq!(t.dim(), 3);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\mu"), "got: {latex}");
    assert!(
        latex.contains("\\bm C^{-\\mathsf{T}}") && latex.contains("\\operatorname{tr} \\bm C"),
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\partial"),
        "S should be fully expanded: {latex}"
    );
}

#[test]
fn isochoric_stress_direct_diff_by_c_runs_chain_rule() {
    let src = r#"
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
J = det(F)
C = Tensor("\bm C", order=2, dim=3)
C = F.T * F
C_bar = Tensor("\bar{\bm C}", order=2, dim=3)
C_bar = J^(-(2/3)) * C
I_1_bar = Scalar("\bar{I}_1")
I_1_bar = tr(C_bar)
Psi = Scalar("\Psi")
Psi = 1/2 * mu * I_1_bar
S = Tensor("\bm S", order=2, dim=3)
S = 2 * diff(Psi, C)
display(S, mode=symbol)
"#;
    let (outputs, interp) = run_source_with_env(src).unwrap();
    match interp.get("S") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 2);
            assert_eq!(t.dim(), 3);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\mu"), "got: {latex}");
    assert!(
        latex.contains("\\bm C^{-\\mathsf{T}}") && latex.contains("\\operatorname{tr} \\bm C"),
        "got: {latex}"
    );
    assert!(
        latex.contains("\\mu \\, {J}^{-\\frac{2}{3}} \\, \\left("),
        "J^(-2/3) should be factored once: {latex}"
    );
    assert!(
        latex.contains("\\left( \\operatorname{tr} \\bm C \\right) \\, \\bm C^{-\\mathsf{T}}"),
        "trace coefficient should be grouped before tensor factor: {latex}"
    );
    assert_eq!(
        latex.matches("{J}^{-\\frac{2}{3}}").count(),
        1,
        "got: {latex}"
    );
    assert!(!latex.contains("J \\, J"), "got: {latex}");
    assert!(!latex.contains("+ -"), "got: {latex}");
    assert!(
        !latex.contains("\\partial"),
        "direct diff(Psi, C) should apply the chain rule: {latex}"
    );
}

#[test]
fn isochoric_cbar_derivative_symbol_includes_projection_terms() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
J = det(F)
C = Tensor("\bm C", order=2, dim=3)
C = F.T * F
C_bar = Tensor("\bar{\bm C}", order=2, dim=3)
C_bar = J^(-(2/3)) * C
D = diff(C_bar, C)
display(D, mode=symbol)
"#;
    let (outputs, interp) = run_source_with_env(src).unwrap();
    match interp.get("D") {
        Some(Value::Tensor(t)) => assert_eq!(t.order(), 4),
        other => panic!("expected Tensor Diff, got {other:?}"),
    }
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\mathbb D ="), "got: {latex}");
    assert!(latex.contains("\\mathbb{I}"), "got: {latex}");
    assert!(latex.contains("\\otimes"), "got: {latex}");
    assert!(
        latex.contains("\\bm C^{-\\mathsf{T}}") && latex.contains("{J}^{-\\frac{2}{3}}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("{J}^{-\\frac{2}{3}} \\, \\left(")
            && latex.contains("\\frac{1}{3} \\, \\bm C \\otimes"),
        "got: {latex}"
    );
    assert_eq!(
        latex.matches("{J}^{-\\frac{2}{3}}").count(),
        1,
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\frac{{J}^{-\\frac{2}{3}}}{3}"),
        "got: {latex}"
    );
    assert!(!latex.contains("\\det \\bm C"), "got: {latex}");
    assert!(
        !latex.contains("\\partial"),
        "P should be fully expanded: {latex}"
    );
}

#[test]
fn cauchy_green_determinant_displays_through_jacobian() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
J = det(F)
C = Tensor("\bm C", order=2, dim=3)
C = F.T * F
display(det(C), mode=symbol)
display(det(C)^(-1/3), mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("{J}^{2}"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("{J}^{-\\frac{2}{3}}"),
        "got: {}",
        outputs[1].latex
    );
    assert!(
        !outputs[1].latex.contains("\\det \\bm C"),
        "got: {}",
        outputs[1].latex
    );
}

#[test]
fn negative_fraction_exponent_renders_sign_outside_fraction() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
J = det(F)
C = Tensor("\bm C", order=2, dim=3)
C = F.T * F
C_bar = Tensor("\bar{\bm C}", order=2, dim=3)
C_bar = J^(-2/3) * C
display(C_bar, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("{J}^{-\\frac{2}{3}}"), "got: {latex}");
    assert!(!latex.contains("\\frac{-2}{3}"), "got: {latex}");
}

#[test]
fn colon_rejects_scalar_operand() {
    let src = format!("{PRELUDE}\nx = mu : C");
    assert!(run_source(&src).is_err());
}

#[test]
fn otimes_is_outer_alias() {
    let src = format!("{PRELUDE}\nO = otimes(F, F)\ndisplay(O, mode=symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F \\otimes \\bm F"),
        "got: {}",
        outputs[0].latex
    );
    match interp.get("O") {
        Some(Value::Tensor(t)) => assert_eq!(t.order(), 4),
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn diff_by_scalar() {
    // ∂W/∂mu for W = mu/2 (I1 - 3): the coefficient (I1 - 3)/2 survives.
    let src = format!("{PRELUDE}\nW = mu/2 * (I1 - 3)\na = diff(W, mu)\ndisplay(a, mode=symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(matches!(interp.get("a"), Some(Value::Scalar(_))));
    let latex = &outputs[0].latex;
    // Display back-substitutes I1 = tr(C): the result reads (I1 - 3)/2.
    assert!(
        latex.contains("I1") && latex.contains("- 3"),
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\mu"),
        "mu must not survive d/dmu: {latex}"
    );
}

#[test]
fn diff_by_scalar_log_term() {
    // ∂(mu log J)/∂mu = log J (J back-substituted in display)
    let src =
        format!("{PRELUDE}\nJ = det(F)\nW = mu * log(J)\na = diff(W, mu)\nexport(a, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\log J");
}

#[test]
fn diff_by_scalar_of_independent_expr_is_zero() {
    let src = format!("{PRELUDE}\nW = lambda * (I1 - 3)\na = diff(W, mu)\nexport(a, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "0");
}
