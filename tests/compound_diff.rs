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
    let src = format!(
        "{PRELUDE}\nW = mu/2 * (I1 - 3)\nS = 2 * diff(W, C)\ndisplay(S, mode=symbol)"
    );
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
    let src = format!(
        "{PRELUDE}\nW = log(det(C))\nS = diff(W, C)\nexport(S, format=latex)"
    );
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
        latex.contains("\\frac{\\mu}{2}") && latex.contains("^{-\\mathsf{T}}"),
        "log J term must give mu/2 C^-T: {latex}"
    );
    // The original variable F must not appear bare via det(F):
    assert!(!latex.contains("\\det \\bm F"), "det F must be rewritten: {latex}");
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
    let src = format!(
        "{PRELUDE}\nW = mu/2 * (I1 - 3)\nG = C : diff(W, C)\ndisplay(G, mode=symbol)"
    );
    let outputs = run_source(&src).unwrap();
    assert!(outputs[0].latex.contains(" : "), "got: {}", outputs[0].latex);
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
    let src = format!(
        "{PRELUDE}\nW = mu/2 * (I1 - 3)\na = diff(W, mu)\ndisplay(a, mode=symbol)"
    );
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(matches!(interp.get("a"), Some(Value::Scalar(_))));
    let latex = &outputs[0].latex;
    // Display back-substitutes I1 = tr(C): the result reads (I1 - 3)/2.
    assert!(
        latex.contains("I1") && latex.contains("- 3"),
        "got: {latex}"
    );
    assert!(!latex.contains("\\mu"), "mu must not survive d/dmu: {latex}");
}

#[test]
fn diff_by_scalar_log_term() {
    // ∂(mu log J)/∂mu = log J (J back-substituted in display)
    let src = format!(
        "{PRELUDE}\nJ = det(F)\nW = mu * log(J)\na = diff(W, mu)\nexport(a, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\log J");
}

#[test]
fn diff_by_scalar_of_independent_expr_is_zero() {
    let src = format!(
        "{PRELUDE}\nW = lambda * (I1 - 3)\na = diff(W, mu)\nexport(a, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "0");
}
