//! Tests for the hand-written spectral Hill–CR derivation chain end to end.

use tensorforge::interpreter::Value;
use tensorforge::run_source;
use tensorforge::run_source_with_env;

const PRELUDE: &str = r#"
mu = Scalar("\mu")
kappa = Scalar("\kappa")
m = Scalar("m")
n = Scalar("n")
F = Tensor("\bm F", order=2, dim=3)
lam = Var("\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
C = Sum(lambda[a]^2 * N[a] & N[a], a)
E = Sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * Diff(E, C)
W = mu * (E : E) + kappa/2 * Tr(E)^2
"#;

// ---- scalar function library -------------------------------------------------

#[test]
fn scalar_functions_and_derivatives() {
    let src = r#"
mu = Scalar("\mu")
x = sinh(mu)
dx = Diff(x, mu)
y = exp(mu)
dy = Diff(y, mu)
z = sqrt(mu)
dx.show(symbol)
dy.show(symbol)
z.show(symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("\\cosh \\mu"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("e^{\\mu}"),
        "got: {}",
        outputs[1].latex
    );
    assert!(
        outputs[2].latex.contains("\\sqrt{\\mu}"),
        "got: {}",
        outputs[2].latex
    );
}

// ---- generalized strain ------------------------------------------------------

#[test]
fn cr_strain_symbol_display() {
    let src = format!("{PRELUDE}\nE.show(symbol)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    // E = Σ_a (λ_a^m − λ_a^{−n})/(m+n) N_a ⊗ N_a
    assert!(latex.contains("\\sum_{a=1}^{3}"), "got: {latex}");
    assert!(latex.contains("{{\\lambda}_{a}}^{m}"), "got: {latex}");
    assert!(latex.contains("{{\\lambda}_{a}}^{-n}"), "got: {latex}");
    assert!(latex.contains("m + n"), "got: {latex}");
    assert!(
        latex.contains("{\\bm N}_{a} \\otimes {\\bm N}_{a}"),
        "got: {latex}"
    );
}

#[test]
fn manual_strain_is_symmetric_and_order_2() {
    let (_, interp) = run_source_with_env(PRELUDE).unwrap();
    match interp.get("E") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 2);
            assert!(t.is_symmetric());
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn q_is_a_real_fourth_order_expression() {
    let src = format!("{PRELUDE}\nQ.show(symbol)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\mathbb Q = \\sum_{a=1}^{3}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("\\sum_{\\substack{b=1 \\\\ b\\ne a}}^{3}"),
        "got: {latex}"
    );
    assert!(!latex.contains("\\partial"), "got: {latex}");
}

#[test]
fn hencky_strain_symbol_display() {
    let src = r#"
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
H = Sum(log(lambda[a]) * N[a] & N[a], a)
H.show(symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("\\log {\\lambda}_{a}"),
        "got: {}",
        outputs[0].latex
    );
}

// ---- Hill energy: T = dW/dE --------------------------------------------------

#[test]
fn thermodynamic_force_from_quadratic_energy() {
    // T = ∂W/∂E = 2μ E + κ Tr(E) I
    let src = format!("{PRELUDE}\nT = Diff(W, E)\nT.show(symbol)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("2 \\, \\mu \\, \\bm E"), "got: {latex}");
    assert!(
        latex.contains("\\kappa \\, \\left( \\operatorname{tr} \\bm E \\right) \\, \\bm I"),
        "got: {latex}"
    );
}

// ---- stress: S = T : Q --------------------------------------------------------

#[test]
fn second_pk_stress_is_t_double_dot_q() {
    let src = format!("{PRELUDE}\nT = Diff(W, E)\nS = T : Q\nS.show(symbol)");
    let outputs = run_source(&src).unwrap();
    // The contraction distributes through the real fourth-order Q expression.
    assert!(
        outputs[0].latex.contains("\\sum_{a=1}^{3}"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0].latex.contains("2 \\, \\mu \\, \\bm E"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0]
            .latex
            .contains("\\kappa \\, \\left( \\operatorname{tr} \\bm E \\right) \\, \\bm I"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0].latex.contains("b\\ne a"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn strain_diff_by_c_generates_q() {
    let src = format!("{PRELUDE}\nQ = 2 * Diff(E, C)\nQ.show(symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        !outputs[0].latex.contains("\\partial"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0].latex.contains("b\\ne a"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn second_pk_stress_symbol_formula() {
    // S_a = (E'(λ_a)/λ_a) (2μ E(λ_a) + κ Σ_b E(λ_b))
    let src = format!("{PRELUDE}\nT = Diff(W, E)\nS = T : Q\nS.show(symbol)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    // CR derivative: m λ^{m−1} + n λ^{−n−1}
    assert!(latex.contains("{{\\lambda}_{a}}^{m - 1}"), "got: {latex}");
    assert!(latex.contains("{{\\lambda}_{a}}^{-n - 1}"), "got: {latex}");
    assert!(latex.contains("2 \\, \\mu \\, \\bm E"), "got: {latex}");
    assert!(
        latex.contains("\\kappa \\, \\left( \\operatorname{tr} \\bm E \\right) \\, \\bm I"),
        "got: {latex}"
    );
    assert!(latex.contains("b\\ne a"), "got: {latex}");
}

#[test]
fn shipped_example_runs_end_to_end() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/start.tens"))
        .unwrap();
    let outputs = run_source(&src).unwrap();
    assert!(!outputs.is_empty());
    assert!(outputs.iter().all(|o| o.error.is_none()));
    assert!(outputs.iter().any(|o| o.header.starts_with("S.show")));
}
