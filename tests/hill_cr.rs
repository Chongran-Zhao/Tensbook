//! Tests for generalized strains, spectral calculus, and the scalar
//! function library — the Hill–CR derivation chain end to end.

use tensorforge::interpreter::Value;
use tensorforge::run_source;
use tensorforge::run_source_with_env;

const PRELUDE: &str = r#"
mu = Scalar("\mu")
kappa = Scalar("\kappa")
m = Scalar("m")
n = Scalar("n")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
E = gstrain(C, scale=CR, m=m, n=n)
W = mu * ddot(E, E) + kappa/2 * tr(E)^2
"#;

// ---- scalar function library -------------------------------------------------

#[test]
fn scalar_functions_and_derivatives() {
    let src = r#"
mu = Scalar("\mu")
x = sinh(mu)
dx = diff(x, mu)
y = exp(mu)
dy = diff(y, mu)
z = sqrt(mu)
display(dx, mode=symbol)
display(dy, mode=symbol)
display(z, mode=symbol)
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
fn cr_strain_spectral_display() {
    let src = format!("{PRELUDE}\ndisplay(E, mode=spectral)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    // E = Σ_a (λ_a^m − λ_a^{−n})/(m+n) M_a
    assert!(latex.contains("\\sum_{a=1}^{3}"), "got: {latex}");
    assert!(latex.contains("{\\lambda_{a}}^{m}"), "got: {latex}");
    assert!(latex.contains("{\\lambda_{a}}^{-n}"), "got: {latex}");
    assert!(latex.contains("m + n"), "got: {latex}");
    assert!(latex.contains("\\bm M_a"), "got: {latex}");
}

#[test]
fn gstrain_is_symmetric_and_order_2() {
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
fn gstrain_requires_symmetric_base() {
    let src = r#"
m = Scalar("m")
n = Scalar("n")
F = Tensor("\bm F", order=2, dim=3)
E = gstrain(F, scale=CR, m=m, n=n)
"#;
    assert!(run_source(src).is_err());
}

#[test]
fn hencky_strain_spectral_display() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
H = gstrain(C, scale=Hencky)
display(H, mode=spectral)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("\\log \\lambda_{a}"),
        "got: {}",
        outputs[0].latex
    );
}

// ---- Hill energy: T = dW/dE --------------------------------------------------

#[test]
fn thermodynamic_force_from_quadratic_energy() {
    // T = ∂W/∂E = 2μ E + κ tr(E) I
    let src = format!("{PRELUDE}\nT = diff(W, E)\ndisplay(T, mode=symbol)");
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
    let src = format!("{PRELUDE}\nT = diff(W, E)\nS = 2 * diff(W, C)\ndisplay(S, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    // After back-substitution the symbol display reads S = T : Q.
    assert!(
        outputs[0].latex.contains("\\bm T : \\mathbb{Q}"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn second_pk_stress_spectral_formula() {
    // S_a = (E'(λ_a)/λ_a) (2μ E(λ_a) + κ Σ_b E(λ_b))
    let src = format!("{PRELUDE}\nS = 2 * diff(W, C)\ndisplay(S, mode=spectral)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    // CR derivative: m λ^{m−1} + n λ^{−n−1}
    assert!(latex.contains("{\\lambda_{a}}^{m - 1}"), "got: {latex}");
    assert!(latex.contains("{\\lambda_{a}}^{-n - 1}"), "got: {latex}");
    // the invariant tr(E) becomes a spectral sum over b
    assert!(latex.contains("\\sum_{b=1}^{3}"), "got: {latex}");
    // factors 2μ and κ present
    assert!(latex.contains("2 \\, \\mu"), "got: {latex}");
    assert!(latex.contains("\\kappa"), "got: {latex}");
}

#[test]
fn diff_by_strain_rejects_dependence_outside_strain() {
    // W depends on F through det(F) outside E: must error, not return 0.
    let src = format!("{PRELUDE}\nW2 = W + log(det(F))\nT = diff(W2, E)");
    assert!(run_source(&src).is_err());
}

#[test]
fn hill_cr_example_runs_end_to_end() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/hill_cr.tens"
    ))
    .unwrap();
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs.len(), 7);
    assert!(outputs.iter().all(|o| o.error.is_none()));
}
