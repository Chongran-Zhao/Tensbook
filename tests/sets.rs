//! Indexed families (`ScalarSet`/`VectorSet`), element access `lambda[a]`,
//! and the `sum(body, a)` spectral-sum builtin.

use tensorforge::metadata::{DisplayCapabilityState, ValueKind};
use tensorforge::{run_source, run_source_with_env};

const PRELUDE: &str = r#"
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
"#;

const EIG_PRELUDE: &str = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
lambda = eigvals(C, "\lambda")
N = eigvecs(C, "\bm N")
"#;

#[test]
fn spectral_decomposition_written_by_hand() {
    let src = format!("{PRELUDE}\nC = sum(lambda[a] * N[a] & N[a], a)\ndisplay(C)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex,
        "\\bm C = \\sum_{a=1}^{3} {\\lambda}_{a} \\, {\\bm N}_{a} \\otimes {\\bm N}_{a}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn concrete_and_abstract_element_access() {
    let src = format!("{PRELUDE}\nx = lambda[2]\nexport(x, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "{\\lambda}_{2}");

    let src = format!("{PRELUDE}\nx = lambda[5]\nexport(x, format=latex)");
    let err = run_source(&src).unwrap_err();
    assert!(err.message.contains("out of range"), "got: {}", err.message);
}

#[test]
fn scalar_sum_uses_spec_sum() {
    let src = format!("{PRELUDE}\nI1 = sum(lambda[a], a)\nexport(I1, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\sum_{a=1}^{3} {\\lambda}_{a}");
}

#[test]
fn display_set_shows_family_and_index_range() {
    let src = format!("{PRELUDE}\ndisplay(lambda)\ndisplay(N)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex,
        "{\\lambda}_{a}\\quad \\text{with } a=1,2,3"
    );
    assert_eq!(outputs[1].latex, "{\\bm N}_{a}\\quad \\text{with } a=1,2,3");
}

#[test]
fn set_display_rejects_non_symbol_modes() {
    let src = format!("{PRELUDE}\ndisplay(lambda, mode=matrix)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message
            .contains("set display only supports mode=symbol"),
        "got: {}",
        err.message
    );
}

#[test]
fn sum_requires_the_index_in_the_body() {
    let src = format!("{PRELUDE}\nI1 = sum(lambda[a], b)\ndisplay(I1)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("does not mention"),
        "got: {}",
        err.message
    );
}

#[test]
fn hill_strain_from_function_and_sets() {
    // The user-facing composition: a scale function declared via Var,
    // applied to eigenvalue set elements, summed against eigenprojectors.
    let src = format!(
        "{PRELUDE}\nlam = Var(\"\\hat{{\\lambda}}\")\nm = Scalar(\"m\")\nn = Scalar(\"n\")\n\
         Ecr = (lam^m - lam^(-n))/(m + n)\n\
         E = sum(Ecr(lambda[a]) * N[a] & N[a], a)\ndisplay(E)"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.starts_with("\\bm E = \\sum_{a=1}^{3}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("{{\\lambda}_{a}}^{m}"),
        "scale function must be applied at lambda_a: {latex}"
    );
    assert!(
        latex.contains("{\\bm N}_{a} \\otimes {\\bm N}_{a}"),
        "got: {latex}"
    );
    // The function's bound variable must not leak into the display.
    assert!(!latex.contains("\\hat"), "got: {latex}");
}

#[test]
fn indexing_a_non_set_errors() {
    let src = "mu = Scalar(\"\\mu\")\nx = mu[1]\ndisplay(x)";
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("requires a tensor"),
        "got: {}",
        err.message
    );
}

#[test]
fn eigval_set_derivative_uses_projectors() {
    let src = format!("{EIG_PRELUDE}\nW = sum(lambda[a]^2, a)\nS = diff(W, C)\ndisplay(S)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex,
        "\\bm S = \\sum_{a=1}^{3} 2 \\, {\\lambda}_{a} \\, {\\bm N}_{a} \\otimes {\\bm N}_{a}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn eig_sets_require_symmetric_base() {
    let src = "F = Tensor(\"\\bm F\", order=2, dim=3)\nlambda = eigvals(F, \"\\lambda\")";
    let err = run_source(src).unwrap_err();
    assert!(
        err.message
            .contains("provably symmetric second-order tensor"),
        "got: {}",
        err.message
    );
}

#[test]
fn manual_spectral_strain_diff_generates_q() {
    let src = r#"
m = Scalar("m")
n = Scalar("n")
lam = Var("\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)
display(Q)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(!latex.contains("\\partial"), "got: {latex}");
    assert!(
        latex.contains("\\mathbb Q = \\sum_{a=1}^{3}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("{{\\lambda}_{a}}^{2} - {{\\lambda}_{b}}^{2}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("\\sum_{\\substack{b=1 \\\\ b\\ne a}}^{3}"),
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\frac{\\frac"),
        "nested fractions are not allowed: {latex}"
    );
    assert!(
        !latex.contains("2 \\, \\frac"),
        "off-diagonal coefficient should be flattened: {latex}"
    );
    assert!(
        latex.contains(
            "\\frac{m \\, {{\\lambda}_{a}}^{m - 1} + n \\, {{\\lambda}_{a}}^{-n - 1}}{\\left( m + n \\right) \\, {\\lambda}_{a}}"
        ),
        "diagonal coefficient should be flattened: {latex}"
    );
    assert!(
        latex.contains("{\\bm N}_{a} \\otimes {\\bm N}_{a}"),
        "got: {latex}"
    );
    assert!(!latex.contains("\\bm M_a"), "got: {latex}");
}

#[test]
fn metadata_records_manual_spectral_q_as_fourth_order_tensor() {
    let src = r#"
m = Scalar("m")
n = Scalar("n")
lam = Var("\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)
"#;
    let (_, interp) = run_source_with_env(src).unwrap();
    let q = interp.symbol_info("Q").expect("Q metadata");
    assert_eq!(q.kind, ValueKind::Tensor { order: 4, dim: 3 });
    let ch = q.characteristic.as_ref().expect("tensor characteristic");
    assert_eq!(ch.order, 4);
    assert_eq!(ch.dim, 3);
    assert!(ch.derived);
    assert!(ch.sum);
    assert!(ch.outer_like);
    assert!(ch.boxtimes_like);
    assert!(q
        .display_modes
        .iter()
        .any(|m| { m.mode == "block_components" && m.state == DisplayCapabilityState::Available }));
}

#[test]
fn manual_spectral_q_block_components_renders_for_derived_order_four_tensor() {
    let src = r#"
m = Scalar("m")
n = Scalar("n")
lam = Var("\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)
display(Q, mode=block_components)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\mathbb Q = \\begin{bmatrix}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("\\sum_{a=1}^{3}")
            && latex.contains("\\sum_{\\substack{b=1 \\\\ b\\ne a}}^{3}"),
        "got: {latex}"
    );
    assert!(
        latex.contains("{\\bm N}_{ai}") || latex.contains("{\\bm N}_{a i}"),
        "block entries should componentize vector-set factors: {latex}"
    );
    assert!(!latex.contains("not supported"), "got: {latex}");
}

#[test]
fn manual_spectral_q_supports_hencky_and_polynomial_scales() {
    let src = format!(
        "{PRELUDE}\nC = sum(lambda[a]^2 * N[a] & N[a], a)\n\
         lam = Var(\"\\lambda\")\nEh = log(lam)\nEp = lam^2\n\
         H = sum(Eh(lambda[a]) * N[a] & N[a], a)\n\
         P = sum(Ep(lambda[a]) * N[a] & N[a], a)\n\
         Qh = 2 * diff(H, C)\nQp = 2 * diff(P, C)\n\
         display(Qh)\ndisplay(Qp)"
    );
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\log {\\lambda}_{a}"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0].latex.contains("\\frac{1}{{{\\lambda}_{a}}^{2}}"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("2 \\, {\\lambda}_{a}"),
        "got: {}",
        outputs[1].latex
    );
    assert!(
        !outputs[0].latex.contains("\\partial"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        !outputs[1].latex.contains("\\partial"),
        "got: {}",
        outputs[1].latex
    );
}

#[test]
fn manual_spectral_diff_requires_matching_c_decomposition() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
lam = Var("\lambda")
Escale = lam^2
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
E = sum(Escale(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message
            .contains("matching stretch spectral decomposition"),
        "got: {}",
        err.message
    );
}

#[test]
fn manual_spectral_diff_rejects_mismatched_sets() {
    let src = r#"
lambda = ScalarSet("\lambda", dim=3)
mu = ScalarSet("\mu", dim=3)
N = VectorSet("\bm N", dim=3)
M = VectorSet("\bm M", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
E1 = sum(mu[a]^2 * N[a] & N[a], a)
Q1 = 2 * diff(E1, C)
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("same stretch set as C"),
        "got: {}",
        err.message
    );

    let src = r#"
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
M = VectorSet("\bm M", dim=3)
C = sum(lambda[a]^2 * N[a] & N[a], a)
E2 = sum(lambda[a]^2 * M[a] & M[a], a)
Q2 = 2 * diff(E2, C)
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("same vector set and dimension as C"),
        "got: {}",
        err.message
    );
}
