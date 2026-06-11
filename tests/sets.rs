//! Indexed families (`ScalarSet`/`VectorSet`), element access `lambda[a]`,
//! and the `sum(body, a)` spectral-sum builtin.

use tensorforge::run_source;

const PRELUDE: &str = r#"
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)
"#;

#[test]
fn spectral_decomposition_written_by_hand() {
    let src = format!("{PRELUDE}\nC = sum(lambda[a] * outer(N[a], N[a]), a)\ndisplay(C)");
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
         E = sum(Ecr(lambda[a]) * outer(N[a], N[a]), a)\ndisplay(E)"
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
        err.message.contains("not a declared set"),
        "got: {}",
        err.message
    );
}
