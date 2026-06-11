//! User-defined scalar functions: `Var("\lambda")` declares a function
//! argument; expressions mentioning it apply with call syntax.

use tensorforge::run_source;

/// Drop the `X = ` display left-hand side so outputs of differently named
/// variables can be compared.
fn rhs(latex: &str) -> &str {
    latex.split_once(" = ").map(|(_, r)| r).unwrap_or(latex)
}

const PRELUDE: &str = r#"
lam = Var("\lambda")
m = Scalar("m")
n = Scalar("n")
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
C = F.T*F
"#;

#[test]
fn function_application_substitutes_the_var() {
    let src = format!(
        "{PRELUDE}\nEcr = (lam^m - lam^(-n))/(m + n)\nx = Scalar(\"x\")\n\
         y = Ecr(x)\nexport(y, format=latex)"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("{x}^{m}"), "got: {latex}");
    assert!(
        !latex.contains("\\lambda"),
        "var must be substituted: {latex}"
    );
}

#[test]
fn legacy_gstrain_custom_scale_matches_builtin_cr() {
    // Legacy compatibility: the old gstrain shortcut still accepts a
    // user-defined scale and matches the built-in CR scale.
    let src = format!(
        "{PRELUDE}\nEcr = (lam^m - lam^(-n))/(m + n)\n\
         E1 = gstrain(C, scale=Ecr)\nE2 = gstrain(C, scale=CR, m=m, n=n)\n\
         W1 = mu * ddot(E1, E1)\nW2 = mu * ddot(E2, E2)\n\
         S1 = 2 * diff(W1, C)\nS2 = 2 * diff(W2, C)\n\
         display(E1, mode=spectral)\ndisplay(E2, mode=spectral)\n\
         display(S1, mode=spectral)\ndisplay(S2, mode=spectral)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        rhs(&outputs[0].latex),
        rhs(&outputs[1].latex),
        "strain displays differ"
    );
    assert_eq!(
        rhs(&outputs[2].latex),
        rhs(&outputs[3].latex),
        "stress displays differ"
    );
}

#[test]
fn legacy_gstrain_custom_hencky_scale_matches_builtin() {
    let src = format!(
        "{PRELUDE}\nEh = log(lam)\n\
         E1 = gstrain(C, scale=Eh)\nE2 = gstrain(C, scale=Hencky)\n\
         display(E1, mode=spectral)\ndisplay(E2, mode=spectral)"
    );
    let outputs = run_source(&src).unwrap();
    assert_eq!(rhs(&outputs[0].latex), rhs(&outputs[1].latex));
}

#[test]
fn applying_a_non_function_errors() {
    let src = format!("{PRELUDE}\ng = mu(m)\ndisplay(g)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("not a function"),
        "got: {}",
        err.message
    );
}

#[test]
fn diff_of_function_by_its_var() {
    // d/dλ of a user function works through the ordinary scalar diff path.
    let src = format!("{PRELUDE}\nEh = log(lam)\ndE = diff(Eh, lam)\nexport(dE, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\frac{1}{\\lambda}",
        "got: {}",
        outputs[0].latex
    );
}
