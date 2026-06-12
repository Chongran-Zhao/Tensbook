//! User-defined scalar functions: `Var("\lambda")` declares a function
//! argument; expressions mentioning it apply with call syntax.

use tensorforge::run_source;

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
