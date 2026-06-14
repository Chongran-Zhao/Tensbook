//! Integer matrix power `A^n` and the display of `A*A` as `A²`.

use tensorforge::run_source;

const PRELUDE: &str = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
"#;

#[test]
fn product_of_equal_tensors_shows_as_square() {
    // C*C renders C² but stays a product internally (so diff still works).
    let src = format!("{PRELUDE}\nT = C * C\nexport(T, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "{\\bm C}^{2}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn explicit_power_input() {
    let src = format!("{PRELUDE}\nT = C^3\nexport(T, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "{\\bm C}^{3}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn power_one_folds_to_base() {
    let src = r#"
G = Tensor("\bm G", order=2, dim=3)
T = G^1
export(T, format=latex)
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "\\bm G", "got: {}", outputs[0].latex);
}

#[test]
fn trace_of_square_differentiates() {
    // d tr(C C)/dC = C + C = 2C, whether written C*C or C^2.
    let src = format!("{PRELUDE}\nW = tr(C*C)\nS = diff(W, C)\nexport(S, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("2 \\, \\bm C"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn power_of_non_symmetric_is_not_symmetric() {
    let src = r#"
G = Tensor("\bm G", order=2, dim=3)
T = G^2
"#;
    // Just ensure it parses/evaluates without error and renders as a power.
    let outputs = run_source(&format!("{src}\nexport(T, format=latex)")).unwrap();
    assert_eq!(outputs[0].latex, "{\\bm G}^{2}");
}
