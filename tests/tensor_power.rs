//! Integer matrix power `A^n` and the display of `A*A` as `A²`.

use tensorforge::run_source;

const PRELUDE: &str = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
"#;

#[test]
fn product_of_equal_tensors_shows_as_square() {
    // C*C renders C² but stays a product internally (so diff still works).
    let src = format!("{PRELUDE}\nT = C * C\nT.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm T = {\\bm C}^{2}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn explicit_power_input() {
    let src = format!("{PRELUDE}\nT = C^3\nT.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm T = {\\bm C}^{3}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn power_one_folds_to_base() {
    let src = r#"
G = Tensor("\bm G", order=2, dim=3)
T = G^1
T.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm T = \\bm G",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn trace_of_square_differentiates() {
    // d Tr(C C)/dC = C + C = 2C, whether written C*C or C^2.
    let src = format!("{PRELUDE}\nW = Tr(C*C)\nS = Diff(W, C)\nS.show()");
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
    let outputs = run_source(&format!("{src}\nT.show()")).unwrap();
    assert_eq!(outputs[0].latex, "\\bm T = {\\bm G}^{2}");
}
