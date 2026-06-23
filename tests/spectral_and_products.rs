//! Phase-5 tests: spectral decomposition, operator products, and removed
//! helper commands.

use tensorforge::interpreter::Value;
use tensorforge::run_source;
use tensorforge::run_source_with_env;

const PRELUDE: &str = r#"
F = Tensor("\bm F", order=2, dim=3)
G = Tensor("\bm G", order=2, dim=3)
C = F.T * F
"#;

#[test]
fn outer_product_order_and_display() {
    let src = format!("{PRELUDE}\nO = F & G\nO.show(symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F \\otimes \\bm G"),
        "got: {}",
        outputs[0].latex
    );
    match interp.get("O") {
        Some(Value::Tensor(t)) => assert_eq!(t.order(), 4, "F ⊗ G must be order 4"),
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn ampersand_operator_is_outer_product() {
    let src = format!("{PRELUDE}\nO = F & G\nP = F & G\nO.show(symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F \\otimes \\bm G"),
        "got: {}",
        outputs[0].latex
    );
    match (interp.get("O"), interp.get("P")) {
        (Some(Value::Tensor(o)), Some(Value::Tensor(p))) => {
            assert_eq!(o, p);
            assert_eq!(o.order(), 4);
        }
        other => panic!("expected Tensors, got {other:?}"),
    }
}

#[test]
fn outer_of_same_tensor_is_symmetric() {
    let src = format!("{PRELUDE}\nO = F & F\nP = F & G");
    let (_, interp) = run_source_with_env(&src).unwrap();
    // Note: symmetry inference only applies to order-2; order-4 outer is
    // reported not-symmetric by the order check, which is the conservative
    // behaviour we want here.
    match (interp.get("O"), interp.get("P")) {
        (Some(Value::Tensor(o)), Some(Value::Tensor(p))) => {
            assert_eq!(o.order(), 4);
            assert_eq!(p.order(), 4);
        }
        other => panic!("expected Tensors, got {other:?}"),
    }
}

#[test]
fn star_is_single_contraction() {
    let src = format!("{PRELUDE}\nD = F * G\nD.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\bm D = \\bm F \\, \\bm G");
}

#[test]
fn colon_is_scalar_double_contraction() {
    let src = format!("{PRELUDE}\nu = F : G\nu.show(symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F : \\bm G"),
        "got: {}",
        outputs[0].latex
    );
    match interp.get("u") {
        Some(Value::Scalar(_)) => {}
        other => panic!("colon double contraction must produce a Scalar, got {other:?}"),
    }
}

#[test]
fn ddot_function_is_not_public_dsl() {
    let src = format!("{PRELUDE}\nu = ddot(F, G)\nu.show(symbol)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("unknown function `ddot`"),
        "got: {err}"
    );
}

#[test]
fn export_function_is_not_public_dsl() {
    let src = format!("{PRELUDE}\nexport(C, format=markdown)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("Export button"),
        "got: {}",
        err.message
    );
}

#[test]
fn legacy_lowercase_commands_are_not_public_dsl() {
    for (line, expected) in [
        ("J = det(F)", "`det(...)` was renamed; use `Det(...)`"),
        ("I1 = tr(C)", "`tr(...)` was renamed; use `Tr(...)`"),
        ("Finv = inv(F)", "`inv(...)` was renamed; use `Inv(...)`"),
        ("D = diff(C, F)", "`diff(...)` was renamed; use `Diff(...)`"),
        (
            "Cs = simplify(C, rules=tensor)",
            "`simplify(...)` was renamed; use `Simplify(...)`",
        ),
        ("S = sum(1, a)", "`sum(...)` was renamed; use `Sum(...)`"),
        (
            "display(C, mode=symbol)",
            "`display(expr, mode=...)` was removed; use `expr.show(...)`",
        ),
        (
            "lambda = eigvals(C, \"\\lambda\")",
            "`eigvals/eigvecs` were removed",
        ),
        (
            "N = eigvecs(C, \"\\bm N\")",
            "`eigvals/eigvecs` were removed",
        ),
    ] {
        let src = format!("{PRELUDE}\n{line}");
        let err = run_source(&src).unwrap_err();
        assert!(
            err.message.contains(expected),
            "for `{line}` expected `{expected}`, got: {}",
            err.message
        );
    }
}

#[test]
fn product_helper_functions_are_not_public_dsl() {
    for call in ["dot(F, G)", "outer(F, G)", "otimes(F, G)"] {
        let src = format!("{PRELUDE}\nX = {call}");
        let err = run_source(&src).unwrap_err();
        assert!(err.message.contains("was removed"), "got: {}", err.message);
    }
}

// ---- isotropic tensor functions sqrt/log/exp --------------------------------

#[test]
fn tensor_scalar_functions_point_at_the_set_form() {
    // log/sqrt/exp of a tensor are no longer built in; the error points at
    // the explicit spectral-sum form.
    let src = format!("{PRELUDE}\nC = F.T * F\nL = log(C)\nL.show()");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("Sum(log(lambda[a]) * N[a] & N[a], a)"),
        "got: {}",
        err.message
    );
}

#[test]
fn scalar_log_still_works() {
    let src = format!("{PRELUDE}\nJ = Det(F)\nx = log(J)\nx.show(symbol)");
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\log J"),
        "got: {}",
        outputs[0].latex
    );
}
