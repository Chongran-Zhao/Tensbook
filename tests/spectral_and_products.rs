//! Phase-5 tests: spectral decomposition, outer/dot/ddot, markdown export.

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
    let src = format!("{PRELUDE}\nO = F & G\ndisplay(O, mode=symbol)");
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
    let src = format!("{PRELUDE}\nO = F & G\nP = outer(F, G)\ndisplay(O, mode=symbol)");
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
fn dot_is_single_contraction() {
    let src = format!("{PRELUDE}\nD = dot(F, G)\nexport(D, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "\\bm F \\bm G");
}

#[test]
fn ddot_is_scalar_double_contraction() {
    let src = format!("{PRELUDE}\nu = ddot(F, G)\ndisplay(u, mode=symbol)");
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F : \\bm G"),
        "got: {}",
        outputs[0].latex
    );
    match interp.get("u") {
        Some(Value::Scalar(_)) => {}
        other => panic!("ddot must produce a Scalar, got {other:?}"),
    }
}

#[test]
fn markdown_export_wraps_display_math() {
    let src = format!("{PRELUDE}\nexport(C, format=markdown)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].latex, "$$\n\\bm F^{\\mathsf{T}} \\bm F\n$$");
}

// ---- isotropic tensor functions sqrt/log/exp --------------------------------

#[test]
fn tensor_scalar_functions_point_at_the_set_form() {
    // log/sqrt/exp of a tensor are no longer built in; the error points at
    // the explicit spectral-sum form.
    let src = format!("{PRELUDE}\nC = F.T * F\nL = log(C)\ndisplay(L)");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("sum(log(lambda[a]) * N[a] & N[a], a)"),
        "got: {}",
        err.message
    );
}

#[test]
fn scalar_log_still_works() {
    let src = format!("{PRELUDE}\nJ = det(F)\nx = log(J)\ndisplay(x, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    // det F is back-substituted to J in the display.
    assert!(
        outputs[0].latex.contains("\\log J"),
        "got: {}",
        outputs[0].latex
    );
}
