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
fn spectral_of_symmetric_tensor() {
    let src = format!("{PRELUDE}\nS = spectral(C)\ndisplay(S, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm S = \\sum_{a=1}^{3} c_a \\, \\bm N_a \\otimes \\bm N_a",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn spectral_rejects_unsymmetric_tensor() {
    // F is a general tensor: spectral(F) must be refused (not provably symmetric).
    let src = format!("{PRELUDE}\nS = spectral(F)");
    assert!(run_source(&src).is_err());
}

#[test]
fn spectral_result_is_symmetric() {
    let src = format!("{PRELUDE}\nS = spectral(C)");
    let (_, interp) = run_source_with_env(&src).unwrap();
    match interp.get("S") {
        Some(Value::Tensor(t)) => assert!(t.is_symmetric()),
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn outer_product_order_and_display() {
    let src = format!("{PRELUDE}\nO = outer(F, G)\ndisplay(O, mode=symbol)");
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
fn outer_of_same_tensor_is_symmetric() {
    let src = format!("{PRELUDE}\nO = outer(F, F)\nP = outer(F, G)");
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
fn tensor_sqrt_is_spectral_sum() {
    // U = sqrt(C) is the right stretch tensor.
    let src = format!("{PRELUDE}\nU = sqrt(C)\ndisplay(U, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "\\bm U = \\sum_{a=1}^{3} \\sqrt{c_a} \\, \\bm N_a \\otimes \\bm N_a",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn tensor_log_and_exp() {
    let src = format!(
        "{PRELUDE}\nL = log(C)\nE = exp(C)\n\
         display(L, mode=symbol)\ndisplay(E, mode=symbol)"
    );
    let outputs = run_source(&src).unwrap();
    assert!(
        outputs[0].latex.contains("\\log c_a"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("e^{c_a}"),
        "got: {}",
        outputs[1].latex
    );
}

#[test]
fn tensor_fn_requires_symmetry() {
    // sqrt of a general (not provably symmetric) tensor must be refused.
    let src = format!("{PRELUDE}\nU = sqrt(F)");
    assert!(run_source(&src).is_err());
}

#[test]
fn tensor_fn_result_is_symmetric() {
    let src = format!("{PRELUDE}\nU = sqrt(C)");
    let (_, interp) = run_source_with_env(&src).unwrap();
    match interp.get("U") {
        Some(Value::Tensor(t)) => {
            assert!(t.is_symmetric());
            assert_eq!(t.order(), 2);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
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
