//! MVP integration tests, covering the acceptance checklist:
//! 1. Scalar parses; 2. Tensor parses; 3. `C = F.T * F` parses;
//! 4. C is inferred symmetric; 5. display(C, mode=symbol);
//! 6. display(C, mode=components) is a 3x3 matrix; 7. export(C, format=latex);
//! 8. the full neo_hookean.tens example runs end to end.

use tensorforge::interpreter::Value;
use tensorforge::tensor::TensorExpr;
use tensorforge::{run_source, run_source_with_env};

const PRELUDE: &str = r#"
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
"#;

#[test]
fn scalar_parses() {
    let (_, interp) = run_source_with_env(r#"mu = Scalar("\mu")"#).unwrap();
    match interp.get("mu") {
        Some(Value::Scalar(s)) => {
            assert_eq!(**s, tensorforge::symbolic::ScalarExpr::sym("\\mu", "\\mu"));
        }
        other => panic!("expected Scalar, got {other:?}"),
    }
}

#[test]
fn markdown_note_blocks_are_ignored_by_parser() {
    let src = r#"
```notes
# Model notes

Inline math $W = \mu I_1$ and display math:

$$
\bm C = \bm F^\mathsf{T}\bm F
$$
```

mu = Scalar("\mu")
display(mu, mode=symbol)
```notes
## More notes

- These lines are Markdown, not TensorForge DSL.
- Symbols like $$, #, and `code` stay inside the note block.
```
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0].latex.contains("\\mu"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn note_blocks_may_contain_fenced_code() {
    // A markdown code fence inside a note: its closing bare ``` must not
    // terminate the note block.
    let src = r#"
```notes
# Note with code

```rust
let x = 1;
```

more prose after the inner fence
```
mu = Scalar("\mu")
display(mu, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0].latex.contains("\\mu"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn unterminated_markdown_note_block_is_a_parse_error() {
    let src = r#"
```notes
# Missing close
mu = Scalar("\mu")
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.to_string().contains("unterminated notes block"),
        "got: {err}"
    );
}

#[test]
fn tensor_parses_with_order_and_dim() {
    let (_, interp) = run_source_with_env(r#"F = Tensor("\bm F", order=2, dim=3)"#).unwrap();
    match interp.get("F") {
        Some(Value::Tensor(t)) => {
            assert_eq!(t.order(), 2);
            assert_eq!(t.dim(), 3);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn identity_tensor_parses() {
    let (_, interp) =
        run_source_with_env(r#"I = Tensor("\bm I", order=2, dim=3, identity=true)"#).unwrap();
    match interp.get("I") {
        Some(Value::Tensor(t)) => {
            assert!(t.is_identity());
            assert!(t.is_symmetric());
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn ftf_parses_as_matmul_of_transpose() {
    let (_, interp) = run_source_with_env(PRELUDE).unwrap();
    match interp.get("C") {
        Some(Value::Tensor(t)) => match &**t {
            TensorExpr::MatMul(a, b) => {
                assert!(matches!(&**a, TensorExpr::Transpose(inner) if **inner == **b));
            }
            other => panic!("expected MatMul(Transpose(F), F), got {other:?}"),
        },
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn ftf_is_inferred_symmetric() {
    let (_, interp) = run_source_with_env(PRELUDE).unwrap();
    match interp.get("C") {
        Some(Value::Tensor(t)) => {
            assert!(t.is_symmetric(), "C = F.T * F must be inferred symmetric");
            assert_eq!(t.order(), 2);
            assert_eq!(t.dim(), 3);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn general_product_is_not_inferred_symmetric() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
G = Tensor("\bm G", order=2, dim=3)
P = F * G
"#;
    let (_, interp) = run_source_with_env(src).unwrap();
    match interp.get("P") {
        Some(Value::Tensor(t)) => assert!(!t.is_symmetric()),
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn display_symbol_mode() {
    let src = format!("{PRELUDE}\ndisplay(C, mode=symbol)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].header, "display C, mode=symbol");
    // Displayed as `\bm C = \bm F^{\mathsf{T}} \bm F`
    assert!(
        outputs[0].latex.contains("\\bm C ="),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[0].latex.contains("\\bm F^{\\mathsf{T}} \\bm F"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn display_components_mode_is_3x3() {
    let src = format!("{PRELUDE}\ndisplay(C, mode=components)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\begin{bmatrix}"), "got: {latex}");
    assert!(latex.contains("\\end{bmatrix}"), "got: {latex}");
    // 3 rows -> 2 row separators; 3 columns -> 2 `&` per row, 6 total.
    assert_eq!(latex.matches("\\\\").count(), 2, "got: {latex}");
    assert_eq!(latex.matches('&').count(), 6, "got: {latex}");
    // C_12 entry: (F^T F)_{12} = F_11 F_12 + F_21 F_22 + F_31 F_32
    assert!(
        latex.contains("F_{11} \\, F_{12} + F_{21} \\, F_{22} + F_{31} \\, F_{32}"),
        "got: {latex}"
    );
    // Diagonal entry C_11 = F_11^2-ish sum: F_11 F_11 + F_21 F_21 + F_31 F_31
    assert!(
        latex.contains("F_{11} \\, F_{11} + F_{21} \\, F_{21} + F_{31} \\, F_{31}"),
        "got: {latex}"
    );
}

#[test]
fn export_latex() {
    let src = format!("{PRELUDE}\nexport(C, format=latex)");
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs[0].header, "export C, format=latex");
    assert_eq!(outputs[0].latex, "\\bm F^{\\mathsf{T}} \\bm F");
}

#[test]
fn scalar_invariants_render() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
J = det(F)
I1 = tr(C)
display(J, mode=symbol)
display(I1, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("\\det \\bm F"),
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("\\operatorname{tr}"),
        "got: {}",
        outputs[1].latex
    );
}

#[test]
fn neo_hookean_example_runs_end_to_end() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/neo_hookean.tens"
    ))
    .unwrap();
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert_eq!(outputs.len(), 12);

    // W is a scalar expression containing mu, lambda, log, det, tr.
    match interp.get("W") {
        Some(Value::Scalar(_)) => {}
        other => panic!("expected W to be a Scalar, got {other:?}"),
    }
    let w_out = outputs
        .iter()
        .find(|o| o.header.starts_with("display W"))
        .unwrap();
    // J = det F and I1 = tr C are back-substituted in the display.
    for needle in ["\\mu", "\\lambda", "\\log J", "I1"] {
        assert!(
            w_out.latex.contains(needle),
            "missing {needle} in: {}",
            w_out.latex
        );
    }
}

#[test]
fn declared_label_survives_reassignment() {
    // I1 = Scalar("I_1") declares the display label; the later
    // I1 = tr(C) reassignment keeps using it on the display LHS.
    let src = r#"
I1 = Scalar("I_1")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
I1 = tr(C)
display(I1, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.starts_with("I_1 = "),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn underived_single_char_tensor_gets_bold_lhs() {
    // C was never declared with a label: the LHS is synthesized as \bm C.
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
display(C, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.starts_with("\\bm C = "),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn type_errors_are_reported() {
    // scalar has no transpose
    assert!(run_source(r#"mu = Scalar("\mu")"#).is_ok());
    assert!(run_source(
        r#"
mu = Scalar("\mu")
x = mu.T
"#
    )
    .is_err());

    // det of a scalar is rejected
    assert!(run_source(
        r#"
mu = Scalar("\mu")
x = det(mu)
"#
    )
    .is_err());

    // dimension mismatch is rejected
    assert!(run_source(
        r#"
A = Tensor("\bm A", order=2, dim=3)
B = Tensor("\bm B", order=2, dim=2)
X = A * B
"#
    )
    .is_err());

    // undefined variable
    assert!(run_source("X = Y * Y").is_err());
}
