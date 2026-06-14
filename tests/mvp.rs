//! MVP integration tests, covering the acceptance checklist:
//! 1. Scalar parses; 2. Tensor parses; 3. `C = F.T * F` parses;
//! 4. C is inferred symmetric; 5. display(C, mode=symbol);
//! 6. display(C, mode=components) is a 3x3 matrix; 7. export(C, format=latex);
//! 8. the full Mooney-Rivlin uniaxial example runs end to end.

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
    // New note sentinels do not collide with ordinary Markdown code fences.
    let src = r#"
<!-- tensorforge:note -->
# Note with code

```
bare fence works too
```

```rust
let x = 1;
```

more prose after the inner fence
<!-- /tensorforge:note -->
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
fn tens_blocks_execute_inside_markdown_document() {
    let src = r#"
# Markdown first

This prose is not TensorForge DSL.

<!-- tensorforge:tens -->
mu = Scalar("\mu")
display(mu, mode=symbol)
<!-- /tensorforge:tens -->

More Markdown with $W = \mu I_1$.
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
fn display_does_not_substitute_aliases_from_the_same_tens_block() {
    let src = r#"
<!-- tensorforge:tens -->
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
display(C)
<!-- /tensorforge:tens -->

<!-- tensorforge:tens -->
I1 = Scalar("I_1")
I2 = Scalar("I_2")
I1 = tr(C)
I2 = 0.5 * ((tr(C))^2 - tr(C*C))
display(I2)
<!-- /tensorforge:tens -->

<!-- tensorforge:tens -->
Psi = Scalar("\Psi")
Psi = I1 + I2
display(Psi)
<!-- /tensorforge:tens -->
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs.len(), 3);
    assert!(
        outputs[1].latex.contains("\\operatorname{tr} \\bm C"),
        "got: {}",
        outputs[1].latex
    );
    assert!(
        !outputs[1].latex.contains("{I_1}^{2}"),
        "got: {}",
        outputs[1].latex
    );
    assert!(
        outputs[2].latex.contains("I_1 + I_2"),
        "got: {}",
        outputs[2].latex
    );
}

#[test]
fn bracketed_display_calls_share_a_preview_row() {
    let src = r#"
I1 = Scalar("I_1")
I2 = Scalar("I_2")
I1 = 1
I2 = 2
[display(I1) display(I2)]
display(I1)
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].row, outputs[1].row);
    assert!(outputs[0].row.is_some());
    assert_eq!(outputs[2].row, None);
}

#[test]
fn unterminated_tens_block_is_a_parse_error() {
    let src = r#"
# Markdown first

<!-- tensorforge:tens -->
mu = Scalar("\mu")
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.to_string().contains("unterminated tens block"),
        "got: {err}"
    );
}

#[test]
fn legacy_note_blocks_still_parse() {
    let src = r#"
```notes
# Legacy note

```rust
let x = 1;
```
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
        outputs[0].latex.contains("\\bm F^{\\mathsf{T}} \\, \\bm F"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn symbol_display_preserves_transpose_in_user_definition() {
    let src = r#"
lam = Var("\lambda")
F = Tensor("\bm F", order=2, dim=3)
F[1][1] = lam
F[2][2] = lam^(-1/2)
F[3][3] = lam^(-1/2)
C = F.T * F
display(C, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    assert!(
        outputs[0].latex.contains("\\bm F^{\\mathsf{T}} \\, \\bm F"),
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn display_uses_named_definitions() {
    let src = r#"
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
I1 = Scalar("I_1")
I1 = tr(C)
W = I1 - 3
display(W, mode=symbol)
export(W, format=latex)
"#;
    let outputs = run_source(src).unwrap();
    for output in &outputs {
        assert!(output.latex.contains("I_1 - 3"), "got: {}", output.latex);
        assert!(
            !output.latex.contains("\\operatorname{tr}"),
            "got: {}",
            output.latex
        );
    }
}

#[test]
fn display_expands_derived_results_but_keeps_base_aliases() {
    let src = r#"
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
I1 = Scalar("I_1")
I1 = tr(C)
Psi = mu/2 * (I1 - 3)
S = 2 * diff(Psi, C)
P = F * S
display(P, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\bm P = \\mu \\, \\bm F"), "got: {latex}");
    assert!(!latex.contains("\\bm S"), "got: {latex}");
}

#[test]
fn display_hoists_scalar_coefficients_out_of_matrix_products() {
    let src = r#"
C1 = Scalar("C_1")
C2 = Scalar("C_2")
F = Tensor("\bm F", order=2, dim=3)
C = Tensor("\bm C", order=2, dim=3, symmetric=true)
I = Tensor("\bm I", order=2, dim=3, identity=true)
S = 2 * (C1 * I + C2 * C)
P = F * S
display(P, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("\\bm P = 2 \\, \\bm F \\, \\left("),
        "got: {latex}"
    );
    assert!(
        !latex.contains("\\bm F \\, 2"),
        "scalar coefficient should not sit after F: {latex}"
    );
}

#[test]
fn display_simplifies_transposes_of_named_symmetric_definitions() {
    let src = r#"
C1 = Scalar("C_1")
C2 = Scalar("C_2")
I1 = Scalar("I_1")
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
I = Tensor("\bm I", order=2, dim=3, identity=true)
S = 2 * (C1 * I + C2/2 * (2 * I1 * I - (C.T + C.T)))
display(S, mode=symbol)
"#;
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("C_2 \\, \\left( I_1 \\, \\bm I - \\bm C \\right)"),
        "got: {latex}"
    );
    assert!(!latex.contains("\\bm C^{\\mathsf{T}}"), "got: {latex}");
    assert!(!latex.contains("\\bm C + \\bm C"), "got: {latex}");
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
    assert_eq!(outputs[0].latex, "\\bm F^{\\mathsf{T}} \\, \\bm F");
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
fn mooney_rivlin_uniaxial_example_runs_end_to_end() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/mooney_rivlin_uniaxial.tens"
    ))
    .unwrap();
    let (outputs, interp) = run_source_with_env(&src).unwrap();
    assert_eq!(outputs.len(), 8);

    match interp.get("Psi") {
        Some(Value::Scalar(_)) => {}
        other => panic!("expected Psi to be a Scalar, got {other:?}"),
    }
    let s = interp.get("S").expect("S should be defined");
    assert!(matches!(s, Value::Tensor(_)));

    let s_out = outputs
        .iter()
        .find(|o| o.header.starts_with("display S"))
        .unwrap();
    assert!(
        !s_out.latex.contains("\\partial"),
        "S should be derived instead of displayed as an unevaluated derivative: {}",
        s_out.latex
    );
    for needle in ["\\bm S", "C_1", "C_2", "\\bm I", "\\bm C"] {
        assert!(
            s_out.latex.contains(needle),
            "missing {needle} in: {}",
            s_out.latex
        );
    }
}

#[test]
fn start_example_runs_as_a_notebook_tour() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/start.tens"))
        .unwrap();
    let outputs = run_source(&src).unwrap();
    assert_eq!(outputs.len(), 16);
    assert!(outputs.iter().all(|o| o.error.is_none()));
    assert!(
        outputs.iter().any(|o| o.header.starts_with("display P11")),
        "start example should show component access"
    );
    assert!(
        outputs.iter().any(|o| o.header.starts_with("display Q")),
        "start example should show spectral differentiation"
    );
    assert!(
        outputs.iter().any(|o| o.row.is_some()),
        "start example should demonstrate side-by-side display rows"
    );
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
