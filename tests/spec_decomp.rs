//! Component-filled tensors (`F[1][1] = ...`) and the computed spectral
//! decomposition `[a, b] = Spec_Decomp(C)` for diagonal symbolic tensors.

use tensbook::run_source;

/// Uniaxial incompressible stretch: F = diag(λ, λ^{-1/2}, λ^{-1/2}).
const PRELUDE: &str = r#"
lam = Var("\lambda")
F = Tensor("\bm F", order=2, dim=3)
F[1][1] = lam
F[2][2] = lam^(-1/2)
F[3][3] = lam^(-1/2)
C = F.T*F
c = ScalarSet("c", dim=3)
N = VectorSet("\bm N", dim=3)
[c, N] = Spec_Decomp(C)
"#;

#[test]
fn filled_tensor_matrix_display() {
    let src = "lam = Var(\"\\lambda\")\nF = Tensor(\"\\bm F\", order=2, dim=3)\n\
               F[1][1] = lam\nF[2][2] = 1\nF.show(matrix)";
    let outputs = run_source(src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("\\lambda & 0 & 0"), "got: {latex}");
    assert!(latex.contains("0 & 1 & 0"), "got: {latex}");
    assert!(
        latex.contains("0 & 0 & 0"),
        "unset entries are zero: {latex}"
    );
}

#[test]
fn derived_product_entries_simplify() {
    let src = format!("{PRELUDE}\nC.show(matrix)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("{\\lambda}^{2} & 0 & 0"), "got: {latex}");
    assert!(latex.contains("{\\lambda}^{-1}"), "got: {latex}");
}

#[test]
fn spec_decomp_fills_eigenvalues_and_basis() {
    let src = format!("{PRELUDE}\nx = c[1]\nx.show()\nv = N[2]\nv.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "x = {\\lambda}^{2}",
        "got: {}",
        outputs[0].latex
    );
    assert!(
        outputs[1].latex.contains("{\\bm N}_{2}"),
        "got: {}",
        outputs[1].latex
    );
}

#[test]
fn spectral_sum_expands_to_explicit_matrix() {
    // Hencky strain of uniaxial incompressible stretch:
    // E = Σ log(c_a) N_a ⊗ N_a → diag(2 ln λ, −ln λ, −ln λ) with c_a = λ_a².
    let src = format!("{PRELUDE}\nEh = Sum(log(c[a]) * N[a] & N[a], a)\nEh.show(matrix)");
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(
        latex.contains("2 \\, \\log \\lambda & 0 & 0"),
        "got: {latex}"
    );
    assert!(latex.contains("-\\log \\lambda"), "got: {latex}");
}

#[test]
fn spec_decomp_requires_diagonal() {
    let src = "lam = Var(\"\\lambda\")\nF = Tensor(\"\\bm F\", order=2, dim=2)\n\
               F[1][1] = lam\nF[1][2] = lam\nF[2][2] = lam\n\
               c = ScalarSet(\"c\", dim=2)\nN = VectorSet(\"\\bm N\", dim=2)\n\
               [c, N] = Spec_Decomp(F)";
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("requires a diagonal tensor"),
        "got: {}",
        err.message
    );
}

#[test]
fn spec_decomp_requires_declared_sets() {
    let src = "lam = Var(\"\\lambda\")\nF = Tensor(\"\\bm F\", order=2, dim=3)\n\
               F[1][1] = lam\n[c, N] = Spec_Decomp(F)";
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("declare `c` first"),
        "got: {}",
        err.message
    );
}

#[test]
fn component_assignment_rejects_derived_targets() {
    let src = format!("{PRELUDE}\nC[1][1] = lam\nC.show()");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("derived expression"),
        "got: {}",
        err.message
    );
}

#[test]
fn bare_flag_error_suggests_keyword_form() {
    let src = "F = Tensor(\"\\bm F\", order=2, dim=3, identity)";
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("identity=true"),
        "got: {}",
        err.message
    );
}

#[test]
fn tensor_component_read() {
    // C = diag(λ², λ⁻¹, λ⁻¹); read individual components as scalars.
    let src =
        format!("{PRELUDE}\nx = C[1][1]\nx.show()\ny = C[2][2]\ny.show()\nz = C[1][2]\nz.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "x = {\\lambda}^{2}",
        "got: {}",
        outputs[0].latex
    );
    assert_eq!(
        outputs[1].latex, "y = {\\lambda}^{-1}",
        "got: {}",
        outputs[1].latex
    );
    assert_eq!(
        outputs[2].latex, "z = 0",
        "off-diagonal is zero: {}",
        outputs[2].latex
    );
}

#[test]
fn display_component_read_uses_component_lhs() {
    let src = r#"
lam = Var("\lambda")
P = Tensor("\bm P", order=2, dim=3)
P[1][1] = lam
P[1][1].show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].header, "P[1][1].show()");
    assert_eq!(outputs[0].latex, "{\\bm P}_{11} = \\lambda");
}

#[test]
fn component_index_out_of_range_errors() {
    let src = format!("{PRELUDE}\nx = C[4][1]\nx.show()");
    let err = run_source(&src).unwrap_err();
    assert!(err.message.contains("out of range"), "got: {}", err.message);
}
