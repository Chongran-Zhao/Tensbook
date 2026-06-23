use tensorforge::run_source;

#[test]
fn integrate_basic_and_affine_functions() {
    let src = r#"
x = Var("x")
a = Scalar("a")
b = Scalar("b")
I1 = Integrate(x^3, x)
I2 = Integrate(1/x, x)
I3 = Integrate(exp(2*x + b), x)
I4 = Integrate(sin(3*x), x)
I5 = Integrate(cosh(4*x), x)
I1.show()
I2.show()
I3.show()
I4.show()
I5.show()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\frac{1}{4}") && outputs[0].latex.contains("{x}^{4}"));
    assert!(outputs[1].latex.contains("\\log x"));
    assert!(outputs[2].latex.contains("\\frac{1}{2}") && outputs[2].latex.contains("e^"));
    assert!(outputs[2].latex.contains("2"));
    assert!(outputs[3].latex.contains("-\\frac{1}{3}") || outputs[3].latex.contains("\\frac{-"));
    assert!(outputs[3].latex.contains("\\cos"));
    assert!(outputs[4].latex.contains("\\sinh"));
}

#[test]
fn integrate_chain_rule_and_formal_fallback() {
    let src = r#"
x = Var("x")
u = x^2 + 1
I1 = Integrate(Diff(u, x) * exp(u), x)
I2 = Integrate(exp(-2/x)/x, x)
I1.show()
I2.show()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("e^"));
    assert!(outputs[1].latex.contains("\\int"));
    assert!(outputs[1].latex.contains("e^{"));
}

#[test]
fn classify_ode_and_pde_examples() {
    let src = r#"
x = Var("x")
y = Var("y")
f = Function("f", x)
u = Function("u", x, y)
ode = Equation(Derivative(f, x) + sin(x), 0)
nonlinear = Equation(Derivative(f, x, order=2) + x*Derivative(f, x) + f^2, 0)
pde = Equation(Derivative(u, x, order=4) + Derivative(u, y, order=4), exp(-(x^2 + y^2)))
c1 = ClassifyODE(ode, f, x)
c2 = ClassifyODE(nonlinear, f, x)
c3 = ClassifyODE(pde, u, x)
c1.show(details)
c2.show()
c3.show()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{ODE}"));
    assert!(outputs[0].latex.contains("\\text{linear}"));
    assert!(outputs[0].latex.contains("\\text{nonhomogeneous}"));
    assert!(outputs[1].latex.contains("2"));
    assert!(outputs[1].latex.contains("\\text{nonlinear}"));
    assert!(outputs[2].latex.contains("\\text{PDE}"));
    assert!(outputs[2].latex.contains("4"));
}

#[test]
fn solve_linear_first_order_keeps_nonelementary_integral() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(x^2 * Derivative(y, x) + 2*y, 5*x)
sol = SolveODE(eq, y, x)
sol.show(steps)
sol.show(solution)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{integrating factor}"));
    assert!(outputs[0].latex.contains("e^{-2") || outputs[0].latex.contains("-2"));
    assert!(outputs[1].latex.contains("\\int"));
    assert!(outputs[1].latex.contains("C\\_1") || outputs[1].latex.contains("C_1"));
}

#[test]
fn solve_separable_first_order() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(3*y^2*Derivative(y, x), cos(x))
sol = SolveODE(eq, y, x)
sol.show(solution)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("{y}^{3}"));
    assert!(outputs[0].latex.contains("\\sin x"));
}

#[test]
fn solve_exact_first_order_with_initial_condition() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation((2 + x^2*y)*Derivative(y, x) + x*y^2, 0)
sol = SolveODE(eq, y, x, ic=IC(y(1), 2))
sol.show(steps)
sol.show(solution)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{exactness check}"));
    assert!(outputs[1].latex.contains("2 \\, y"));
    assert!(outputs[1].latex.contains("6"));
}

#[test]
fn applied_math_ode_example_runs_end_to_end() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/applied-math-ode.tens"
    ))
    .unwrap();
    let outputs = run_source(&src).unwrap();
    assert!(outputs.len() >= 20, "got {} outputs", outputs.len());
    assert!(outputs.iter().all(|o| o.error.is_none()));
    assert!(outputs.iter().any(|o| o.latex.contains("\\text{PDE}")));
    assert!(outputs.iter().any(|o| o.latex.contains("\\int")));
    assert!(outputs
        .iter()
        .any(|o| o.latex.contains("\\text{exactness check}")));
    assert!(outputs
        .iter()
        .any(|o| o.latex.contains("\\text{unsupported")));
    assert!(outputs
        .iter()
        .any(|o| o.header.starts_with("linear_sol.show")));
    assert!(outputs.iter().any(|o| o.header.starts_with("sep_sol.show")));
    assert!(outputs
        .iter()
        .any(|o| o.header.starts_with("exact_sol.show")));
}
