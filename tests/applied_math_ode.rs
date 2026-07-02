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
ode_problem = ODE(ode, f, x)
nonlinear_problem = ODE(nonlinear, f, x)
pde_problem = ODE(pde, u, x)
ode_problem.show(classification)
nonlinear_problem.show(classification)
pde_problem.show(classification)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{ODE}"));
    assert!(outputs[0].latex.contains("\\text{linear}"));
    assert!(outputs[0].latex.contains("\\text{non-homogeneous}"));
    assert!(!outputs[0].latex.contains("\\frac{d f}{d x}"));
    assert!(!outputs[0]
        .latex
        .contains("\\text{no explicit boundary condition}"));
    assert!(!outputs[0]
        .latex
        .contains("\\text{linear integrating factor}"));
    assert!(!outputs[0].latex.contains("\\text{separable}"));
    assert!(!outputs[0].latex.contains("\\text{exact}"));
    assert!(outputs[1].latex.contains("2"));
    assert!(outputs[1].latex.contains("\\text{nonlinear}"));
    assert!(outputs[2].latex.contains("\\text{PDE}"));
    assert!(outputs[2].latex.contains("4"));
}

#[test]
fn ode_show_and_classify_include_boundary_condition_status() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x) + 2*y, exp(x))
plain = ODE(eq, y, x)
with_bc = ODE(eq, y, x, BoundaryCondition(y(0), 1))
plain.show()
plain.show(boundary)
plain.show(classification)
plain.show(methods)
with_bc.show()
with_bc.show(boundary)
with_bc.show(classification)
"#;
    let outputs = run_source(src).unwrap();
    assert!(!outputs[0]
        .latex
        .contains("\\text{no explicit boundary condition}"));
    assert!(outputs[0].latex.contains("\\frac{d y}{d x}"));
    assert!(outputs[1]
        .latex
        .contains("\\text{no explicit boundary condition}"));
    assert!(!outputs[2]
        .latex
        .contains("\\text{no explicit boundary condition}"));
    assert!(!outputs[2].latex.contains("\\frac{d y}{d x}"));
    assert!(!outputs[2]
        .latex
        .contains("\\text{linear integrating factor}"));
    assert!(outputs[3]
        .latex
        .contains("\\text{linear integrating factor}"));
    assert!(!outputs[4].latex.contains("y\\left( 0 \\right) = 1"));
    assert!(outputs[5].latex.contains("y\\left( 0 \\right) = 1"));
    assert!(!outputs[6].latex.contains("y\\left( 0 \\right) = 1"));
    assert!(!outputs[6].latex.contains("\\frac{d y}{d x}"));
    assert!(outputs[6].latex.contains("\\text{linear}"));
}

#[test]
fn solve_linear_first_order_keeps_nonelementary_integral() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(x^2 * Derivative(y, x) + 2*y, 5*x)
ode = ODE(eq, y, x)
ode.solve(details=true)
ode.solve()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\mu ="));
    assert!(!outputs[0].latex.contains("\\text{ODE}"));
    assert!(!outputs[0].latex.contains("y\\left"));
    assert!(!outputs[0].latex.contains("\\frac{5 \\, x}{{x}^{2}}"));
    assert!(!outputs[0].latex.contains("\\frac{x}{{x}^{2}}"));
    assert!(outputs[0].latex.contains("e^{-2") || outputs[0].latex.contains("-2"));
    assert!(outputs[1].latex.contains("\\int"));
    assert!(!outputs[1].latex.contains("y\\left"));
    assert!(outputs[1].latex.contains("C\\_1") || outputs[1].latex.contains("C_1"));
}

#[test]
fn solve_separable_first_order() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(3*y^2*Derivative(y, x), cos(x))
ode = ODE(eq, y, x)
ode.show(classification)
ode.solve()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{nonlinear}"));
    assert!(outputs[0].latex.contains("\\text{non-homogeneous}"));
    assert!(!outputs[0].latex.contains("\\text{n/a}"));
    assert!(!outputs[0].latex.contains("\\text{separable}"));
    assert!(!outputs[0].latex.contains("\\text{exact}"));
    assert!(outputs[1].latex.contains("{y}^{3}"));
    assert!(outputs[1].latex.contains("\\sin x"));
}

#[test]
fn solve_method_keyword_selects_available_solver() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(3*y^2*Derivative(y, x), cos(x))
ode = ODE(eq, y, x)
ode.solve()
ode.solve(method=auto)
ode.solve(method=separable)
ode.solve(method=exact)
ode.solve(method=linear)
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, outputs[1].latex);
    assert!(outputs[2].latex.contains("{y}^{3}"));
    assert!(outputs[2].latex.contains("\\sin x"));
    assert!(outputs[3].latex.contains("C\\_1") || outputs[3].latex.contains("C_1"));
    assert!(outputs[4]
        .latex
        .contains("\\text{unsupported: requested method `linear` is not available"));
    assert!(outputs[4].latex.contains("separable"));
    assert!(outputs[4].latex.contains("exact"));
}

#[test]
fn solve_method_keyword_rejects_unknown_method() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x), x)
ode = ODE(eq, y, x)
ode.solve(method=fancy)
"#;
    let err = run_source(src).unwrap_err();
    assert!(err.message.contains("unknown ODE solve method `fancy`"));
    assert!(err
        .message
        .contains("expected auto, linear, separable, exact, characteristic"));
}

#[test]
fn solve_second_order_constant_homogeneous_characteristic_cases() {
    let src = r#"
x = Var("x")
y = Function("y", x)
real = ODE(Equation(Derivative(y, x, order=2) + 5*Derivative(y, x) + 6*y, 0), y, x)
repeated = ODE(Equation(Derivative(y, x, order=2) + 6*Derivative(y, x) + 9*y, 0), y, x)
complex = ODE(Equation(Derivative(y, x, order=2) + 2*Derivative(y, x) + 5*y, 0), y, x)
real.show(classification)
real.show(methods)
real.solve(method=characteristic, details=true)
repeated.solve(method=characteristic)
complex.solve(method=characteristic)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{order 2}"));
    assert!(outputs[0].latex.contains("\\text{linear}"));
    assert!(outputs[0].latex.contains("\\text{homogeneous}"));
    assert!(outputs[1].latex.contains("\\text{characteristic}"));
    assert!(outputs[2].latex.contains("\\lambda^2 + 5\\lambda + 6 = 0"));
    assert!(outputs[2].latex.contains("\\lambda_1=-2"));
    assert!(outputs[2].latex.contains("\\lambda_2=-3"));
    assert!(outputs[3].latex.contains("C_2 \\, x"));
    assert!(outputs[4].latex.contains("\\cos"));
    assert!(outputs[4].latex.contains("\\sin"));
}

#[test]
fn solve_second_order_with_two_boundary_conditions() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x, order=2) + y, 0)
ode = ODE(eq, y, x, BoundaryCondition(y(0), 1), BoundaryCondition(Derivative(y, x), 0, 0))
ode.show(boundary)
ode.solve(method=characteristic)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("y\\left( 0 \\right) = 1"));
    assert!(outputs[0].latex.contains("\\frac{d y}{d x}"));
    assert!(outputs[1].latex.contains("\\cos x"));
    assert!(!outputs[1].latex.contains("C_1"));
    assert!(!outputs[1].latex.contains("C_2"));
}

#[test]
fn solve_second_order_undetermined_and_variation() {
    use tensorforge::interpreter::OutputDetail;
    let src = r#"
x = Var("x")
y = Function("y", x)
ode = ODE(Equation(Derivative(y, x, order=2) + y, x^2), y, x)
ode.show(methods)
ode.solve(method=undetermined, details=true)
ode.solve(method=variation, details=true)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0]
        .latex
        .contains("\\text{undetermined coefficients}"));
    assert!(outputs[1].latex.contains("particular ansatz"));
    assert!(outputs[1].latex.contains("{x}^{2} - 2") || outputs[1].latex.contains("-2 + {x}^{2}"));
    match outputs[1].detail.as_ref().expect("undetermined steps") {
        OutputDetail::OdeSteps { steps } => {
            assert!(steps.iter().any(|s| s.label == "particular solution"));
        }
        other => panic!("expected ODE steps, got {other:?}"),
    }
    assert!(outputs[2].latex.contains("Wronskian"));
    assert!(outputs[2].latex.contains("V_1"));
    assert!(outputs[2].latex.contains("V_2"));
}

#[test]
fn solve_power_series_examples() {
    let src = r#"
x = Var("x")
y = Function("y", x)
first = ODE(Equation(Derivative(y, x) + 2*y, 0), y, x)
second = ODE(Equation(Derivative(y, x, order=2) + x*Derivative(y, x) + y, 0), y, x)
first.solve(method=power_series, terms=5, details=true)
second.show(methods)
second.solve(method=power_series, terms=6, details=true)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("C_{n+1}=-\\frac{2C_n}{n+1}"));
    assert!(outputs[1].latex.contains("\\text{power series}"));
    assert!(outputs[2].latex.contains("C_{n+2}=-\\frac{C_n}{n+2}"));
}

#[test]
fn solve_frobenius_bessel_indicial_roots() {
    let src = r#"
x = Var("x")
lambda = Var("\lambda")
y = Function("y", x)
bessel = ODE(Equation(x^2*Derivative(y, x, order=2) + x*Derivative(y, x) + (x^2 - lambda^2)*y, 0), y, x)
bessel.show(methods)
bessel.solve(method=frobenius, details=true)
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\text{frobenius}"));
    assert!(outputs[1].latex.contains("r^2-\\lambda^2=0"));
    assert!(outputs[1].latex.contains("r=\\pm \\lambda"));
    assert!(outputs[1].latex.contains("J_{\\lambda}"));
}

#[test]
fn separable_and_exact_details_are_stepwise() {
    use tensorforge::interpreter::OutputDetail;
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(3*y^2*Derivative(y, x), cos(x))
ode = ODE(eq, y, x)
ode.solve(method=separable, details=true)
ode.solve(method=exact, details=true)
"#;
    let outputs = run_source(src).unwrap();
    match outputs[0].detail.as_ref().expect("separable steps") {
        OutputDetail::OdeSteps { steps } => {
            assert!(steps.iter().any(|s| s.label == "original equation"));
            assert!(steps.iter().any(|s| s.label == "separate variables"));
            assert!(steps.iter().any(|s| s.label == "integrate both sides"));
            assert!(steps.iter().any(|s| s.label == "implicit solution"));
            assert!(steps.iter().all(|s| !s.latex.is_empty()));
        }
        other => panic!("expected separable steps, got {other:?}"),
    }
    match outputs[1].detail.as_ref().expect("exact steps") {
        OutputDetail::OdeSteps { steps } => {
            assert!(steps.iter().any(|s| s.label == "original equation"));
            assert!(steps.iter().any(|s| s.label == "differential form"));
            assert!(steps.iter().any(|s| s.label == "identify M and N"));
            assert!(steps.iter().any(|s| s.label == "exactness check"));
            assert!(steps.iter().any(|s| s.label == "potential"));
            assert!(steps.iter().any(|s| s.label == "implicit solution"));
            assert!(steps.iter().all(|s| !s.latex.is_empty()));
        }
        other => panic!("expected exact steps, got {other:?}"),
    }
}

#[test]
fn solve_exact_first_order_with_initial_condition() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation((2 + x^2*y)*Derivative(y, x) + x*y^2, 0)
ode = ODE(eq, y, x, BoundaryCondition(y(1), 2))
ode.solve(details=true)
ode.solve()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("2 \\, x \\, y = 2 \\, x \\, y"));
    assert!(outputs[1].latex.contains("2 \\, y"));
    assert!(outputs[1].latex.contains("6"));
}

#[test]
fn old_ode_entry_points_report_migration_hint() {
    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x), 0)
sol = SolveODE(eq, y, x, ic=IC(y(0), 1))
"#;
    let err = run_source(src).unwrap_err();
    assert!(err.message.contains("ODE(eq, y, x"), "got: {}", err.message);
    assert!(
        err.message.contains("BoundaryCondition"),
        "got: {}",
        err.message
    );

    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x), 0)
kind = ClassifyODE(eq, y, x)
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("ODE(eq, y, x).show(classification)"),
        "got: {}",
        err.message
    );

    let src = r#"
x = Var("x")
y = Function("y", x)
eq = Equation(Derivative(y, x), 0)
ode = ODE(eq, y, x)
ode.classify()
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("ODE.show(classification)"),
        "got: {}",
        err.message
    );
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
        .any(|o| o.latex.contains("2 \\, x \\, y = 2 \\, x \\, y")));
    assert!(outputs
        .iter()
        .any(|o| o.latex.contains("\\text{unsupported")));
    assert!(outputs
        .iter()
        .any(|o| o.header.starts_with("linear_problem.solve")));
    assert!(outputs
        .iter()
        .any(|o| o.header.starts_with("sep_problem.solve")));
    assert!(outputs
        .iter()
        .any(|o| o.header.starts_with("exact_problem.solve")));
}

#[test]
fn structured_detail_for_classification_methods_and_steps() {
    use tensorforge::interpreter::OutputDetail;
    let src = r#"
x = Var("x")
y = Function("y", x)
p = ODE(Equation(x^2*Derivative(y, x) + 2*y, 5*x), y, x)
pbc = ODE(Equation(Derivative(y, x) + 2*y, exp(x)), y, x, BoundaryCondition(y(0), 1))
p.show()
p.show(classification)
p.show(boundary)
pbc.show(boundary)
p.show(methods)
p.solve(details=true)
"#;
    let outputs = run_source(src).unwrap();

    // bare .show() stays plain math (no structured chrome).
    assert!(outputs[0].detail.is_none());

    // classification carries no boundary — that belongs to show(boundary).
    match outputs[1].detail.as_ref().expect("classification detail") {
        OutputDetail::OdeClassification {
            kind,
            order,
            linear,
            homogeneous,
        } => {
            assert_eq!(kind, "ODE");
            assert_eq!(*order, 1);
            assert!(*linear);
            assert!(!*homogeneous);
        }
        other => panic!("expected classification detail, got {other:?}"),
    }

    // show(boundary): none for `p`, present for `pbc`.
    match outputs[2].detail.as_ref().expect("boundary detail") {
        OutputDetail::OdeBoundary { boundary } => assert!(boundary.is_none()),
        other => panic!("expected boundary detail, got {other:?}"),
    }
    match outputs[3].detail.as_ref().expect("boundary detail") {
        OutputDetail::OdeBoundary { boundary } => {
            assert!(boundary
                .as_deref()
                .unwrap()
                .contains("y\\left( 0 \\right) = 1"));
        }
        other => panic!("expected boundary detail, got {other:?}"),
    }

    match outputs[4].detail.as_ref().expect("methods detail") {
        OutputDetail::OdeMethods { available, default } => {
            assert_eq!(default, "linear integrating factor");
            assert!(available.iter().any(|m| m == "linear integrating factor"));
        }
        other => panic!("expected methods detail, got {other:?}"),
    }

    match outputs[5].detail.as_ref().expect("steps detail") {
        OutputDetail::OdeSteps { steps } => {
            assert_eq!(steps.first().unwrap().label, "standard form");
            assert_eq!(steps.last().unwrap().label, "solution");
            assert!(steps.iter().any(|s| s.label == "integrating factor"));
            assert!(steps.iter().all(|s| !s.latex.is_empty()));
        }
        other => panic!("expected steps detail, got {other:?}"),
    }
}
