//! User-defined scalar functions: `Var("\lambda")` declares a function
//! argument; expressions mentioning it apply with call syntax.

use tensorforge::run_source;

const PRELUDE: &str = r#"
lam = Var("\lambda")
m = Scalar("m")
n = Scalar("n")
mu = Scalar("\mu")
F = Tensor("\bm F", order=2, dim=3)
C = F.T*F
"#;

#[test]
fn function_application_substitutes_the_var() {
    let src = format!(
        "{PRELUDE}\nEcr = (lam^m - lam^(-n))/(m + n)\nx = Scalar(\"x\")\n\
         y = Ecr(x)\ny.show()"
    );
    let outputs = run_source(&src).unwrap();
    let latex = &outputs[0].latex;
    assert!(latex.contains("{x}^{m}"), "got: {latex}");
    assert!(
        !latex.contains("\\lambda"),
        "var must be substituted: {latex}"
    );
}

#[test]
fn applying_a_non_function_errors() {
    let src = format!("{PRELUDE}\ng = mu(m)\ng.show()");
    let err = run_source(&src).unwrap_err();
    assert!(
        err.message.contains("not a function"),
        "got: {}",
        err.message
    );
}

#[test]
fn diff_of_function_by_its_var() {
    // d/dλ of a user function works through the ordinary scalar diff path.
    let src = format!("{PRELUDE}\nEh = log(lam)\ndE = Diff(Eh, lam)\ndE.show()");
    let outputs = run_source(&src).unwrap();
    assert_eq!(
        outputs[0].latex, "dE = \\frac{1}{\\lambda}",
        "got: {}",
        outputs[0].latex
    );
}

#[test]
fn diff_order_keyword_repeats_scalar_derivative() {
    let src = r#"
x = Scalar("x")
f = exp(x) + x^3
d2f = Diff(f, x, order=2)
d2f.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "d2f = e^{x} + 6 \\, x");
}

#[test]
fn diff_order_keyword_supports_repeated_partials() {
    let src = r#"
x = Scalar("x")
y = Scalar("y")
f = x^3 * y + y^2
fxx = Diff(f, x, order=2)
fxx.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "fxx = 6 \\, x \\, y");
}

#[test]
fn diff_order_zero_returns_original_expression() {
    let src = r#"
x = Scalar("x")
f = x^2 + 1
g = Diff(f, x, order=0)
g.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "g = {x}^{2} + 1");
}

#[test]
fn tanh_derivative_uses_builtin_rule() {
    let src = r#"
x = Scalar("x")
dx = Diff(tanh(x), x)
dx.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "dx = 1 - {\\left( \\tanh x \\right)}^{2}");
}

#[test]
fn unknown_function_renders_as_function_of_var() {
    let src = r#"
x = Var("x")
y = Function("y", x)
y.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "y = y\\left( x \\right)");
}

#[test]
fn derivative_command_builds_formal_derivatives() {
    let src = r#"
x = Var("x")
y = Function("y", x)
dy = Derivative(y, x)
d2y = Derivative(y, x, order=2)
dy.show()
d2y.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "dy = \\frac{d y}{d x}");
    assert_eq!(outputs[1].latex, "d2y = \\frac{d^{2} y}{d x^{2}}");
}

#[test]
fn derivative_command_builds_mixed_partials() {
    let src = r#"
x = Var("x")
z = Var("z")
u = Function("u", x, z)
ux = Derivative(u, x)
uxzz = Derivative(Derivative(u, x), z, order=2)
ux.show()
uxzz.show()
"#;
    let outputs = run_source(src).unwrap();
    assert!(outputs[0].latex.contains("\\partial"));
    assert!(outputs[0].latex.contains("\\partial x"));
    assert!(outputs[1].latex.contains("\\partial^{3}"));
    assert!(outputs[1].latex.contains("\\partial x"));
    assert!(outputs[1].latex.contains("\\partial z^{2}"));
}

#[test]
fn derivative_rejects_explicit_expression() {
    let src = r#"
x = Var("x")
f = x^2
df = Derivative(f, x)
"#;
    let err = run_source(src).unwrap_err();
    assert!(err.message.contains("use `Diff"), "got: {}", err.message);
}

#[test]
fn diff_rejects_unknown_function_derivatives() {
    let src = r#"
x = Var("x")
y = Function("y", x)
dy = Diff(y, x)
"#;
    let err = run_source(src).unwrap_err();
    assert!(
        err.message.contains("use `Derivative"),
        "got: {}",
        err.message
    );
}

#[test]
fn unknown_function_application_substitutes_its_var_argument() {
    let src = r#"
x = Var("x")
t = Var("t")
y = Function("y", x)
z = y(t)
z.show()
"#;
    let outputs = run_source(src).unwrap();
    assert_eq!(outputs[0].latex, "z = y\\left( t \\right)");
}
