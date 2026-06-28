use tensorforge::interpreter::OutputDetail;
use tensorforge::run_source;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn plots_single_scalar_curve() {
    let outputs = run_source(
        r#"
x = Var("x")
sin(x).plot(-pi, pi)
"#,
    )
    .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].latex, "\\sin x");
    let OutputDetail::Plot(plot) = outputs[0].detail.as_ref().expect("plot detail") else {
        panic!("expected plot detail");
    };
    assert_eq!(plot.x_label, "x");
    assert!(approx(plot.x_range[0], -std::f64::consts::PI, 1e-12));
    assert!(approx(plot.x_range[1], std::f64::consts::PI, 1e-12));
    assert_eq!(plot.series.len(), 1);
    assert!(plot.y_range[0] < -0.9);
    assert!(plot.y_range[1] > 0.9);
    assert!(plot.series[0]
        .segments
        .iter()
        .flatten()
        .any(|point| point[0].abs() < 0.01 && point[1].abs() < 0.01));
}

#[test]
fn plot_rejects_unbound_symbols() {
    let err = run_source(
        r#"
x = Var("x")
a = Scalar("a")
(a*x).plot(0, 1)
"#,
    )
    .unwrap_err();
    assert!(err.message.contains("`a` is unbound"));
}

#[test]
fn plot_rejects_expression_context() {
    let err = run_source(
        r#"
x = Var("x")
p = sin(x).plot(0, 1)
"#,
    )
    .unwrap_err();
    assert!(err.message.contains("`.plot(...)` is an output statement"));
}
