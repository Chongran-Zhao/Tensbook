//! Applied-math ODE/PDE classification and symbolic solvers.

use crate::differentiation::diff_scalar_by_scalar;
use crate::error::Error;
use crate::integration::{depends_on, integrate, scalar_symbol_name};
use crate::renderer::latex::scalar_to_latex;
use crate::simplifier::{simplify_scalar, RuleSet};
use crate::symbolic::{fold_add, fold_mul, fold_pow, fold_sub, ScalarExpr};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub lhs: Rc<ScalarExpr>,
    pub rhs: Rc<ScalarExpr>,
    pub residual: Rc<ScalarExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCondition {
    pub function: Rc<ScalarExpr>,
    pub point: Rc<ScalarExpr>,
    pub value: Rc<ScalarExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeProblem {
    pub equation: Equation,
    pub target: Rc<ScalarExpr>,
    pub independent: Rc<ScalarExpr>,
    pub boundary_conditions: Vec<BoundaryCondition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeClassification {
    pub equation: Equation,
    pub kind: DifferentialKind,
    pub order: usize,
    pub linear: bool,
    pub homogeneous: bool,
    pub subtypes: Vec<String>,
    pub reasons: Vec<String>,
    pub boundary_conditions: Vec<BoundaryCondition>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DifferentialKind {
    Ode,
    Pde,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution {
    pub method: String,
    pub equation: Equation,
    pub solution: Option<Equation>,
    pub steps: Vec<SolutionStep>,
    pub warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolutionStep {
    pub label: String,
    pub equation: Option<Equation>,
    pub expr: Option<Rc<ScalarExpr>>,
    pub latex: Option<String>,
}

impl SolutionStep {
    fn equation(label: &str, equation: Equation) -> Self {
        SolutionStep {
            label: label.to_string(),
            equation: Some(equation),
            expr: None,
            latex: None,
        }
    }

    fn expr(label: &str, expr: Rc<ScalarExpr>) -> Self {
        SolutionStep {
            label: label.to_string(),
            equation: None,
            expr: Some(expr),
            latex: None,
        }
    }

    fn latex(label: &str, latex: String) -> Self {
        SolutionStep {
            label: label.to_string(),
            equation: None,
            expr: None,
            latex: Some(latex),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMethod {
    Auto,
    Linear,
    Separable,
    Exact,
    Characteristic,
    Undetermined,
    Variation,
    PowerSeries,
    Frobenius,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveConfig {
    pub method: SolveMethod,
    pub about: Option<Rc<ScalarExpr>>,
    pub terms: usize,
}

impl Default for SolveConfig {
    fn default() -> Self {
        SolveConfig {
            method: SolveMethod::Auto,
            about: None,
            terms: 6,
        }
    }
}

impl SolveMethod {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "linear" => Some(Self::Linear),
            "separable" => Some(Self::Separable),
            "exact" => Some(Self::Exact),
            "characteristic" => Some(Self::Characteristic),
            "undetermined" => Some(Self::Undetermined),
            "variation" => Some(Self::Variation),
            "power_series" => Some(Self::PowerSeries),
            "frobenius" => Some(Self::Frobenius),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Linear => "linear",
            Self::Separable => "separable",
            Self::Exact => "exact",
            Self::Characteristic => "characteristic",
            Self::Undetermined => "undetermined",
            Self::Variation => "variation",
            Self::PowerSeries => "power_series",
            Self::Frobenius => "frobenius",
        }
    }
}

#[derive(Clone)]
struct Target {
    name: String,
    args: Vec<Rc<ScalarExpr>>,
}

#[derive(Clone)]
struct FirstOrderForm {
    m: Rc<ScalarExpr>,
    n: Rc<ScalarExpr>,
}

#[derive(Clone)]
struct SecondOrderLinearForm {
    a2: Rc<ScalarExpr>,
    a1: Rc<ScalarExpr>,
    a0: Rc<ScalarExpr>,
    forcing: Rc<ScalarExpr>,
}

#[derive(Clone, Copy)]
struct ConstantSecondOrder {
    a: f64,
    b: f64,
}

type ExactPotential = (Rc<ScalarExpr>, Vec<SolutionStep>);

struct SeparableForm {
    left_integrand: Rc<ScalarExpr>,
    right_integrand: Rc<ScalarExpr>,
    left_integral: Rc<ScalarExpr>,
    right_integral: Rc<ScalarExpr>,
}

struct ImplicitSolveParams<'a> {
    eq: &'a Equation,
    method: &'a str,
    independent: &'a Rc<ScalarExpr>,
    target: &'a Target,
    left: Rc<ScalarExpr>,
    right_without_c: Rc<ScalarExpr>,
    ic: Option<BoundaryCondition>,
    steps: Vec<SolutionStep>,
}

struct BoundaryLinearRow {
    c1: Rc<ScalarExpr>,
    c2: Rc<ScalarExpr>,
    rest: Rc<ScalarExpr>,
    value: Rc<ScalarExpr>,
}

pub fn equation(lhs: Rc<ScalarExpr>, rhs: Rc<ScalarExpr>) -> Equation {
    let residual = simplify_scalar(
        &Rc::new(ScalarExpr::Sub(lhs.clone(), rhs.clone())),
        RuleSet::Continuum,
    );
    Equation { lhs, rhs, residual }
}

pub fn classify(
    eq: &Equation,
    target: &Rc<ScalarExpr>,
    independent: &Rc<ScalarExpr>,
) -> Result<OdeClassification, Error> {
    let target = target_from_expr(target)?;
    let x_name = scalar_symbol_name(independent)
        .ok_or_else(|| Error::msg("ODE expects an independent variable such as x"))?;
    let occurrences = target_occurrences(&eq.residual, &target.name);
    let order = occurrences
        .iter()
        .map(|orders| orders.iter().sum())
        .max()
        .unwrap_or(0);
    let kind = if target.args.len() > 1
        || occurrences.iter().any(|orders| {
            orders.iter().enumerate().any(|(i, order)| {
                *order > 0
                    && target
                        .args
                        .get(i)
                        .is_some_and(|a| scalar_symbol_name(a).is_some_and(|name| name != x_name))
            })
        }) {
        DifferentialKind::Pde
    } else {
        DifferentialKind::Ode
    };
    let linear = target_degree(&eq.residual, &target.name).is_some_and(|degree| degree <= 1);
    let homogeneous = {
        let zeroed = simplify_scalar(&zero_target(&eq.residual, &target.name), RuleSet::Continuum);
        is_zero(&zeroed)
    };
    let mut subtypes = Vec::new();
    if matches!(kind, DifferentialKind::Ode) && order == 1 {
        if linear_first_order_coeffs(&eq.residual, &target, x_name).is_some() {
            subtypes.push("linear".to_string());
        }
        if separable_form(&eq.residual, &target, independent).is_some() {
            subtypes.push("separable".to_string());
        }
        if exact_form(&eq.residual, &target, independent)?.is_some() {
            subtypes.push("exact".to_string());
        }
    }
    if matches!(kind, DifferentialKind::Ode) && order == 2 && linear {
        if let Some(form) = second_order_linear_form(&eq.residual, &target) {
            if constant_second_order_coefficients(&form).is_some() {
                subtypes.push("constant_coefficients".to_string());
            }
            if is_regular_singular_candidate(&eq.residual, &target, independent) {
                subtypes.push("regular_singular".to_string());
            } else {
                subtypes.push("ordinary_point".to_string());
            }
        }
    }
    let mut reasons = Vec::new();
    reasons.push(format!("{} independent variable(s)", target.args.len()));
    reasons.push(format!("highest derivative order = {order}"));
    reasons.push(if linear {
        "target function appears linearly".to_string()
    } else {
        "target function appears nonlinearly".to_string()
    });
    Ok(OdeClassification {
        equation: eq.clone(),
        kind,
        order,
        linear,
        homogeneous,
        subtypes,
        reasons,
        boundary_conditions: Vec::new(),
    })
}

pub fn problem(
    equation: Equation,
    target: Rc<ScalarExpr>,
    independent: Rc<ScalarExpr>,
    boundary_conditions: Vec<BoundaryCondition>,
) -> OdeProblem {
    OdeProblem {
        equation,
        target,
        independent,
        boundary_conditions,
    }
}

pub fn classify_problem(problem: &OdeProblem) -> Result<OdeClassification, Error> {
    let mut class = classify(&problem.equation, &problem.target, &problem.independent)?;
    class.boundary_conditions = problem.boundary_conditions.clone();
    Ok(class)
}

pub fn solve_problem(problem: &OdeProblem) -> Result<OdeSolution, Error> {
    solve_problem_with_config(problem, SolveConfig::default())
}

pub fn solve_problem_with_method(
    problem: &OdeProblem,
    method: SolveMethod,
) -> Result<OdeSolution, Error> {
    solve_problem_with_config(
        problem,
        SolveConfig {
            method,
            ..SolveConfig::default()
        },
    )
}

pub fn solve_problem_with_config(
    problem: &OdeProblem,
    config: SolveConfig,
) -> Result<OdeSolution, Error> {
    solve(
        &problem.equation,
        &problem.target,
        &problem.independent,
        problem.boundary_conditions.clone(),
        &config,
    )
}

pub fn solve(
    eq: &Equation,
    target: &Rc<ScalarExpr>,
    independent: &Rc<ScalarExpr>,
    boundary_conditions: Vec<BoundaryCondition>,
    config: &SolveConfig,
) -> Result<OdeSolution, Error> {
    let class = classify(eq, target, independent)?;
    if matches!(class.kind, DifferentialKind::Pde) {
        return Ok(OdeSolution {
            method: "unsupported".to_string(),
            equation: eq.clone(),
            solution: None,
            steps: vec![],
            warning: Some("PDE solving is not implemented".to_string()),
        });
    }
    let target = target_from_expr(target)?;
    let x_name = scalar_symbol_name(independent)
        .ok_or_else(|| Error::msg("ODE.solve expects an independent variable such as x"))?;
    let method = config.method;
    let first_boundary_condition = boundary_conditions.first().cloned();

    if matches!(method, SolveMethod::Auto | SolveMethod::Linear) {
        if let Some((p, q, form)) = linear_first_order_coeffs(&eq.residual, &target, x_name) {
            return solve_linear_first_order(
                eq,
                &target,
                independent,
                p,
                q,
                form,
                first_boundary_condition.clone(),
            );
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Separable) {
        if let Some(form) = separable_form(&eq.residual, &target, independent) {
            return solve_implicit_pair(ImplicitSolveParams {
                eq,
                method: "separable",
                independent,
                target: &target,
                left: form.left_integral.clone(),
                right_without_c: form.right_integral.clone(),
                ic: first_boundary_condition.clone(),
                steps: separable_solution_steps(&form, independent, &target),
            });
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Exact) {
        if let Some((phi, steps)) = exact_form(&eq.residual, &target, independent)? {
            return solve_implicit_pair(ImplicitSolveParams {
                eq,
                method: "exact",
                independent,
                target: &target,
                left: phi,
                right_without_c: Rc::new(ScalarExpr::Num(0.0)),
                ic: first_boundary_condition.clone(),
                steps,
            });
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Characteristic) {
        if let Some(form) = second_order_linear_form(&eq.residual, &target) {
            if let Some(coeffs) = constant_second_order_homogeneous(&form) {
                return Ok(solve_characteristic_second_order(
                    eq,
                    &target,
                    independent,
                    coeffs,
                    &boundary_conditions,
                ));
            }
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Undetermined) {
        if let Some(form) = second_order_linear_form(&eq.residual, &target) {
            if let Some(coeffs) = constant_second_order_coefficients(&form) {
                if !is_zero(&form.forcing) {
                    if let Some(solution) = solve_undetermined_second_order(
                        eq,
                        &target,
                        independent,
                        &form.forcing,
                        coeffs,
                        &boundary_conditions,
                    ) {
                        return Ok(solution);
                    }
                }
            }
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Variation) {
        if let Some(form) = second_order_linear_form(&eq.residual, &target) {
            if let Some(coeffs) = constant_second_order_coefficients(&form) {
                if !is_zero(&form.forcing) {
                    return Ok(solve_variation_second_order(
                        eq,
                        &target,
                        independent,
                        &form.forcing,
                        coeffs,
                    ));
                }
            }
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::PowerSeries) {
        if let Some(solution) = solve_power_series(eq, &target, independent, config) {
            return Ok(solution);
        }
    }
    if matches!(method, SolveMethod::Auto | SolveMethod::Frobenius) {
        if let Some(solution) = solve_frobenius(eq, &target, independent, config) {
            return Ok(solution);
        }
    }

    if !matches!(method, SolveMethod::Auto) {
        return Ok(unavailable_method_solution(eq, method, &class));
    }

    Ok(OdeSolution {
        method: "unsupported".to_string(),
        equation: eq.clone(),
        solution: None,
        steps: vec![],
        warning: Some(unsupported_method_reason(&class)),
    })
}

fn solve_linear_first_order(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    p: Rc<ScalarExpr>,
    q: Rc<ScalarExpr>,
    form: FirstOrderForm,
    ic: Option<BoundaryCondition>,
) -> Result<OdeSolution, Error> {
    let mu = Rc::new(ScalarExpr::Func {
        name: "exp".to_string(),
        arg: integrate(&p, independent)?,
    });
    let integral = integrate(
        &Rc::new(ScalarExpr::Mul(mu.clone(), q.clone())),
        independent,
    )?;
    let c = constant();
    let rhs = simplify_scalar(
        &Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Add(integral.clone(), c.clone())),
            mu.clone(),
        )),
        RuleSet::Continuum,
    );
    let mut solution = equation(base_unknown(target), rhs.clone());
    let mut warning = None;
    if let Some(ic) = ic {
        match solve_constant_from_explicit(&rhs, target, independent, &ic, &c) {
            Some(value) => {
                solution = equation(
                    base_unknown(target),
                    simplify_scalar(
                        &subst_symbol(&rhs, constant_name(), &value),
                        RuleSet::Continuum,
                    ),
                );
            }
            None => {
                warning = Some("initial condition could not be applied symbolically".to_string())
            }
        }
    }
    Ok(OdeSolution {
        method: "linear first-order integrating factor".to_string(),
        equation: eq.clone(),
        solution: Some(solution),
        steps: vec![
            SolutionStep::equation(
                "standard form",
                equation(
                    Rc::new(ScalarExpr::Add(
                        derivative_unknown(target, 1),
                        Rc::new(ScalarExpr::Mul(p.clone(), base_unknown(target))),
                    )),
                    q,
                ),
            ),
            SolutionStep::expr("coefficient of derivative", form.n),
            SolutionStep::equation("integrating factor", equation(symbol("\\mu"), mu)),
        ],
        warning,
    })
}

fn solve_implicit_pair(params: ImplicitSolveParams<'_>) -> Result<OdeSolution, Error> {
    let ImplicitSolveParams {
        eq,
        method,
        independent,
        target,
        left,
        right_without_c,
        ic,
        mut steps,
    } = params;
    let c = constant();
    let mut rhs = simplify_scalar(
        &Rc::new(ScalarExpr::Add(right_without_c.clone(), c.clone())),
        RuleSet::Continuum,
    );
    let mut warning = None;
    if let Some(ic) = ic {
        if let Some(value) =
            solve_constant_from_implicit(&left, &right_without_c, target, independent, &ic)
        {
            rhs = simplify_scalar(
                &Rc::new(ScalarExpr::Add(right_without_c, value)),
                RuleSet::Continuum,
            );
        } else {
            warning = Some("initial condition could not be applied symbolically".to_string());
        }
    }
    let solution = equation(left, rhs);
    steps.push(SolutionStep::equation(
        "implicit solution",
        solution.clone(),
    ));
    Ok(OdeSolution {
        method: method.to_string(),
        equation: eq.clone(),
        solution: Some(solution),
        steps,
        warning,
    })
}

fn second_order_linear_form(
    residual: &Rc<ScalarExpr>,
    target: &Target,
) -> Option<SecondOrderLinearForm> {
    let y = base_unknown(target);
    let dy = derivative_unknown(target, 1);
    let ddy = derivative_unknown(target, 2);
    let mut a2 = Rc::new(ScalarExpr::Num(0.0));
    let mut a1 = Rc::new(ScalarExpr::Num(0.0));
    let mut a0 = Rc::new(ScalarExpr::Num(0.0));
    let mut rest = Rc::new(ScalarExpr::Num(0.0));
    for term in additive_terms(residual) {
        if let Some(coeff) = factor_atom(&term.expr, &ddy) {
            if contains_target(&coeff, &target.name) {
                return None;
            }
            a2 = fold_add(a2, signed(term.sign, coeff));
        } else if let Some(coeff) = factor_atom(&term.expr, &dy) {
            if contains_target(&coeff, &target.name) {
                return None;
            }
            a1 = fold_add(a1, signed(term.sign, coeff));
        } else if let Some(coeff) = factor_atom(&term.expr, &y) {
            if contains_target(&coeff, &target.name) {
                return None;
            }
            a0 = fold_add(a0, signed(term.sign, coeff));
        } else {
            if contains_target(&term.expr, &target.name) {
                return None;
            }
            rest = fold_add(rest, signed(term.sign, term.expr));
        }
    }
    (!is_zero(&a2)).then(|| SecondOrderLinearForm {
        a2: simplify_scalar(&a2, RuleSet::Continuum),
        a1: simplify_scalar(&a1, RuleSet::Continuum),
        a0: simplify_scalar(&a0, RuleSet::Continuum),
        forcing: simplify_scalar(&Rc::new(ScalarExpr::Neg(rest)), RuleSet::Continuum),
    })
}

fn constant_second_order_coefficients(form: &SecondOrderLinearForm) -> Option<ConstantSecondOrder> {
    let a2 = numeric_scalar_value(&form.a2)?;
    if a2 == 0.0 {
        return None;
    }
    Some(ConstantSecondOrder {
        a: numeric_scalar_value(&form.a1)? / a2,
        b: numeric_scalar_value(&form.a0)? / a2,
    })
}

fn constant_second_order_homogeneous(form: &SecondOrderLinearForm) -> Option<ConstantSecondOrder> {
    if is_zero(&form.forcing) {
        constant_second_order_coefficients(form)
    } else {
        None
    }
}

fn solve_characteristic_second_order(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    coeffs: ConstantSecondOrder,
    bcs: &[BoundaryCondition],
) -> OdeSolution {
    let mut rhs = homogeneous_solution_rhs(coeffs, independent);
    let mut warning = None;
    if let Some(applied) = apply_second_order_boundaries(&rhs, target, independent, bcs) {
        rhs = applied;
    } else if !bcs.is_empty() {
        warning = Some("boundary conditions could not be applied symbolically".to_string());
    }
    let solution = equation(base_unknown(target), rhs);
    OdeSolution {
        method: "characteristic".to_string(),
        equation: eq.clone(),
        solution: Some(solution),
        steps: characteristic_steps(coeffs),
        warning,
    }
}

fn characteristic_steps(coeffs: ConstantSecondOrder) -> Vec<SolutionStep> {
    let disc = coeffs.a * coeffs.a - 4.0 * coeffs.b;
    let mut steps = vec![SolutionStep::latex(
        "characteristic equation",
        format!(
            "\\lambda^2 + {}\\lambda + {} = 0",
            num_latex(coeffs.a),
            num_latex(coeffs.b)
        ),
    )];
    if disc > 0.0 {
        let r1 = (-coeffs.a + disc.sqrt()) / 2.0;
        let r2 = (-coeffs.a - disc.sqrt()) / 2.0;
        steps.push(SolutionStep::latex(
            "two distinct real roots",
            format!(
                "\\lambda_1={},\\qquad \\lambda_2={}",
                num_latex(r1),
                num_latex(r2)
            ),
        ));
    } else if disc == 0.0 {
        let r = -coeffs.a / 2.0;
        steps.push(SolutionStep::latex(
            "repeated real root",
            format!("\\lambda_1=\\lambda_2={}", num_latex(r)),
        ));
    } else {
        let alpha = -coeffs.a / 2.0;
        let beta = (-disc).sqrt() / 2.0;
        steps.push(SolutionStep::latex(
            "complex roots",
            format!("\\lambda={}\\pm{}i", num_latex(alpha), num_latex(beta)),
        ));
    }
    steps
}

fn homogeneous_solution_rhs(
    coeffs: ConstantSecondOrder,
    independent: &Rc<ScalarExpr>,
) -> Rc<ScalarExpr> {
    let disc = coeffs.a * coeffs.a - 4.0 * coeffs.b;
    let c1 = symbol("C_1");
    let c2 = symbol("C_2");
    if disc > 0.0 {
        let r1 = (-coeffs.a + disc.sqrt()) / 2.0;
        let r2 = (-coeffs.a - disc.sqrt()) / 2.0;
        let u1 = exp_linear(r1, independent);
        let u2 = exp_linear(r2, independent);
        simplify_scalar(
            &Rc::new(ScalarExpr::Add(fold_mul(&c1, &u1), fold_mul(&c2, &u2))),
            RuleSet::Continuum,
        )
    } else if disc == 0.0 {
        let r = -coeffs.a / 2.0;
        let factor = simplify_scalar(
            &Rc::new(ScalarExpr::Add(c1, fold_mul(&c2, independent))),
            RuleSet::Continuum,
        );
        simplify_scalar(
            &fold_mul(&factor, &exp_linear(r, independent)),
            RuleSet::Continuum,
        )
    } else {
        let alpha = -coeffs.a / 2.0;
        let beta = (-disc).sqrt() / 2.0;
        let cos = Rc::new(ScalarExpr::Func {
            name: "cos".to_string(),
            arg: scale_arg(beta, independent),
        });
        let sin = Rc::new(ScalarExpr::Func {
            name: "sin".to_string(),
            arg: scale_arg(beta, independent),
        });
        let oscillation = simplify_scalar(
            &Rc::new(ScalarExpr::Add(fold_mul(&c1, &cos), fold_mul(&c2, &sin))),
            RuleSet::Continuum,
        );
        simplify_scalar(
            &fold_mul(&exp_linear(alpha, independent), &oscillation),
            RuleSet::Continuum,
        )
    }
}

fn solve_undetermined_second_order(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    forcing: &Rc<ScalarExpr>,
    coeffs: ConstantSecondOrder,
    bcs: &[BoundaryCondition],
) -> Option<OdeSolution> {
    let x_name = scalar_symbol_name(independent)?;
    let forcing_coeffs = polynomial_coeffs(forcing, x_name)?;
    if coeffs.b == 0.0 {
        return None;
    }
    let particular_coeffs = polynomial_particular_coeffs(&forcing_coeffs, coeffs)?;
    let particular = polynomial_expr(&particular_coeffs, independent);
    let mut rhs = simplify_scalar(
        &Rc::new(ScalarExpr::Add(
            homogeneous_solution_rhs(coeffs, independent),
            particular.clone(),
        )),
        RuleSet::Continuum,
    );
    let mut warning = None;
    if let Some(applied) = apply_second_order_boundaries(&rhs, target, independent, bcs) {
        rhs = applied;
    } else if !bcs.is_empty() {
        warning = Some("boundary conditions could not be applied symbolically".to_string());
    }
    let solution = equation(base_unknown(target), rhs);
    Some(OdeSolution {
        method: "undetermined coefficients".to_string(),
        equation: eq.clone(),
        solution: Some(solution),
        steps: vec![
            SolutionStep::latex(
                "homogeneous solution",
                format!(
                    "y_H = {}",
                    ode_scalar_to_latex(&homogeneous_solution_rhs(coeffs, independent))
                ),
            ),
            SolutionStep::latex(
                "particular ansatz",
                format!(
                    "y_p = {}",
                    polynomial_template_latex(forcing_coeffs.len() - 1)
                ),
            ),
            SolutionStep::expr("particular solution", particular),
        ],
        warning,
    })
}

fn solve_variation_second_order(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    forcing: &Rc<ScalarExpr>,
    coeffs: ConstantSecondOrder,
) -> OdeSolution {
    let u1 = symbol("u_1");
    let u2 = symbol("u_2");
    let w = symbol("W");
    let x_latex = ode_scalar_to_latex(independent);
    let g_latex = ode_scalar_to_latex(forcing);
    let rhs = Rc::new(ScalarExpr::Add(
        homogeneous_solution_rhs(coeffs, independent),
        symbol("u_p"),
    ));
    OdeSolution {
        method: "variation of parameters".to_string(),
        equation: eq.clone(),
        solution: Some(equation(base_unknown(target), rhs)),
        steps: vec![
            SolutionStep::latex(
                "homogeneous basis",
                format!(
                    "u_1,u_2\\ \\text{{come from }}\\ {}",
                    ode_scalar_to_latex(&homogeneous_solution_rhs(coeffs, independent))
                ),
            ),
            SolutionStep::latex(
                "Wronskian",
                format!(
                    "{} = \\begin{{vmatrix}}u_1&u_2\\\\u_1'&u_2'\\end{{vmatrix}}",
                    ode_scalar_to_latex(&w)
                ),
            ),
            SolutionStep::latex(
                "parameters",
                format!(
                    "V_1=-\\int \\frac{{{}\\,{}}}{{W}}\\,d{},\\qquad V_2=\\int \\frac{{{}\\,{}}}{{W}}\\,d{}",
                    ode_scalar_to_latex(&u2),
                    g_latex,
                    x_latex,
                    ode_scalar_to_latex(&u1),
                    g_latex,
                    x_latex
                ),
            ),
            SolutionStep::latex(
                "particular solution",
                "u_p=V_1u_1+V_2u_2".to_string(),
            ),
        ],
        warning: Some(
            "variation of parameters is shown formally; automatic integration is not attempted"
                .to_string(),
        ),
    }
}

fn solve_power_series(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    config: &SolveConfig,
) -> Option<OdeSolution> {
    let about = config
        .about
        .clone()
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(0.0)));
    let terms = config.terms.max(2);
    let series = power_series_latex(target, independent, &about, terms);
    let mut steps = vec![
        SolutionStep::latex("assume power series", series.clone()),
        SolutionStep::latex(
            "differentiate",
            "y'=\\sum_{n=1}^{\\infty}nC_n(x-a)^{n-1},\\qquad y''=\\sum_{n=2}^{\\infty}n(n-1)C_n(x-a)^{n-2}".to_string(),
        ),
    ];
    let recurrence = lecture_power_series_recurrence(eq, target, independent)
        .unwrap_or_else(|| "\\text{match coefficients to obtain a recurrence for }C_n".to_string());
    steps.push(SolutionStep::latex("recurrence", recurrence));
    let solution_latex = format!("{} + \\cdots", finite_series_latex(independent, terms));
    Some(OdeSolution {
        method: "power series".to_string(),
        equation: eq.clone(),
        solution: Some(equation(
            base_unknown(target),
            symbol_with_latex("series", &solution_latex),
        )),
        steps,
        warning: Some("power series output is a finite displayed expansion".to_string()),
    })
}

fn solve_frobenius(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    config: &SolveConfig,
) -> Option<OdeSolution> {
    let about = config
        .about
        .clone()
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(0.0)));
    if !is_zero(&about) {
        return None;
    }
    let bessel = bessel_parameter(&eq.residual, target, independent);
    let (indicial, roots, solution) = if let Some(lambda) = bessel {
        let lam = ode_scalar_to_latex(&lambda);
        (
            format!("r^2-{}^2=0", lam),
            format!("r=\\pm {}", lam),
            format!("A J_{{{}}}(x)+B J_{{-{}}}(x)", lam, lam),
        )
    } else if is_regular_singular_candidate(&eq.residual, target, independent) {
        (
            "r(r-1)+p(0)r+q(0)=0".to_string(),
            "\\text{classify roots as distinct, repeated, or integer-spaced}".to_string(),
            "x^r\\sum_{n=0}^{\\infty}C_nx^n".to_string(),
        )
    } else {
        return None;
    };
    Some(OdeSolution {
        method: "frobenius".to_string(),
        equation: eq.clone(),
        solution: Some(equation(
            base_unknown(target),
            symbol_with_latex("frobenius", &solution),
        )),
        steps: vec![
            SolutionStep::latex(
                "Frobenius ansatz",
                "y=x^r\\sum_{n=0}^{\\infty}C_nx^n".to_string(),
            ),
            SolutionStep::latex("indicial equation", indicial),
            SolutionStep::latex("indicial roots", roots),
        ],
        warning: Some(
            "Frobenius output is formal; full recurrence expansion is not implemented".to_string(),
        ),
    })
}

fn linear_first_order_coeffs(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    x_name: &str,
) -> Option<(Rc<ScalarExpr>, Rc<ScalarExpr>, FirstOrderForm)> {
    let form = first_order_form(residual, target)?;
    if contains_target(&form.n, &target.name) {
        return None;
    }
    let y = base_unknown(target);
    let mut a0 = Rc::new(ScalarExpr::Num(0.0));
    let mut c = Rc::new(ScalarExpr::Num(0.0));
    for term in additive_terms(&form.m) {
        if let Some(coeff) = factor_atom(&term.expr, &y) {
            if contains_target(&coeff, &target.name) {
                return None;
            }
            a0 = fold_add(a0, signed(term.sign, coeff));
        } else if contains_target(&term.expr, &target.name) {
            return None;
        } else {
            c = fold_add(c, signed(term.sign, term.expr));
        }
    }
    if depends_on(&form.n, &target.name) || !depends_on_or_const(&form.n, x_name) {
        return None;
    }
    let p = simplify_scalar(
        &Rc::new(ScalarExpr::Div(a0, form.n.clone())),
        RuleSet::Continuum,
    );
    let q = simplify_scalar(
        &Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Neg(c)), form.n.clone())),
        RuleSet::Continuum,
    );
    Some((p, q, form))
}

fn first_order_form(residual: &Rc<ScalarExpr>, target: &Target) -> Option<FirstOrderForm> {
    let dy = derivative_unknown(target, 1);
    let mut n = Rc::new(ScalarExpr::Num(0.0));
    let mut m = Rc::new(ScalarExpr::Num(0.0));
    for term in additive_terms(residual) {
        if let Some(coeff) = factor_atom(&term.expr, &dy) {
            n = fold_add(n, signed(term.sign, coeff));
        } else {
            if target_occurrences(&term.expr, &target.name)
                .iter()
                .any(|orders| orders.iter().sum::<usize>() > 1)
            {
                return None;
            }
            m = fold_add(m, signed(term.sign, term.expr));
        }
    }
    (!is_zero(&n)).then_some(FirstOrderForm {
        m: simplify_scalar(&m, RuleSet::Continuum),
        n: simplify_scalar(&n, RuleSet::Continuum),
    })
}

fn separable_form(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> Option<SeparableForm> {
    let x_name = scalar_symbol_name(independent)?;
    let form = first_order_form(residual, target)?;
    let y_var = target_symbol(target);
    let n = replace_base_target_with_symbol(&form.n, target, &y_var);
    let m = replace_base_target_with_symbol(&form.m, target, &y_var);
    let y_name = scalar_symbol_name(&y_var)?;

    if depends_only_on(&n, y_name, x_name) && depends_only_on(&m, x_name, y_name) {
        let right_integrand = simplify_scalar(&Rc::new(ScalarExpr::Neg(m)), RuleSet::Continuum);
        let left_integral = integrate(&n, &y_var).ok()?;
        let right_integral = integrate(&right_integrand, independent).ok()?;
        return Some(SeparableForm {
            left_integrand: n,
            right_integrand,
            left_integral,
            right_integral,
        });
    }

    if is_one(&n) {
        let rhs = simplify_scalar(&Rc::new(ScalarExpr::Neg(m)), RuleSet::Continuum);
        let (gx, hy) = split_xy_product(&rhs, x_name, y_name)?;
        let left_integrand = simplify_scalar(
            &Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), hy)),
            RuleSet::Continuum,
        );
        let left_integral = integrate(&left_integrand, &y_var).ok()?;
        let right_integral = integrate(&gx, independent).ok()?;
        return Some(SeparableForm {
            left_integrand,
            right_integrand: gx,
            left_integral,
            right_integral,
        });
    }
    None
}

fn separable_solution_steps(
    form: &SeparableForm,
    independent: &Rc<ScalarExpr>,
    target: &Target,
) -> Vec<SolutionStep> {
    let y_latex = scalar_to_latex(&target_symbol(target));
    let x_latex = scalar_to_latex(independent);
    vec![
        SolutionStep::latex(
            "separate variables",
            format!(
                "{}\\, d{} = {}\\, d{}",
                ode_scalar_to_latex(&form.left_integrand),
                y_latex,
                ode_scalar_to_latex(&form.right_integrand),
                x_latex,
            ),
        ),
        SolutionStep::latex(
            "integrate both sides",
            format!(
                "\\int {}\\, d{} = \\int {}\\, d{}",
                ode_scalar_to_latex(&form.left_integrand),
                y_latex,
                ode_scalar_to_latex(&form.right_integrand),
                x_latex,
            ),
        ),
    ]
}

fn differential_form_latex(m: &Rc<ScalarExpr>, n: &Rc<ScalarExpr>) -> String {
    format!(
        "{}\\, dx + {}\\, dy = 0",
        ode_scalar_to_latex(m),
        ode_scalar_to_latex(n),
    )
}

fn identify_mn_latex(m: &Rc<ScalarExpr>, n: &Rc<ScalarExpr>) -> String {
    format!(
        "M = {},\\qquad N = {}",
        ode_scalar_to_latex(m),
        ode_scalar_to_latex(n),
    )
}

fn unavailable_method_solution(
    eq: &Equation,
    method: SolveMethod,
    class: &OdeClassification,
) -> OdeSolution {
    let available = supported_method_labels(class);
    let available_text = if available.is_empty() {
        "none".to_string()
    } else {
        available.join(", ")
    };
    OdeSolution {
        method: "unsupported".to_string(),
        equation: eq.clone(),
        solution: None,
        steps: vec![],
        warning: Some(format!(
            "requested method `{}` is not available; available methods: {}",
            method.label(),
            available_text
        )),
    }
}

fn exact_form(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> Result<Option<ExactPotential>, Error> {
    let Some(form) = first_order_form(residual, target) else {
        return Ok(None);
    };
    let Some(x_name) = scalar_symbol_name(independent) else {
        return Ok(None);
    };
    let y_var = target_symbol(target);
    let Some(y_name) = scalar_symbol_name(&y_var) else {
        return Ok(None);
    };
    let m = replace_base_target_with_symbol(&form.m, target, &y_var);
    let n = replace_base_target_with_symbol(&form.n, target, &y_var);
    let my = simplify_scalar(&diff_scalar_by_scalar(&m, y_name)?, RuleSet::Continuum);
    let nx = simplify_scalar(&diff_scalar_by_scalar(&n, x_name)?, RuleSet::Continuum);
    if my != nx {
        return Ok(None);
    }
    let phi_m = {
        let int_m = integrate(&m, independent)?;
        let int_m_y = simplify_scalar(&diff_scalar_by_scalar(&int_m, y_name)?, RuleSet::Continuum);
        let g_prime = if scalar_equiv_commutative(&n, &int_m_y) {
            Rc::new(ScalarExpr::Num(0.0))
        } else {
            simplify_scalar(&fold_sub(n.clone(), int_m_y), RuleSet::Continuum)
        };
        let g = integrate(&g_prime, &y_var)?;
        simplify_scalar(&Rc::new(ScalarExpr::Add(int_m, g)), RuleSet::Continuum)
    };
    let phi_n = {
        let int_n = integrate(&n, &y_var)?;
        let int_n_x = simplify_scalar(&diff_scalar_by_scalar(&int_n, x_name)?, RuleSet::Continuum);
        let f_prime = if scalar_equiv_commutative(&m, &int_n_x) {
            Rc::new(ScalarExpr::Num(0.0))
        } else {
            simplify_scalar(&fold_sub(m.clone(), int_n_x), RuleSet::Continuum)
        };
        let f = integrate(&f_prime, independent)?;
        simplify_scalar(&Rc::new(ScalarExpr::Add(int_n, f)), RuleSet::Continuum)
    };
    let phi = if contains_formal_integral(&phi_n) && !contains_formal_integral(&phi_m) {
        phi_m
    } else {
        phi_n
    };
    Ok(Some((
        phi.clone(),
        vec![
            SolutionStep::latex("differential form", differential_form_latex(&m, &n)),
            SolutionStep::latex("identify M and N", identify_mn_latex(&m, &n)),
            SolutionStep::equation("exactness check", equation(my, nx)),
            SolutionStep::expr("potential", phi.clone()),
        ],
    )))
}

pub fn render_equation(eq: &Equation) -> String {
    format!(
        "{} = {}",
        ode_scalar_to_latex(&eq.lhs),
        ode_scalar_to_latex(&eq.rhs)
    )
}

fn ode_scalar_to_latex(expr: &Rc<ScalarExpr>) -> String {
    scalar_to_latex(&ode_display_expr(expr))
}

fn ode_display_expr(expr: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    match &**expr {
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _)
        | ScalarExpr::SetElem { .. } => expr.clone(),
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(ode_display_expr(a), ode_display_expr(b))),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(ode_display_expr(a), ode_display_expr(b))),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(ode_display_expr(a), ode_display_expr(b))),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(ode_display_expr(a), ode_display_expr(b))),
        ScalarExpr::Pow(base, exp) => Rc::new(ScalarExpr::Pow(
            ode_display_expr(base),
            ode_display_expr(exp),
        )),
        ScalarExpr::Neg(inner) => Rc::new(ScalarExpr::Neg(ode_display_expr(inner))),
        ScalarExpr::Log(inner) => Rc::new(ScalarExpr::Log(ode_display_expr(inner))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: ode_display_expr(arg),
        }),
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } => {
            if derivative_orders.iter().all(|order| *order == 0) {
                Rc::new(ScalarExpr::Sym {
                    name: name.clone(),
                    latex: name.clone(),
                })
            } else {
                Rc::new(ScalarExpr::UnknownFunc {
                    name: name.clone(),
                    args: args.clone(),
                    derivative_orders: derivative_orders.clone(),
                })
            }
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => Rc::new(ScalarExpr::Integral {
            integrand: ode_display_expr(integrand),
            variable: variable.clone(),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: ode_display_expr(body),
            index: index.clone(),
            dim: *dim,
        }),
    }
}

pub fn render_boundary_condition(bc: &BoundaryCondition) -> String {
    if let ScalarExpr::UnknownFunc {
        args,
        derivative_orders,
        ..
    } = &*bc.function
    {
        if derivative_orders.iter().sum::<usize>() > 0 && args.len() == 1 {
            return format!(
                "\\left. {} \\right|_{{{}={}}} = {}",
                scalar_to_latex(&bc.function),
                scalar_to_latex(&args[0]),
                scalar_to_latex(&bc.point),
                scalar_to_latex(&bc.value)
            );
        }
    }
    format!(
        "{} = {}",
        scalar_to_latex(&bc.function),
        scalar_to_latex(&bc.value)
    )
}

fn render_boundary_condition_status(bcs: &[BoundaryCondition]) -> String {
    if bcs.is_empty() {
        "\\text{no explicit boundary condition}".to_string()
    } else {
        bcs.iter()
            .map(render_boundary_condition)
            .collect::<Vec<_>>()
            .join(",\\quad ")
    }
}

pub fn render_problem(problem: &OdeProblem, mode: &str) -> Result<String, Error> {
    match mode {
        "symbol" | "equation" => Ok(render_equation(&problem.equation)),
        "boundary" => Ok(render_boundary_condition_status(
            &problem.boundary_conditions,
        )),
        "classification" => {
            let class = classify_problem(problem)?;
            Ok(render_classification(&class, "symbol"))
        }
        "methods" => {
            let class = classify_problem(problem)?;
            Ok(render_available_methods(&class))
        }
        _ => Err(Error::msg(format!("unknown ODE display mode `{mode}`"))),
    }
}

pub fn render_classification(class: &OdeClassification, mode: &str) -> String {
    let _ = mode;
    classification_summary(class)
}

fn classification_values(class: &OdeClassification) -> (String, String, String, String) {
    let kind = match class.kind {
        DifferentialKind::Ode => "ODE",
        DifferentialKind::Pde => "PDE",
    };
    let linear = if class.linear { "linear" } else { "nonlinear" };
    let homogeneous = if class.homogeneous {
        "homogeneous"
    } else {
        "non-homogeneous"
    };
    (
        kind.to_string(),
        class.order.to_string(),
        linear.to_string(),
        homogeneous.to_string(),
    )
}

fn classification_summary(class: &OdeClassification) -> String {
    let (kind, order, linear, homogeneous) = classification_values(class);
    let parts = [kind, format!("order {order}"), linear, homogeneous];
    let cols = "c".repeat(parts.len());
    let row = parts
        .iter()
        .map(|part| format!("\\text{{{}}}", escape_text(part)))
        .collect::<Vec<_>>()
        .join(" & ");
    format!("\\begin{{array}}{{{cols}}}{row}\\end{{array}}")
}

pub fn render_available_methods(class: &OdeClassification) -> String {
    let methods = supported_method_labels(class);
    let available = if methods.is_empty() {
        "\\text{none}".to_string()
    } else {
        methods
            .iter()
            .map(|method| format!("\\text{{{}}}", escape_text(method)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let default = methods
        .first()
        .map(|method| format!("\\text{{{}}}", escape_text(method)))
        .unwrap_or_else(|| {
            format!(
                "\\text{{{}}}",
                escape_text(&unsupported_method_reason(class))
            )
        });
    format!(
        "\\begin{{array}}{{rl}}\n\\text{{available methods}} & {}\\\\[3pt]\n\\text{{default solve path}} & {}\n\\end{{array}}",
        available, default
    )
}

fn supported_method_labels(class: &OdeClassification) -> Vec<String> {
    if !matches!(class.kind, DifferentialKind::Ode) {
        return Vec::new();
    }
    if class.order == 2 && class.linear {
        let mut methods = Vec::new();
        if class
            .subtypes
            .iter()
            .any(|item| item == "constant_coefficients")
            && class.homogeneous
        {
            methods.push("characteristic".to_string());
        }
        if class
            .subtypes
            .iter()
            .any(|item| item == "constant_coefficients")
            && !class.homogeneous
        {
            methods.push("undetermined coefficients".to_string());
            methods.push("variation of parameters".to_string());
        }
        if class.subtypes.iter().any(|item| item == "ordinary_point") {
            methods.push("power series".to_string());
        }
        if class.subtypes.iter().any(|item| item == "regular_singular") {
            methods.push("frobenius".to_string());
        }
        return methods;
    }
    if class.order != 1 {
        return Vec::new();
    }
    let mut methods = Vec::new();
    for subtype in ["linear", "separable", "exact"] {
        if class.subtypes.iter().any(|item| item == subtype) {
            methods.push(match subtype {
                "linear" => "linear integrating factor".to_string(),
                "separable" => "separable".to_string(),
                "exact" => "exact".to_string(),
                _ => unreachable!(),
            });
        }
    }
    methods
}

fn unsupported_method_reason(class: &OdeClassification) -> String {
    match class.kind {
        DifferentialKind::Pde => "PDE solving is not implemented".to_string(),
        DifferentialKind::Ode if class.order != 1 => {
            "only first-order and supported second-order linear ODE solvers are implemented"
                .to_string()
        }
        DifferentialKind::Ode => "no supported symbolic method matched".to_string(),
    }
}

// ---- structured payloads for the ODE result UI ----------------------------
// These mirror the LaTeX renderers above but return plain data, so the front
// end can render badges / numbered steps instead of one opaque math blob. The
// LaTeX renderers are kept byte-for-byte identical (the CLI and tests rely on
// them); these are purely additive.

/// One labelled derivation row for the structured `solve(details=true)` UI.
#[derive(Debug, Clone, PartialEq)]
pub struct OdeStep {
    pub label: String,
    pub latex: String,
}

impl OdeStep {
    fn new(label: &str, latex: String) -> Self {
        OdeStep {
            label: label.to_string(),
            latex,
        }
    }
}

/// Classification fields behind `show(classification)`. The boundary condition
/// is deliberately excluded — it belongs to `show(boundary)`.
pub struct OdeClassInfo {
    pub kind: String,
    pub order: usize,
    pub linear: bool,
    pub homogeneous: bool,
}

pub fn classification_info(problem: &OdeProblem) -> Result<OdeClassInfo, Error> {
    let class = classify_problem(problem)?;
    Ok(OdeClassInfo {
        kind: match class.kind {
            DifferentialKind::Ode => "ODE",
            DifferentialKind::Pde => "PDE",
        }
        .to_string(),
        order: class.order,
        linear: class.linear,
        homogeneous: class.homogeneous,
    })
}

/// Rendered boundary-condition LaTeX behind `show(boundary)`, or `None` when no
/// explicit condition was given.
pub fn boundary_info(problem: &OdeProblem) -> Option<String> {
    (!problem.boundary_conditions.is_empty()).then(|| {
        problem
            .boundary_conditions
            .iter()
            .map(render_boundary_condition)
            .collect::<Vec<_>>()
            .join(",\\quad ")
    })
}

/// Available + default solver labels behind `show(methods)`.
pub struct OdeMethodsInfo {
    pub available: Vec<String>,
    pub default: String,
}

pub fn methods_info(problem: &OdeProblem) -> Result<OdeMethodsInfo, Error> {
    let class = classify_problem(problem)?;
    let available = supported_method_labels(&class);
    let default = available
        .first()
        .cloned()
        .unwrap_or_else(|| unsupported_method_reason(&class));
    Ok(OdeMethodsInfo { available, default })
}

/// Labelled derivation steps parallel to `render_solution_details`.
pub fn solution_steps(sol: &OdeSolution) -> Vec<OdeStep> {
    match sol.method.as_str() {
        "linear first-order integrating factor" => linear_process_steps(sol),
        "separable" => separable_process_steps(sol),
        "exact" => exact_process_steps(sol),
        _ => generic_process_steps(sol),
    }
}

fn step_latex(steps: Vec<OdeStep>) -> Vec<String> {
    steps.into_iter().map(|step| step.latex).collect()
}

fn push_warning_step(steps: &mut Vec<OdeStep>, sol: &OdeSolution) {
    if let Some(warning) = &sol.warning {
        steps.push(OdeStep::new(
            "note",
            format!("\\text{{Note: {}}}", escape_text(warning)),
        ));
    }
}

pub fn render_solution(sol: &OdeSolution, mode: &str) -> String {
    match mode {
        "steps" => {
            let rows = solution_step_rows(sol, false);
            format!(
                "\\begin{{aligned}}\n{}\n\\end{{aligned}}",
                rows.join("\\\\\n")
            )
        }
        _ => {
            if let Some(eq) = &sol.solution {
                render_equation(eq)
            } else {
                format!(
                    "\\text{{unsupported: {}}}",
                    escape_text(
                        sol.warning
                            .as_deref()
                            .unwrap_or("no symbolic method matched")
                    )
                )
            }
        }
    }
}

pub fn render_solution_details(sol: &OdeSolution, class: &OdeClassification) -> String {
    let _ = class;
    format!(
        "\\begin{{gathered}}\n{}\n\\end{{gathered}}",
        render_solution_process(sol)
    )
}

fn solution_step_rows(sol: &OdeSolution, include_solution: bool) -> Vec<String> {
    let mut rows = vec![format!(
        "\\text{{method}} &= \\text{{{}}}",
        escape_text(&sol.method)
    )];
    for step in &sol.steps {
        if let Some(eq) = &step.equation {
            rows.push(format!(
                "\\text{{{}}} &: {}",
                escape_text(&step.label),
                render_equation(eq)
            ));
        } else if let Some(expr) = &step.expr {
            rows.push(format!(
                "\\text{{{}}} &= {}",
                escape_text(&step.label),
                ode_scalar_to_latex(expr)
            ));
        } else if let Some(latex) = &step.latex {
            rows.push(format!(
                "\\text{{{}}} &: {}",
                escape_text(&step.label),
                latex
            ));
        }
    }
    if include_solution {
        if let Some(solution) = &sol.solution {
            rows.push(format!(
                "\\text{{solution}} &: {}",
                render_equation(solution)
            ));
        }
    }
    if let Some(warning) = &sol.warning {
        rows.push(format!(
            "\\text{{warning}} &= \\text{{{}}}",
            escape_text(warning)
        ));
    }
    rows
}

fn render_solution_process(sol: &OdeSolution) -> String {
    match sol.method.as_str() {
        "linear first-order integrating factor" => render_linear_solution_process(sol),
        "separable" => render_separable_solution_process(sol),
        "exact" => render_exact_solution_process(sol),
        _ => render_generic_solution_process(sol),
    }
}

fn render_linear_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(linear_process_steps(sol)))
}

fn linear_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
    let mut steps = Vec::new();
    if let Some(standard) = step_equation(sol, "standard form") {
        steps.push(OdeStep::new("standard form", render_equation(standard)));
    }
    if let Some(mu) = step_equation(sol, "integrating factor") {
        steps.push(OdeStep::new("integrating factor", render_equation(mu)));
    }
    if let Some(standard) = step_equation(sol, "standard form") {
        let q = ode_scalar_to_latex(&standard.rhs);
        let target = sol
            .solution
            .as_ref()
            .map(|solution| ode_scalar_to_latex(&solution.lhs))
            .unwrap_or_else(|| "\\text{unknown}".to_string());
        let independent = sol
            .solution
            .as_ref()
            .and_then(|solution| one_variable_unknown_arg_latex(&solution.lhs))
            .unwrap_or_else(|| "x".to_string());
        steps.push(OdeStep::new(
            "exact derivative",
            format!(
                "\\frac{{d}}{{d {}}}\\left(\\mu\\, {}\\right)=\\mu\\, {}",
                independent, target, q
            ),
        ));
        steps.push(OdeStep::new(
            "integrate",
            format!(
                "\\mu\\, {}=\\int \\mu\\, {}\\, d{}+C_1",
                target, q, independent
            ),
        ));
    }
    if let Some(solution) = &sol.solution {
        steps.push(OdeStep::new("solution", render_equation(solution)));
    }
    push_warning_step(&mut steps, sol);
    steps
}

fn render_separable_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(separable_process_steps(sol)))
}

fn separable_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
    let mut steps = Vec::new();
    steps.push(OdeStep::new(
        "original equation",
        render_equation(&sol.equation),
    ));
    for step in &sol.steps {
        if let Some(eq) = &step.equation {
            steps.push(OdeStep::new(&step.label, render_equation(eq)));
        } else if let Some(expr) = &step.expr {
            steps.push(OdeStep::new(&step.label, ode_scalar_to_latex(expr)));
        } else if let Some(latex) = &step.latex {
            steps.push(OdeStep::new(&step.label, latex.clone()));
        }
    }
    if sol.solution.is_none() {
        steps.push(OdeStep::new("unsupported", unsupported_text(sol)));
    }
    push_warning_step(&mut steps, sol);
    steps
}

fn render_exact_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(exact_process_steps(sol)))
}

fn exact_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
    let mut steps = Vec::new();
    steps.push(OdeStep::new(
        "original equation",
        render_equation(&sol.equation),
    ));
    for step in &sol.steps {
        if let Some(eq) = &step.equation {
            steps.push(OdeStep::new(&step.label, render_equation(eq)));
        } else if let Some(expr) = &step.expr {
            steps.push(OdeStep::new(&step.label, ode_scalar_to_latex(expr)));
        } else if let Some(latex) = &step.latex {
            steps.push(OdeStep::new(&step.label, latex.clone()));
        }
    }
    if sol.solution.is_none() {
        steps.push(OdeStep::new("unsupported", unsupported_text(sol)));
    }
    push_warning_step(&mut steps, sol);
    steps
}

fn render_generic_solution_process(sol: &OdeSolution) -> String {
    let mut rows = Vec::new();
    for step in &sol.steps {
        if let Some(eq) = &step.equation {
            rows.push(format!("\\text{{{}:}}", escape_text(&step.label)));
            rows.push(render_equation(eq));
        } else if let Some(expr) = &step.expr {
            rows.push(format!(
                "\\text{{{}: }}{}",
                escape_text(&step.label),
                ode_scalar_to_latex(expr)
            ));
        } else if let Some(latex) = &step.latex {
            rows.push(format!("\\text{{{}:}}", escape_text(&step.label)));
            rows.push(latex.clone());
        }
    }
    if rows.is_empty() {
        rows.push(unsupported_text(sol));
    } else {
        push_final_solution(&mut rows, sol);
    }
    push_warning(&mut rows, sol);
    aligned_process(rows)
}

fn generic_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
    let mut steps = Vec::new();
    for step in &sol.steps {
        if let Some(eq) = &step.equation {
            steps.push(OdeStep::new(&step.label, render_equation(eq)));
        } else if let Some(expr) = &step.expr {
            steps.push(OdeStep::new(&step.label, ode_scalar_to_latex(expr)));
        } else if let Some(latex) = &step.latex {
            steps.push(OdeStep::new(&step.label, latex.clone()));
        }
    }
    if steps.is_empty() {
        steps.push(OdeStep::new("unsupported", unsupported_text(sol)));
    } else if let Some(solution) = &sol.solution {
        steps.push(OdeStep::new("solution", render_equation(solution)));
    }
    push_warning_step(&mut steps, sol);
    steps
}

fn step_equation<'a>(sol: &'a OdeSolution, label: &str) -> Option<&'a Equation> {
    sol.steps
        .iter()
        .find(|step| step.label == label)
        .and_then(|step| step.equation.as_ref())
}

fn one_variable_unknown_arg_latex(expr: &Rc<ScalarExpr>) -> Option<String> {
    match &**expr {
        ScalarExpr::UnknownFunc {
            args,
            derivative_orders,
            ..
        } if args.len() == 1 && derivative_orders.iter().all(|order| *order == 0) => {
            Some(scalar_to_latex(&args[0]))
        }
        _ => None,
    }
}

fn push_final_solution(rows: &mut Vec<String>, sol: &OdeSolution) {
    if let Some(solution) = &sol.solution {
        rows.push(render_equation(solution));
    }
}

fn push_warning(rows: &mut Vec<String>, sol: &OdeSolution) {
    if let Some(warning) = &sol.warning {
        rows.push(format!("\\text{{Note: {}}}", escape_text(warning)));
    }
}

fn unsupported_text(sol: &OdeSolution) -> String {
    format!(
        "\\text{{Unsupported: {}}}",
        escape_text(
            sol.warning
                .as_deref()
                .unwrap_or("no symbolic method matched")
        )
    )
}

fn aligned_process(rows: Vec<String>) -> String {
    if rows.is_empty() {
        return "\\text{No symbolic steps available.}".to_string();
    }
    format!(
        "\\begin{{aligned}}\n&{}\n\\end{{aligned}}",
        rows.join("\\\\[3pt]\n&")
    )
}

fn target_from_expr(expr: &Rc<ScalarExpr>) -> Result<Target, Error> {
    match &**expr {
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } if derivative_orders.iter().all(|order| *order == 0) => Ok(Target {
            name: name.clone(),
            args: args.clone(),
        }),
        _ => Err(Error::msg(
            "ODE target must be an unknown function such as y = Function(\"y\", x)",
        )),
    }
}

fn base_unknown(target: &Target) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::UnknownFunc {
        name: target.name.clone(),
        args: target.args.clone(),
        derivative_orders: vec![0; target.args.len()],
    })
}

fn derivative_unknown(target: &Target, order: usize) -> Rc<ScalarExpr> {
    let mut orders = vec![0; target.args.len()];
    if !orders.is_empty() {
        orders[0] = order;
    }
    Rc::new(ScalarExpr::UnknownFunc {
        name: target.name.clone(),
        args: target.args.clone(),
        derivative_orders: orders,
    })
}

fn target_symbol(target: &Target) -> Rc<ScalarExpr> {
    symbol(&target.name)
}

fn symbol(name: &str) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Sym {
        name: name.to_string(),
        latex: name.to_string(),
    })
}

fn constant() -> Rc<ScalarExpr> {
    symbol(constant_name())
}

fn constant_name() -> &'static str {
    "C_1"
}

fn target_occurrences(expr: &Rc<ScalarExpr>, target_name: &str) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    collect_target_occurrences(expr, target_name, &mut out);
    out
}

fn collect_target_occurrences(expr: &Rc<ScalarExpr>, target_name: &str, out: &mut Vec<Vec<usize>>) {
    match &**expr {
        ScalarExpr::UnknownFunc {
            name,
            derivative_orders,
            ..
        } if name == target_name => out.push(derivative_orders.clone()),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            collect_target_occurrences(a, target_name, out);
            collect_target_occurrences(b, target_name, out);
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            collect_target_occurrences(a, target_name, out);
        }
        ScalarExpr::UnknownFunc { args, .. } => {
            for arg in args {
                collect_target_occurrences(arg, target_name, out);
            }
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => {
            collect_target_occurrences(integrand, target_name, out);
            collect_target_occurrences(variable, target_name, out);
        }
        ScalarExpr::SpecSum { body, .. } => collect_target_occurrences(body, target_name, out),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::SetElem { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _) => {}
    }
}

fn contains_target(expr: &Rc<ScalarExpr>, target_name: &str) -> bool {
    !target_occurrences(expr, target_name).is_empty()
}

fn target_degree(expr: &Rc<ScalarExpr>, target_name: &str) -> Option<usize> {
    match &**expr {
        ScalarExpr::UnknownFunc { name, args, .. } if name == target_name => {
            if args.iter().any(|arg| contains_target(arg, target_name)) {
                None
            } else {
                Some(1)
            }
        }
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => Some(0),
        ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) => {
            Some(target_degree(a, target_name)?.max(target_degree(b, target_name)?))
        }
        ScalarExpr::Mul(a, b) => {
            let degree = target_degree(a, target_name)? + target_degree(b, target_name)?;
            (degree <= 1).then_some(degree)
        }
        ScalarExpr::Div(a, b) => {
            let bd = target_degree(b, target_name)?;
            if bd > 0 {
                None
            } else {
                target_degree(a, target_name)
            }
        }
        ScalarExpr::Pow(a, b) => {
            let ad = target_degree(a, target_name)?;
            let bd = target_degree(b, target_name)?;
            if bd > 0 {
                None
            } else if ad == 0 {
                Some(0)
            } else if matches!(&**b, ScalarExpr::Num(n) if *n == 1.0) {
                Some(1)
            } else {
                None
            }
        }
        ScalarExpr::Neg(a) => target_degree(a, target_name),
        ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            (target_degree(a, target_name)? == 0).then_some(0)
        }
        ScalarExpr::UnknownFunc { args, .. } => {
            for arg in args {
                if target_degree(arg, target_name)? > 0 {
                    return None;
                }
            }
            Some(0)
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => {
            if target_degree(integrand, target_name)? > 0
                || target_degree(variable, target_name)? > 0
            {
                None
            } else {
                Some(0)
            }
        }
        ScalarExpr::SpecSum { body, .. } => target_degree(body, target_name),
        ScalarExpr::Det(_) | ScalarExpr::Tr(_) | ScalarExpr::Ddot(_, _) => Some(0),
    }
}

fn zero_target(expr: &Rc<ScalarExpr>, target_name: &str) -> Rc<ScalarExpr> {
    match &**expr {
        ScalarExpr::UnknownFunc { name, .. } if name == target_name => {
            Rc::new(ScalarExpr::Num(0.0))
        }
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(
            zero_target(a, target_name),
            zero_target(b, target_name),
        )),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(
            zero_target(a, target_name),
            zero_target(b, target_name),
        )),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(
            zero_target(a, target_name),
            zero_target(b, target_name),
        )),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(
            zero_target(a, target_name),
            zero_target(b, target_name),
        )),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(
            zero_target(a, target_name),
            zero_target(b, target_name),
        )),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(zero_target(a, target_name))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(zero_target(a, target_name))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: zero_target(arg, target_name),
        }),
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } => Rc::new(ScalarExpr::UnknownFunc {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| zero_target(arg, target_name))
                .collect(),
            derivative_orders: derivative_orders.clone(),
        }),
        ScalarExpr::Integral {
            integrand,
            variable,
        } => Rc::new(ScalarExpr::Integral {
            integrand: zero_target(integrand, target_name),
            variable: zero_target(variable, target_name),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: zero_target(body, target_name),
            index: index.clone(),
            dim: *dim,
        }),
        _ => expr.clone(),
    }
}

#[derive(Clone)]
struct SignedTerm {
    sign: f64,
    expr: Rc<ScalarExpr>,
}

fn additive_terms(expr: &Rc<ScalarExpr>) -> Vec<SignedTerm> {
    let mut out = Vec::new();
    collect_additive_terms(expr, 1.0, &mut out);
    out
}

fn collect_additive_terms(expr: &Rc<ScalarExpr>, sign: f64, out: &mut Vec<SignedTerm>) {
    match &**expr {
        ScalarExpr::Add(a, b) => {
            collect_additive_terms(a, sign, out);
            collect_additive_terms(b, sign, out);
        }
        ScalarExpr::Sub(a, b) => {
            collect_additive_terms(a, sign, out);
            collect_additive_terms(b, -sign, out);
        }
        ScalarExpr::Neg(a) => collect_additive_terms(a, -sign, out),
        _ => out.push(SignedTerm {
            sign,
            expr: expr.clone(),
        }),
    }
}

fn signed(sign: f64, expr: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    if sign == 1.0 {
        expr
    } else {
        Rc::new(ScalarExpr::Neg(expr))
    }
}

fn factor_atom(expr: &Rc<ScalarExpr>, atom: &Rc<ScalarExpr>) -> Option<Rc<ScalarExpr>> {
    if simplify_scalar(expr, RuleSet::Continuum) == simplify_scalar(atom, RuleSet::Continuum) {
        return Some(Rc::new(ScalarExpr::Num(1.0)));
    }
    if let ScalarExpr::Div(num, den) = &**expr {
        if let Some(coeff) = factor_atom(num, atom) {
            if !contains_atom(den, atom) {
                return Some(Rc::new(ScalarExpr::Div(coeff, den.clone())));
            }
        }
    }
    let factors = product_factors(expr);
    let mut hit = None;
    let mut rest = Vec::new();
    for (i, factor) in factors.iter().enumerate() {
        if hit.is_none()
            && simplify_scalar(factor, RuleSet::Continuum)
                == simplify_scalar(atom, RuleSet::Continuum)
        {
            hit = Some(i);
        } else {
            rest.push((*factor).clone());
        }
    }
    hit.map(|_| multiply_all(rest))
}

fn contains_atom(expr: &Rc<ScalarExpr>, atom: &Rc<ScalarExpr>) -> bool {
    if simplify_scalar(expr, RuleSet::Continuum) == simplify_scalar(atom, RuleSet::Continuum) {
        return true;
    }
    match &**expr {
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => contains_atom(a, atom) || contains_atom(b, atom),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            contains_atom(a, atom)
        }
        ScalarExpr::UnknownFunc { args, .. } => args.iter().any(|arg| contains_atom(arg, atom)),
        ScalarExpr::Integral {
            integrand,
            variable,
        } => contains_atom(integrand, atom) || contains_atom(variable, atom),
        ScalarExpr::SpecSum { body, .. } => contains_atom(body, atom),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::SetElem { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _) => false,
    }
}

fn product_factors(expr: &Rc<ScalarExpr>) -> Vec<&Rc<ScalarExpr>> {
    match &**expr {
        ScalarExpr::Mul(a, b) => {
            let mut out = product_factors(a);
            out.extend(product_factors(b));
            out
        }
        _ => vec![expr],
    }
}

fn multiply_all(factors: Vec<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    factors
        .into_iter()
        .reduce(|a, b| Rc::new(ScalarExpr::Mul(a, b)))
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(1.0)))
}

fn depends_on_or_const(expr: &Rc<ScalarExpr>, allowed: &str) -> bool {
    let vars = scalar_symbols(expr);
    vars.into_iter().all(|v| v == allowed)
}

fn depends_only_on(expr: &Rc<ScalarExpr>, allowed: &str, forbidden: &str) -> bool {
    depends_on(expr, allowed) && !depends_on(expr, forbidden) && depends_on_or_const(expr, allowed)
}

fn scalar_symbols(expr: &Rc<ScalarExpr>) -> Vec<String> {
    let mut out = Vec::new();
    collect_scalar_symbols(expr, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_scalar_symbols(expr: &Rc<ScalarExpr>, out: &mut Vec<String>) {
    match &**expr {
        ScalarExpr::Sym { name, .. } => out.push(name.clone()),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            collect_scalar_symbols(a, out);
            collect_scalar_symbols(b, out);
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            collect_scalar_symbols(a, out);
        }
        ScalarExpr::UnknownFunc { args, .. } => {
            for arg in args {
                collect_scalar_symbols(arg, out);
            }
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => {
            collect_scalar_symbols(integrand, out);
            collect_scalar_symbols(variable, out);
        }
        ScalarExpr::SpecSum { body, .. } => collect_scalar_symbols(body, out),
        ScalarExpr::Num(_)
        | ScalarExpr::SetElem { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _) => {}
    }
}

fn split_xy_product(
    expr: &Rc<ScalarExpr>,
    x_name: &str,
    y_name: &str,
) -> Option<(Rc<ScalarExpr>, Rc<ScalarExpr>)> {
    let mut gx = Vec::new();
    let mut hy = Vec::new();
    for factor in product_factors(expr) {
        let x = depends_on(factor, x_name);
        let y = depends_on(factor, y_name);
        match (x, y) {
            (true, false) => gx.push(factor.clone()),
            (false, true) => hy.push(factor.clone()),
            (false, false) => gx.push(factor.clone()),
            (true, true) => return None,
        }
    }
    if hy.is_empty() {
        return None;
    }
    Some((multiply_all(gx), multiply_all(hy)))
}

fn replace_base_target_with_symbol(
    expr: &Rc<ScalarExpr>,
    target: &Target,
    replacement: &Rc<ScalarExpr>,
) -> Rc<ScalarExpr> {
    match &**expr {
        ScalarExpr::UnknownFunc {
            name,
            derivative_orders,
            ..
        } if name == &target.name && derivative_orders.iter().all(|order| *order == 0) => {
            replacement.clone()
        }
        ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(
            replace_base_target_with_symbol(a, target, replacement),
            replace_base_target_with_symbol(b, target, replacement),
        )),
        ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(
            replace_base_target_with_symbol(a, target, replacement),
            replace_base_target_with_symbol(b, target, replacement),
        )),
        ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(
            replace_base_target_with_symbol(a, target, replacement),
            replace_base_target_with_symbol(b, target, replacement),
        )),
        ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(
            replace_base_target_with_symbol(a, target, replacement),
            replace_base_target_with_symbol(b, target, replacement),
        )),
        ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(
            replace_base_target_with_symbol(a, target, replacement),
            replace_base_target_with_symbol(b, target, replacement),
        )),
        ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(replace_base_target_with_symbol(
            a,
            target,
            replacement,
        ))),
        ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(replace_base_target_with_symbol(
            a,
            target,
            replacement,
        ))),
        ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
            name: name.clone(),
            arg: replace_base_target_with_symbol(arg, target, replacement),
        }),
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } => Rc::new(ScalarExpr::UnknownFunc {
            name: name.clone(),
            args: args
                .iter()
                .map(|arg| replace_base_target_with_symbol(arg, target, replacement))
                .collect(),
            derivative_orders: derivative_orders.clone(),
        }),
        ScalarExpr::Integral {
            integrand,
            variable,
        } => Rc::new(ScalarExpr::Integral {
            integrand: replace_base_target_with_symbol(integrand, target, replacement),
            variable: replace_base_target_with_symbol(variable, target, replacement),
        }),
        ScalarExpr::SpecSum { body, index, dim } => Rc::new(ScalarExpr::SpecSum {
            body: replace_base_target_with_symbol(body, target, replacement),
            index: index.clone(),
            dim: *dim,
        }),
        _ => expr.clone(),
    }
}

fn scale_arg(coeff: f64, arg: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    simplify_scalar(
        &fold_mul(&Rc::new(ScalarExpr::Num(coeff)), arg),
        RuleSet::Continuum,
    )
}

fn exp_linear(coeff: f64, independent: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    if coeff == 0.0 {
        return Rc::new(ScalarExpr::Num(1.0));
    }
    Rc::new(ScalarExpr::Func {
        name: "exp".to_string(),
        arg: scale_arg(coeff, independent),
    })
}

fn symbol_with_latex(name: &str, latex: &str) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Sym {
        name: name.to_string(),
        latex: latex.to_string(),
    })
}

fn num_latex(value: f64) -> String {
    scalar_to_latex(&ScalarExpr::Num(value))
}

fn constant2_name() -> &'static str {
    "C_2"
}

fn polynomial_coeffs(expr: &Rc<ScalarExpr>, x_name: &str) -> Option<Vec<f64>> {
    fn add(mut a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
        if a.len() < b.len() {
            a.resize(b.len(), 0.0);
        }
        for (i, value) in b.into_iter().enumerate() {
            a[i] += value;
        }
        trim_poly(a)
    }
    fn scale(mut p: Vec<f64>, c: f64) -> Vec<f64> {
        for value in &mut p {
            *value *= c;
        }
        trim_poly(p)
    }
    fn mul(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, av) in a.iter().enumerate() {
            for (j, bv) in b.iter().enumerate() {
                out[i + j] += av * bv;
            }
        }
        trim_poly(out)
    }
    let expr = simplify_scalar(expr, RuleSet::Continuum);
    match &*expr {
        ScalarExpr::Num(n) => Some(vec![*n]),
        ScalarExpr::Sym { name, .. } if name == x_name => Some(vec![0.0, 1.0]),
        ScalarExpr::Sym { .. } => None,
        ScalarExpr::Add(a, b) => Some(add(
            polynomial_coeffs(a, x_name)?,
            polynomial_coeffs(b, x_name)?,
        )),
        ScalarExpr::Sub(a, b) => Some(add(
            polynomial_coeffs(a, x_name)?,
            scale(polynomial_coeffs(b, x_name)?, -1.0),
        )),
        ScalarExpr::Neg(a) => Some(scale(polynomial_coeffs(a, x_name)?, -1.0)),
        ScalarExpr::Mul(a, b) => Some(mul(
            polynomial_coeffs(a, x_name)?,
            polynomial_coeffs(b, x_name)?,
        )),
        ScalarExpr::Div(a, b) => {
            let den = numeric_scalar_value(b)?;
            Some(scale(polynomial_coeffs(a, x_name)?, 1.0 / den))
        }
        ScalarExpr::Pow(base, exp) => {
            let ScalarExpr::Num(n) = &**exp else {
                return None;
            };
            if n.fract() != 0.0 || *n < 0.0 {
                return None;
            }
            let mut out = vec![1.0];
            let base = polynomial_coeffs(base, x_name)?;
            for _ in 0..(*n as usize) {
                out = mul(out, base.clone());
            }
            Some(out)
        }
        _ => None,
    }
}

fn trim_poly(mut values: Vec<f64>) -> Vec<f64> {
    while values.len() > 1 && values.last().is_some_and(|v| v.abs() < 1e-12) {
        values.pop();
    }
    values
}

fn polynomial_particular_coeffs(forcing: &[f64], coeffs: ConstantSecondOrder) -> Option<Vec<f64>> {
    if coeffs.b == 0.0 {
        return None;
    }
    let mut p = vec![0.0; forcing.len()];
    for k in (0..forcing.len()).rev() {
        let next1 = if k + 1 < p.len() {
            coeffs.a * (k as f64 + 1.0) * p[k + 1]
        } else {
            0.0
        };
        let next2 = if k + 2 < p.len() {
            (k as f64 + 2.0) * (k as f64 + 1.0) * p[k + 2]
        } else {
            0.0
        };
        p[k] = (forcing[k] - next1 - next2) / coeffs.b;
    }
    Some(trim_poly(p))
}

fn polynomial_expr(coeffs: &[f64], x: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    let mut acc: Option<Rc<ScalarExpr>> = None;
    for (degree, coeff) in coeffs.iter().enumerate().rev() {
        if coeff.abs() < 1e-12 {
            continue;
        }
        let power = fold_pow(x.clone(), degree as f64);
        let term = fold_mul(&Rc::new(ScalarExpr::Num(coeff.abs())), &power);
        acc = Some(match (acc, *coeff < 0.0) {
            (None, false) => term,
            (None, true) => Rc::new(ScalarExpr::Neg(term)),
            (Some(prev), false) => fold_add(prev, term),
            (Some(prev), true) => fold_sub(prev, term),
        });
    }
    simplify_scalar(
        &acc.unwrap_or_else(|| Rc::new(ScalarExpr::Num(0.0))),
        RuleSet::Continuum,
    )
}

fn polynomial_template_latex(degree: usize) -> String {
    (0..=degree)
        .map(|i| {
            if i == 0 {
                "A_0".to_string()
            } else if i == 1 {
                "A_1x".to_string()
            } else {
                format!("A_{i}x^{i}")
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn apply_second_order_boundaries(
    rhs: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    bcs: &[BoundaryCondition],
) -> Option<Rc<ScalarExpr>> {
    if bcs.len() < 2 {
        return None;
    }
    let row1 = boundary_linear_row(rhs, target, independent, &bcs[0])?;
    let row2 = boundary_linear_row(rhs, target, independent, &bcs[1])?;
    let b1 = simplify_scalar(
        &Rc::new(ScalarExpr::Sub(row1.value.clone(), row1.rest.clone())),
        RuleSet::Continuum,
    );
    let b2 = simplify_scalar(
        &Rc::new(ScalarExpr::Sub(row2.value.clone(), row2.rest.clone())),
        RuleSet::Continuum,
    );
    let det = simplify_scalar(
        &Rc::new(ScalarExpr::Sub(
            fold_mul(&row1.c1, &row2.c2),
            fold_mul(&row1.c2, &row2.c1),
        )),
        RuleSet::Continuum,
    );
    if is_zero(&det) {
        return None;
    }
    let c1_value = simplify_scalar(
        &Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Sub(
                fold_mul(&b1, &row2.c2),
                fold_mul(&row1.c2, &b2),
            )),
            det.clone(),
        )),
        RuleSet::Continuum,
    );
    let c2_value = simplify_scalar(
        &Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Sub(
                fold_mul(&row1.c1, &b2),
                fold_mul(&b1, &row2.c1),
            )),
            det,
        )),
        RuleSet::Continuum,
    );
    let out = subst_symbol(rhs, constant_name(), &c1_value);
    Some(simplify_scalar(
        &subst_symbol(&out, constant2_name(), &c2_value),
        RuleSet::Continuum,
    ))
}

fn boundary_linear_row(
    rhs: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    bc: &BoundaryCondition,
) -> Option<BoundaryLinearRow> {
    let order = boundary_derivative_order(target, bc)?;
    let x_name = scalar_symbol_name(independent)?;
    let expr = if order == 0 {
        rhs.clone()
    } else if order == 1 {
        diff_scalar_by_scalar(rhs, x_name).ok()?
    } else {
        return None;
    };
    let at_point = subst_symbol(&expr, x_name, &bc.point);
    let (c1, rest) = linear_coeff_const(&at_point, constant_name())?;
    let (c2, rest) = linear_coeff_const(&rest, constant2_name())?;
    Some(BoundaryLinearRow {
        c1,
        c2,
        rest,
        value: bc.value.clone(),
    })
}

fn boundary_derivative_order(target: &Target, bc: &BoundaryCondition) -> Option<usize> {
    match &*bc.function {
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } if name == &target.name && args.len() == 1 => Some(derivative_orders.iter().sum()),
        _ => None,
    }
}

fn power_series_latex(
    target: &Target,
    independent: &Rc<ScalarExpr>,
    about: &Rc<ScalarExpr>,
    _terms: usize,
) -> String {
    let y = scalar_to_latex(&base_unknown(target));
    let x = ode_scalar_to_latex(independent);
    let a = ode_scalar_to_latex(about);
    format!("{y}=\\sum_{{n=0}}^{{\\infty}}C_n({x}-{a})^n")
}

fn finite_series_latex(independent: &Rc<ScalarExpr>, terms: usize) -> String {
    let x = ode_scalar_to_latex(independent);
    (0..terms)
        .map(|n| {
            if n == 0 {
                "C_0".to_string()
            } else if n == 1 {
                format!("C_1{x}")
            } else {
                format!("C_{n}{x}^{n}")
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

fn lecture_power_series_recurrence(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> Option<String> {
    let x_name = scalar_symbol_name(independent)?;
    if let Some((p, q, _)) = linear_first_order_coeffs(&eq.residual, target, x_name) {
        if is_zero(&q) && numeric_scalar_value(&p) == Some(2.0) {
            return Some("(n+1)C_{n+1}+2C_n=0,\\qquad C_{n+1}=-\\frac{2C_n}{n+1}".to_string());
        }
    }
    let form = second_order_linear_form(&eq.residual, target)?;
    if is_one(&form.a2)
        && scalar_equiv_commutative(&form.a1, independent)
        && is_one(&form.a0)
        && is_zero(&form.forcing)
    {
        return Some("(n+2)(n+1)C_{n+2}+(n+1)C_n=0,\\qquad C_{n+2}=-\\frac{C_n}{n+2}".to_string());
    }
    None
}

fn is_regular_singular_candidate(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> bool {
    let Some(form) = second_order_linear_form(residual, target) else {
        return false;
    };
    is_power_of_var(&form.a2, independent, 2.0) && factor_atom(&form.a1, independent).is_some()
}

fn bessel_parameter(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> Option<Rc<ScalarExpr>> {
    let form = second_order_linear_form(residual, target)?;
    if !is_power_of_var(&form.a2, independent, 2.0)
        || !scalar_equiv_commutative(&form.a1, independent)
        || !is_zero(&form.forcing)
    {
        return None;
    }
    let x2 = fold_pow(independent.clone(), 2.0);
    match &*simplify_scalar(&form.a0, RuleSet::Continuum) {
        ScalarExpr::Sub(a, b) if scalar_equiv_commutative(a, &x2) => match &**b {
            ScalarExpr::Pow(base, exp) if matches!(&**exp, ScalarExpr::Num(n) if *n == 2.0) => {
                Some(base.clone())
            }
            _ => None,
        },
        _ => None,
    }
}

fn is_power_of_var(expr: &Rc<ScalarExpr>, var: &Rc<ScalarExpr>, power: f64) -> bool {
    let expected = fold_pow(var.clone(), power);
    scalar_equiv_commutative(expr, &expected)
}

fn solve_constant_from_explicit(
    rhs: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    ic: &BoundaryCondition,
    _c: &Rc<ScalarExpr>,
) -> Option<Rc<ScalarExpr>> {
    validate_ic(target, ic)?;
    let x_name = scalar_symbol_name(independent)?;
    let at_x = subst_symbol(rhs, x_name, &ic.point);
    let (coeff, rest) = linear_coeff_const(&at_x, constant_name())?;
    Some(simplify_scalar(
        &Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Sub(ic.value.clone(), rest)),
            coeff.clone(),
        )),
        RuleSet::Continuum,
    ))
    .filter(|_| !is_zero(&coeff))
}

fn solve_constant_from_implicit(
    left: &Rc<ScalarExpr>,
    right_without_c: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    ic: &BoundaryCondition,
) -> Option<Rc<ScalarExpr>> {
    validate_ic(target, ic)?;
    let x_name = scalar_symbol_name(independent)?;
    let y_name = target.name.as_str();
    let left_at = subst_symbol(&subst_symbol(left, x_name, &ic.point), y_name, &ic.value);
    let right_at = subst_symbol(right_without_c, x_name, &ic.point);
    Some(simplify_scalar(
        &Rc::new(ScalarExpr::Sub(left_at, right_at)),
        RuleSet::Continuum,
    ))
}

fn validate_ic(target: &Target, ic: &BoundaryCondition) -> Option<()> {
    match &*ic.function {
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } if name == &target.name
            && args.len() == 1
            && derivative_orders.iter().all(|order| *order == 0) =>
        {
            Some(())
        }
        _ => None,
    }
}

fn linear_coeff_const(
    expr: &Rc<ScalarExpr>,
    symbol: &str,
) -> Option<(Rc<ScalarExpr>, Rc<ScalarExpr>)> {
    let atom = Rc::new(ScalarExpr::Sym {
        name: symbol.to_string(),
        latex: symbol.to_string(),
    });
    let mut coeff = Rc::new(ScalarExpr::Num(0.0));
    let mut rest = Rc::new(ScalarExpr::Num(0.0));
    for term in additive_terms(expr) {
        if let Some(c) = factor_atom(&term.expr, &atom) {
            if depends_on(&c, symbol) {
                return None;
            }
            coeff = fold_add(coeff, signed(term.sign, c));
        } else if depends_on(&term.expr, symbol) {
            return None;
        } else {
            rest = fold_add(rest, signed(term.sign, term.expr));
        }
    }
    Some((
        simplify_scalar(&coeff, RuleSet::Continuum),
        simplify_scalar(&rest, RuleSet::Continuum),
    ))
}

fn subst_symbol(expr: &Rc<ScalarExpr>, name: &str, replacement: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    crate::substitute::subst_scalar_sym(expr, name, replacement)
}

fn is_zero(expr: &Rc<ScalarExpr>) -> bool {
    matches!(&*simplify_scalar(expr, RuleSet::Continuum), ScalarExpr::Num(n) if *n == 0.0)
}

fn is_one(expr: &Rc<ScalarExpr>) -> bool {
    matches!(&*simplify_scalar(expr, RuleSet::Continuum), ScalarExpr::Num(n) if *n == 1.0)
}

fn escape_text(text: &str) -> String {
    text.replace('\\', "\\textbackslash{}")
        .replace('_', "\\_")
        .replace('&', "\\&")
}

fn contains_formal_integral(expr: &Rc<ScalarExpr>) -> bool {
    match &**expr {
        ScalarExpr::Integral { .. } => true,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => contains_formal_integral(a) || contains_formal_integral(b),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            contains_formal_integral(a)
        }
        ScalarExpr::UnknownFunc { args, .. } => args.iter().any(contains_formal_integral),
        ScalarExpr::SpecSum { body, .. } => contains_formal_integral(body),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::SetElem { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _) => false,
    }
}

fn scalar_equiv_commutative(a: &Rc<ScalarExpr>, b: &Rc<ScalarExpr>) -> bool {
    canonical_scalar_key(&simplify_scalar(a, RuleSet::Continuum))
        == canonical_scalar_key(&simplify_scalar(b, RuleSet::Continuum))
}

fn canonical_scalar_key(expr: &Rc<ScalarExpr>) -> String {
    match &**expr {
        ScalarExpr::Add(a, b) => {
            let mut terms = [canonical_scalar_key(a), canonical_scalar_key(b)];
            terms.sort();
            format!("add({})", terms.join(","))
        }
        ScalarExpr::Sub(a, b) => {
            let mut terms = [
                canonical_scalar_key(a),
                format!("neg({})", canonical_scalar_key(b)),
            ];
            terms.sort();
            format!("add({})", terms.join(","))
        }
        ScalarExpr::Mul(_, _) => {
            let mut coeff = 1.0;
            let mut factors = Vec::new();
            for factor in product_factors(expr) {
                if let Some(n) = numeric_scalar_value(factor) {
                    coeff *= n;
                } else {
                    factors.push(canonical_scalar_key(factor));
                }
            }
            if coeff != 1.0 || factors.is_empty() {
                factors.push(format!("num({coeff:.12})"));
            }
            factors.sort();
            format!("mul({})", factors.join(","))
        }
        ScalarExpr::Div(a, b) => {
            if let Some(den) = numeric_scalar_value(b) {
                return canonical_scalar_key(&Rc::new(ScalarExpr::Mul(
                    Rc::new(ScalarExpr::Num(1.0 / den)),
                    a.clone(),
                )));
            }
            format!(
                "div({},{})",
                canonical_scalar_key(a),
                canonical_scalar_key(b)
            )
        }
        ScalarExpr::Pow(a, b) => {
            format!(
                "pow({},{})",
                canonical_scalar_key(a),
                canonical_scalar_key(b)
            )
        }
        ScalarExpr::Neg(a) => format!("neg({})", canonical_scalar_key(a)),
        ScalarExpr::Log(a) => format!("log({})", canonical_scalar_key(a)),
        ScalarExpr::Func { name, arg } => format!("func({name},{})", canonical_scalar_key(arg)),
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } => format!(
            "unknown({name},{:?},{})",
            derivative_orders,
            args.iter()
                .map(canonical_scalar_key)
                .collect::<Vec<_>>()
                .join(",")
        ),
        ScalarExpr::Integral {
            integrand,
            variable,
        } => format!(
            "integral({},{})",
            canonical_scalar_key(integrand),
            canonical_scalar_key(variable)
        ),
        other => format!("{other:?}"),
    }
}

fn numeric_scalar_value(expr: &Rc<ScalarExpr>) -> Option<f64> {
    match &**expr {
        ScalarExpr::Num(n) => Some(*n),
        ScalarExpr::Neg(a) => numeric_scalar_value(a).map(|n| -n),
        ScalarExpr::Div(a, b) => Some(numeric_scalar_value(a)? / numeric_scalar_value(b)?),
        _ => None,
    }
}
