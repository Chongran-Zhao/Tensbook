//! Applied-math ODE/PDE classification and symbolic solvers.

pub(crate) use crate::differentiation::diff_scalar_by_scalar;
pub(crate) use crate::error::Error;
pub(crate) use crate::integration::{depends_on, integrate, scalar_symbol_name};
pub(crate) use crate::renderer::latex::scalar_to_latex;
pub(crate) use crate::simplifier::{simplify_scalar, RuleSet};
pub(crate) use crate::symbolic::{fold_add, fold_mul, fold_pow, fold_sub, ScalarExpr};
pub(crate) use std::rc::Rc;

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
    pub(crate) fn equation(label: &str, equation: Equation) -> Self {
        SolutionStep {
            label: label.to_string(),
            equation: Some(equation),
            expr: None,
            latex: None,
        }
    }

    pub(crate) fn expr(label: &str, expr: Rc<ScalarExpr>) -> Self {
        SolutionStep {
            label: label.to_string(),
            equation: None,
            expr: Some(expr),
            latex: None,
        }
    }

    pub(crate) fn latex(label: &str, latex: String) -> Self {
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

    pub(crate) fn label(self) -> &'static str {
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
pub(crate) struct Target {
    pub(crate) name: String,
    pub(crate) args: Vec<Rc<ScalarExpr>>,
}

#[derive(Clone)]
pub(crate) struct FirstOrderForm {
    pub(crate) m: Rc<ScalarExpr>,
    pub(crate) n: Rc<ScalarExpr>,
}

#[derive(Clone)]
pub(crate) struct SecondOrderLinearForm {
    pub(crate) a2: Rc<ScalarExpr>,
    pub(crate) a1: Rc<ScalarExpr>,
    pub(crate) a0: Rc<ScalarExpr>,
    pub(crate) forcing: Rc<ScalarExpr>,
}

#[derive(Clone, Copy)]
pub(crate) struct ConstantSecondOrder {
    pub(crate) a: f64,
    pub(crate) b: f64,
}

pub(crate) type ExactPotential = (Rc<ScalarExpr>, Vec<SolutionStep>);

pub(crate) struct SeparableForm {
    pub(crate) left_integrand: Rc<ScalarExpr>,
    pub(crate) right_integrand: Rc<ScalarExpr>,
    pub(crate) left_integral: Rc<ScalarExpr>,
    pub(crate) right_integral: Rc<ScalarExpr>,
}

pub(crate) struct ImplicitSolveParams<'a> {
    pub(crate) eq: &'a Equation,
    pub(crate) method: &'a str,
    pub(crate) independent: &'a Rc<ScalarExpr>,
    pub(crate) target: &'a Target,
    pub(crate) left: Rc<ScalarExpr>,
    pub(crate) right_without_c: Rc<ScalarExpr>,
    pub(crate) ic: Option<BoundaryCondition>,
    pub(crate) steps: Vec<SolutionStep>,
}

pub(crate) struct BoundaryLinearRow {
    pub(crate) c1: Rc<ScalarExpr>,
    pub(crate) c2: Rc<ScalarExpr>,
    pub(crate) rest: Rc<ScalarExpr>,
    pub(crate) value: Rc<ScalarExpr>,
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

mod render;
mod solve;
mod util;

pub use render::*;
pub use solve::*;
pub(crate) use util::*;
