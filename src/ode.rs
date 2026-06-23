//! Applied-math ODE/PDE classification and first-order symbolic solvers.

use crate::differentiation::diff_scalar_by_scalar;
use crate::error::Error;
use crate::integration::{depends_on, integrate, scalar_symbol_name};
use crate::renderer::latex::scalar_to_latex;
use crate::simplifier::{simplify_scalar, RuleSet};
use crate::symbolic::{fold_add, fold_sub, ScalarExpr};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub lhs: Rc<ScalarExpr>,
    pub rhs: Rc<ScalarExpr>,
    pub residual: Rc<ScalarExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialCondition {
    pub function: Rc<ScalarExpr>,
    pub point: Rc<ScalarExpr>,
    pub value: Rc<ScalarExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OdeClassification {
    pub kind: DifferentialKind,
    pub order: usize,
    pub linear: bool,
    pub homogeneous: Option<bool>,
    pub subtypes: Vec<String>,
    pub reasons: Vec<String>,
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
        .ok_or_else(|| Error::msg("ClassifyODE expects an independent variable such as x"))?;
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
    let homogeneous = linear.then(|| {
        let zeroed = simplify_scalar(&zero_target(&eq.residual, &target.name), RuleSet::Continuum);
        is_zero(&zeroed)
    });
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
    let mut reasons = Vec::new();
    reasons.push(format!("{} independent variable(s)", target.args.len()));
    reasons.push(format!("highest derivative order = {order}"));
    reasons.push(if linear {
        "target function appears linearly".to_string()
    } else {
        "target function appears nonlinearly".to_string()
    });
    Ok(OdeClassification {
        kind,
        order,
        linear,
        homogeneous,
        subtypes,
        reasons,
    })
}

pub fn solve(
    eq: &Equation,
    target: &Rc<ScalarExpr>,
    independent: &Rc<ScalarExpr>,
    ic: Option<InitialCondition>,
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
        .ok_or_else(|| Error::msg("SolveODE expects an independent variable such as x"))?;

    if let Some((p, q, form)) = linear_first_order_coeffs(&eq.residual, &target, x_name) {
        return solve_linear_first_order(eq, &target, independent, p, q, form, ic);
    }
    if let Some((left, right)) = separable_form(&eq.residual, &target, independent) {
        return solve_implicit_pair(
            eq,
            "separable",
            independent,
            &target,
            left,
            right,
            ic,
            vec![],
        );
    }
    if let Some((phi, steps)) = exact_form(&eq.residual, &target, independent)? {
        return solve_implicit_pair(
            eq,
            "exact",
            independent,
            &target,
            phi,
            Rc::new(ScalarExpr::Num(0.0)),
            ic,
            steps,
        );
    }

    Ok(OdeSolution {
        method: "unsupported".to_string(),
        equation: eq.clone(),
        solution: None,
        steps: vec![],
        warning: Some(
            "first-order nonlinear, not separable, not exact; integrating-factor search is not implemented"
                .to_string(),
        ),
    })
}

fn solve_linear_first_order(
    eq: &Equation,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    p: Rc<ScalarExpr>,
    q: Rc<ScalarExpr>,
    form: FirstOrderForm,
    ic: Option<InitialCondition>,
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
            SolutionStep {
                label: "standard form".to_string(),
                equation: Some(equation(
                    Rc::new(ScalarExpr::Add(
                        derivative_unknown(target, 1),
                        Rc::new(ScalarExpr::Mul(p.clone(), base_unknown(target))),
                    )),
                    q,
                )),
                expr: None,
            },
            SolutionStep {
                label: "coefficient of derivative".to_string(),
                equation: None,
                expr: Some(form.n),
            },
            SolutionStep {
                label: "integrating factor".to_string(),
                equation: Some(equation(symbol("\\mu"), mu)),
                expr: None,
            },
        ],
        warning,
    })
}

fn solve_implicit_pair(
    eq: &Equation,
    method: &str,
    independent: &Rc<ScalarExpr>,
    target: &Target,
    left: Rc<ScalarExpr>,
    right_without_c: Rc<ScalarExpr>,
    ic: Option<InitialCondition>,
    mut steps: Vec<SolutionStep>,
) -> Result<OdeSolution, Error> {
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
    steps.push(SolutionStep {
        label: "implicit solution".to_string(),
        equation: Some(solution.clone()),
        expr: None,
    });
    Ok(OdeSolution {
        method: method.to_string(),
        equation: eq.clone(),
        solution: Some(solution),
        steps,
        warning,
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
) -> Option<(Rc<ScalarExpr>, Rc<ScalarExpr>)> {
    let x_name = scalar_symbol_name(independent)?;
    let form = first_order_form(residual, target)?;
    let y_var = target_symbol(target);
    let n = replace_base_target_with_symbol(&form.n, target, &y_var);
    let m = replace_base_target_with_symbol(&form.m, target, &y_var);
    let y_name = scalar_symbol_name(&y_var)?;

    if depends_only_on(&n, y_name, x_name) && depends_only_on(&m, x_name, y_name) {
        let left = integrate(&n, &y_var).ok()?;
        let right = integrate(&Rc::new(ScalarExpr::Neg(m)), independent).ok()?;
        return Some((left, right));
    }

    if is_one(&n) {
        let rhs = simplify_scalar(&Rc::new(ScalarExpr::Neg(m)), RuleSet::Continuum);
        let (gx, hy) = split_xy_product(&rhs, x_name, y_name)?;
        let left = integrate(
            &Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), hy)),
            &y_var,
        )
        .ok()?;
        let right = integrate(&gx, independent).ok()?;
        return Some((left, right));
    }
    None
}

fn exact_form(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> Result<Option<(Rc<ScalarExpr>, Vec<SolutionStep>)>, Error> {
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
        phi,
        vec![
            SolutionStep {
                label: "differential form M + N y' = 0".to_string(),
                equation: Some(equation(
                    m.clone(),
                    Rc::new(ScalarExpr::Neg(Rc::new(ScalarExpr::Mul(
                        n.clone(),
                        derivative_unknown(target, 1),
                    )))),
                )),
                expr: None,
            },
            SolutionStep {
                label: "exactness check".to_string(),
                equation: Some(equation(my, nx)),
                expr: None,
            },
        ],
    )))
}

pub fn render_equation(eq: &Equation) -> String {
    format!(
        "{} = {}",
        scalar_to_latex(&eq.lhs),
        scalar_to_latex(&eq.rhs)
    )
}

pub fn render_classification(class: &OdeClassification, mode: &str) -> String {
    let kind = match class.kind {
        DifferentialKind::Ode => "ODE",
        DifferentialKind::Pde => "PDE",
    };
    let linear = if class.linear { "linear" } else { "nonlinear" };
    let homogeneous = match class.homogeneous {
        Some(true) => "homogeneous",
        Some(false) => "nonhomogeneous",
        None => "not applicable",
    };
    let subtypes = if class.subtypes.is_empty() {
        "none".to_string()
    } else {
        class.subtypes.join(", ")
    };
    let mut rows = vec![
        format!("\\text{{type}} &= \\text{{{kind}}}"),
        format!("\\text{{order}} &= {}", class.order),
        format!("\\text{{linearity}} &= \\text{{{linear}}}"),
        format!("\\text{{homogeneity}} &= \\text{{{homogeneous}}}"),
        format!("\\text{{first-order subtype}} &= \\text{{{subtypes}}}"),
    ];
    if mode == "details" {
        for reason in &class.reasons {
            rows.push(format!(
                "\\text{{reason}} &= \\text{{{}}}",
                escape_text(reason)
            ));
        }
    }
    format!(
        "\\begin{{aligned}}\n{}\n\\end{{aligned}}",
        rows.join("\\\\\n")
    )
}

pub fn render_solution(sol: &OdeSolution, mode: &str) -> String {
    match mode {
        "steps" => {
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
                        scalar_to_latex(expr)
                    ));
                }
            }
            if let Some(warning) = &sol.warning {
                rows.push(format!(
                    "\\text{{warning}} &= \\text{{{}}}",
                    escape_text(warning)
                ));
            }
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

fn solve_constant_from_explicit(
    rhs: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
    ic: &InitialCondition,
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
    ic: &InitialCondition,
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

fn validate_ic(target: &Target, ic: &InitialCondition) -> Option<()> {
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
            let mut terms = vec![canonical_scalar_key(a), canonical_scalar_key(b)];
            terms.sort();
            format!("add({})", terms.join(","))
        }
        ScalarExpr::Sub(a, b) => {
            let mut terms = vec![
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
