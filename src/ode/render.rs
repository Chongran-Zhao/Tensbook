//! LaTeX rendering and structured UI payloads for ODE results.

use super::*;

pub fn render_equation(eq: &Equation) -> String {
    format!(
        "{} = {}",
        ode_scalar_to_latex(&eq.lhs),
        ode_scalar_to_latex(&eq.rhs)
    )
}

pub(crate) fn ode_scalar_to_latex(expr: &Rc<ScalarExpr>) -> String {
    scalar_to_latex(&ode_display_expr(expr))
}

pub(crate) fn ode_display_expr(expr: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
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

pub(crate) fn render_boundary_condition_status(bcs: &[BoundaryCondition]) -> String {
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

pub(crate) fn classification_values(class: &OdeClassification) -> (String, String, String, String) {
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

pub(crate) fn classification_summary(class: &OdeClassification) -> String {
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

pub(crate) fn supported_method_labels(class: &OdeClassification) -> Vec<String> {
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

pub(crate) fn unsupported_method_reason(class: &OdeClassification) -> String {
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

pub(crate) fn step_latex(steps: Vec<OdeStep>) -> Vec<String> {
    steps.into_iter().map(|step| step.latex).collect()
}

pub(crate) fn push_warning_step(steps: &mut Vec<OdeStep>, sol: &OdeSolution) {
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

pub(crate) fn solution_step_rows(sol: &OdeSolution, include_solution: bool) -> Vec<String> {
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

pub(crate) fn render_solution_process(sol: &OdeSolution) -> String {
    match sol.method.as_str() {
        "linear first-order integrating factor" => render_linear_solution_process(sol),
        "separable" => render_separable_solution_process(sol),
        "exact" => render_exact_solution_process(sol),
        _ => render_generic_solution_process(sol),
    }
}

pub(crate) fn render_linear_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(linear_process_steps(sol)))
}

pub(crate) fn linear_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
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

pub(crate) fn render_separable_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(separable_process_steps(sol)))
}

pub(crate) fn separable_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
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

pub(crate) fn render_exact_solution_process(sol: &OdeSolution) -> String {
    aligned_process(step_latex(exact_process_steps(sol)))
}

pub(crate) fn exact_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
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

pub(crate) fn render_generic_solution_process(sol: &OdeSolution) -> String {
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

pub(crate) fn generic_process_steps(sol: &OdeSolution) -> Vec<OdeStep> {
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

pub(crate) fn step_equation<'a>(sol: &'a OdeSolution, label: &str) -> Option<&'a Equation> {
    sol.steps
        .iter()
        .find(|step| step.label == label)
        .and_then(|step| step.equation.as_ref())
}

pub(crate) fn one_variable_unknown_arg_latex(expr: &Rc<ScalarExpr>) -> Option<String> {
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

pub(crate) fn push_final_solution(rows: &mut Vec<String>, sol: &OdeSolution) {
    if let Some(solution) = &sol.solution {
        rows.push(render_equation(solution));
    }
}

pub(crate) fn push_warning(rows: &mut Vec<String>, sol: &OdeSolution) {
    if let Some(warning) = &sol.warning {
        rows.push(format!("\\text{{Note: {}}}", escape_text(warning)));
    }
}

pub(crate) fn unsupported_text(sol: &OdeSolution) -> String {
    format!(
        "\\text{{Unsupported: {}}}",
        escape_text(
            sol.warning
                .as_deref()
                .unwrap_or("no symbolic method matched")
        )
    )
}

pub(crate) fn aligned_process(rows: Vec<String>) -> String {
    if rows.is_empty() {
        return "\\text{No symbolic steps available.}".to_string();
    }
    format!(
        "\\begin{{aligned}}\n&{}\n\\end{{aligned}}",
        rows.join("\\\\[3pt]\n&")
    )
}
