//! First/second-order symbolic solvers and solvable-form detection.

use super::*;

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

pub(crate) fn solve_linear_first_order(
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

pub(crate) fn solve_implicit_pair(params: ImplicitSolveParams<'_>) -> Result<OdeSolution, Error> {
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

pub(crate) fn second_order_linear_form(
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

pub(crate) fn constant_second_order_coefficients(
    form: &SecondOrderLinearForm,
) -> Option<ConstantSecondOrder> {
    let a2 = numeric_scalar_value(&form.a2)?;
    if a2 == 0.0 {
        return None;
    }
    Some(ConstantSecondOrder {
        a: numeric_scalar_value(&form.a1)? / a2,
        b: numeric_scalar_value(&form.a0)? / a2,
    })
}

pub(crate) fn constant_second_order_homogeneous(
    form: &SecondOrderLinearForm,
) -> Option<ConstantSecondOrder> {
    if is_zero(&form.forcing) {
        constant_second_order_coefficients(form)
    } else {
        None
    }
}

pub(crate) fn solve_characteristic_second_order(
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

pub(crate) fn characteristic_steps(coeffs: ConstantSecondOrder) -> Vec<SolutionStep> {
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

pub(crate) fn homogeneous_solution_rhs(
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

pub(crate) fn solve_undetermined_second_order(
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

pub(crate) fn solve_variation_second_order(
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

pub(crate) fn solve_power_series(
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

pub(crate) fn solve_frobenius(
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

pub(crate) fn linear_first_order_coeffs(
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

pub(crate) fn first_order_form(
    residual: &Rc<ScalarExpr>,
    target: &Target,
) -> Option<FirstOrderForm> {
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

pub(crate) fn separable_form(
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

pub(crate) fn separable_solution_steps(
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

pub(crate) fn differential_form_latex(m: &Rc<ScalarExpr>, n: &Rc<ScalarExpr>) -> String {
    format!(
        "{}\\, dx + {}\\, dy = 0",
        ode_scalar_to_latex(m),
        ode_scalar_to_latex(n),
    )
}

pub(crate) fn identify_mn_latex(m: &Rc<ScalarExpr>, n: &Rc<ScalarExpr>) -> String {
    format!(
        "M = {},\\qquad N = {}",
        ode_scalar_to_latex(m),
        ode_scalar_to_latex(n),
    )
}

pub(crate) fn unavailable_method_solution(
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

pub(crate) fn exact_form(
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
