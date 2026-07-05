//! Scalar-expression utilities shared by the classifiers and solvers.

use super::*;

pub(crate) fn target_from_expr(expr: &Rc<ScalarExpr>) -> Result<Target, Error> {
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

pub(crate) fn base_unknown(target: &Target) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::UnknownFunc {
        name: target.name.clone(),
        args: target.args.clone(),
        derivative_orders: vec![0; target.args.len()],
    })
}

pub(crate) fn derivative_unknown(target: &Target, order: usize) -> Rc<ScalarExpr> {
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

pub(crate) fn target_symbol(target: &Target) -> Rc<ScalarExpr> {
    symbol(&target.name)
}

pub(crate) fn symbol(name: &str) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Sym {
        name: name.to_string(),
        latex: name.to_string(),
    })
}

pub(crate) fn constant() -> Rc<ScalarExpr> {
    symbol(constant_name())
}

pub(crate) fn constant_name() -> &'static str {
    "C_1"
}

pub(crate) fn target_occurrences(expr: &Rc<ScalarExpr>, target_name: &str) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    collect_target_occurrences(expr, target_name, &mut out);
    out
}

pub(crate) fn collect_target_occurrences(
    expr: &Rc<ScalarExpr>,
    target_name: &str,
    out: &mut Vec<Vec<usize>>,
) {
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

pub(crate) fn contains_target(expr: &Rc<ScalarExpr>, target_name: &str) -> bool {
    !target_occurrences(expr, target_name).is_empty()
}

pub(crate) fn target_degree(expr: &Rc<ScalarExpr>, target_name: &str) -> Option<usize> {
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

pub(crate) fn zero_target(expr: &Rc<ScalarExpr>, target_name: &str) -> Rc<ScalarExpr> {
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
pub(crate) struct SignedTerm {
    pub(crate) sign: f64,
    pub(crate) expr: Rc<ScalarExpr>,
}

pub(crate) fn additive_terms(expr: &Rc<ScalarExpr>) -> Vec<SignedTerm> {
    let mut out = Vec::new();
    collect_additive_terms(expr, 1.0, &mut out);
    out
}

pub(crate) fn collect_additive_terms(expr: &Rc<ScalarExpr>, sign: f64, out: &mut Vec<SignedTerm>) {
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

pub(crate) fn signed(sign: f64, expr: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    if sign == 1.0 {
        expr
    } else {
        Rc::new(ScalarExpr::Neg(expr))
    }
}

pub(crate) fn factor_atom(expr: &Rc<ScalarExpr>, atom: &Rc<ScalarExpr>) -> Option<Rc<ScalarExpr>> {
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

pub(crate) fn contains_atom(expr: &Rc<ScalarExpr>, atom: &Rc<ScalarExpr>) -> bool {
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

pub(crate) fn product_factors(expr: &Rc<ScalarExpr>) -> Vec<&Rc<ScalarExpr>> {
    match &**expr {
        ScalarExpr::Mul(a, b) => {
            let mut out = product_factors(a);
            out.extend(product_factors(b));
            out
        }
        _ => vec![expr],
    }
}

pub(crate) fn multiply_all(factors: Vec<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    factors
        .into_iter()
        .reduce(|a, b| Rc::new(ScalarExpr::Mul(a, b)))
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(1.0)))
}

pub(crate) fn depends_on_or_const(expr: &Rc<ScalarExpr>, allowed: &str) -> bool {
    let vars = scalar_symbols(expr);
    vars.into_iter().all(|v| v == allowed)
}

pub(crate) fn depends_only_on(expr: &Rc<ScalarExpr>, allowed: &str, forbidden: &str) -> bool {
    depends_on(expr, allowed) && !depends_on(expr, forbidden) && depends_on_or_const(expr, allowed)
}

pub(crate) fn scalar_symbols(expr: &Rc<ScalarExpr>) -> Vec<String> {
    let mut out = Vec::new();
    collect_scalar_symbols(expr, &mut out);
    out.sort();
    out.dedup();
    out
}

pub(crate) fn collect_scalar_symbols(expr: &Rc<ScalarExpr>, out: &mut Vec<String>) {
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

pub(crate) fn split_xy_product(
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

pub(crate) fn replace_base_target_with_symbol(
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

pub(crate) fn scale_arg(coeff: f64, arg: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    simplify_scalar(
        &fold_mul(&Rc::new(ScalarExpr::Num(coeff)), arg),
        RuleSet::Continuum,
    )
}

pub(crate) fn exp_linear(coeff: f64, independent: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    if coeff == 0.0 {
        return Rc::new(ScalarExpr::Num(1.0));
    }
    Rc::new(ScalarExpr::Func {
        name: "exp".to_string(),
        arg: scale_arg(coeff, independent),
    })
}

pub(crate) fn symbol_with_latex(name: &str, latex: &str) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Sym {
        name: name.to_string(),
        latex: latex.to_string(),
    })
}

pub(crate) fn num_latex(value: f64) -> String {
    scalar_to_latex(&ScalarExpr::Num(value))
}

pub(crate) fn constant2_name() -> &'static str {
    "C_2"
}

pub(crate) fn polynomial_coeffs(expr: &Rc<ScalarExpr>, x_name: &str) -> Option<Vec<f64>> {
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

pub(crate) fn trim_poly(mut values: Vec<f64>) -> Vec<f64> {
    while values.len() > 1 && values.last().is_some_and(|v| v.abs() < 1e-12) {
        values.pop();
    }
    values
}

pub(crate) fn polynomial_particular_coeffs(
    forcing: &[f64],
    coeffs: ConstantSecondOrder,
) -> Option<Vec<f64>> {
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

pub(crate) fn polynomial_expr(coeffs: &[f64], x: &Rc<ScalarExpr>) -> Rc<ScalarExpr> {
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

pub(crate) fn polynomial_template_latex(degree: usize) -> String {
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

pub(crate) fn apply_second_order_boundaries(
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

pub(crate) fn boundary_linear_row(
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

pub(crate) fn boundary_derivative_order(target: &Target, bc: &BoundaryCondition) -> Option<usize> {
    match &*bc.function {
        ScalarExpr::UnknownFunc {
            name,
            args,
            derivative_orders,
        } if name == &target.name && args.len() == 1 => Some(derivative_orders.iter().sum()),
        _ => None,
    }
}

pub(crate) fn power_series_latex(
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

pub(crate) fn finite_series_latex(independent: &Rc<ScalarExpr>, terms: usize) -> String {
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

pub(crate) fn lecture_power_series_recurrence(
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

pub(crate) fn is_regular_singular_candidate(
    residual: &Rc<ScalarExpr>,
    target: &Target,
    independent: &Rc<ScalarExpr>,
) -> bool {
    let Some(form) = second_order_linear_form(residual, target) else {
        return false;
    };
    is_power_of_var(&form.a2, independent, 2.0) && factor_atom(&form.a1, independent).is_some()
}

pub(crate) fn bessel_parameter(
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

pub(crate) fn is_power_of_var(expr: &Rc<ScalarExpr>, var: &Rc<ScalarExpr>, power: f64) -> bool {
    let expected = fold_pow(var.clone(), power);
    scalar_equiv_commutative(expr, &expected)
}

pub(crate) fn solve_constant_from_explicit(
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

pub(crate) fn solve_constant_from_implicit(
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

pub(crate) fn validate_ic(target: &Target, ic: &BoundaryCondition) -> Option<()> {
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

pub(crate) fn linear_coeff_const(
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

pub(crate) fn subst_symbol(
    expr: &Rc<ScalarExpr>,
    name: &str,
    replacement: &Rc<ScalarExpr>,
) -> Rc<ScalarExpr> {
    crate::substitute::subst_scalar_sym(expr, name, replacement)
}

pub(crate) fn is_zero(expr: &Rc<ScalarExpr>) -> bool {
    matches!(&*simplify_scalar(expr, RuleSet::Continuum), ScalarExpr::Num(n) if *n == 0.0)
}

pub(crate) fn is_one(expr: &Rc<ScalarExpr>) -> bool {
    matches!(&*simplify_scalar(expr, RuleSet::Continuum), ScalarExpr::Num(n) if *n == 1.0)
}

pub(crate) fn escape_text(text: &str) -> String {
    text.replace('\\', "\\textbackslash{}")
        .replace('_', "\\_")
        .replace('&', "\\&")
}

pub(crate) fn contains_formal_integral(expr: &Rc<ScalarExpr>) -> bool {
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

pub(crate) fn scalar_equiv_commutative(a: &Rc<ScalarExpr>, b: &Rc<ScalarExpr>) -> bool {
    canonical_scalar_key(&simplify_scalar(a, RuleSet::Continuum))
        == canonical_scalar_key(&simplify_scalar(b, RuleSet::Continuum))
}

pub(crate) fn canonical_scalar_key(expr: &Rc<ScalarExpr>) -> String {
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

pub(crate) fn numeric_scalar_value(expr: &Rc<ScalarExpr>) -> Option<f64> {
    match &**expr {
        ScalarExpr::Num(n) => Some(*n),
        ScalarExpr::Neg(a) => numeric_scalar_value(a).map(|n| -n),
        ScalarExpr::Div(a, b) => Some(numeric_scalar_value(a)? / numeric_scalar_value(b)?),
        _ => None,
    }
}
