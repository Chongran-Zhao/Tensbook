//! Conservative rule-based scalar integration.

use crate::differentiation::{diff_scalar_by_scalar, scalar_mentions_scalar};
use crate::error::Error;
use crate::simplifier::{simplify_scalar, RuleSet};
use crate::symbolic::{fold_mul, fold_pow, ScalarExpr};
use std::rc::Rc;

pub fn integrate(
    expr: &Rc<ScalarExpr>,
    variable: &Rc<ScalarExpr>,
) -> Result<Rc<ScalarExpr>, Error> {
    let var = scalar_symbol_name(variable).ok_or_else(|| {
        Error::msg("Integrate expects a scalar variable such as x: Integrate(expr, x)")
    })?;
    Ok(simplify_scalar(
        &integrate_by_name(expr, var, variable)?,
        RuleSet::Continuum,
    ))
}

pub fn formal_integral(expr: Rc<ScalarExpr>, variable: Rc<ScalarExpr>) -> Rc<ScalarExpr> {
    Rc::new(ScalarExpr::Integral {
        integrand: expr,
        variable,
    })
}

pub fn scalar_symbol_name(expr: &ScalarExpr) -> Option<&str> {
    match expr {
        ScalarExpr::Sym { name, .. } => Some(name),
        _ => None,
    }
}

pub fn depends_on(expr: &ScalarExpr, var: &str) -> bool {
    scalar_mentions_scalar(expr, var)
}

fn integrate_by_name(
    expr: &Rc<ScalarExpr>,
    var: &str,
    variable: &Rc<ScalarExpr>,
) -> Result<Rc<ScalarExpr>, Error> {
    let expr = simplify_scalar(expr, RuleSet::Algebra);
    if !depends_on(&expr, var) {
        return Ok(Rc::new(ScalarExpr::Mul(expr, variable.clone())));
    }
    match &*expr {
        ScalarExpr::Add(a, b) => Ok(Rc::new(ScalarExpr::Add(
            integrate_by_name(a, var, variable)?,
            integrate_by_name(b, var, variable)?,
        ))),
        ScalarExpr::Sub(a, b) => Ok(Rc::new(ScalarExpr::Sub(
            integrate_by_name(a, var, variable)?,
            integrate_by_name(b, var, variable)?,
        ))),
        ScalarExpr::Sym { name, .. } if name == var => Ok(Rc::new(ScalarExpr::Div(
            Rc::new(ScalarExpr::Pow(
                variable.clone(),
                Rc::new(ScalarExpr::Num(2.0)),
            )),
            Rc::new(ScalarExpr::Num(2.0)),
        ))),
        ScalarExpr::Pow(base, exp) => integrate_power(base, exp, var, variable)
            .unwrap_or_else(|| Ok(formal_integral(expr.clone(), variable.clone()))),
        ScalarExpr::Div(a, b) => {
            if is_one(a) && is_var(b, var) {
                return Ok(Rc::new(ScalarExpr::Log(variable.clone())));
            }
            if !depends_on(a, var) {
                if let Some(inv_power) = reciprocal_power_as_power(b, var) {
                    let body = Rc::new(ScalarExpr::Mul(a.clone(), inv_power));
                    return integrate_by_name(&body, var, variable);
                }
            }
            if same_expr(&diff_scalar_by_scalar(b, var)?, a) {
                return Ok(Rc::new(ScalarExpr::Log(b.clone())));
            }
            try_product_rule(&expr, var, variable)
                .unwrap_or_else(|| Ok(formal_integral(expr.clone(), variable.clone())))
        }
        ScalarExpr::Func { name, arg } => integrate_known_func(name, arg, var, variable)
            .unwrap_or_else(|| Ok(formal_integral(expr.clone(), variable.clone()))),
        ScalarExpr::Mul(_, _) => try_product_rule(&expr, var, variable)
            .unwrap_or_else(|| Ok(formal_integral(expr.clone(), variable.clone()))),
        _ => Ok(formal_integral(expr.clone(), variable.clone())),
    }
}

fn integrate_power(
    base: &Rc<ScalarExpr>,
    exp: &Rc<ScalarExpr>,
    var: &str,
    variable: &Rc<ScalarExpr>,
) -> Option<Result<Rc<ScalarExpr>, Error>> {
    let ScalarExpr::Num(n) = &**exp else {
        return None;
    };
    if !is_var(base, var) {
        return None;
    }
    if *n == -1.0 {
        return Some(Ok(Rc::new(ScalarExpr::Log(variable.clone()))));
    }
    let next = n + 1.0;
    Some(Ok(Rc::new(ScalarExpr::Div(
        Rc::new(ScalarExpr::Pow(
            variable.clone(),
            Rc::new(ScalarExpr::Num(next)),
        )),
        Rc::new(ScalarExpr::Num(next)),
    ))))
}

fn integrate_known_func(
    name: &str,
    arg: &Rc<ScalarExpr>,
    var: &str,
    _variable: &Rc<ScalarExpr>,
) -> Option<Result<Rc<ScalarExpr>, Error>> {
    let darg = diff_scalar_by_scalar(arg, var).ok()?;
    if !is_nonzero_constant_wrt(&darg, var) {
        return None;
    }
    let anti = antiderivative_for_func(name, arg)?;
    Some(Ok(Rc::new(ScalarExpr::Div(anti, darg))))
}

fn antiderivative_for_func(name: &str, arg: &Rc<ScalarExpr>) -> Option<Rc<ScalarExpr>> {
    let func = |n: &str| {
        Rc::new(ScalarExpr::Func {
            name: n.to_string(),
            arg: arg.clone(),
        })
    };
    match name {
        "exp" => Some(func("exp")),
        "sin" => Some(Rc::new(ScalarExpr::Neg(func("cos")))),
        "cos" => Some(func("sin")),
        "sinh" => Some(func("cosh")),
        "cosh" => Some(func("sinh")),
        "tan" => Some(Rc::new(ScalarExpr::Neg(Rc::new(ScalarExpr::Log(func(
            "cos",
        )))))),
        _ => None,
    }
}

fn try_product_rule(
    expr: &Rc<ScalarExpr>,
    var: &str,
    variable: &Rc<ScalarExpr>,
) -> Option<Result<Rc<ScalarExpr>, Error>> {
    let factors = product_factors(expr);
    let (constant, dependent): (Vec<_>, Vec<_>) =
        factors.into_iter().partition(|f| !depends_on(f, var));
    if !constant.is_empty() && !dependent.is_empty() {
        let const_part = multiply_all(constant);
        let dep_part = multiply_all(dependent);
        return Some(
            integrate_by_name(&dep_part, var, variable)
                .map(|inner| simplify_scalar(&fold_mul(&const_part, &inner), RuleSet::Continuum)),
        );
    }

    if let Some(value) = try_chain_product(expr, var) {
        return Some(value);
    }
    None
}

fn try_chain_product(expr: &Rc<ScalarExpr>, var: &str) -> Option<Result<Rc<ScalarExpr>, Error>> {
    let factors = product_factors(expr);
    for (i, factor) in factors.iter().enumerate() {
        match &***factor {
            ScalarExpr::Func { name, arg } => {
                let darg = diff_scalar_by_scalar(arg, var).ok()?;
                let Some(j) = factors
                    .iter()
                    .enumerate()
                    .find_map(|(j, f)| (j != i && same_expr(f, &darg)).then_some(j))
                else {
                    continue;
                };
                let rest = rest_product(&factors, &[i, j]);
                if depends_on(&rest, var) {
                    continue;
                }
                let anti = antiderivative_for_func(name, arg)?;
                return Some(Ok(simplify_scalar(
                    &fold_mul(&rest, &anti),
                    RuleSet::Continuum,
                )));
            }
            ScalarExpr::Pow(base, exp) => {
                let ScalarExpr::Num(n) = &**exp else {
                    continue;
                };
                if *n == -1.0 {
                    continue;
                }
                let dbase = diff_scalar_by_scalar(base, var).ok()?;
                let Some(j) = factors
                    .iter()
                    .enumerate()
                    .find_map(|(j, f)| (j != i && same_expr(f, &dbase)).then_some(j))
                else {
                    continue;
                };
                let rest = rest_product(&factors, &[i, j]);
                if depends_on(&rest, var) {
                    continue;
                }
                let anti = Rc::new(ScalarExpr::Div(
                    fold_pow(base.clone(), n + 1.0),
                    Rc::new(ScalarExpr::Num(n + 1.0)),
                ));
                return Some(Ok(simplify_scalar(
                    &fold_mul(&rest, &anti),
                    RuleSet::Continuum,
                )));
            }
            _ => {}
        }
    }
    None
}

fn is_nonzero_constant_wrt(expr: &Rc<ScalarExpr>, var: &str) -> bool {
    !depends_on(expr, var) && !matches!(&**expr, ScalarExpr::Num(n) if *n == 0.0)
}

fn is_var(expr: &Rc<ScalarExpr>, var: &str) -> bool {
    matches!(&**expr, ScalarExpr::Sym { name, .. } if name == var)
}

fn reciprocal_power_as_power(expr: &Rc<ScalarExpr>, var: &str) -> Option<Rc<ScalarExpr>> {
    if is_var(expr, var) {
        return Some(Rc::new(ScalarExpr::Pow(
            expr.clone(),
            Rc::new(ScalarExpr::Num(-1.0)),
        )));
    }
    match &**expr {
        ScalarExpr::Pow(base, exp) if is_var(base, var) => {
            let ScalarExpr::Num(n) = &**exp else {
                return None;
            };
            Some(Rc::new(ScalarExpr::Pow(
                base.clone(),
                Rc::new(ScalarExpr::Num(-n)),
            )))
        }
        _ => None,
    }
}

fn is_one(expr: &Rc<ScalarExpr>) -> bool {
    matches!(&**expr, ScalarExpr::Num(n) if *n == 1.0)
}

fn same_expr(a: &Rc<ScalarExpr>, b: &Rc<ScalarExpr>) -> bool {
    simplify_scalar(a, RuleSet::Continuum) == simplify_scalar(b, RuleSet::Continuum)
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

fn multiply_all(factors: Vec<&Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    factors
        .into_iter()
        .cloned()
        .reduce(|a, b| Rc::new(ScalarExpr::Mul(a, b)))
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(1.0)))
}

fn rest_product(factors: &[&Rc<ScalarExpr>], skip: &[usize]) -> Rc<ScalarExpr> {
    factors
        .iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .map(|(_, f)| (*f).clone())
        .reduce(|a, b| Rc::new(ScalarExpr::Mul(a, b)))
        .unwrap_or_else(|| Rc::new(ScalarExpr::Num(1.0)))
}
