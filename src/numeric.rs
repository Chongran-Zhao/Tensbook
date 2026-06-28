//! Numeric evaluation for plotting.
//!
//! This is intentionally small and conservative: unsupported symbolic nodes,
//! unbound symbols, and non-finite results return `None` so plot sampling can
//! create a gap instead of guessing.

use crate::symbolic::ScalarExpr;
use std::collections::BTreeSet;
use std::f64::consts::{E, PI};

pub fn eval_at(expr: &ScalarExpr, var: &str, x: f64) -> Option<f64> {
    finite(match expr {
        ScalarExpr::Num(n) => *n,
        ScalarExpr::Sym { name, .. } if name == var => x,
        ScalarExpr::Sym { name, .. } if name == "pi" => PI,
        ScalarExpr::Sym { name, .. } if name == "e" => E,
        ScalarExpr::Sym { .. } => return None,
        ScalarExpr::Neg(a) => -eval_at(a, var, x)?,
        ScalarExpr::Add(a, b) => eval_at(a, var, x)? + eval_at(b, var, x)?,
        ScalarExpr::Sub(a, b) => eval_at(a, var, x)? - eval_at(b, var, x)?,
        ScalarExpr::Mul(a, b) => eval_at(a, var, x)? * eval_at(b, var, x)?,
        ScalarExpr::Div(a, b) => eval_at(a, var, x)? / eval_at(b, var, x)?,
        ScalarExpr::Pow(a, b) => eval_at(a, var, x)?.powf(eval_at(b, var, x)?),
        ScalarExpr::Log(a) => eval_at(a, var, x)?.ln(),
        ScalarExpr::Func { name, arg } => {
            let value = eval_at(arg, var, x)?;
            match name.as_str() {
                "sin" => value.sin(),
                "cos" => value.cos(),
                "tan" => value.tan(),
                "exp" => value.exp(),
                "sqrt" => value.sqrt(),
                "sinh" => value.sinh(),
                "cosh" => value.cosh(),
                "tanh" => value.tanh(),
                _ => return None,
            }
        }
        ScalarExpr::UnknownFunc { .. }
        | ScalarExpr::Integral { .. }
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _)
        | ScalarExpr::SpecSum { .. }
        | ScalarExpr::SetElem { .. } => return None,
    })
}

pub fn symbol_names(expr: &ScalarExpr) -> Vec<String> {
    let mut names = BTreeSet::new();
    collect_symbol_names(expr, &mut names);
    names.into_iter().collect()
}

pub fn unbound_symbols(expr: &ScalarExpr, abscissa: &str) -> Vec<String> {
    symbol_names(expr)
        .into_iter()
        .filter(|name| name != abscissa && name != "pi" && name != "e")
        .collect()
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn collect_symbol_names(expr: &ScalarExpr, names: &mut BTreeSet<String>) {
    match expr {
        ScalarExpr::Sym { name, .. } => {
            names.insert(name.clone());
        }
        ScalarExpr::Num(_) => {}
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            collect_symbol_names(a, names);
        }
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            collect_symbol_names(a, names);
            collect_symbol_names(b, names);
        }
        ScalarExpr::UnknownFunc { args, .. } => {
            for arg in args {
                collect_symbol_names(arg, names);
            }
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => {
            collect_symbol_names(integrand, names);
            collect_symbol_names(variable, names);
        }
        ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _)
        | ScalarExpr::SpecSum { .. }
        | ScalarExpr::SetElem { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn sym(name: &str) -> Rc<ScalarExpr> {
        Rc::new(ScalarExpr::sym(name, name))
    }

    fn num(value: f64) -> Rc<ScalarExpr> {
        Rc::new(ScalarExpr::Num(value))
    }

    fn func(name: &str, arg: Rc<ScalarExpr>) -> ScalarExpr {
        ScalarExpr::Func {
            name: name.to_string(),
            arg,
        }
    }

    #[test]
    fn evaluates_basic_numeric_scalar_expressions() {
        let x = sym("x");
        let sin_x = func("sin", x.clone());
        assert_eq!(eval_at(&sin_x, "x", 0.0), Some(0.0));

        let square = ScalarExpr::Pow(x.clone(), num(2.0));
        assert_eq!(eval_at(&square, "x", 3.0), Some(9.0));

        let exp_zero = func("exp", num(0.0));
        assert_eq!(eval_at(&exp_zero, "x", 1.0), Some(1.0));
    }

    #[test]
    fn non_finite_values_become_gaps() {
        let reciprocal = ScalarExpr::Div(num(1.0), sym("x"));
        assert_eq!(eval_at(&reciprocal, "x", 0.0), None);

        let sqrt_negative = func("sqrt", num(-1.0));
        assert_eq!(eval_at(&sqrt_negative, "x", 0.0), None);
    }

    #[test]
    fn constants_and_unbound_symbols_are_distinguished() {
        let pi = ScalarExpr::sym("pi", "\\pi");
        assert_eq!(eval_at(&pi, "x", 0.0), Some(PI));

        let e = ScalarExpr::sym("e", "e");
        assert_eq!(eval_at(&e, "x", 0.0), Some(E));

        let a_times_x = ScalarExpr::Mul(sym("a"), sym("x"));
        assert_eq!(unbound_symbols(&a_times_x, "x"), vec!["a".to_string()]);
    }
}
