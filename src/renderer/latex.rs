//! Symbol-mode LaTeX rendering for scalar and tensor expressions.

use crate::symbolic::{extract_coeff, ScalarExpr};
use crate::tensor::TensorExpr;
use std::rc::Rc;

/// Render a scalar expression to LaTeX.
pub fn scalar_to_latex(expr: &ScalarExpr) -> String {
    render_scalar(expr)
}

/// Render a tensor expression to LaTeX (symbolic, e.g. `\bm F^{\mathsf{T}} \bm F`).
pub fn tensor_to_latex(expr: &TensorExpr) -> String {
    render_tensor(expr)
}

// ---- scalar -------------------------------------------------------------

// Precedence levels: Add/Sub = 1, Mul/Neg = 2, Pow = 3, atoms (incl. \frac) = 4.
fn sprec(expr: &ScalarExpr) -> u8 {
    match expr {
        ScalarExpr::Add(..) | ScalarExpr::Sub(..) | ScalarExpr::Ddot(..) => 1,
        ScalarExpr::Mul(..) | ScalarExpr::Neg(..) | ScalarExpr::SpecSum { .. } => 2,
        ScalarExpr::Pow(..)
        | ScalarExpr::Log(..)
        | ScalarExpr::Det(..)
        | ScalarExpr::Tr(..)
        | ScalarExpr::Func { .. } => 3,
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Div(..)
        | ScalarExpr::SetElem { .. } => 4,
    }
}

fn paren(s: String) -> String {
    format!("\\left( {s} \\right)")
}

fn render_scalar(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Sym { latex, .. } => latex.clone(),
        ScalarExpr::Num(n) => render_num(*n),
        ScalarExpr::Add(a, b) => format!("{} + {}", render_scalar(a), render_scalar(b)),
        ScalarExpr::Sub(a, b) => {
            let rhs = if sprec(b) <= 1 {
                paren(render_scalar(b))
            } else {
                render_scalar(b)
            };
            format!("{} - {}", render_scalar(a), rhs)
        }
        ScalarExpr::Mul(a, b) => {
            let lhs = if sprec(a) < 2 {
                paren(render_scalar(a))
            } else {
                render_scalar(a)
            };
            let rhs = if sprec(b) < 2 {
                paren(render_scalar(b))
            } else {
                render_scalar(b)
            };
            format!("{lhs} \\, {rhs}")
        }
        ScalarExpr::Div(a, b) => render_fraction(a, b),
        ScalarExpr::Pow(base, exp) => {
            let b = match &**base {
                ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => {
                    render_scalar(base)
                }
                _ => paren(render_scalar(base)),
            };
            format!("{{{b}}}^{{{}}}", render_scalar(exp))
        }
        ScalarExpr::Neg(a) => {
            let inner = if sprec(a) < 2 {
                paren(render_scalar(a))
            } else {
                render_scalar(a)
            };
            format!("-{inner}")
        }
        ScalarExpr::Log(a) => {
            let inner = match &**a {
                ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => {
                    render_scalar(a)
                }
                _ => paren(render_scalar(a)),
            };
            format!("\\log {inner}")
        }
        ScalarExpr::Func { name, arg } => {
            let inner = render_scalar(arg);
            match name.as_str() {
                "sqrt" => format!("\\sqrt{{{inner}}}"),
                "exp" => format!("e^{{{inner}}}"),
                "sinh" | "cosh" | "tanh" | "sin" | "cos" | "tan" => {
                    let wrapped = match &**arg {
                        ScalarExpr::Sym { .. }
                        | ScalarExpr::Num(_)
                        | ScalarExpr::SetElem { .. } => inner,
                        _ => paren(inner),
                    };
                    format!("\\{name} {wrapped}")
                }
                other => format!("\\operatorname{{{other}}}\\left( {inner} \\right)"),
            }
        }
        ScalarExpr::SetElem { latex, index, .. } => {
            format!("{{{latex}}}_{{{}}}", index.latex())
        }
        ScalarExpr::SpecSum { body, index, dim } => {
            format!("\\sum_{{{index}=1}}^{{{dim}}} {}", render_scalar(body))
        }
        ScalarExpr::Det(t) => {
            let inner = match &**t {
                TensorExpr::Var { .. } => render_tensor(t),
                _ => paren(render_tensor(t)),
            };
            format!("\\det {inner}")
        }
        ScalarExpr::Tr(t) => {
            let inner = match &**t {
                TensorExpr::Var { .. } => render_tensor(t),
                _ => paren(render_tensor(t)),
            };
            format!("\\operatorname{{tr}} {inner}")
        }
        // A : B — the double-contraction display symbol from the spec.
        ScalarExpr::Ddot(a, b) => {
            let lhs = if tprec(a) < 2 {
                paren(render_tensor(a))
            } else {
                render_tensor(a)
            };
            let rhs = if tprec(b) < 2 {
                paren(render_tensor(b))
            } else {
                render_tensor(b)
            };
            format!("{lhs} : {rhs}")
        }
    }
}

fn render_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        return format!("{}", n as i64);
    }
    // Render simple rationals as fractions (e.g. -2/3 rather than
    // -0.6666666666666666) — numeric folding loses the Div structure.
    for den in 2..=12u32 {
        let num = n * den as f64;
        if (num - num.round()).abs() < 1e-9 && num.round().abs() < 1e6 {
            let num = num.round() as i64;
            let sign = if num < 0 { "-" } else { "" };
            return format!("{sign}\\frac{{{}}}{{{den}}}", num.abs());
        }
    }
    format!("{n}")
}

fn render_fraction(num: &ScalarExpr, den: &ScalarExpr) -> String {
    if let Some(tex) = render_numeric_denominator_as_factor(num, den) {
        return tex;
    }
    let (num_neg, num_tex) = render_abs_if_negative(num);
    let (den_neg, den_tex) = render_abs_if_negative(den);
    let frac = format!("\\frac{{{num_tex}}}{{{den_tex}}}");
    if num_neg ^ den_neg {
        format!("-{frac}")
    } else {
        frac
    }
}

fn render_numeric_denominator_as_factor(num: &ScalarExpr, den: &ScalarExpr) -> Option<String> {
    let den_value = numeric_literal(den)?;
    if den_value == 0.0 || numeric_literal(num).is_some() || !can_render_as_product_factor(num) {
        return None;
    }

    let (num_neg, num_tex) = render_abs_as_product_factor(num);
    let sign = if (den_value < 0.0) ^ num_neg { "-" } else { "" };
    Some(format!(
        "{sign}\\frac{{1}}{{{}}} \\, {num_tex}",
        render_num(den_value.abs())
    ))
}

fn numeric_literal(expr: &ScalarExpr) -> Option<f64> {
    match expr {
        ScalarExpr::Num(n) => Some(*n),
        ScalarExpr::Neg(inner) => numeric_literal(inner).map(|n| -n),
        _ => None,
    }
}

fn can_render_as_product_factor(expr: &ScalarExpr) -> bool {
    !matches!(expr, ScalarExpr::Add(..) | ScalarExpr::Sub(..))
}

fn render_abs_as_product_factor(expr: &ScalarExpr) -> (bool, String) {
    match expr {
        ScalarExpr::Neg(inner) => (true, render_product_factor(inner)),
        _ => (false, render_product_factor(expr)),
    }
}

fn render_product_factor(expr: &ScalarExpr) -> String {
    match expr {
        // `(tr C) x` — same convention as tensor coefficients.
        ScalarExpr::Tr(_) => paren(render_scalar(expr)),
        _ if sprec(expr) < 2 => paren(render_scalar(expr)),
        _ => render_scalar(expr),
    }
}

fn render_abs_if_negative(expr: &ScalarExpr) -> (bool, String) {
    match expr {
        ScalarExpr::Num(n) if *n < 0.0 => (true, render_num(-n)),
        ScalarExpr::Neg(inner) => (true, render_scalar(inner)),
        _ => (false, render_scalar(expr)),
    }
}

fn render_tensor_coefficient(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Mul(a, b) => {
            let mut factors = Vec::new();
            collect_scalar_product_factors(a, &mut factors);
            collect_scalar_product_factors(b, &mut factors);
            factors
                .into_iter()
                .map(render_tensor_coefficient_factor)
                .collect::<Vec<_>>()
                .join(" \\, ")
        }
        _ => render_tensor_coefficient_factor(expr),
    }
}

fn collect_scalar_product_factors<'a>(expr: &'a ScalarExpr, out: &mut Vec<&'a ScalarExpr>) {
    match expr {
        ScalarExpr::Mul(a, b) => {
            collect_scalar_product_factors(a, out);
            collect_scalar_product_factors(b, out);
        }
        _ => out.push(expr),
    }
}

fn render_tensor_coefficient_factor(expr: &ScalarExpr) -> String {
    match expr {
        ScalarExpr::Tr(_) => paren(render_scalar(expr)),
        _ if sprec(expr) < 2 => paren(render_scalar(expr)),
        _ => render_scalar(expr),
    }
}

// ---- tensor -------------------------------------------------------------

// Precedence: Add/Sub = 1, MatMul/ScalarMul/Neg/Outer = 2, atoms = 3.
fn tprec(expr: &TensorExpr) -> u8 {
    match expr {
        TensorExpr::Add(..) | TensorExpr::Sub(..) => 1,
        TensorExpr::MatMul(..)
        | TensorExpr::ScalarMul(..)
        | TensorExpr::Neg(..)
        | TensorExpr::Outer(..)
        | TensorExpr::BoxTimes(..) => 2,
        TensorExpr::Var { .. }
        | TensorExpr::Transpose(..)
        | TensorExpr::Inverse(..)
        | TensorExpr::InverseTranspose(..)
        | TensorExpr::Diff { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. }
        | TensorExpr::Filled { .. } => 3,
        TensorExpr::SumIdx { .. } => 1,
    }
}

fn superscripted(t: &TensorExpr, sup: &str) -> String {
    let base = match t {
        TensorExpr::Var { .. } | TensorExpr::Filled { .. } => render_tensor(t),
        _ => paren(render_tensor(t)),
    };
    format!("{base}^{{{sup}}}")
}

fn render_tensor(expr: &TensorExpr) -> String {
    match expr {
        TensorExpr::Var { latex, .. } => latex.clone(),
        TensorExpr::SetElem { latex, index, .. } => {
            format!("{{{latex}}}_{{{}}}", index.latex())
        }
        TensorExpr::Filled { latex, .. } => latex.clone(),
        TensorExpr::SumIdx {
            index,
            range,
            exclude,
            body,
        } => {
            let lower = match exclude {
                Some(ex) => format!("\\substack{{{index}=1 \\\\ {index}\\ne {ex}}}"),
                None => format!("{index}=1"),
            };
            format!("\\sum_{{{lower}}}^{{{range}}} {}", render_tensor(body))
        }
        TensorExpr::Transpose(t) => superscripted(t, "\\mathsf{T}"),
        TensorExpr::Inverse(t) => superscripted(t, "-1"),
        TensorExpr::InverseTranspose(t) => superscripted(t, "-\\mathsf{T}"),
        TensorExpr::Identity4 { .. } => "\\mathbb{I}".to_string(),
        TensorExpr::Diff {
            num,
            den,
            num_label,
        } => {
            let num_tex = num_label.clone().unwrap_or_else(|| {
                let inner = render_tensor(num);
                if tprec(num) < 3 {
                    paren(inner)
                } else {
                    inner
                }
            });
            format!(
                "\\frac{{\\partial {num_tex}}}{{\\partial {}}}",
                render_tensor(den)
            )
        }
        TensorExpr::MatMul(a, b) => {
            // Outer-product operands must be parenthesized: matrix products
            // render as juxtaposition, so `(N_a ⊗ N_a) T` would otherwise
            // read as `N_a ⊗ (N_a T)`.
            let needs_paren = |t: &TensorExpr| tprec(t) < 2 || matches!(t, TensorExpr::Outer(..));
            let lhs = if needs_paren(a) {
                paren(render_tensor(a))
            } else {
                render_tensor(a)
            };
            let rhs = if needs_paren(b) {
                paren(render_tensor(b))
            } else {
                render_tensor(b)
            };
            format!("{lhs} {rhs}")
        }
        TensorExpr::Outer(a, b) => {
            let lhs = if tprec(a) < 2 {
                paren(render_tensor(a))
            } else {
                render_tensor(a)
            };
            let rhs = if tprec(b) < 2 {
                paren(render_tensor(b))
            } else {
                render_tensor(b)
            };
            format!("{lhs} \\otimes {rhs}")
        }
        TensorExpr::BoxTimes(a, b) => {
            let lhs = if tprec(a) <= 2 {
                paren(render_tensor(a))
            } else {
                render_tensor(a)
            };
            let rhs = if tprec(b) <= 2 {
                paren(render_tensor(b))
            } else {
                render_tensor(b)
            };
            format!("{lhs} \\boxtimes {rhs}")
        }
        TensorExpr::Add(a, b) => {
            if let Some(rhs) = render_negative_tensor(b) {
                format!("{} - {}", render_tensor(a), rhs)
            } else {
                format!("{} + {}", render_tensor(a), render_tensor(b))
            }
        }
        TensorExpr::Sub(a, b) => {
            let rhs = if tprec(b) <= 1 {
                paren(render_tensor(b))
            } else {
                render_tensor(b)
            };
            format!("{} - {}", render_tensor(a), rhs)
        }
        TensorExpr::ScalarMul(s, t) => {
            let s_tex = if sprec(s) < 2 {
                paren(scalar_to_latex(s))
            } else {
                render_tensor_coefficient(s)
            };
            let t_tex = if tprec(t) < 2 {
                paren(render_tensor(t))
            } else {
                render_tensor(t)
            };
            format!("{s_tex} \\, {t_tex}")
        }
        TensorExpr::Neg(t) => {
            let inner = if tprec(t) < 2 {
                paren(render_tensor(t))
            } else {
                render_tensor(t)
            };
            format!("-{inner}")
        }
    }
}

fn render_negative_tensor(expr: &TensorExpr) -> Option<String> {
    match expr {
        TensorExpr::Neg(t) => Some(if tprec(t) < 2 {
            paren(render_tensor(t))
        } else {
            render_tensor(t)
        }),
        TensorExpr::ScalarMul(s, t) => {
            let positive = positive_negative_scalar(s)?;
            let s_tex = if sprec(&positive) < 2 {
                paren(render_scalar(&positive))
            } else {
                render_tensor_coefficient(&positive)
            };
            let t_tex = if tprec(t) < 2 {
                paren(render_tensor(t))
            } else {
                render_tensor(t)
            };
            Some(format!("{s_tex} \\, {t_tex}"))
        }
        _ => None,
    }
}

fn positive_negative_scalar(expr: &ScalarExpr) -> Option<ScalarExpr> {
    match expr {
        ScalarExpr::Neg(inner) => return Some((**inner).clone()),
        ScalarExpr::Num(n) if *n < 0.0 => return Some((*numeric_coeff_expr(-n)).clone()),
        _ => {}
    }

    let expr = Rc::new(expr.clone());
    let (coeff, rest) = extract_coeff(&expr);
    if coeff < 0.0 {
        Some((*scalar_with_coeff(-coeff, rest)).clone())
    } else {
        None
    }
}

fn scalar_with_coeff(coeff: f64, rest: Option<Rc<ScalarExpr>>) -> Rc<ScalarExpr> {
    match rest {
        None => numeric_coeff_expr(coeff),
        Some(r) if coeff == 1.0 => r,
        Some(r) => Rc::new(ScalarExpr::Mul(numeric_coeff_expr(coeff), r)),
    }
}

fn numeric_coeff_expr(coeff: f64) -> Rc<ScalarExpr> {
    for den in 1..=12 {
        let num = (coeff * den as f64).round();
        if (coeff - num / den as f64).abs() < 1e-12 {
            if den == 1 {
                return Rc::new(ScalarExpr::Num(num));
            }
            return Rc::new(ScalarExpr::Div(
                Rc::new(ScalarExpr::Num(num)),
                Rc::new(ScalarExpr::Num(den as f64)),
            ));
        }
    }
    Rc::new(ScalarExpr::Num(coeff))
}
