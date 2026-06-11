//! Symbol-mode LaTeX rendering for scalar and tensor expressions.

use crate::symbolic::ScalarExpr;
use crate::tensor::TensorExpr;

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
        ScalarExpr::Mul(..) | ScalarExpr::Neg(..) => 2,
        ScalarExpr::Pow(..) | ScalarExpr::Log(..) | ScalarExpr::Det(..) | ScalarExpr::Tr(..) => 3,
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::Div(..) => 4,
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
        ScalarExpr::Div(a, b) => {
            format!("\\frac{{{}}}{{{}}}", render_scalar(a), render_scalar(b))
        }
        ScalarExpr::Pow(base, exp) => {
            let b = match &**base {
                ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => render_scalar(base),
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
                ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => render_scalar(a),
                _ => paren(render_scalar(a)),
            };
            format!("\\log {inner}")
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
        format!("{}", n as i64)
    } else {
        format!("{n}")
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
        | TensorExpr::Outer(..) => 2,
        TensorExpr::Var { .. }
        | TensorExpr::Transpose(..)
        | TensorExpr::Inverse(..)
        | TensorExpr::InverseTranspose(..)
        | TensorExpr::Diff { .. }
        | TensorExpr::Spectral { .. } => 3,
    }
}

fn superscripted(t: &TensorExpr, sup: &str) -> String {
    let base = match t {
        TensorExpr::Var { .. } => render_tensor(t),
        _ => paren(render_tensor(t)),
    };
    format!("{base}^{{{sup}}}")
}

fn render_tensor(expr: &TensorExpr) -> String {
    match expr {
        TensorExpr::Var { latex, .. } => latex.clone(),
        TensorExpr::Transpose(t) => superscripted(t, "\\mathsf{T}"),
        TensorExpr::Inverse(t) => superscripted(t, "-1"),
        TensorExpr::InverseTranspose(t) => superscripted(t, "-\\mathsf{T}"),
        TensorExpr::Diff { num, den, num_label } => {
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
        // Σ_{a=1}^{dim} λ_a N_a ⊗ N_a — eigenvalue symbol derived from the
        // decomposed tensor's (lowercased) letter, e.g. C → c_a.
        TensorExpr::Spectral { base, base_latex } => {
            let letter = base_latex
                .trim()
                .rsplit(' ')
                .next()
                .unwrap_or("t")
                .to_lowercase();
            format!(
                "\\sum_{{a=1}}^{{{}}} {}_a \\, \\bm N_a \\otimes \\bm N_a",
                base.dim(),
                letter
            )
        }
        TensorExpr::Add(a, b) => format!("{} + {}", render_tensor(a), render_tensor(b)),
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
                scalar_to_latex(s)
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
