//! The interpreter: evaluates the syntactic AST into semantic values,
//! maintains the environment, performs type checking, and executes
//! `display` / `export` commands.

use crate::ast::{BinOp, Expr, Stmt, UnOp};
use crate::differentiation::{
    diff_block_components, diff_component_equation, diff_scalar_by_tensor,
};
use crate::error::Error;
use crate::renderer::components::tensor_to_component_matrix;
use crate::renderer::latex::{scalar_to_latex, tensor_to_latex};
use crate::simplifier::{simplify_scalar, simplify_tensor, RuleSet};
use crate::symbolic::ScalarExpr;
use crate::tensor::{TensorExpr, TensorProperties};
use std::collections::HashMap;
use std::rc::Rc;

/// A semantic value: scalars and tensors are distinct types and cannot be
/// mixed except where mathematics allows (scalar * tensor, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(Rc<ScalarExpr>),
    Tensor(Rc<TensorExpr>),
}

/// One line of output produced by `display(...)` or `export(...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// e.g. `display C, mode=symbol` or `export C, format=latex`
    pub header: String,
    /// LaTeX payload.
    pub latex: String,
}

#[derive(Default)]
pub struct Interpreter {
    env: HashMap<String, Value>,
    /// The display name (`C`, `W`, ...) each value was last assigned to,
    /// keyed by source variable name, used in display headers and `\bm C = ...`.
    outputs: Vec<Output>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a whole program; returns the outputs of all display/export calls.
    pub fn run(&mut self, stmts: &[Stmt]) -> Result<Vec<Output>, Error> {
        for stmt in stmts {
            self.exec(stmt)?;
        }
        Ok(std::mem::take(&mut self.outputs))
    }

    /// Look up a variable's evaluated value.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.env.get(name)
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<(), Error> {
        match stmt {
            Stmt::Assign { name, expr } => {
                let value = self.eval(expr)?;
                self.env.insert(name.clone(), value);
                Ok(())
            }
            Stmt::Expr(Expr::Call {
                callee,
                args,
                kwargs,
            }) if callee == "display" || callee == "export" => {
                self.exec_output(callee, args, kwargs)
            }
            Stmt::Expr(expr) => {
                // Evaluate for the side effect of error checking.
                self.eval(expr)?;
                Ok(())
            }
        }
    }

    // ---- evaluation ------------------------------------------------------

    fn eval(&mut self, expr: &Expr) -> Result<Value, Error> {
        match expr {
            Expr::Num(n) => Ok(Value::Scalar(Rc::new(ScalarExpr::Num(*n)))),
            Expr::Ident(name) => self
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| Error::msg(format!("undefined variable `{name}`"))),
            Expr::Str(_) | Expr::Bool(_) => Err(Error::msg(
                "string/bool literals are only valid as arguments to Scalar/Tensor/display/export",
            )),
            Expr::Field { target, name } => {
                let value = self.eval(target)?;
                match (value, name.as_str()) {
                    (Value::Tensor(t), "T") => {
                        Ok(Value::Tensor(Rc::new(TensorExpr::transpose(t)?)))
                    }
                    (Value::Scalar(_), "T") => {
                        Err(Error::msg("`.T` is not defined for scalars"))
                    }
                    (_, other) => Err(Error::msg(format!("unknown property `.{other}`"))),
                }
            }
            Expr::Unary { op: UnOp::Neg, expr } => match self.eval(expr)? {
                Value::Scalar(s) => Ok(Value::Scalar(Rc::new(ScalarExpr::Neg(s)))),
                Value::Tensor(t) => Ok(Value::Tensor(Rc::new(TensorExpr::Neg(t)))),
            },
            Expr::Binary { op, lhs, rhs } => {
                let l = self.eval(lhs)?;
                let r = self.eval(rhs)?;
                self.eval_binary(*op, l, r)
            }
            Expr::Call {
                callee,
                args,
                kwargs,
            } => self.eval_call(callee, args, kwargs),
        }
    }

    fn eval_binary(&self, op: BinOp, l: Value, r: Value) -> Result<Value, Error> {
        use BinOp::*;
        match (op, l, r) {
            // scalar ∘ scalar
            (op, Value::Scalar(a), Value::Scalar(b)) => {
                let node = match op {
                    Add => ScalarExpr::Add(a, b),
                    Sub => ScalarExpr::Sub(a, b),
                    Mul => ScalarExpr::Mul(a, b),
                    Div => ScalarExpr::Div(a, b),
                    Pow => ScalarExpr::Pow(a, b),
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            // tensor ∘ tensor
            (Mul, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            (Add, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::add(a, b)?)))
            }
            (Sub, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::sub(a, b)?)))
            }
            // scalar * tensor (either side)
            (Mul, Value::Scalar(s), Value::Tensor(t))
            | (Mul, Value::Tensor(t), Value::Scalar(s)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::ScalarMul(s, t))))
            }
            // tensor / scalar = (1/s) * tensor
            (Div, Value::Tensor(t), Value::Scalar(s)) => {
                let inv = ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), s);
                Ok(Value::Tensor(Rc::new(TensorExpr::ScalarMul(
                    Rc::new(inv),
                    t,
                ))))
            }
            (op, l, r) => Err(Error::msg(format!(
                "operator `{}` is not defined between {} and {}",
                op_name(op),
                kind(&l),
                kind(&r)
            ))),
        }
    }

    fn eval_call(
        &mut self,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        match callee {
            "Scalar" => self.builtin_scalar(args, kwargs),
            "Tensor" => self.builtin_tensor(args, kwargs),
            "det" | "tr" => {
                let t = self.expect_tensor_arg(callee, args, kwargs)?;
                if t.order() != 2 {
                    return Err(Error::msg(format!(
                        "`{callee}` requires a second-order tensor, got order {}",
                        t.order()
                    )));
                }
                let node = if callee == "det" {
                    ScalarExpr::Det(t)
                } else {
                    ScalarExpr::Tr(t)
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            "log" => {
                if args.len() != 1 || !kwargs.is_empty() {
                    return Err(Error::msg("`log` takes exactly one argument"));
                }
                match self.eval(&args[0])? {
                    Value::Scalar(s) => Ok(Value::Scalar(Rc::new(ScalarExpr::Log(s)))),
                    Value::Tensor(_) => Err(Error::msg(
                        "`log` of a tensor requires spectral decomposition (not in MVP); \
                         argument must be a scalar",
                    )),
                }
            }
            "diff" => self.builtin_diff(args, kwargs),
            "outer" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::outer(a, b)?)))
            }
            // dot(A, B): single contraction — same operation as `A * B` for
            // second-order tensors.
            "dot" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            // ddot(A, B): double contraction A : B, a scalar.
            "ddot" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                if a.order() != 2 || b.order() != 2 {
                    return Err(Error::msg(
                        "`ddot` requires two second-order tensors in this phase",
                    ));
                }
                if a.dim() != b.dim() {
                    return Err(Error::msg(format!(
                        "dimension mismatch in ddot: {} vs {}",
                        a.dim(),
                        b.dim()
                    )));
                }
                Ok(Value::Scalar(Rc::new(ScalarExpr::Ddot(a, b))))
            }
            "spectral" => {
                if args.len() != 1 || !kwargs.is_empty() {
                    return Err(Error::msg("`spectral` takes exactly one argument"));
                }
                let t = match self.eval(&args[0])? {
                    Value::Tensor(t) => t,
                    Value::Scalar(_) => {
                        return Err(Error::msg("`spectral` requires a tensor argument"))
                    }
                };
                // Eigenvalue symbol comes from the subject variable name when
                // available (spectral(C) → c_a), else from the tensor's latex.
                let label = match &args[0] {
                    Expr::Ident(name) => format!("\\bm {name}"),
                    _ => tensor_to_latex(&t),
                };
                Ok(Value::Tensor(Rc::new(TensorExpr::spectral(t, label)?)))
            }
            "inv" => {
                let t = self.expect_tensor_arg(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::inverse(t)?)))
            }
            "simplify" => {
                if args.len() != 1 {
                    return Err(Error::msg("`simplify` expects one expression argument"));
                }
                let mut rules = RuleSet::Continuum;
                for (key, value) in kwargs {
                    match (key.as_str(), value) {
                        ("rules", Expr::Ident(name)) | ("rules", Expr::Str(name)) => {
                            rules = RuleSet::parse(name)?;
                        }
                        (other, _) => {
                            return Err(Error::msg(format!(
                                "unknown keyword `{other}` for `simplify`"
                            )))
                        }
                    }
                }
                match self.eval(&args[0])? {
                    Value::Scalar(s) => Ok(Value::Scalar(simplify_scalar(&s, rules))),
                    Value::Tensor(t) => Ok(Value::Tensor(simplify_tensor(&t, rules))),
                }
            }
            "display" | "export" => Err(Error::msg(format!(
                "`{callee}` is a statement and cannot be used inside an expression"
            ))),
            other => Err(Error::msg(format!("unknown function `{other}`"))),
        }
    }

    fn expect_tensor_arg(
        &mut self,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Rc<TensorExpr>, Error> {
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg(format!("`{callee}` takes exactly one argument")));
        }
        match self.eval(&args[0])? {
            Value::Tensor(t) => Ok(t),
            Value::Scalar(_) => Err(Error::msg(format!(
                "`{callee}` requires a tensor argument, got a scalar"
            ))),
        }
    }

    fn expect_two_tensors(
        &mut self,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(Rc<TensorExpr>, Rc<TensorExpr>), Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(format!(
                "`{callee}` takes exactly two tensor arguments"
            )));
        }
        let a = match self.eval(&args[0])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => {
                return Err(Error::msg(format!(
                    "`{callee}` requires tensor arguments, got a scalar"
                )))
            }
        };
        let b = match self.eval(&args[1])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => {
                return Err(Error::msg(format!(
                    "`{callee}` requires tensor arguments, got a scalar"
                )))
            }
        };
        Ok((a, b))
    }

    // ---- builtins: declarations ------------------------------------------

    /// `diff(expr, X)` — symbolic derivative with respect to a declared
    /// second-order tensor variable.
    ///
    /// - scalar / tensor  → evaluated immediately to a tensor expression;
    /// - tensor / tensor  → an opaque order-4 `Diff` node (component
    ///   formula rendered on demand via the index engine).
    fn builtin_diff(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg("`diff` takes exactly two arguments: diff(expr, X)"));
        }
        let num = self.eval(&args[0])?;
        let den = match self.eval(&args[1])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => {
                return Err(Error::msg(
                    "diff with respect to a scalar is not supported yet; \
                     the denominator must be a tensor variable",
                ))
            }
        };
        if !matches!(&*den, TensorExpr::Var { .. }) {
            return Err(Error::msg(
                "diff is only supported with respect to a declared tensor variable \
                 (e.g. F), not a compound expression",
            ));
        }
        if den.order() != 2 {
            return Err(Error::msg(format!(
                "diff denominator must be a second-order tensor, got order {}",
                den.order()
            )));
        }
        match num {
            Value::Scalar(s) => Ok(Value::Tensor(diff_scalar_by_tensor(&s, &den)?)),
            Value::Tensor(t) => {
                if t.order() != 2 {
                    return Err(Error::msg(format!(
                        "tensor-by-tensor diff currently requires a second-order \
                         numerator, got order {}",
                        t.order()
                    )));
                }
                // Label the numerator with its source variable name (if any)
                // so the symbol display reads ∂C/∂F rather than ∂(FᵀF)/∂F.
                let num_label = match &args[0] {
                    Expr::Ident(name) => self.display_lhs(name),
                    _ => None,
                };
                Ok(Value::Tensor(Rc::new(TensorExpr::Diff {
                    num: t,
                    den,
                    num_label,
                })))
            }
        }
    }

    fn builtin_scalar(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Scalar` takes exactly one argument: Scalar(\"<latex>\")",
            ));
        }
        let latex = match &args[0] {
            Expr::Str(s) => s.clone(),
            _ => return Err(Error::msg("`Scalar` expects a string LaTeX name")),
        };
        Ok(Value::Scalar(Rc::new(ScalarExpr::Sym {
            name: latex.clone(),
            latex,
        })))
    }

    fn builtin_tensor(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 1 {
            return Err(Error::msg(
                "`Tensor` expects a string LaTeX name as its first argument",
            ));
        }
        let latex = match &args[0] {
            Expr::Str(s) => s.clone(),
            _ => return Err(Error::msg("`Tensor` expects a string LaTeX name")),
        };
        let mut order: Option<usize> = None;
        let mut dim: Option<usize> = None;
        let mut props = TensorProperties::default();
        for (key, value) in kwargs {
            match key.as_str() {
                "order" => order = Some(expect_usize(value, "order")?),
                "dim" => dim = Some(expect_usize(value, "dim")?),
                "identity" => props.identity = expect_bool(value, "identity")?,
                "symmetric" => props.symmetric = expect_bool(value, "symmetric")?,
                "antisymmetric" => props.antisymmetric = expect_bool(value, "antisymmetric")?,
                "orthogonal" => props.orthogonal = expect_bool(value, "orthogonal")?,
                "isotropic" => props.isotropic = expect_bool(value, "isotropic")?,
                other => {
                    return Err(Error::msg(format!("unknown Tensor property `{other}`")))
                }
            }
        }
        let order = order.ok_or_else(|| Error::msg("`Tensor` requires `order=<n>`"))?;
        let dim = dim.ok_or_else(|| Error::msg("`Tensor` requires `dim=<n>`"))?;
        if order == 0 {
            return Err(Error::msg(
                "order-0 tensors should be declared with `Scalar(...)`",
            ));
        }
        if props.symmetric && props.antisymmetric {
            return Err(Error::msg(
                "a tensor cannot be both symmetric and antisymmetric (unless zero)",
            ));
        }
        Ok(Value::Tensor(Rc::new(TensorExpr::Var {
            name: latex.clone(),
            latex,
            order,
            dim,
            props,
        })))
    }

    // ---- display / export -------------------------------------------------

    fn exec_output(
        &mut self,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(), Error> {
        if args.len() != 1 {
            return Err(Error::msg(format!(
                "`{callee}` expects exactly one expression argument"
            )));
        }
        // The argument's source name (if a bare identifier) labels the output.
        let subject = match &args[0] {
            Expr::Ident(name) => Some(name.clone()),
            _ => None,
        };
        let value = self.eval(&args[0])?;

        let mut mode = "symbol".to_string();
        let mut format = "latex".to_string();
        for (key, raw) in kwargs {
            // display(C, mode=symbol): mode values arrive as bare identifiers.
            let val = match raw {
                Expr::Ident(s) | Expr::Str(s) => s.clone(),
                other => {
                    return Err(Error::msg(format!(
                        "invalid value for `{key}`: {other:?}"
                    )))
                }
            };
            match key.as_str() {
                "mode" if callee == "display" => mode = val,
                "format" if callee == "export" => format = val,
                other => {
                    return Err(Error::msg(format!(
                        "unknown keyword `{other}` for `{callee}`"
                    )))
                }
            }
        }

        let latex = match callee {
            "display" => self.render_display(&value, &subject, &mode)?,
            "export" => self.render_export(&value, &format)?,
            _ => unreachable!(),
        };
        let label = subject.as_deref().unwrap_or("<expr>");
        let detail = if callee == "display" {
            format!("mode={mode}")
        } else {
            format!("format={format}")
        };
        self.outputs.push(Output {
            header: format!("{callee} {label}, {detail}"),
            latex,
        });
        Ok(())
    }

    fn render_display(
        &self,
        value: &Value,
        subject: &Option<String>,
        mode: &str,
    ) -> Result<String, Error> {
        let lhs = subject
            .as_ref()
            .and_then(|name| self.display_lhs(name))
            .map(|tex| format!("{tex} = "))
            .unwrap_or_default();
        match (value, mode) {
            (Value::Scalar(s), "symbol") => Ok(format!("{lhs}{}", scalar_to_latex(s))),
            (Value::Tensor(t), "symbol") => Ok(format!("{lhs}{}", tensor_to_latex(t))),
            // Derivative components use the abstract-index engine and carry
            // their own ∂C_ij/∂F_mn left-hand side.
            (Value::Tensor(t), "components" | "matrix" | "block_components")
                if matches!(&**t, TensorExpr::Diff { .. }) =>
            {
                let TensorExpr::Diff { num, den, num_label } = &**t else {
                    unreachable!()
                };
                let label = num_label.clone().unwrap_or_else(|| tensor_to_latex(num));
                if mode == "block_components" {
                    diff_block_components(num, &label, den)
                } else {
                    diff_component_equation(num, &label, den)
                }
            }
            (Value::Tensor(t), "components" | "matrix") => {
                Ok(format!("{lhs}{}", tensor_to_component_matrix(t)?))
            }
            (Value::Tensor(_), "block_components") => Err(Error::msg(
                "block_components is only available for fourth-order derivative \
                 variables (e.g. A = diff(P, F))",
            )),
            (Value::Scalar(_), "components" | "matrix") => Err(Error::msg(
                "component display is only defined for tensors",
            )),
            (_, other) => Err(Error::msg(format!(
                "unknown display mode `{other}` (supported: symbol, components, \
                 matrix, block_components)"
            ))),
        }
    }

    fn render_export(&self, value: &Value, format: &str) -> Result<String, Error> {
        let latex = match value {
            Value::Scalar(s) => scalar_to_latex(s),
            Value::Tensor(t) => tensor_to_latex(t),
        };
        match format {
            "latex" => Ok(latex),
            // Markdown: LaTeX in a $$ display-math block.
            "markdown" => Ok(format!("$$\n{latex}\n$$")),
            other => Err(Error::msg(format!(
                "unknown export format `{other}` (supported: latex, markdown)"
            ))),
        }
    }

    /// LaTeX for the left-hand side of `display(X, ...)`: a bold symbol made
    /// from the variable name for tensors (`C` -> `\bm C`), the plain name
    /// otherwise. Multi-character tensor names (e.g. `dCdF`) stay plain —
    /// `\bm dCdF` would bold only the `d`.
    fn display_lhs(&self, name: &str) -> Option<String> {
        match self.env.get(name)? {
            Value::Tensor(_) if name.chars().count() == 1 => Some(format!("\\bm {name}")),
            _ => Some(name.to_string()),
        }
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Scalar(_) => "Scalar",
        Value::Tensor(_) => "Tensor",
    }
}

fn op_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => "^",
    }
}

fn expect_usize(expr: &Expr, what: &str) -> Result<usize, Error> {
    match expr {
        Expr::Num(n) if n.fract() == 0.0 && *n >= 0.0 => Ok(*n as usize),
        _ => Err(Error::msg(format!("`{what}` must be a non-negative integer"))),
    }
}

fn expect_bool(expr: &Expr, what: &str) -> Result<bool, Error> {
    match expr {
        Expr::Bool(b) => Ok(*b),
        _ => Err(Error::msg(format!("`{what}` must be true or false"))),
    }
}
