//! The interpreter: evaluates the syntactic AST into semantic values,
//! maintains the environment, performs type checking, and executes
//! `display` / `export` commands.

use crate::ast::{BinOp, Expr, Stmt, UnOp};
use crate::differentiation::{
    diff_block_components, diff_component_equation, diff_scalar_by_scalar, diff_scalar_by_tensor,
    diff_tensor_by_tensor,
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
    /// 1-based source line of the display/export statement.
    pub line: usize,
    /// `Some(message)` if this statement failed (per-block error recovery).
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Interpreter {
    env: HashMap<String, Value>,
    /// LaTeX display labels declared via `Scalar("...")` / `Tensor("...")`,
    /// keyed by variable name. Labels survive reassignment, so
    /// `I1 = Scalar("I_1")` followed by `I1 = tr(C)` still displays as
    /// `I_1 = ...`.
    labels: HashMap<String, String>,
    /// Definitions for display-time back-substitution, in insertion order:
    /// `C = F.T * F` lets later displays show `\bm C` instead of `FᵀF`.
    defs: Vec<crate::substitute::Def>,
    /// Symbol names declared as function arguments via `Var("...")`, in
    /// declaration order. A scalar expression mentioning exactly one of
    /// these is a function of it and can be applied with call syntax.
    fn_vars: Vec<String>,
    outputs: Vec<Output>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run a whole program, stopping at the first error (CLI behavior).
    pub fn run(&mut self, stmts: &[Stmt]) -> Result<Vec<Output>, Error> {
        for stmt in stmts {
            self.exec(stmt)
                .map_err(|e| Error::new(e.message, e.line.or(Some(stmt.line()))))?;
        }
        Ok(std::mem::take(&mut self.outputs))
    }

    /// Run a whole program with per-statement error recovery (UI behavior):
    /// a failing statement produces an error Output (tagged with its line)
    /// and execution continues with the next statement.
    pub fn run_lenient(&mut self, stmts: &[Stmt]) -> Vec<Output> {
        for stmt in stmts {
            if let Err(e) = self.exec(stmt) {
                self.outputs.push(Output {
                    header: format!("line {}", e.line.unwrap_or(stmt.line())),
                    latex: String::new(),
                    line: e.line.unwrap_or(stmt.line()),
                    error: Some(e.message),
                });
            }
        }
        std::mem::take(&mut self.outputs)
    }

    /// Look up a variable's evaluated value.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.env.get(name)
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<(), Error> {
        match stmt {
            Stmt::Assign { name, expr, .. } => {
                let value = self.eval(expr)?;
                // A direct Scalar("...")/Tensor("...") declaration also
                // registers the display label for this variable name.
                if let Expr::Call { callee, args, .. } = expr {
                    if (callee == "Scalar" || callee == "Tensor" || callee == "Var")
                        && !args.is_empty()
                    {
                        if let Expr::Str(latex) = &args[0] {
                            self.labels.insert(name.clone(), latex.clone());
                        }
                    }
                }
                // Register compound definitions for display-time
                // back-substitution (declared leaves substitute trivially
                // and are skipped).
                let is_leaf = matches!(
                    &value,
                    Value::Tensor(t) if matches!(&**t, TensorExpr::Var { .. })
                ) || matches!(
                    &value,
                    Value::Scalar(s) if matches!(&**s, ScalarExpr::Sym { .. } | ScalarExpr::Num(_))
                );
                if !is_leaf {
                    if let Some(latex) = self.display_label(name, &value) {
                        // Re-registering a name drops its previous definition.
                        self.defs.retain(|d| d.latex != latex);
                        self.defs.push(crate::substitute::Def {
                            latex,
                            value: value.clone(),
                        });
                    }
                }
                self.env.insert(name.clone(), value);
                Ok(())
            }
            Stmt::Expr(
                Expr::Call {
                    callee,
                    args,
                    kwargs,
                },
                line,
            ) if callee == "display" || callee == "export" => {
                self.exec_output(callee, args, kwargs, *line)
            }
            Stmt::Expr(expr, _) => {
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
                    (Value::Scalar(_), "T") => Err(Error::msg("`.T` is not defined for scalars")),
                    (_, other) => Err(Error::msg(format!("unknown property `.{other}`"))),
                }
            }
            Expr::Unary {
                op: UnOp::Neg,
                expr,
            } => match self.eval(expr)? {
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
            // A : B — double contraction (2:2 scalar, 2:4 / 4:2 order 2)
            (Ddot, Value::Tensor(a), Value::Tensor(b)) => match (a.order(), b.order()) {
                (2, 2) => {
                    if a.dim() != b.dim() {
                        return Err(Error::msg(format!(
                            "dimension mismatch in `:`: {} vs {}",
                            a.dim(),
                            b.dim()
                        )));
                    }
                    Ok(Value::Scalar(Rc::new(ScalarExpr::Ddot(a, b))))
                }
                (2, 4) => Ok(simplified_tensor_value(Rc::new(TensorExpr::ddot_tq(a, b)?))),
                (4, 2) => Ok(simplified_tensor_value(Rc::new(TensorExpr::ddot_tq(b, a)?))),
                (oa, ob) => Err(Error::msg(format!(
                    "`:` supports 2:2, 2:4 and 4:2 contractions, got orders \
                         {oa} and {ob}"
                ))),
            },
            (Ddot, l, r) => Err(Error::msg(format!(
                "`:` is not defined between {} and {}",
                kind(&l),
                kind(&r)
            ))),
            // scalar ∘ scalar
            (op, Value::Scalar(a), Value::Scalar(b)) => {
                let node = match op {
                    Add => ScalarExpr::Add(a, b),
                    Sub => ScalarExpr::Sub(a, b),
                    Mul => ScalarExpr::Mul(a, b),
                    Div => ScalarExpr::Div(a, b),
                    Pow => ScalarExpr::Pow(a, b),
                    Ddot => unreachable!("handled above"),
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            // tensor ∘ tensor
            (Mul, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(simplified_tensor_value(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            (Add, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(simplified_tensor_value(Rc::new(TensorExpr::add(a, b)?)))
            }
            (Sub, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(simplified_tensor_value(Rc::new(TensorExpr::sub(a, b)?)))
            }
            // scalar * tensor (either side), with coefficient folding so
            // 2 * (½ T : Q) reads T : Q.
            (Mul, Value::Scalar(s), Value::Tensor(t))
            | (Mul, Value::Tensor(t), Value::Scalar(s)) => {
                let out = if let TensorExpr::ScalarMul(s2, inner) = &*t {
                    let merged = crate::symbolic::fold_mul(&s, s2);
                    if matches!(&*merged, ScalarExpr::Num(x) if *x == 1.0) {
                        inner.clone()
                    } else {
                        Rc::new(TensorExpr::ScalarMul(merged, inner.clone()))
                    }
                } else {
                    Rc::new(TensorExpr::ScalarMul(s, t))
                };
                Ok(simplified_tensor_value(out))
            }
            // tensor / scalar = (1/s) * tensor
            (Div, Value::Tensor(t), Value::Scalar(s)) => {
                let inv = ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), s);
                Ok(simplified_tensor_value(Rc::new(TensorExpr::ScalarMul(
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
            "Var" => self.builtin_var(args, kwargs),
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
            "log" | "sqrt" | "exp" | "sinh" | "cosh" | "tanh" => {
                if args.len() != 1 || !kwargs.is_empty() {
                    return Err(Error::msg(format!("`{callee}` takes exactly one argument")));
                }
                match self.eval(&args[0])? {
                    Value::Scalar(s) => match callee {
                        "log" => Ok(Value::Scalar(Rc::new(ScalarExpr::Log(s)))),
                        other => Ok(Value::Scalar(Rc::new(ScalarExpr::Func {
                            name: other.to_string(),
                            arg: s,
                        }))),
                    },
                    Value::Tensor(t) => {
                        // Isotropic tensor function through the spectral form.
                        let label = match &args[0] {
                            Expr::Ident(name) => format!("\\bm {name}"),
                            _ => tensor_to_latex(&t),
                        };
                        Ok(Value::Tensor(Rc::new(TensorExpr::spectral_fn(
                            callee, t, label,
                        )?)))
                    }
                }
            }
            // gstrain(C, scale=CR, m=..., n=...): generalized Lagrangian
            // strain E(C) = Σ E(λ_a) M_a (Hill's family).
            "gstrain" => {
                if args.len() != 1 {
                    return Err(Error::msg(
                        "`gstrain` expects one tensor argument: gstrain(C, scale=..., ...)",
                    ));
                }
                let base = match self.eval(&args[0])? {
                    Value::Tensor(t) => t,
                    Value::Scalar(_) => {
                        return Err(Error::msg("`gstrain` requires a tensor argument"))
                    }
                };
                let mut kind: Option<String> = None;
                let mut m: Option<Rc<ScalarExpr>> = None;
                let mut n: Option<Rc<ScalarExpr>> = None;
                for (key, raw) in kwargs {
                    match key.as_str() {
                        "scale" => match raw {
                            Expr::Ident(s) | Expr::Str(s) => kind = Some(s.clone()),
                            other => {
                                return Err(Error::msg(format!(
                                    "invalid scale: {other:?} (expected CR, SethHill, or Hencky)"
                                )))
                            }
                        },
                        "m" => match self.eval(raw)? {
                            Value::Scalar(s) => m = Some(s),
                            _ => return Err(Error::msg("`m` must be a scalar")),
                        },
                        "n" => match self.eval(raw)? {
                            Value::Scalar(s) => n = Some(s),
                            _ => return Err(Error::msg("`n` must be a scalar")),
                        },
                        other => {
                            return Err(Error::msg(format!("unknown gstrain keyword `{other}`")))
                        }
                    }
                }
                let scale = match kind.as_deref() {
                    Some("CR") => crate::tensor::Scale::CR {
                        m: m.ok_or_else(|| Error::msg("CR strain requires m=..."))?,
                        n: n.ok_or_else(|| Error::msg("CR strain requires n=..."))?,
                    },
                    Some("SethHill") => crate::tensor::Scale::SethHill {
                        m: m.ok_or_else(|| Error::msg("Seth–Hill strain requires m=..."))?,
                    },
                    Some("Hencky") => crate::tensor::Scale::Hencky,
                    // A user-defined function of one `Var` argument.
                    Some(other) => match self.env.get(other).cloned() {
                        Some(Value::Scalar(body)) => {
                            let vars = self.free_fn_vars(&body);
                            let var = match vars.as_slice() {
                                [v] => v.clone(),
                                [] => {
                                    return Err(Error::msg(format!(
                                        "scale `{other}` is not a function: it mentions no \
                                         `Var(...)` argument"
                                    )))
                                }
                                _ => {
                                    return Err(Error::msg(format!(
                                        "scale `{other}` mentions several `Var(...)` arguments \
                                         ({}); a scale function has one variable",
                                        vars.join(", ")
                                    )))
                                }
                            };
                            let dbody = simplify_scalar(
                                &diff_scalar_by_scalar(&body, &var)?,
                                RuleSet::Continuum,
                            );
                            crate::tensor::Scale::Custom { var, body, dbody }
                        }
                        _ => {
                            return Err(Error::msg(format!(
                                "unknown scale `{other}` (supported: CR, SethHill, Hencky, \
                                 or a function declared via Var)"
                            )))
                        }
                    },
                    None => {
                        return Err(Error::msg(
                            "`gstrain` requires scale=CR | SethHill | Hencky or a \
                             user function of a Var argument",
                        ))
                    }
                };
                Ok(Value::Tensor(Rc::new(TensorExpr::gen_strain(
                    base,
                    scale,
                    "\\bm E".to_string(),
                )?)))
            }
            "diff" => self.builtin_diff(args, kwargs),
            "outer" | "otimes" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::outer(a, b)?)))
            }
            // dot(A, B): single contraction — same operation as `A * B` for
            // second-order tensors.
            "dot" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            // ddot(A, B): double contraction. 2:2 → scalar; 2:4 / 4:2 → order 2.
            "ddot" => {
                let (a, b) = self.expect_two_tensors(callee, args, kwargs)?;
                match (a.order(), b.order()) {
                    (2, 2) => {
                        if a.dim() != b.dim() {
                            return Err(Error::msg(format!(
                                "dimension mismatch in ddot: {} vs {}",
                                a.dim(),
                                b.dim()
                            )));
                        }
                        Ok(Value::Scalar(Rc::new(ScalarExpr::Ddot(a, b))))
                    }
                    (2, 4) => Ok(simplified_tensor_value(Rc::new(TensorExpr::ddot_tq(a, b)?))),
                    (4, 2) => Ok(simplified_tensor_value(Rc::new(TensorExpr::ddot_tq(b, a)?))),
                    (oa, ob) => Err(Error::msg(format!(
                        "`ddot` supports 2:2, 2:4 and 4:2 contractions, got orders \
                         {oa} and {ob}"
                    ))),
                }
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
            // A user-defined name bound to a scalar expression in a declared
            // `Var` argument is a function: `E(x)` substitutes the argument.
            other => match self.env.get(other).cloned() {
                Some(Value::Scalar(body)) => self.apply_scalar_function(other, &body, args, kwargs),
                Some(Value::Tensor(_)) => {
                    Err(Error::msg(format!("`{other}` is a tensor, not a function")))
                }
                None => Err(Error::msg(format!("unknown function `{other}`"))),
            },
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

    /// `diff(expr, X)` — symbolic derivative.
    ///
    /// Denominator forms:
    /// - declared scalar symbol (`diff(W, mu)`) → scalar derivative;
    /// - tensor variable or compound tensor expression (`diff(W, F)`,
    ///   `diff(W, C)` with `C = F.T * F`) → tensor-valued derivative; a
    ///   compound denominator is treated as the independent variable and
    ///   matched structurally, with a strict independence check rejecting
    ///   hidden dependence (e.g. `det(F)` inside `diff(…, C)`).
    /// - tensor / tensor → opaque order-4 `Diff` node.
    fn builtin_diff(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`diff` takes exactly two arguments: diff(expr, X)",
            ));
        }
        let num = self.eval(&args[0])?;
        let den = self.eval(&args[1])?;

        // --- scalar denominator: d/d mu -----------------------------------
        let den = match den {
            Value::Scalar(ds) => {
                let ScalarExpr::Sym { name, .. } = &*ds else {
                    return Err(Error::msg(
                        "diff with respect to a scalar requires a declared scalar \
                         symbol (e.g. mu), not a compound expression",
                    ));
                };
                return match num {
                    Value::Scalar(s) => Ok(Value::Scalar(simplify_scalar(
                        &diff_scalar_by_scalar(&s, name)?,
                        RuleSet::Continuum,
                    ))),
                    Value::Tensor(_) => Err(Error::msg(
                        "differentiating a tensor with respect to a scalar is not \
                         supported yet",
                    )),
                };
            }
            Value::Tensor(t) => t,
        };

        if den.order() != 2 {
            return Err(Error::msg(format!(
                "diff denominator must be a second-order tensor, got order {}",
                den.order()
            )));
        }
        match num {
            Value::Scalar(s) => Ok(Value::Tensor(simplify_tensor(
                &diff_scalar_by_tensor(&s, &den)?,
                RuleSet::Continuum,
            ))),
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
                let den_label = match &args[1] {
                    Expr::Ident(name) => self.display_lhs(name),
                    _ => None,
                };
                Ok(Value::Tensor(simplify_tensor(
                    &diff_tensor_by_tensor(&t, &den, num_label, den_label)?,
                    RuleSet::Continuum,
                )))
            }
        }
    }

    /// `Var("\lambda")` — declare a scalar *function argument*. Any scalar
    /// expression mentioning it becomes a function of it (applicable with
    /// call syntax, usable as a `gstrain` scale).
    fn builtin_var(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Var` takes exactly one argument: Var(\"<latex>\")",
            ));
        }
        let latex = match &args[0] {
            Expr::Str(s) => s.clone(),
            _ => return Err(Error::msg("`Var` expects a string LaTeX name")),
        };
        if !self.fn_vars.contains(&latex) {
            self.fn_vars.push(latex.clone());
        }
        Ok(Value::Scalar(Rc::new(ScalarExpr::Sym {
            name: latex.clone(),
            latex,
        })))
    }

    /// The declared `Var` arguments that occur free in `body`.
    fn free_fn_vars(&self, body: &ScalarExpr) -> Vec<String> {
        self.fn_vars
            .iter()
            .filter(|v| crate::differentiation::scalar_mentions_scalar(body, v))
            .cloned()
            .collect()
    }

    /// Apply a user scalar function: substitute its single `Var` argument.
    fn apply_scalar_function(
        &mut self,
        name: &str,
        body: &Rc<ScalarExpr>,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg(format!(
                "`{name}` is a function of one variable and takes exactly one argument"
            )));
        }
        let vars = self.free_fn_vars(body);
        let var = match vars.as_slice() {
            [v] => v.clone(),
            [] => {
                return Err(Error::msg(format!(
                    "`{name}` is not a function: it mentions no `Var(...)` argument"
                )))
            }
            _ => {
                return Err(Error::msg(format!(
                    "`{name}` mentions several `Var(...)` arguments ({}); functions of \
                     one variable only are supported",
                    vars.join(", ")
                )))
            }
        };
        match self.eval(&args[0])? {
            Value::Scalar(x) => Ok(Value::Scalar(crate::substitute::subst_scalar_sym(
                body, &var, &x,
            ))),
            Value::Tensor(_) => Err(Error::msg(format!(
                "`{name}` is a scalar function; applying it to a tensor is not \
                 supported yet (use gstrain for spectral application)"
            ))),
        }
    }

    fn builtin_scalar(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
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

    fn builtin_tensor(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
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
                other => return Err(Error::msg(format!("unknown Tensor property `{other}`"))),
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
        line: usize,
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
                other => return Err(Error::msg(format!("invalid value for `{key}`: {other:?}"))),
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

        // Back-substitute registered definitions (most recent first) so the
        // display shows \bm C rather than the expanded FᵀF. Internal values
        // stay expanded; this is presentation only. Derivative nodes and
        // spectral mode are exempt: they need the expanded structure.
        let skip_subst = mode == "spectral"
            || matches!(&value, Value::Tensor(t) if matches!(&**t, TensorExpr::Diff { .. }));
        let value = if skip_subst {
            value
        } else {
            crate::substitute::substitute(&value, &self.defs)
        };

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
            line,
            error: None,
        });
        Ok(())
    }

    /// Display label for a variable about to be (re)assigned `value`:
    /// declared label first, then synthesized \bm for single-char tensor
    /// names, then the plain name.
    fn display_label(&self, name: &str, value: &Value) -> Option<String> {
        if let Some(label) = self.labels.get(name) {
            return Some(label.clone());
        }
        match value {
            Value::Tensor(t) if name.chars().count() == 1 => Some(default_tensor_label(name, t)),
            _ => Some(name.to_string()),
        }
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
            // Principal-axis form: E, T(E), and S = T : Q expand to Σ_a ... M_a.
            (Value::Tensor(t), "spectral") => Ok(format!(
                "{lhs}{}",
                crate::renderer::spectral::tensor_to_spectral(t)?
            )),
            (Value::Scalar(_), "spectral") => {
                Err(Error::msg("spectral display is only defined for tensors"))
            }
            // Derivative components use the abstract-index engine and carry
            // their own ∂C_ij/∂F_mn left-hand side.
            (Value::Tensor(t), "components" | "matrix" | "block_components")
                if matches!(&**t, TensorExpr::Diff { .. }) =>
            {
                let TensorExpr::Diff {
                    num,
                    den,
                    num_label,
                } = &**t
                else {
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
            (Value::Scalar(_), "components" | "matrix") => {
                Err(Error::msg("component display is only defined for tensors"))
            }
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

    /// LaTeX for the left-hand side of `display(X, ...)`, in order of
    /// preference:
    /// 1. the label declared via `Scalar("...")`/`Tensor("...")` for this
    ///    name (kept across reassignment);
    /// 2. a tensor symbol synthesized from a single-character tensor name
    ///    (`C` -> `\bm C`, fourth-order `A` -> `\mathbb A`);
    /// 3. the plain variable name (multi-character names stay plain —
    ///    `\bm dCdF` would bold only the `d`).
    fn display_lhs(&self, name: &str) -> Option<String> {
        let value = self.env.get(name)?;
        if let Some(label) = self.labels.get(name) {
            return Some(label.clone());
        }
        match value {
            Value::Tensor(t) if name.chars().count() == 1 => Some(default_tensor_label(name, t)),
            _ => Some(name.to_string()),
        }
    }
}

fn default_tensor_label(name: &str, tensor: &Rc<TensorExpr>) -> String {
    if tensor.order() == 4 {
        format!("\\mathbb {name}")
    } else {
        format!("\\bm {name}")
    }
}

fn simplified_tensor_value(t: Rc<TensorExpr>) -> Value {
    Value::Tensor(simplify_tensor(&t, RuleSet::Continuum))
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
        BinOp::Ddot => ":",
    }
}

fn expect_usize(expr: &Expr, what: &str) -> Result<usize, Error> {
    match expr {
        Expr::Num(n) if n.fract() == 0.0 && *n >= 0.0 => Ok(*n as usize),
        _ => Err(Error::msg(format!(
            "`{what}` must be a non-negative integer"
        ))),
    }
}

fn expect_bool(expr: &Expr, what: &str) -> Result<bool, Error> {
    match expr {
        Expr::Bool(b) => Ok(*b),
        _ => Err(Error::msg(format!("`{what}` must be true or false"))),
    }
}
