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
    /// Direct symbol-display values before continuum simplification. This
    /// preserves user-facing definitions such as `C = F.T * F` even when the
    /// computational value simplifies `F.T` to `F` for a component-filled
    /// symmetric `F`.
    display_values: HashMap<String, Value>,
    /// Symbol names declared as function arguments via `Var("...")`, in
    /// declaration order. A scalar expression mentioning exactly one of
    /// these is a function of it and can be applied with call syntax.
    fn_vars: Vec<String>,
    /// Indexed families declared via `ScalarSet("...", dim=n)` /
    /// `VectorSet("...", dim=n)`, keyed by variable name. Elements are
    /// accessed as `name[a]` (abstract index) or `name[1]` (concrete).
    sets: HashMap<String, SetDecl>,
    outputs: Vec<Output>,
}

struct SetDecl {
    latex: String,
    /// `true` for VectorSet (order-1 elements), `false` for ScalarSet.
    vector: bool,
    dim: usize,
    /// The decomposed tensor when declared via `eigvals(...)`/`eigvecs(...)`.
    base: Option<Rc<TensorExpr>>,
    /// Concrete element values filled in by `[a, b] = Spec_Decomp(C)`:
    /// scalar expressions for a ScalarSet, order-1 `Filled` vectors for a
    /// VectorSet. `name[1]` resolves to the value; `name[a]` stays abstract.
    values: Option<Vec<Value>>,
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
                // Set declarations bind into the sets registry, not the
                // value environment: only `name[index]` elements are values.
                if let Expr::Call {
                    callee,
                    args,
                    kwargs,
                } = expr
                {
                    if callee == "ScalarSet" || callee == "VectorSet" {
                        return self.declare_set(name, callee, args, kwargs);
                    }
                    if callee == "eigvals" || callee == "eigvecs" {
                        return self.declare_eig_set(name, callee, args, kwargs);
                    }
                }
                let value = self.eval(expr)?;
                let display_value = self.eval_display_value(expr).ok();
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
                if let Some(display_value) =
                    display_value.filter(|v| !is_leaf && value_contains_transpose(v))
                {
                    self.display_values.insert(name.clone(), display_value);
                } else {
                    self.display_values.remove(name);
                }
                self.env.insert(name.clone(), value);
                Ok(())
            }
            Stmt::AssignComponent {
                name,
                indices,
                expr,
                ..
            } => self.exec_assign_component(name, indices, expr),
            Stmt::AssignPair {
                first,
                second,
                expr,
                ..
            } => self.exec_assign_pair(first, second, expr),
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

    /// `F[1][1] = expr` — set one component of a declared tensor, turning
    /// it into (or updating) a component-filled tensor. Unassigned
    /// components are zero.
    fn exec_assign_component(
        &mut self,
        name: &str,
        indices: &[Expr],
        expr: &Expr,
    ) -> Result<(), Error> {
        let rhs = match self.eval(expr)? {
            Value::Scalar(s) => s,
            Value::Tensor(_) => {
                return Err(Error::msg("a tensor component must be a scalar expression"))
            }
        };
        let (latex, order, dim, mut entries) = match self.env.get(name) {
            Some(Value::Tensor(t)) => match &**t {
                TensorExpr::Var {
                    latex, order, dim, ..
                } if (1..=2).contains(order) => (
                    latex.clone(),
                    *order,
                    *dim,
                    (0..dim.pow(*order as u32))
                        .map(|_| Rc::new(ScalarExpr::Num(0.0)))
                        .collect(),
                ),
                TensorExpr::Filled {
                    latex,
                    order,
                    dim,
                    entries,
                } => (latex.clone(), *order, *dim, entries.clone()),
                _ => {
                    return Err(Error::msg(format!(
                        "`{name}` is a derived expression; components can only be \
                         assigned on a declared Tensor"
                    )))
                }
            },
            Some(Value::Scalar(_)) => {
                return Err(Error::msg(format!("`{name}` is a scalar, not a tensor")))
            }
            None => {
                return Err(Error::msg(format!(
                    "undefined tensor `{name}`; declare it first with Tensor(...)"
                )))
            }
        };
        if indices.len() != order {
            return Err(Error::msg(format!(
                "`{name}` has order {order} and takes {order} index group(s), got {}",
                indices.len()
            )));
        }
        let mut flat = 0usize;
        for idx in indices {
            let k = match idx {
                Expr::Num(n) if n.fract() == 0.0 && *n >= 1.0 && (*n as usize) <= dim => {
                    *n as usize
                }
                _ => {
                    return Err(Error::msg(format!(
                        "component indices must be integers in 1..={dim}"
                    )))
                }
            };
            flat = flat * dim + (k - 1);
        }
        entries[flat] = rhs;
        self.env.insert(
            name.to_string(),
            Value::Tensor(Rc::new(TensorExpr::Filled {
                latex,
                order,
                dim,
                entries,
            })),
        );
        Ok(())
    }

    /// `[a, b] = Spec_Decomp(C)` — symbolic eigendecomposition of a
    /// component-filled (diagonal) tensor into two pre-declared sets.
    fn exec_assign_pair(&mut self, first: &str, second: &str, expr: &Expr) -> Result<(), Error> {
        let Expr::Call {
            callee,
            args,
            kwargs,
        } = expr
        else {
            return Err(Error::msg(
                "`[a, b] = ...` expects Spec_Decomp(C) on the right-hand side",
            ));
        };
        if callee != "Spec_Decomp" {
            return Err(Error::msg(format!(
                "`[a, b] = ...` expects Spec_Decomp(C), got `{callee}`"
            )));
        }
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg("`Spec_Decomp` takes exactly one tensor"));
        }
        let c = match self.eval(&args[0])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => return Err(Error::msg("`Spec_Decomp` requires a tensor argument")),
        };
        if c.order() != 2 {
            return Err(Error::msg("`Spec_Decomp` requires a second-order tensor"));
        }
        let dim = c.dim();
        // The targets must be pre-declared sets of matching dimension.
        match self.sets.get(first) {
            Some(d) if !d.vector && d.dim == dim => {}
            Some(d) if d.vector => {
                return Err(Error::msg(format!(
                    "`{first}` must be a ScalarSet (the principal values)"
                )))
            }
            Some(_) => return Err(Error::msg("set dimension does not match the tensor")),
            None => {
                return Err(Error::msg(format!(
                    "declare `{first}` first: {first} = ScalarSet(\"<latex>\", dim={dim})"
                )))
            }
        }
        match self.sets.get(second) {
            Some(d) if d.vector && d.dim == dim => {}
            Some(d) if !d.vector => {
                return Err(Error::msg(format!(
                    "`{second}` must be a VectorSet (the principal directions)"
                )))
            }
            Some(_) => return Err(Error::msg("set dimension does not match the tensor")),
            None => {
                return Err(Error::msg(format!(
                    "declare `{second}` first: {second} = VectorSet(\"<latex>\", dim={dim})"
                )))
            }
        }
        // Compute the entries and check diagonality (in the working basis).
        let mut diag = Vec::with_capacity(dim);
        for i in 0..dim {
            for j in 0..dim {
                let e = simplify_scalar(
                    &Rc::new(crate::renderer::components::component(&c, i, j)?),
                    RuleSet::Continuum,
                );
                if i == j {
                    diag.push(e);
                } else if !matches!(&*e, ScalarExpr::Num(n) if *n == 0.0) {
                    return Err(Error::msg(format!(
                        "symbolic eigendecomposition currently requires a diagonal \
                         tensor in the working basis; C_{{{}{}}} = {}",
                        i + 1,
                        j + 1,
                        scalar_to_latex(&e)
                    )));
                }
            }
        }
        // Eigenvalues: the diagonal entries. Eigenvectors: the standard basis.
        let vec_latex = self.sets[second].latex.clone();
        let basis: Vec<Value> = (0..dim)
            .map(|k| {
                let entries = (0..dim)
                    .map(|i| Rc::new(ScalarExpr::Num(if i == k { 1.0 } else { 0.0 })))
                    .collect();
                Value::Tensor(Rc::new(TensorExpr::Filled {
                    latex: format!("{{{vec_latex}}}_{{{}}}", k + 1),
                    order: 1,
                    dim,
                    entries,
                }))
            })
            .collect();
        self.sets.get_mut(first).unwrap().values =
            Some(diag.into_iter().map(Value::Scalar).collect());
        self.sets.get_mut(second).unwrap().values = Some(basis);
        Ok(())
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
            Expr::Index { target, index } => self.eval_index(target, index),
            Expr::Call {
                callee,
                args,
                kwargs,
            } => self.eval_call(callee, args, kwargs),
        }
    }

    fn eval_display_value(&mut self, expr: &Expr) -> Result<Value, Error> {
        match expr {
            Expr::Field { target, name } => {
                let value = self.eval_display_value(target)?;
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
            } => match self.eval_display_value(expr)? {
                Value::Scalar(s) => Ok(Value::Scalar(Rc::new(ScalarExpr::Neg(s)))),
                Value::Tensor(t) => Ok(Value::Tensor(Rc::new(TensorExpr::Neg(t)))),
            },
            Expr::Binary { op, lhs, rhs } => {
                let l = self.eval_display_value(lhs)?;
                let r = self.eval_display_value(rhs)?;
                self.eval_binary_display(*op, l, r)
            }
            _ => self.eval(expr),
        }
    }

    fn eval_binary_display(&self, op: BinOp, l: Value, r: Value) -> Result<Value, Error> {
        use BinOp::*;
        match (op, l, r) {
            (Mul, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            (Add, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::add(a, b)?)))
            }
            (Sub, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::sub(a, b)?)))
            }
            (Mul, Value::Scalar(s), Value::Tensor(t))
            | (Mul, Value::Tensor(t), Value::Scalar(s)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::ScalarMul(s, t))))
            }
            (Div, Value::Tensor(t), Value::Scalar(s)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::ScalarMul(
                    Rc::new(ScalarExpr::Div(Rc::new(ScalarExpr::Num(1.0)), s)),
                    t,
                ))))
            }
            (Ddot, Value::Tensor(a), Value::Tensor(b)) => match (a.order(), b.order()) {
                (2, 2) => Ok(Value::Scalar(Rc::new(ScalarExpr::Ddot(a, b)))),
                (2, 4) => Ok(Value::Tensor(Rc::new(TensorExpr::ddot_tq(a, b)?))),
                (4, 2) => Ok(Value::Tensor(Rc::new(TensorExpr::ddot_tq(b, a)?))),
                (oa, ob) => Err(Error::msg(format!(
                    "`:` supports 2:2, 2:4 and 4:2 contractions, got orders \
                     {oa} and {ob}"
                ))),
            },
            (Outer, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(Value::Tensor(Rc::new(TensorExpr::outer(a, b)?)))
            }
            (op, Value::Scalar(a), Value::Scalar(b)) => {
                let node = match op {
                    Add => ScalarExpr::Add(a, b),
                    Sub => ScalarExpr::Sub(a, b),
                    Mul => ScalarExpr::Mul(a, b),
                    Div => ScalarExpr::Div(a, b),
                    Pow => ScalarExpr::Pow(a, b),
                    Outer | Ddot => unreachable!("handled above"),
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            (op, l, r) => self.eval_binary(op, l, r),
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
            // A & B — outer product.
            (Outer, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(simplified_tensor_value(Rc::new(TensorExpr::outer(a, b)?)))
            }
            (Outer, l, r) => Err(Error::msg(format!(
                "`&` is not defined between {} and {}",
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
                    Outer | Ddot => unreachable!("handled above"),
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            // tensor ∘ tensor
            (Mul, Value::Tensor(a), Value::Tensor(b)) => {
                Ok(simplified_tensor_value(Rc::new(TensorExpr::matmul(a, b)?)))
            }
            // A^n — integer matrix power.
            (Pow, Value::Tensor(a), Value::Scalar(s)) => {
                let exp = match &*s {
                    ScalarExpr::Num(n) if n.fract() == 0.0 && *n >= 1.0 && *n < 64.0 => *n as u32,
                    _ => {
                        return Err(Error::msg(
                            "tensor `^` requires a small positive integer exponent (A^2, A^3)",
                        ))
                    }
                };
                Ok(simplified_tensor_value(Rc::new(TensorExpr::power(a, exp)?)))
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
            "sum" => self.builtin_sum(args, kwargs),
            "ScalarSet" | "VectorSet" | "eigvals" | "eigvecs" => Err(Error::msg(format!(
                "`{callee}` declares a set and must be used in an assignment, \
                 e.g. lambda = {callee}(...)"
            ))),
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
                    Value::Tensor(_) => Err(Error::msg(format!(
                        "`{callee}` of a tensor is not supported; write the spectral \
                         form explicitly, e.g. sum({callee}(lambda[a]) * N[a] & N[a], a)"
                    ))),
                }
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
    /// call syntax, e.g. `Ecr(lambda[a])` inside a spectral sum).
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

    /// `lambda = ScalarSet("\lambda", dim=3)` / `N = VectorSet("\bm N", dim=3)`.
    fn declare_set(
        &mut self,
        name: &str,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(), Error> {
        let latex = match args {
            [Expr::Str(s)] => s.clone(),
            _ => {
                return Err(Error::msg(format!(
                    "`{callee}` expects a string LaTeX name: {callee}(\"<latex>\", dim=<n>)"
                )))
            }
        };
        let mut dim: Option<usize> = None;
        for (key, value) in kwargs {
            match key.as_str() {
                "dim" => dim = Some(expect_usize(value, "dim")?),
                other => return Err(Error::msg(format!("unknown {callee} keyword `{other}`"))),
            }
        }
        let dim = dim.ok_or_else(|| Error::msg(format!("`{callee}` requires `dim=<n>`")))?;
        self.labels.insert(name.to_string(), latex.clone());
        self.sets.insert(
            name.to_string(),
            SetDecl {
                latex,
                vector: callee == "VectorSet",
                dim,
                base: None,
                values: None,
            },
        );
        Ok(())
    }

    /// `lambda = eigvals(C, "\lambda")` / `N = eigvecs(C, "\bm N")` —
    /// eigenvalue/eigenvector sets bound to a provably symmetric tensor.
    fn declare_eig_set(
        &mut self,
        name: &str,
        callee: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(), Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(format!(
                "`{callee}` expects a tensor and a LaTeX name: {callee}(C, \"<latex>\")"
            )));
        }
        let base = match self.eval(&args[0])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => {
                return Err(Error::msg(format!("`{callee}` requires a tensor argument")))
            }
        };
        if base.order() != 2 || !base.is_symmetric() {
            return Err(Error::msg(format!(
                "`{callee}` requires a provably symmetric second-order tensor \
                 (e.g. C = F.T * F)"
            )));
        }
        let Expr::Str(latex) = &args[1] else {
            return Err(Error::msg(format!(
                "`{callee}` expects a string LaTeX name as its second argument"
            )));
        };
        self.labels.insert(name.to_string(), latex.clone());
        self.sets.insert(
            name.to_string(),
            SetDecl {
                latex: latex.clone(),
                vector: callee == "eigvecs",
                dim: base.dim(),
                base: Some(base),
                values: None,
            },
        );
        Ok(())
    }

    /// The eigenvector-set LaTeX registered for `base`, if any.
    fn eigvec_latex_for(&self, base: &Rc<TensorExpr>) -> Option<String> {
        self.sets
            .values()
            .find(|d| d.vector && d.base.as_ref() == Some(base))
            .map(|d| d.latex.clone())
    }

    /// `lambda[a]` / `N[1]` — a set element.
    fn eval_index(&mut self, target: &Expr, index: &Expr) -> Result<Value, Error> {
        // Flatten an index chain: `P[1][1]` -> base = P, idxs = [1, 1];
        // `lambda[a]` -> base = lambda, idxs = [a].
        let mut idx_exprs = vec![index];
        let mut base = target;
        while let Expr::Index {
            target: inner,
            index: i,
        } = base
        {
            idx_exprs.push(i);
            base = inner;
        }
        idx_exprs.reverse();

        // A declared set takes exactly one index; anything else is a tensor
        // component read `T[i][j]`.
        let is_set = matches!(base, Expr::Ident(name) if self.sets.contains_key(name));
        if !is_set {
            return self.eval_component_read(base, &idx_exprs);
        }
        let Expr::Ident(set_name) = base else {
            unreachable!("is_set guarantees an Ident");
        };
        if idx_exprs.len() != 1 {
            return Err(Error::msg(format!(
                "set `{set_name}` takes a single index, got {}",
                idx_exprs.len()
            )));
        }
        let index = idx_exprs[0];
        let decl = self.sets.get(set_name).ok_or_else(|| {
            Error::msg(format!(
                "`{set_name}` is not a declared set; declare it with \
                 ScalarSet(...)/VectorSet(...)"
            ))
        })?;
        let idx = match index {
            Expr::Ident(i) => crate::symbolic::SetIndex::Sym(i.clone()),
            Expr::Num(n) if n.fract() == 0.0 && *n >= 1.0 && (*n as usize) <= decl.dim => {
                crate::symbolic::SetIndex::Num(*n as usize)
            }
            Expr::Num(n) => {
                return Err(Error::msg(format!(
                    "index {n} is out of range for `{set_name}` (1..={})",
                    decl.dim
                )))
            }
            _ => {
                return Err(Error::msg(
                    "a set index must be an index name or an integer literal",
                ))
            }
        };
        // A concrete index on a valued set resolves to the stored value.
        if let (crate::symbolic::SetIndex::Num(k), Some(values)) = (&idx, &decl.values) {
            return Ok(values[*k - 1].clone());
        }
        if decl.vector {
            Ok(Value::Tensor(Rc::new(TensorExpr::SetElem {
                latex: decl.latex.clone(),
                order: 1,
                dim: decl.dim,
                index: idx,
                set_dim: decl.dim,
                base: decl.base.clone(),
            })))
        } else {
            let eig = decl.base.clone().map(|base| {
                let vec_latex = self.eigvec_latex_for(&base).unwrap_or_else(|| {
                    // No eigenvector set declared (yet): derive a display
                    // name; differentiation will still produce projectors.
                    "\\bm N".to_string()
                });
                crate::symbolic::EigLink { base, vec_latex }
            });
            Ok(Value::Scalar(Rc::new(ScalarExpr::SetElem {
                latex: decl.latex.clone(),
                index: idx,
                set_dim: decl.dim,
                eig,
            })))
        }
    }

    /// `T[i][j]` — read a component of a tensor expression as a scalar.
    fn eval_component_read(&mut self, base: &Expr, idx_exprs: &[&Expr]) -> Result<Value, Error> {
        let t = match self.eval(base)? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => {
                return Err(Error::msg(
                    "component indexing `[i][j]` requires a tensor (or a declared set)",
                ))
            }
        };
        let order = t.order();
        if idx_exprs.len() != order {
            return Err(Error::msg(format!(
                "a tensor of order {order} needs {order} component index(es), got {}",
                idx_exprs.len()
            )));
        }
        if order != 2 {
            return Err(Error::msg(
                "component read currently supports second-order tensors",
            ));
        }
        let dim = t.dim();
        let i = component_index(idx_exprs[0], dim)?;
        let j = component_index(idx_exprs[1], dim)?;
        let entry = crate::renderer::components::component(&t, i - 1, j - 1)?;
        Ok(Value::Scalar(simplify_scalar(
            &Rc::new(entry),
            RuleSet::Continuum,
        )))
    }

    /// `sum(body, a)` — sum the body over the abstract set index `a`.
    fn builtin_sum(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`sum` expects two arguments: sum(<expr>, <index>)",
            ));
        }
        let Expr::Ident(index) = &args[1] else {
            return Err(Error::msg("the second argument of `sum` is an index name"));
        };
        match self.eval(&args[0])? {
            Value::Tensor(t) => {
                let range = tensor_index_range(&t, index).ok_or_else(|| {
                    Error::msg(format!(
                        "the summand does not mention any set element with index `{index}`"
                    ))
                })?;
                Ok(Value::Tensor(Rc::new(TensorExpr::SumIdx {
                    index: index.clone(),
                    range,
                    exclude: None,
                    body: t,
                })))
            }
            Value::Scalar(s) => {
                let range = scalar_index_range(&s, index).ok_or_else(|| {
                    Error::msg(format!(
                        "the summand does not mention any set element with index `{index}`"
                    ))
                })?;
                Ok(Value::Scalar(Rc::new(ScalarExpr::SpecSum {
                    body: s,
                    index: index.clone(),
                    dim: range,
                })))
            }
        }
    }

    /// Look up the concrete value of a set element by its display latex.
    fn set_value(&self, latex: &str, vector: bool, k: usize) -> Option<Value> {
        self.sets
            .values()
            .find(|d| d.vector == vector && d.latex == latex && d.values.is_some())
            .and_then(|d| d.values.as_ref().unwrap().get(k - 1).cloned())
    }

    /// Expand spectral sums into explicit term chains and resolve valued
    /// set elements, for concrete (matrix-mode) display. `bound` maps
    /// in-scope summation indices to their current concrete value.
    fn instantiate_tensor(
        &self,
        t: &Rc<TensorExpr>,
        bound: &HashMap<String, usize>,
    ) -> Result<Rc<TensorExpr>, Error> {
        use crate::symbolic::SetIndex;
        Ok(match &**t {
            TensorExpr::SumIdx {
                index,
                range,
                exclude,
                body,
            } => {
                let mut bound = bound.clone();
                let mut acc: Option<Rc<TensorExpr>> = None;
                for k in 1..=*range {
                    if let Some(ex) = exclude {
                        if bound.get(ex) == Some(&k) {
                            continue;
                        }
                    }
                    bound.insert(index.clone(), k);
                    let term = self.instantiate_tensor(body, &bound)?;
                    acc = Some(match acc {
                        None => term,
                        Some(prev) => Rc::new(TensorExpr::Add(prev, term)),
                    });
                }
                acc.ok_or_else(|| Error::msg("empty spectral sum"))?
            }
            TensorExpr::SetElem {
                latex,
                order,
                dim,
                index,
                set_dim,
                base,
            } => {
                let k = match index {
                    SetIndex::Num(k) => Some(*k),
                    SetIndex::Sym(name) => bound.get(name).copied(),
                };
                match k {
                    Some(k) => match self.set_value(latex, true, k) {
                        Some(Value::Tensor(v)) => v,
                        _ => Rc::new(TensorExpr::SetElem {
                            latex: latex.clone(),
                            order: *order,
                            dim: *dim,
                            index: SetIndex::Num(k),
                            set_dim: *set_dim,
                            base: base.clone(),
                        }),
                    },
                    None => t.clone(),
                }
            }
            TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::Filled { .. } => {
                t.clone()
            }
            TensorExpr::Transpose(a) => {
                Rc::new(TensorExpr::Transpose(self.instantiate_tensor(a, bound)?))
            }
            TensorExpr::Inverse(a) => {
                Rc::new(TensorExpr::Inverse(self.instantiate_tensor(a, bound)?))
            }
            TensorExpr::InverseTranspose(a) => Rc::new(TensorExpr::InverseTranspose(
                self.instantiate_tensor(a, bound)?,
            )),
            TensorExpr::Diff {
                num,
                den,
                num_label,
            } => Rc::new(TensorExpr::Diff {
                num: self.instantiate_tensor(num, bound)?,
                den: self.instantiate_tensor(den, bound)?,
                num_label: num_label.clone(),
            }),
            TensorExpr::MatMul(a, b) => Rc::new(TensorExpr::MatMul(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
            TensorExpr::Power { base, exp } => Rc::new(TensorExpr::Power {
                base: self.instantiate_tensor(base, bound)?,
                exp: *exp,
            }),
            TensorExpr::Add(a, b) => Rc::new(TensorExpr::Add(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
            TensorExpr::Sub(a, b) => Rc::new(TensorExpr::Sub(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
            TensorExpr::Outer(a, b) => Rc::new(TensorExpr::Outer(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
            TensorExpr::BoxTimes(a, b) => Rc::new(TensorExpr::BoxTimes(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
            TensorExpr::ScalarMul(s, a) => Rc::new(TensorExpr::ScalarMul(
                self.instantiate_scalar(s, bound)?,
                self.instantiate_tensor(a, bound)?,
            )),
            TensorExpr::Neg(a) => Rc::new(TensorExpr::Neg(self.instantiate_tensor(a, bound)?)),
        })
    }

    fn instantiate_scalar(
        &self,
        s: &Rc<ScalarExpr>,
        bound: &HashMap<String, usize>,
    ) -> Result<Rc<ScalarExpr>, Error> {
        use crate::symbolic::SetIndex;
        Ok(match &**s {
            ScalarExpr::SetElem {
                latex,
                index,
                set_dim,
                eig,
            } => {
                let k = match index {
                    SetIndex::Num(k) => Some(*k),
                    SetIndex::Sym(name) => bound.get(name).copied(),
                };
                match k {
                    Some(k) => match self.set_value(latex, false, k) {
                        Some(Value::Scalar(v)) => v,
                        _ => Rc::new(ScalarExpr::SetElem {
                            latex: latex.clone(),
                            index: SetIndex::Num(k),
                            set_dim: *set_dim,
                            eig: eig.clone(),
                        }),
                    },
                    None => s.clone(),
                }
            }
            ScalarExpr::SpecSum { body, index, dim } => {
                let mut bound = bound.clone();
                let mut acc: Option<Rc<ScalarExpr>> = None;
                for k in 1..=*dim {
                    bound.insert(index.clone(), k);
                    let term = self.instantiate_scalar(body, &bound)?;
                    acc = Some(match acc {
                        None => term,
                        Some(prev) => Rc::new(ScalarExpr::Add(prev, term)),
                    });
                }
                acc.ok_or_else(|| Error::msg("empty spectral sum"))?
            }
            ScalarExpr::Sym { .. } | ScalarExpr::Num(_) => s.clone(),
            ScalarExpr::Add(a, b) => Rc::new(ScalarExpr::Add(
                self.instantiate_scalar(a, bound)?,
                self.instantiate_scalar(b, bound)?,
            )),
            ScalarExpr::Sub(a, b) => Rc::new(ScalarExpr::Sub(
                self.instantiate_scalar(a, bound)?,
                self.instantiate_scalar(b, bound)?,
            )),
            ScalarExpr::Mul(a, b) => Rc::new(ScalarExpr::Mul(
                self.instantiate_scalar(a, bound)?,
                self.instantiate_scalar(b, bound)?,
            )),
            ScalarExpr::Div(a, b) => Rc::new(ScalarExpr::Div(
                self.instantiate_scalar(a, bound)?,
                self.instantiate_scalar(b, bound)?,
            )),
            ScalarExpr::Pow(a, b) => Rc::new(ScalarExpr::Pow(
                self.instantiate_scalar(a, bound)?,
                self.instantiate_scalar(b, bound)?,
            )),
            ScalarExpr::Neg(a) => Rc::new(ScalarExpr::Neg(self.instantiate_scalar(a, bound)?)),
            ScalarExpr::Log(a) => Rc::new(ScalarExpr::Log(self.instantiate_scalar(a, bound)?)),
            ScalarExpr::Func { name, arg } => Rc::new(ScalarExpr::Func {
                name: name.clone(),
                arg: self.instantiate_scalar(arg, bound)?,
            }),
            ScalarExpr::Det(t) => Rc::new(ScalarExpr::Det(self.instantiate_tensor(t, bound)?)),
            ScalarExpr::Tr(t) => Rc::new(ScalarExpr::Tr(self.instantiate_tensor(t, bound)?)),
            ScalarExpr::Ddot(a, b) => Rc::new(ScalarExpr::Ddot(
                self.instantiate_tensor(a, bound)?,
                self.instantiate_tensor(b, bound)?,
            )),
        })
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
                 supported; write the spectral form explicitly, e.g. \
                 sum({name}(lambda[a]) * N[a] & N[a], a)"
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

        if let Some(name) = subject.as_deref() {
            if let Some(set) = self.sets.get(name) {
                let mut mode = "symbol".to_string();
                let mut format = "latex".to_string();
                for (key, raw) in kwargs {
                    let val = match raw {
                        Expr::Ident(s) | Expr::Str(s) => s.clone(),
                        other => {
                            return Err(Error::msg(format!("invalid value for `{key}`: {other:?}")))
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
                    "display" => {
                        if mode != "symbol" {
                            return Err(Error::msg("set display only supports mode=symbol"));
                        }
                        render_set_decl(set)
                    }
                    "export" => match format.as_str() {
                        "latex" => render_set_decl(set),
                        "markdown" => format!("$$\n{}\n$$", render_set_decl(set)),
                        other => {
                            return Err(Error::msg(format!(
                                "unknown export format `{other}` (supported: latex, markdown)"
                            )))
                        }
                    },
                    _ => unreachable!(),
                };
                let detail = if callee == "display" {
                    format!("mode={mode}")
                } else {
                    format!("format={format}")
                };
                self.outputs.push(Output {
                    header: format!("{callee} {name}, {detail}"),
                    latex,
                    line,
                    error: None,
                });
                return Ok(());
            }
        }

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

        let value = self.eval(&args[0])?;
        let value = if callee == "display" && mode == "symbol" {
            subject
                .as_deref()
                .and_then(|name| self.display_values.get(name))
                .cloned()
                .unwrap_or(value)
        } else {
            value
        };

        // Back-substitute registered definitions (most recent first) so the
        // display shows \bm C rather than the expanded FᵀF. Internal values
        // stay expanded; this is presentation only. Derivative nodes are
        // exempt: they need the expanded structure.
        let skip_subst =
            matches!(&value, Value::Tensor(t) if matches!(&**t, TensorExpr::Diff { .. }));
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
                // Spectral sums over valued sets expand to explicit,
                // simplified entries; everything else uses the plain
                // component expansion.
                let inst = self.instantiate_tensor(t, &HashMap::new())?;
                // Column-vector display for component-filled vectors.
                if let TensorExpr::Filled {
                    order: 1, entries, ..
                } = &*inst
                {
                    let rows: Vec<String> = entries
                        .iter()
                        .map(|e| scalar_to_latex(&simplify_scalar(e, RuleSet::Continuum)))
                        .collect();
                    return Ok(format!(
                        "{lhs}\\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
                        rows.join(" \\\\\n")
                    ));
                }
                if inst != *t || contains_filled(&inst) {
                    let dim = inst.dim();
                    let mut rows = Vec::with_capacity(dim);
                    for i in 0..dim {
                        let mut row = Vec::with_capacity(dim);
                        for j in 0..dim {
                            let e = simplify_scalar(
                                &Rc::new(crate::renderer::components::component(&inst, i, j)?),
                                RuleSet::Continuum,
                            );
                            row.push(scalar_to_latex(&e));
                        }
                        rows.push(row.join(" & "));
                    }
                    Ok(format!(
                        "{lhs}\\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
                        rows.join(" \\\\\n")
                    ))
                } else {
                    Ok(format!("{lhs}{}", tensor_to_component_matrix(t)?))
                }
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

fn render_set_decl(set: &SetDecl) -> String {
    let values = (1..=set.dim)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{}}}_{{a}}\\quad \\text{{with }} a={values}", set.latex)
}

fn simplified_tensor_value(t: Rc<TensorExpr>) -> Value {
    Value::Tensor(simplify_tensor(&t, RuleSet::Continuum))
}

/// The family size of the (first) set element carrying abstract index
/// `idx` inside a tensor expression, if any.
/// Does the tree contain a component-filled tensor (whose entries should
/// be simplified for matrix display)?
fn contains_filled(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::Filled { .. } => true,
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::Power { base: a, .. } => contains_filled(a),
        TensorExpr::Diff { num, den, .. } => contains_filled(num) || contains_filled(den),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => contains_filled(a) || contains_filled(b),
        TensorExpr::SumIdx { body, .. } => contains_filled(body),
    }
}

fn value_contains_transpose(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor_contains_transpose(t),
        Value::Scalar(_) => false,
    }
}

fn tensor_contains_transpose(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::Transpose(_) | TensorExpr::InverseTranspose(_) => true,
        TensorExpr::Var { .. }
        | TensorExpr::Filled { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. } => false,
        TensorExpr::Inverse(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::SumIdx { body: a, .. }
        | TensorExpr::Power { base: a, .. } => tensor_contains_transpose(a),
        TensorExpr::Diff { num, den, .. } => {
            tensor_contains_transpose(num) || tensor_contains_transpose(den)
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            tensor_contains_transpose(a) || tensor_contains_transpose(b)
        }
    }
}

fn tensor_index_range(t: &TensorExpr, idx: &str) -> Option<usize> {
    use crate::symbolic::SetIndex;
    match t {
        TensorExpr::SetElem {
            index: SetIndex::Sym(i),
            set_dim,
            ..
        } if i == idx => Some(*set_dim),
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => None,
        TensorExpr::Filled { entries, .. } => {
            entries.iter().find_map(|e| scalar_index_range(e, idx))
        }
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a) => tensor_index_range(a, idx),
        TensorExpr::Diff { num, den, .. } => {
            tensor_index_range(num, idx).or_else(|| tensor_index_range(den, idx))
        }
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => {
            tensor_index_range(a, idx).or_else(|| tensor_index_range(b, idx))
        }
        TensorExpr::Power { base, .. } => tensor_index_range(base, idx),
        TensorExpr::SumIdx { index, body, .. } => {
            // An inner sum over the same index binds it.
            if index == idx {
                None
            } else {
                tensor_index_range(body, idx)
            }
        }
        TensorExpr::ScalarMul(s, a) => {
            scalar_index_range(s, idx).or_else(|| tensor_index_range(a, idx))
        }
    }
}

fn scalar_index_range(s: &ScalarExpr, idx: &str) -> Option<usize> {
    use crate::symbolic::SetIndex;
    match s {
        ScalarExpr::SetElem {
            index: SetIndex::Sym(i),
            set_dim,
            ..
        } if i == idx => Some(*set_dim),
        ScalarExpr::Sym { .. } | ScalarExpr::Num(_) | ScalarExpr::SetElem { .. } => None,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            scalar_index_range(a, idx).or_else(|| scalar_index_range(b, idx))
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            scalar_index_range(a, idx)
        }
        ScalarExpr::SpecSum { index, body, .. } => {
            if index == idx {
                None
            } else {
                scalar_index_range(body, idx)
            }
        }
        ScalarExpr::Det(t) | ScalarExpr::Tr(t) => tensor_index_range(t, idx),
        ScalarExpr::Ddot(a, b) => tensor_index_range(a, idx).or_else(|| tensor_index_range(b, idx)),
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
        BinOp::Outer => "&",
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

/// A 1-based component index literal, validated against the tensor dim.
fn component_index(expr: &Expr, dim: usize) -> Result<usize, Error> {
    match expr {
        Expr::Num(n) if n.fract() == 0.0 && *n >= 1.0 && (*n as usize) <= dim => Ok(*n as usize),
        Expr::Num(n) => Err(Error::msg(format!(
            "component index {n} is out of range (1..={dim})"
        ))),
        _ => Err(Error::msg("a component index must be an integer literal")),
    }
}

fn expect_bool(expr: &Expr, what: &str) -> Result<bool, Error> {
    match expr {
        Expr::Bool(b) => Ok(*b),
        _ => Err(Error::msg(format!("`{what}` must be true or false"))),
    }
}
