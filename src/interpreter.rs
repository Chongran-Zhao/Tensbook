//! The interpreter: evaluates the syntactic AST into semantic values,
//! maintains the environment, performs type checking, and executes
//! `.show(...)` output commands.

use crate::ast::{BinOp, Expr, Stmt, UnOp};
use crate::differentiation::{
    diff_block_components, diff_component_equation, diff_scalar_by_scalar, diff_scalar_by_tensor,
    diff_tensor_by_tensor,
};
use crate::error::Error;
use crate::metadata::{
    display_capabilities_for_kind, display_capability_for_kind, tensor_characteristic, value_kind,
    DisplayCapabilityState, SymbolInfo, ValueKind,
};
use crate::ode::{
    BoundaryCondition, Equation, OdeClassification, OdeProblem, OdeSolution, SolveConfig,
    SolveMethod,
};
use crate::renderer::components::{tensor_to_block_component_matrix, tensor_to_component_matrix};
use crate::renderer::latex::{scalar_to_latex, tensor_to_latex};
use crate::simplifier::{simplify_scalar, simplify_tensor, RuleSet};
use crate::symbolic::ScalarExpr;
use crate::tensor::{TensorExpr, TensorProperties};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// A semantic value: scalars and tensors are distinct types and cannot be
/// mixed except where mathematics allows (scalar * tensor, etc.).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(Rc<ScalarExpr>),
    Tensor(Rc<TensorExpr>),
    Equation(Rc<Equation>),
    BoundaryCondition(Rc<BoundaryCondition>),
    OdeProblem(Rc<OdeProblem>),
    OdeClassification(Rc<OdeClassification>),
    OdeSolution(Rc<OdeSolution>),
}

/// Structured payload for outputs the UI renders as chrome (badges, numbered
/// step lists) instead of one opaque LaTeX blob. Only ODE results use this
/// today; `Output::latex` is always populated too, as the fallback and for
/// copy-latex / export / the CLI.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputDetail {
    OdeClassification {
        kind: String,
        order: usize,
        linear: bool,
        homogeneous: bool,
    },
    OdeBoundary {
        boundary: Option<String>,
    },
    OdeMethods {
        available: Vec<String>,
        default: String,
    },
    OdeSteps {
        steps: Vec<crate::ode::OdeStep>,
    },
    Plot(PlotData),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotData {
    pub x_label: String,
    pub x_range: [f64; 2],
    pub y_range: [f64; 2],
    pub series: Vec<PlotSeries>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlotSeries {
    pub label_latex: String,
    pub segments: Vec<Vec<[f64; 2]>>,
}

/// One line of output produced by `.show(...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    /// e.g. `C.show()` or `C.show(matrix)`.
    pub header: String,
    /// LaTeX payload.
    pub latex: String,
    /// 1-based source line of the show statement.
    pub line: usize,
    /// `Some(message)` if this statement failed (per-block error recovery).
    pub error: Option<String>,
    /// Outputs sharing the same row id are rendered side by side in the UI.
    pub row: Option<usize>,
    /// Structured chrome payload (ODE badges / steps), `None` for plain math.
    pub detail: Option<OutputDetail>,
}

#[derive(Debug, Clone)]
struct SolveOptions {
    details: bool,
    config: SolveConfig,
}

#[derive(Default)]
pub struct Interpreter {
    env: HashMap<String, Value>,
    /// LaTeX display labels declared via `Scalar("...")` / `Tensor("...")`,
    /// keyed by variable name. Labels survive reassignment, so
    /// `I1 = Scalar("I_1")` followed by `I1 = Tr(C)` still displays as
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
    /// Names whose definitions are computed results rather than display
    /// aliases. These expand in later displays; foundational notation such as
    /// `C = F.T * F`, `J = Det(F)`, `I1 = Tr(C)` remains aliasable.
    derived_names: HashSet<String>,
    /// Symbol names declared as function arguments via `Var("...")`, in
    /// declaration order. A scalar expression mentioning exactly one of
    /// these is a function of it and can be applied with call syntax.
    fn_vars: Vec<String>,
    /// Indexed families declared via `ScalarSet("...", dim=n)` /
    /// `VectorSet("...", dim=n)`, keyed by variable name. Elements are
    /// accessed as `name[a]` (abstract index) or `name[1]` (concrete).
    sets: HashMap<String, SetDecl>,
    /// UI-facing summary of every named symbol/set after evaluation.
    symbols: HashMap<String, SymbolInfo>,
    outputs: Vec<Output>,
    next_output_row: usize,
}

#[derive(Clone)]
struct SetDecl {
    latex: String,
    /// `true` for VectorSet (order-1 elements), `false` for ScalarSet.
    vector: bool,
    dim: usize,
    /// The decomposed tensor when declared via `Spectral(...)`.
    base: Option<Rc<TensorExpr>>,
    /// Concrete element values filled in by `[a, b] = Spec_Decomp(C)`:
    /// scalar expressions for a ScalarSet, order-1 `Filled` vectors for a
    /// VectorSet. `name[1]` resolves to the value; `name[a]` stays abstract.
    values: Option<Vec<Value>>,
}

struct OutputSubject {
    /// Bare variable name when the output argument is `X`.
    name: Option<String>,
    /// Human-readable source label for the output header.
    header: String,
    /// LaTeX left-hand side for display output.
    lhs: Option<String>,
}

enum DiffTarget {
    Scalar {
        name: String,
    },
    Tensor {
        value: Rc<TensorExpr>,
        label: Option<String>,
    },
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
                    row: None,
                    detail: None,
                });
            }
        }
        std::mem::take(&mut self.outputs)
    }

    /// Look up a variable's evaluated value.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.env.get(name)
    }

    /// Look up a variable or set's UI-facing metadata.
    pub fn symbol_info(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.get(name)
    }

    /// All known variable/set metadata, sorted by source name for stable UI
    /// and test output.
    pub fn symbol_infos(&self) -> Vec<SymbolInfo> {
        let mut infos: Vec<_> = self.symbols.values().cloned().collect();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<(), Error> {
        match stmt {
            Stmt::Assign {
                name, expr, block, ..
            } => {
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
                }
                let value = self.eval(expr)?;
                let display_value = self.eval_display_value(expr).ok();
                // A direct Scalar("...")/Tensor("...") declaration also
                // registers the display label for this variable name.
                if let Expr::Call { callee, args, .. } = expr {
                    if (callee == "Scalar"
                        || callee == "Tensor"
                        || callee == "Var"
                        || callee == "Function")
                        && !args.is_empty()
                    {
                        if let Expr::Str(latex) = &args[0] {
                            self.labels.insert(name.clone(), latex.clone());
                        }
                    }
                }
                let is_leaf = matches!(
                    &value,
                    Value::Tensor(t) if matches!(&**t, TensorExpr::Var { .. })
                ) || matches!(
                    &value,
                    Value::Scalar(s) if matches!(
                        &**s,
                        ScalarExpr::Sym { .. }
                            | ScalarExpr::Num(_)
                            | ScalarExpr::UnknownFunc { .. }
                    )
                );
                let is_derived = self.expr_is_derived_display_result(expr);
                // Register compound definitions for display-time
                // back-substitution (declared leaves substitute trivially
                // and are skipped).
                if let Some(latex) = self.display_label(name, &value) {
                    // Re-registering a name drops its previous definition.
                    self.defs.retain(|d| d.latex != latex);
                    if !is_leaf && !is_derived {
                        self.defs.push(crate::substitute::Def {
                            latex,
                            value: value.clone(),
                            block: *block,
                        });
                    }
                }
                if is_derived {
                    self.derived_names.insert(name.clone());
                } else {
                    self.derived_names.remove(name);
                }
                if let Some(display_value) =
                    display_value.filter(|_| !is_leaf && is_raw_transpose_product_definition(expr))
                {
                    self.display_values.insert(name.clone(), display_value);
                } else {
                    self.display_values.remove(name);
                }
                self.register_value_symbol(name, &value, is_derived);
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
                Expr::MethodCall {
                    target,
                    method,
                    args,
                    kwargs,
                },
                line,
                block,
            ) if method == "show" => self.exec_show(target, args, kwargs, *line, *block, None),
            Stmt::Expr(
                Expr::MethodCall {
                    target,
                    method,
                    args,
                    kwargs,
                },
                line,
                block,
            ) if method == "plot" => self.exec_plot(target, args, kwargs, *line, *block, None),
            Stmt::Expr(
                Expr::MethodCall {
                    target,
                    method,
                    args,
                    kwargs,
                },
                line,
                _block,
            ) if method == "solve" => {
                self.exec_ode_method_output(target, method, args, kwargs, *line)
            }
            Stmt::OutputRow { exprs, line, block } => self.exec_output_row(exprs, *line, *block),
            Stmt::Expr(expr, _, _) => {
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
            other => {
                return Err(Error::msg(format!(
                    "a tensor component must be a scalar expression, got {}",
                    kind(&other)
                )))
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
            Some(other) => {
                return Err(Error::msg(format!(
                    "`{name}` is {}, not a tensor",
                    kind(other)
                )))
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
        let value = Value::Tensor(Rc::new(TensorExpr::Filled {
            latex,
            order,
            dim,
            entries,
        }));
        self.register_value_symbol(name, &value, false);
        self.env.insert(name.to_string(), value);
        Ok(())
    }

    /// `[a, b] = Spec_Decomp(C)` — concrete eigendecomposition of a
    /// component-filled (diagonal) tensor into two pre-declared sets.
    /// `[lambda, N] = Spectral(C, "\lambda", "\bm N")` — symbolic spectral
    /// sets tied to a provably symmetric second-order tensor.
    fn exec_assign_pair(&mut self, first: &str, second: &str, expr: &Expr) -> Result<(), Error> {
        let Expr::Call {
            callee,
            args,
            kwargs,
        } = expr
        else {
            return Err(Error::msg(
                "`[a, b] = ...` expects Spec_Decomp(C) or Spectral(C, \"<scalar>\", \"<vector>\")",
            ));
        };
        match callee.as_str() {
            "Spec_Decomp" => self.exec_spec_decomp_pair(first, second, args, kwargs),
            "Spectral" => self.declare_spectral_pair(first, second, args, kwargs),
            _ => Err(Error::msg(format!(
                "`[a, b] = ...` expects Spec_Decomp(...) or Spectral(...), got `{callee}`"
            ))),
        }
    }

    fn exec_spec_decomp_pair(
        &mut self,
        first: &str,
        second: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(), Error> {
        if args.len() != 1 || !kwargs.is_empty() {
            return Err(Error::msg("`Spec_Decomp` takes exactly one tensor"));
        }
        let c = match self.eval(&args[0])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => return Err(Error::msg("`Spec_Decomp` requires a tensor argument")),
            other => {
                return Err(Error::msg(format!(
                    "`Spec_Decomp` requires a tensor argument, got {}",
                    kind(&other)
                )))
            }
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
        if let Some(decl) = self.sets.get(first).cloned() {
            self.register_set_symbol(first, &decl);
        }
        if let Some(decl) = self.sets.get(second).cloned() {
            self.register_set_symbol(second, &decl);
        }
        Ok(())
    }

    fn declare_spectral_pair(
        &mut self,
        first: &str,
        second: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<(), Error> {
        if args.len() != 3 || !kwargs.is_empty() {
            return Err(Error::msg(format!(
                "`Spectral` expects a tensor, a scalar LaTeX name, and a vector \
                 LaTeX name: [{first}, {second}] = Spectral(C, \"<scalar>\", \"<vector>\")"
            )));
        }
        let base = match self.eval(&args[0])? {
            Value::Tensor(t) => t,
            Value::Scalar(_) => return Err(Error::msg("`Spectral` requires a tensor argument")),
            other => {
                return Err(Error::msg(format!(
                    "`Spectral` requires a tensor argument, got {}",
                    kind(&other)
                )))
            }
        };
        if base.order() != 2 || !base.is_symmetric() {
            return Err(Error::msg(
                "`Spectral` requires a provably symmetric second-order tensor \
                 (e.g. C = F.T * F)",
            ));
        }
        let Expr::Str(scalar_latex) = &args[1] else {
            return Err(Error::msg(
                "`Spectral` expects a string LaTeX name for the scalar set",
            ));
        };
        let Expr::Str(vector_latex) = &args[2] else {
            return Err(Error::msg(
                "`Spectral` expects a string LaTeX name for the vector set",
            ));
        };
        self.labels.insert(first.to_string(), scalar_latex.clone());
        self.labels.insert(second.to_string(), vector_latex.clone());
        self.sets.insert(
            first.to_string(),
            SetDecl {
                latex: scalar_latex.clone(),
                vector: false,
                dim: base.dim(),
                base: Some(base.clone()),
                values: None,
            },
        );
        self.sets.insert(
            second.to_string(),
            SetDecl {
                latex: vector_latex.clone(),
                vector: true,
                dim: base.dim(),
                base: Some(base),
                values: None,
            },
        );
        if let Some(decl) = self.sets.get(first).cloned() {
            self.register_set_symbol(first, &decl);
        }
        if let Some(decl) = self.sets.get(second).cloned() {
            self.register_set_symbol(second, &decl);
        }
        Ok(())
    }

    // ---- evaluation ------------------------------------------------------

    fn eval_identifier(&self, name: &str) -> Result<Value, Error> {
        if let Some(value) = self.env.get(name) {
            return Ok(value.clone());
        }
        match name {
            "pi" if !self.sets.contains_key(name) => Ok(Value::Scalar(Rc::new(ScalarExpr::Sym {
                name: "pi".to_string(),
                latex: "\\pi".to_string(),
            }))),
            "e" if !self.sets.contains_key(name) => Ok(Value::Scalar(Rc::new(ScalarExpr::Sym {
                name: "e".to_string(),
                latex: "e".to_string(),
            }))),
            _ => Err(Error::msg(format!("undefined variable `{name}`"))),
        }
    }

    fn eval(&mut self, expr: &Expr) -> Result<Value, Error> {
        match expr {
            Expr::Num(n) => Ok(Value::Scalar(Rc::new(ScalarExpr::Num(*n)))),
            Expr::Ident(name) => self.eval_identifier(name),
            Expr::Str(_) | Expr::Bool(_) => Err(Error::msg(
                "string/bool literals are only valid as supported command arguments",
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
                other => Err(Error::msg(format!(
                    "unary `-` is not defined for {}",
                    kind(&other)
                ))),
            },
            Expr::Binary { op, lhs, rhs } => {
                let l = self.eval(lhs)?;
                let r = self.eval(rhs)?;
                self.eval_binary(*op, l, r)
            }
            Expr::List(_) => Err(Error::msg(
                "list literals are only valid as a `.plot(...)` target",
            )),
            Expr::Index { target, index } => self.eval_index(target, index),
            Expr::MethodCall { method, .. } if method == "show" || method == "plot" => {
                Err(Error::msg(format!(
                    "`.{method}(...)` is an output statement and cannot be used inside an expression"
                )))
            }
            Expr::MethodCall {
                target,
                method,
                args,
                kwargs,
            } if method == "classify" || method == "solve" => {
                self.eval_ode_method(target, method, args, kwargs)
            }
            Expr::MethodCall { method, .. } => {
                Err(Error::msg(format!("unknown method `.{method}(...)`")))
            }
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
                other => Err(Error::msg(format!(
                    "unary `-` is not defined for {}",
                    kind(&other)
                ))),
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
                (2, 4) => Ok(Value::Tensor(Rc::new(
                    TensorExpr::double_contract_second_fourth(a, b)?,
                ))),
                (4, 2) => Ok(Value::Tensor(Rc::new(
                    TensorExpr::double_contract_second_fourth(b, a)?,
                ))),
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
                (2, 4) => Ok(simplified_tensor_value(Rc::new(
                    TensorExpr::double_contract_second_fourth(a, b)?,
                ))),
                (4, 2) => Ok(simplified_tensor_value(Rc::new(
                    TensorExpr::double_contract_second_fourth(b, a)?,
                ))),
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
            "Function" => self.builtin_function(args, kwargs),
            "Tensor" => self.builtin_tensor(args, kwargs),
            "Sum" => self.builtin_sum(args, kwargs),
            "ScalarSet" | "VectorSet" => Err(Error::msg(format!(
                "`{callee}` declares a set and must be used in an assignment, \
                 e.g. lambda = {callee}(...)"
            ))),
            "Spectral" => Err(Error::msg(
                "`Spectral` returns two sets and must be used as \
                 `[lambda, N] = Spectral(C, \"\\lambda\", \"\\bm N\")`",
            )),
            "Det" | "Tr" => {
                let t = self.expect_tensor_arg(callee, args, kwargs)?;
                if t.order() != 2 {
                    return Err(Error::msg(format!(
                        "`{callee}` requires a second-order tensor, got order {}",
                        t.order()
                    )));
                }
                let node = if callee == "Det" {
                    ScalarExpr::Det(t)
                } else {
                    ScalarExpr::Tr(t)
                };
                Ok(Value::Scalar(Rc::new(node)))
            }
            "log" | "sqrt" | "exp" | "sinh" | "cosh" | "tanh" | "sin" | "cos" | "tan" => {
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
                         form explicitly, e.g. Sum({callee}(lambda[a]) * N[a] & N[a], a)"
                    ))),
                    other => Err(Error::msg(format!(
                        "`{callee}` expects a scalar argument, got {}",
                        kind(&other)
                    ))),
                }
            }
            "Diff" => self.builtin_diff(args, kwargs),
            "Derivative" => self.builtin_derivative(args, kwargs),
            "Equation" => self.builtin_equation(args, kwargs),
            "BoundaryCondition" => self.builtin_boundary_condition(args, kwargs),
            "ODE" => self.builtin_ode(args, kwargs),
            "Integrate" => self.builtin_integrate(args, kwargs),
            "Integral" => self.builtin_integral(args, kwargs),
            "IC" => Err(Error::msg(
                "`IC(...)` was renamed; use `BoundaryCondition(y(x0), y0)`",
            )),
            "ClassifyODE" => Err(Error::msg(
                "`ClassifyODE(eq, y, x)` was removed; use `ODE(eq, y, x).show(classification)`",
            )),
            "SolveODE" => Err(Error::msg(
                "`SolveODE(eq, y, x, ic=...)` was removed; use \
                 `ODE(eq, y, x, BoundaryCondition(...)).solve(details=true)`",
            )),
            "Inv" => {
                let t = self.expect_tensor_arg(callee, args, kwargs)?;
                Ok(Value::Tensor(Rc::new(TensorExpr::inverse(t)?)))
            }
            "Simplify" => {
                if args.len() != 1 {
                    return Err(Error::msg("`Simplify` expects one expression argument"));
                }
                let mut rules = RuleSet::Continuum;
                for (key, value) in kwargs {
                    match (key.as_str(), value) {
                        ("rules", Expr::Ident(name)) | ("rules", Expr::Str(name)) => {
                            rules = RuleSet::parse(name)?;
                        }
                        (other, _) => {
                            return Err(Error::msg(format!(
                                "unknown keyword `{other}` for `Simplify`"
                            )))
                        }
                    }
                }
                match self.eval(&args[0])? {
                    Value::Scalar(s) => Ok(Value::Scalar(simplify_scalar(&s, rules))),
                    Value::Tensor(t) => Ok(Value::Tensor(simplify_tensor(&t, rules))),
                    other => Err(Error::msg(format!(
                        "`Simplify` expects a scalar or tensor, got {}",
                        kind(&other)
                    ))),
                }
            }
            "diff" => Err(renamed_builtin_error("diff", "Diff")),
            "simplify" => Err(renamed_builtin_error("simplify", "Simplify")),
            "sum" => Err(renamed_builtin_error("sum", "Sum")),
            "det" => Err(renamed_builtin_error("det", "Det")),
            "tr" => Err(renamed_builtin_error("tr", "Tr")),
            "inv" => Err(renamed_builtin_error("inv", "Inv")),
            "dot" => Err(Error::msg("`dot(A, B)` was removed; use `A * B`")),
            "outer" | "otimes" => Err(Error::msg(format!(
                "`{callee}(A, B)` was removed; use `A & B`"
            ))),
            "display" => Err(Error::msg(
                "`display(expr, mode=...)` was removed; use `expr.show(...)`",
            )),
            "export" => Err(Error::msg(
                "`export(...)` was removed from the DSL; use the app Export button",
            )),
            "eigvals" | "eigvecs" => Err(Error::msg(
                "`eigvals/eigvecs` were removed; use \
                 `[lambda, N] = Spectral(C, \"\\lambda\", \"\\bm N\")`",
            )),
            // A user-defined name bound to a scalar expression in a declared
            // `Var` argument is a function: `E(x)` substitutes the argument.
            other => match self.env.get(other).cloned() {
                Some(Value::Scalar(body)) => self.apply_scalar_function(other, &body, args, kwargs),
                Some(Value::Tensor(_)) => {
                    Err(Error::msg(format!("`{other}` is a tensor, not a function")))
                }
                Some(value) => Err(Error::msg(format!(
                    "`{other}` is {}, not a function",
                    kind(&value)
                ))),
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
            other => Err(Error::msg(format!(
                "`{callee}` requires a tensor argument, got {}",
                kind(&other)
            ))),
        }
    }

    fn expect_scalar_value(&mut self, expr: &Expr, what: &str) -> Result<Rc<ScalarExpr>, Error> {
        match self.eval(expr)? {
            Value::Scalar(s) => Ok(s),
            other => Err(Error::msg(format!(
                "`{what}` expects a scalar expression, got {}",
                kind(&other)
            ))),
        }
    }

    fn builtin_equation(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Equation` expects two scalar expressions: Equation(lhs, rhs)",
            ));
        }
        let lhs = self.expect_scalar_value(&args[0], "Equation")?;
        let rhs = self.expect_scalar_value(&args[1], "Equation")?;
        Ok(Value::Equation(Rc::new(crate::ode::equation(lhs, rhs))))
    }

    fn builtin_integrate(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Integrate` expects two arguments: Integrate(expr, x)",
            ));
        }
        let expr = self.expect_scalar_value(&args[0], "Integrate")?;
        let var = self.expect_scalar_value(&args[1], "Integrate")?;
        Ok(Value::Scalar(crate::integration::integrate(&expr, &var)?))
    }

    fn builtin_integral(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Integral` expects two arguments: Integral(expr, x)",
            ));
        }
        let expr = self.expect_scalar_value(&args[0], "Integral")?;
        let var = self.expect_scalar_value(&args[1], "Integral")?;
        Ok(Value::Scalar(crate::integration::formal_integral(
            expr, var,
        )))
    }

    fn builtin_boundary_condition(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if !(args.len() == 2 || args.len() == 3) || !kwargs.is_empty() {
            return Err(Error::msg(
                "`BoundaryCondition` expects BoundaryCondition(y(x0), y0) or BoundaryCondition(Derivative(y, x), x0, y0)",
            ));
        }
        let function = self.expect_scalar_value(&args[0], "BoundaryCondition")?;
        let (function, point, value) = if args.len() == 2 {
            let point =
                match &*function {
                    ScalarExpr::UnknownFunc { args, .. } if args.len() == 1 => args[0].clone(),
                    _ => return Err(Error::msg(
                        "`BoundaryCondition` first argument must be a function value such as y(1)",
                    )),
                };
            let value = self.expect_scalar_value(&args[1], "BoundaryCondition")?;
            (function, point, value)
        } else {
            let point = self.expect_scalar_value(&args[1], "BoundaryCondition")?;
            let value = self.expect_scalar_value(&args[2], "BoundaryCondition")?;
            match &*function {
                ScalarExpr::UnknownFunc { .. } => {}
                _ => {
                    return Err(Error::msg(
                        "`BoundaryCondition` first argument must be y(x0) or Derivative(y, x)",
                    ))
                }
            };
            (function, point, value)
        };
        Ok(Value::BoundaryCondition(Rc::new(BoundaryCondition {
            function,
            point,
            value,
        })))
    }

    fn builtin_ode(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() < 3 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`ODE` expects ODE(eq, y, x) or ODE(eq, y, x, BoundaryCondition(...), ...)",
            ));
        }
        let eq = match self.eval(&args[0])? {
            Value::Equation(eq) => eq,
            other => {
                return Err(Error::msg(format!(
                    "`ODE` expects an Equation as its first argument, got {}",
                    kind(&other)
                )))
            }
        };
        let target = self.expect_scalar_value(&args[1], "ODE")?;
        let independent = self.expect_scalar_value(&args[2], "ODE")?;
        let mut boundary_conditions = Vec::new();
        for arg in args.iter().skip(3) {
            match self.eval(arg)? {
                Value::BoundaryCondition(v) => boundary_conditions.push((*v).clone()),
                other => {
                    return Err(Error::msg(format!(
                        "`ODE` boundary arguments expect BoundaryCondition(...), got {}",
                        kind(&other)
                    )))
                }
            }
        }
        Ok(Value::OdeProblem(Rc::new(crate::ode::problem(
            (*eq).clone(),
            target,
            independent,
            boundary_conditions,
        ))))
    }

    fn eval_ode_method(
        &mut self,
        target: &Expr,
        method: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        let problem = match self.eval(target)? {
            Value::OdeProblem(problem) => problem,
            other => {
                return Err(Error::msg(format!(
                    "`.{method}(...)` expects an ODE problem, got {}",
                    kind(&other)
                )))
            }
        };
        match method {
            "classify" => Err(Error::msg(
                "`ODE.classify()` was removed; use `ODE.show(classification)`",
            )),
            "solve" => {
                let options = self.solve_options(args, kwargs)?;
                Ok(Value::OdeSolution(Rc::new(
                    crate::ode::solve_problem_with_config(&problem, options.config)?,
                )))
            }
            _ => Err(Error::msg(format!("unknown ODE method `.{method}(...)`"))),
        }
    }

    fn solve_options(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<SolveOptions, Error> {
        if !args.is_empty() {
            return Err(Error::msg(
                "`ODE.solve(...)` takes only optional keywords such as `details=true`, `method=separable`, `about=0`, or `terms=6`",
            ));
        }
        let mut details = false;
        let mut config = SolveConfig::default();
        for (key, value) in kwargs {
            match key.as_str() {
                "details" => {
                    details = expect_bool(value, "details")?;
                }
                "method" => {
                    config.method = expect_solve_method(value)?;
                }
                "about" => {
                    config.about = Some(self.expect_scalar_value(value, "about")?);
                }
                "terms" => {
                    config.terms = expect_usize(value, "terms")?;
                }
                other => {
                    return Err(Error::msg(format!(
                        "unknown keyword `{other}` for `ODE.solve`"
                    )))
                }
            }
        }
        Ok(SolveOptions { details, config })
    }

    fn exec_ode_method_output(
        &mut self,
        target: &Expr,
        method: &str,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: usize,
    ) -> Result<(), Error> {
        let subject = self.output_subject(target);
        let header = ode_method_header(&subject.header, method, args, kwargs);
        let problem = match self.eval(target)? {
            Value::OdeProblem(problem) => problem,
            other => {
                return Err(Error::msg(format!(
                    "`.{method}(...)` expects an ODE problem, got {}",
                    kind(&other)
                )))
            }
        };
        let (latex, detail) = match method {
            "classify" => {
                return Err(Error::msg(
                    "`ODE.classify()` was removed; use `ODE.show(classification)`",
                ))
            }
            "solve" => {
                let options = self.solve_options(args, kwargs)?;
                let solution = crate::ode::solve_problem_with_config(&problem, options.config)?;
                if options.details {
                    let class = crate::ode::classify_problem(&problem)?;
                    let latex = crate::ode::render_solution_details(&solution, &class);
                    let detail = OutputDetail::OdeSteps {
                        steps: crate::ode::solution_steps(&solution),
                    };
                    (latex, Some(detail))
                } else {
                    (crate::ode::render_solution(&solution, "solution"), None)
                }
            }
            _ => return Err(Error::msg(format!("unknown ODE method `.{method}(...)`"))),
        };
        self.outputs.push(Output {
            header,
            latex,
            line,
            error: None,
            row: None,
            detail,
        });
        Ok(())
    }

    // ---- builtins: declarations ------------------------------------------

    /// `Diff(expr, X, order=n)` — evaluated symbolic derivative. `order` defaults to 1;
    /// higher orders are evaluated by repeated differentiation.
    ///
    /// Denominator forms:
    /// - declared scalar symbol (`Diff(W, mu)`) → scalar derivative;
    /// - tensor variable or compound tensor expression (`Diff(W, F)`,
    ///   `Diff(W, C)` with `C = F.T * F`) → tensor-valued derivative; a
    ///   compound denominator is treated as the independent variable and
    ///   matched structurally, with a strict independence check rejecting
    ///   hidden dependence (e.g. `Det(F)` inside `Diff(..., C)`).
    /// - tensor / tensor → opaque order-4 `Diff` node.
    fn builtin_diff(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() != 2 {
            return Err(Error::msg(
                "`Diff` takes exactly two positional arguments: Diff(expr, X, order=1)",
            ));
        }
        let mut order = 1usize;
        for (key, value) in kwargs {
            match key.as_str() {
                "order" => order = expect_usize(value, "order")?,
                other => return Err(Error::msg(format!("unknown keyword `{other}` for `Diff`"))),
            }
        }

        let den = match self.eval(&args[1])? {
            Value::Scalar(ds) => {
                let ScalarExpr::Sym { name, .. } = &*ds else {
                    return Err(Error::msg(
                        "Diff with respect to a scalar requires a declared scalar \
                         symbol (e.g. mu), not a compound expression",
                    ));
                };
                DiffTarget::Scalar { name: name.clone() }
            }
            Value::Tensor(t) => {
                if t.order() != 2 {
                    return Err(Error::msg(format!(
                        "Diff denominator must be a second-order tensor, got order {}",
                        t.order()
                    )));
                }
                let label = match &args[1] {
                    Expr::Ident(name) => self.display_lhs(name),
                    _ => None,
                };
                DiffTarget::Tensor { value: t, label }
            }
            other => {
                return Err(Error::msg(format!(
                    "Diff denominator must be a scalar or second-order tensor, got {}",
                    kind(&other)
                )))
            }
        };

        let mut value = self.eval(&args[0])?;
        if matches!(&value, Value::Scalar(s) if scalar_contains_unknown_function(s)) {
            return Err(Error::msg(
                "`Diff` evaluates explicit derivatives; use `Derivative(f, x, order=...)` \
                 for formal derivatives of unknown `Function(...)` objects",
            ));
        }
        let first_num_label = match &args[0] {
            Expr::Ident(name) => self.display_lhs(name),
            _ => None,
        };
        for k in 0..order {
            let num_label = (k == 0).then(|| first_num_label.clone()).flatten();
            value = self.diff_once(value, &den, num_label)?;
        }
        Ok(value)
    }

    fn diff_once(
        &self,
        num: Value,
        den: &DiffTarget,
        num_label: Option<String>,
    ) -> Result<Value, Error> {
        match den {
            DiffTarget::Scalar { name } => match num {
                Value::Scalar(s) => Ok(Value::Scalar(simplify_scalar(
                    &diff_scalar_by_scalar(&s, name)?,
                    RuleSet::Continuum,
                ))),
                Value::Tensor(_) => Err(Error::msg(
                    "differentiating a tensor with respect to a scalar is not \
                     supported yet",
                )),
                other => Err(Error::msg(format!(
                    "Diff numerator must be a scalar or tensor, got {}",
                    kind(&other)
                ))),
            },
            DiffTarget::Tensor { value: den, label } => match num {
                Value::Scalar(s) => Ok(Value::Tensor(simplify_tensor(
                    &diff_scalar_by_tensor(&s, den)?,
                    RuleSet::Continuum,
                ))),
                Value::Tensor(t) => {
                    if t.order() != 2 {
                        return Err(Error::msg(format!(
                            "tensor-by-tensor Diff currently requires a second-order \
                             numerator, got order {}",
                            t.order()
                        )));
                    }
                    Ok(Value::Tensor(simplify_tensor(
                        &diff_tensor_by_tensor(&t, den, num_label, label.clone())?,
                        RuleSet::Continuum,
                    )))
                }
                other => Err(Error::msg(format!(
                    "Diff numerator must be a scalar or tensor, got {}",
                    kind(&other)
                ))),
            },
        }
    }

    /// `Derivative(f, x, order=n)` — formal derivative of an unknown
    /// `Function(...)`. This intentionally does not evaluate explicit scalar
    /// expressions; use `Diff(...)` for those.
    fn builtin_derivative(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() != 2 {
            return Err(Error::msg(
                "`Derivative` takes exactly two positional arguments: \
                 Derivative(f, x, order=1)",
            ));
        }
        let mut order = 1usize;
        for (key, value) in kwargs {
            match key.as_str() {
                "order" => order = expect_usize(value, "order")?,
                other => {
                    return Err(Error::msg(format!(
                        "unknown keyword `{other}` for `Derivative`"
                    )))
                }
            }
        }

        let func = self.expect_scalar_value(&args[0], "Derivative")?;
        let variable = self.expect_scalar_value(&args[1], "Derivative")?;
        let var_name = scalar_symbol_name(&variable).ok_or_else(|| {
            Error::msg(
                "`Derivative` expects a declared variable as its second argument; \
                 use `Diff(expr, x)` for explicit scalar differentiation",
            )
        })?;

        let ScalarExpr::UnknownFunc {
            name,
            args: fn_args,
            derivative_orders,
        } = &*func
        else {
            return Err(Error::msg(
                "`Derivative` only constructs formal derivatives of unknown \
                 `Function(...)` objects; use `Diff(expr, x)` for explicit expressions",
            ));
        };

        let Some(index) = fn_args
            .iter()
            .position(|arg| scalar_symbol_name(arg).is_some_and(|arg_name| arg_name == var_name))
        else {
            return Err(Error::msg(format!(
                "`Derivative` variable `{var_name}` is not one of this function's arguments"
            )));
        };

        let mut orders = derivative_orders.clone();
        if index >= orders.len() {
            orders.resize(fn_args.len(), 0);
        }
        orders[index] += order;
        Ok(Value::Scalar(Rc::new(ScalarExpr::UnknownFunc {
            name: name.clone(),
            args: fn_args.clone(),
            derivative_orders: orders,
        })))
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
        if let Some(decl) = self.sets.get(name).cloned() {
            self.register_set_symbol(name, &decl);
        }
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
            other => {
                return Err(Error::msg(format!(
                    "component indexing `[i][j]` requires a tensor, got {}",
                    kind(&other)
                )))
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

    /// `Sum(body, a)` — sum the body over the abstract set index `a`.
    fn builtin_sum(&mut self, args: &[Expr], kwargs: &[(String, Expr)]) -> Result<Value, Error> {
        if args.len() != 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Sum` expects two arguments: Sum(<expr>, <index>)",
            ));
        }
        let Expr::Ident(index) = &args[1] else {
            return Err(Error::msg("the second argument of `Sum` is an index name"));
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
            other => Err(Error::msg(format!(
                "`Sum` expects a scalar or tensor summand, got {}",
                kind(&other)
            ))),
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
            ScalarExpr::UnknownFunc {
                name,
                args,
                derivative_orders,
            } => Rc::new(ScalarExpr::UnknownFunc {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| self.instantiate_scalar(arg, bound))
                    .collect::<Result<Vec<_>, _>>()?,
                derivative_orders: derivative_orders.clone(),
            }),
            ScalarExpr::Integral {
                integrand,
                variable,
            } => Rc::new(ScalarExpr::Integral {
                integrand: self.instantiate_scalar(integrand, bound)?,
                variable: self.instantiate_scalar(variable, bound)?,
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
                 Sum({name}(lambda[a]) * N[a] & N[a], a)"
            ))),
            other => Err(Error::msg(format!(
                "`{name}` is a scalar function; applying it to {} is not supported",
                kind(&other)
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

    /// `Function("y", x)` / `Function("u", x, y)` — declare an unknown
    /// scalar function of declared `Var` arguments.
    fn builtin_function(
        &mut self,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<Value, Error> {
        if args.len() < 2 || !kwargs.is_empty() {
            return Err(Error::msg(
                "`Function` expects a name and at least one variable: Function(\"<latex>\", x)",
            ));
        }
        let latex = match &args[0] {
            Expr::Str(s) => s.clone(),
            _ => return Err(Error::msg("`Function` expects a string LaTeX name")),
        };
        let mut fn_args = Vec::new();
        for arg in &args[1..] {
            let value = self.expect_scalar_value(arg, "Function")?;
            match self.free_fn_vars(&value).as_slice() {
                [_] => fn_args.push(value),
                [] => {
                    return Err(Error::msg(
                        "`Function` arguments must mention declared `Var(...)` symbols",
                    ))
                }
                vars => {
                    return Err(Error::msg(format!(
                        "each `Function` argument must be one `Var(...)`; got {}",
                        vars.join(", ")
                    )))
                }
            }
        }
        Ok(Value::Scalar(Rc::new(ScalarExpr::UnknownFunc {
            name: latex,
            derivative_orders: vec![0; fn_args.len()],
            args: fn_args,
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

    // ---- show output ------------------------------------------------------

    fn exec_show(
        &mut self,
        target: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: usize,
        block: usize,
        row: Option<usize>,
    ) -> Result<(), Error> {
        let mode = show_mode(args, kwargs)?;
        let subject = self.output_subject(target);

        if let Some(name) = subject.name.as_deref() {
            if let Some(set) = self.sets.get(name) {
                self.ensure_set_display_mode(set, &mode)?;
                self.outputs.push(Output {
                    header: show_header(name, &mode, args.is_empty()),
                    latex: render_set_decl(set),
                    line,
                    error: None,
                    row,
                    detail: None,
                });
                return Ok(());
            }
        }

        let value = self.eval(target)?;
        let uses_raw_display_value = mode == "symbol"
            && subject
                .name
                .as_deref()
                .is_some_and(|name| self.display_values.contains_key(name));
        let value = if uses_raw_display_value {
            subject
                .name
                .as_deref()
                .and_then(|name| self.display_values.get(name))
                .cloned()
                .unwrap_or(value)
        } else {
            value
        };

        // Back-substitute registered definitions (most recent first) so the
        // display follows user-defined names such as `C` and `I_1`. Internal
        // values stay expanded; this is presentation only. Derivative nodes
        // are exempt because they need the expanded structure.
        let skip_subst = matches!(&value, Value::Tensor(t) if matches!(&**t, TensorExpr::Diff { .. }))
            || !matches!(&value, Value::Scalar(_) | Value::Tensor(_));
        let value = if skip_subst {
            value
        } else {
            let defs = self.display_defs_for_block(block);
            crate::substitute::substitute(&value, &defs)
        };
        let value = if uses_raw_display_value || skip_subst {
            value
        } else {
            Self::simplify_presentation_value(value)
        };

        let latex = self.render_display(&value, subject.lhs.as_deref(), &mode)?;
        let detail = ode_show_detail(&value, &mode);
        self.outputs.push(Output {
            header: show_header(&subject.header, &mode, args.is_empty()),
            latex,
            line,
            error: None,
            row,
            detail,
        });
        Ok(())
    }

    fn exec_plot(
        &mut self,
        target: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
        line: usize,
        _block: usize,
        row: Option<usize>,
    ) -> Result<(), Error> {
        if !kwargs.is_empty() || args.len() != 2 {
            return Err(Error::msg(
                "`.plot(...)` expects exactly two range bounds: expr.plot(from, to)",
            ));
        }
        let subject = self.output_subject(target);
        let curves = match target {
            Expr::List(items) => {
                if items.is_empty() {
                    return Err(Error::msg("`.plot(...)` needs at least one curve"));
                }
                let mut curves = Vec::with_capacity(items.len());
                for item in items {
                    match self.eval(item)? {
                        Value::Scalar(expr) => curves.push(expr),
                        other => {
                            return Err(Error::msg(format!(
                                "`.plot(...)` expects scalar expressions, got {}",
                                kind(&other)
                            )))
                        }
                    }
                }
                curves
            }
            _ => match self.eval(target)? {
                Value::Scalar(expr) => vec![expr],
                Value::OdeSolution(solution) => {
                    let solution = solution
                        .solution
                        .as_ref()
                        .ok_or_else(|| Error::msg("no closed-form solution to plot"))?;
                    // Only explicit solutions may be plotted: for an implicit
                    // one like `y^3 = sin x + C` the rhs is NOT y(x), and
                    // plotting it would silently draw the wrong curve.
                    if !is_bare_unknown(&solution.lhs) {
                        return Err(Error::msg(
                            "solution is implicit; plot needs an explicit y(x) solution",
                        ));
                    }
                    if !is_numeric_plot_candidate(&solution.rhs) {
                        return Err(Error::msg("solution is not in closed form; cannot plot"));
                    }
                    vec![solution.rhs.clone()]
                }
                other => {
                    return Err(Error::msg(format!(
                        "`.plot(...)` expects a scalar expression or explicit ODE solution, got {}",
                        kind(&other)
                    )))
                }
            },
        };
        let abscissa = self.plot_abscissa(&curves)?;
        for curve in &curves {
            let unbound = crate::numeric::unbound_symbols(curve, &abscissa);
            if let Some(name) = unbound.first() {
                return Err(Error::msg(format!(
                    "`{name}` is unbound; plot needs a concrete function"
                )));
            }
        }
        let from = self.eval_plot_bound(&args[0], "from")?;
        let to = self.eval_plot_bound(&args[1], "to")?;
        if from >= to {
            return Err(Error::msg("plot range requires from < to"));
        }

        let series = curves
            .iter()
            .map(|expr| sample_plot_series(expr, &abscissa, from, to))
            .collect::<Result<Vec<_>, _>>()?;
        let all_y = series
            .iter()
            .flat_map(|series| series.segments.iter())
            .flat_map(|segment| segment.iter().map(|point| point[1]))
            .collect::<Vec<_>>();
        let y_range = robust_y_range(&all_y)?;
        let series = series
            .into_iter()
            .map(|series| PlotSeries {
                label_latex: series.label_latex,
                segments: split_asymptote_jumps(series.segments, y_range),
            })
            .collect::<Vec<_>>();
        let latex = series
            .iter()
            .map(|series| series.label_latex.clone())
            .collect::<Vec<_>>()
            .join(", ");
        self.outputs.push(Output {
            header: format!("{}.plot(...)", plot_header(target, &subject.header)),
            latex,
            line,
            error: None,
            row,
            detail: Some(OutputDetail::Plot(PlotData {
                x_label: abscissa,
                x_range: [from, to],
                y_range,
                series,
            })),
        });
        Ok(())
    }

    fn exec_output_row(&mut self, exprs: &[Expr], line: usize, block: usize) -> Result<(), Error> {
        self.next_output_row += 1;
        let row = Some(self.next_output_row);
        for expr in exprs {
            let Expr::MethodCall {
                target,
                method,
                args,
                kwargs,
            } = expr
            else {
                return Err(Error::new(
                    "output row items must be `.show(...)` or `.plot(...)` calls",
                    Some(line),
                ));
            };
            if method == "show" {
                self.exec_show(target, args, kwargs, line, block, row)?;
            } else if method == "plot" {
                self.exec_plot(target, args, kwargs, line, block, row)?;
            } else {
                return Err(Error::new(
                    "output row items must be `.show(...)` or `.plot(...)` calls",
                    Some(line),
                ));
            }
        }
        Ok(())
    }

    fn plot_abscissa(&self, curves: &[Rc<ScalarExpr>]) -> Result<String, Error> {
        let vars = self
            .fn_vars
            .iter()
            .filter(|var| {
                curves
                    .iter()
                    .any(|curve| crate::differentiation::scalar_mentions_scalar(curve, var))
            })
            .cloned()
            .collect::<Vec<_>>();
        match vars.as_slice() {
            [var] => Ok(var.clone()),
            [] => Err(Error::msg(
                "plot needs a variable; declare one with `Var(\"x\")`",
            )),
            _ => Err(Error::msg(format!(
                "plot needs exactly one variable; found {}",
                vars.join(", ")
            ))),
        }
    }

    fn eval_plot_bound(&mut self, expr: &Expr, name: &str) -> Result<f64, Error> {
        let value = match self.eval(expr)? {
            Value::Scalar(expr) => expr,
            other => {
                return Err(Error::msg(format!(
                    "plot range bound `{name}` must be a scalar constant, got {}",
                    kind(&other)
                )))
            }
        };
        crate::numeric::eval_at(&value, "", 0.0)
            .ok_or_else(|| Error::msg("plot range bounds must be constant"))
    }

    fn display_defs_for_block(&self, block: usize) -> Vec<crate::substitute::Def> {
        if block == 0 {
            return self.defs.clone();
        }
        self.defs
            .iter()
            .filter(|def| def.block != block)
            .cloned()
            .collect()
    }

    fn output_subject(&self, expr: &Expr) -> OutputSubject {
        match expr {
            Expr::Ident(name) => OutputSubject {
                name: Some(name.clone()),
                header: name.clone(),
                lhs: self.display_lhs(name),
            },
            _ => self
                .component_output_subject(expr)
                .unwrap_or_else(|| OutputSubject {
                    name: None,
                    header: "<expr>".to_string(),
                    lhs: None,
                }),
        }
    }

    fn component_output_subject(&self, expr: &Expr) -> Option<OutputSubject> {
        let (name, indices) = flatten_component_subject(expr)?;
        let Value::Tensor(t) = self.env.get(name)? else {
            return None;
        };
        if indices.len() != t.order() {
            return None;
        }
        let mut subscript = String::new();
        let mut header = name.to_string();
        for idx in indices {
            let Expr::Num(n) = idx else {
                return None;
            };
            if n.fract() != 0.0 || *n < 1.0 || (*n as usize) > t.dim() {
                return None;
            }
            let k = *n as usize;
            header.push_str(&format!("[{k}]"));
            subscript.push_str(&k.to_string());
        }
        let lhs = self
            .display_lhs(name)
            .map(|base| format!("{{{base}}}_{{{subscript}}}"));
        Some(OutputSubject {
            name: None,
            header,
            lhs,
        })
    }

    fn simplify_presentation_value(value: Value) -> Value {
        match value {
            Value::Scalar(s) => Value::Scalar(simplify_scalar(&s, RuleSet::Tensor)),
            Value::Tensor(t) => Value::Tensor(simplify_tensor(&t, RuleSet::Tensor)),
            other => other,
        }
    }

    fn register_value_symbol(&mut self, name: &str, value: &Value, derived: bool) {
        let function_like = match value {
            Value::Scalar(s) => !self.free_fn_vars(s).is_empty(),
            Value::Tensor(_) => false,
            _ => false,
        };
        let kind = value_kind(value, function_like);
        let characteristic = match value {
            Value::Tensor(t) => Some(tensor_characteristic(t, derived)),
            _ => None,
        };
        let latex = self
            .display_label(name, value)
            .unwrap_or_else(|| name.to_string());
        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                name: name.to_string(),
                latex,
                display_modes: self.display_capabilities_for_value(value, &kind),
                kind,
                characteristic,
            },
        );
    }

    fn register_set_symbol(&mut self, name: &str, set: &SetDecl) {
        let kind = if set.vector {
            ValueKind::VectorSet { dim: set.dim }
        } else {
            ValueKind::ScalarSet { dim: set.dim }
        };
        self.symbols.insert(
            name.to_string(),
            SymbolInfo {
                name: name.to_string(),
                latex: set.latex.clone(),
                display_modes: display_capabilities_for_kind(&kind),
                kind,
                characteristic: None,
            },
        );
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

    fn expr_is_derived_display_result(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Ident(name) => self.derived_names.contains(name),
            Expr::Unary { expr, .. } => self.expr_is_derived_display_result(expr),
            Expr::Binary { lhs, rhs, .. } => {
                self.expr_is_derived_display_result(lhs) || self.expr_is_derived_display_result(rhs)
            }
            Expr::Field { target, .. } => self.expr_is_derived_display_result(target),
            Expr::Index { target, index } => {
                self.expr_is_derived_display_result(target)
                    || self.expr_is_derived_display_result(index)
            }
            Expr::MethodCall {
                target,
                args,
                kwargs,
                ..
            } => {
                self.expr_is_derived_display_result(target)
                    || args
                        .iter()
                        .any(|arg| self.expr_is_derived_display_result(arg))
                    || kwargs
                        .iter()
                        .any(|(_, value)| self.expr_is_derived_display_result(value))
            }
            Expr::Call {
                callee,
                args,
                kwargs,
            } => {
                callee == "Diff"
                    || args
                        .iter()
                        .any(|arg| self.expr_is_derived_display_result(arg))
                    || kwargs
                        .iter()
                        .any(|(_, value)| self.expr_is_derived_display_result(value))
            }
            Expr::List(items) => items
                .iter()
                .any(|item| self.expr_is_derived_display_result(item)),
            Expr::Num(_) | Expr::Str(_) | Expr::Bool(_) => false,
        }
    }

    fn render_display(
        &self,
        value: &Value,
        lhs: Option<&str>,
        mode: &str,
    ) -> Result<String, Error> {
        self.ensure_value_display_mode(value, mode)?;
        let lhs = lhs.map(|tex| format!("{tex} = ")).unwrap_or_default();
        match (value, mode) {
            (Value::Scalar(s), "symbol") => Ok(format!("{lhs}{}", scalar_to_latex(s))),
            (Value::Tensor(t), "symbol") => Ok(format!("{lhs}{}", tensor_to_latex(t))),
            (Value::Equation(eq), "symbol") => Ok(crate::ode::render_equation(eq)),
            (Value::BoundaryCondition(ic), "symbol") => {
                Ok(crate::ode::render_boundary_condition(ic))
            }
            (Value::OdeClassification(class), "symbol" | "details") => {
                Ok(crate::ode::render_classification(class, mode))
            }
            (Value::OdeSolution(sol), "symbol" | "solution" | "steps") => {
                Ok(crate::ode::render_solution(sol, mode))
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
            (Value::Tensor(t), "block_components") => {
                Ok(format!("{lhs}{}", tensor_to_block_component_matrix(t)?))
            }
            (
                Value::OdeProblem(problem),
                "symbol" | "equation" | "boundary" | "classification" | "methods",
            ) => crate::ode::render_problem(problem, mode),
            (Value::Tensor(_), other) => Err(Error::msg(format!("unknown display mode `{other}`"))),
            (Value::Scalar(_), other) => Err(Error::msg(format!("unknown display mode `{other}`"))),
            (_, other) => Err(Error::msg(format!("unknown display mode `{other}`"))),
        }
    }

    fn ensure_set_display_mode(&self, set: &SetDecl, mode: &str) -> Result<(), Error> {
        let kind = if set.vector {
            ValueKind::VectorSet { dim: set.dim }
        } else {
            ValueKind::ScalarSet { dim: set.dim }
        };
        self.ensure_display_capability(display_capability_for_kind(&kind, mode))
    }

    fn ensure_value_display_mode(&self, value: &Value, mode: &str) -> Result<(), Error> {
        if matches!(
            (value, mode),
            (Value::Tensor(t), "components" | "matrix")
                if matches!(&**t, TensorExpr::Diff { .. })
        ) {
            return Ok(());
        }
        let kind = value_kind(value, false);
        self.ensure_display_capability(display_capability_for_kind(&kind, mode))
    }

    fn display_capabilities_for_value(
        &self,
        value: &Value,
        kind: &ValueKind,
    ) -> Vec<crate::metadata::DisplayCapability> {
        let mut modes = display_capabilities_for_kind(kind);
        if matches!(value, Value::Tensor(t) if matches!(&**t, TensorExpr::Diff { .. })) {
            for mode in ["components", "matrix"] {
                if let Some(cap) = modes.iter_mut().find(|cap| cap.mode == mode) {
                    *cap = crate::metadata::DisplayCapability::available(mode);
                }
            }
        }
        modes
    }

    fn ensure_display_capability(
        &self,
        capability: crate::metadata::DisplayCapability,
    ) -> Result<(), Error> {
        match capability.state {
            DisplayCapabilityState::Available | DisplayCapabilityState::UnsupportedRenderer => {
                Ok(())
            }
            DisplayCapabilityState::InvalidForType => {
                Err(Error::msg(capability.message.unwrap_or_else(|| {
                    format!("mode={} is not available", capability.mode)
                })))
            }
        }
    }

    /// LaTeX for the left-hand side of `X.show(...)`, in order of
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

/// Structured ODE payload for `show(classification)` / `show(methods)`, so the
/// UI can render badges instead of the LaTeX `array`. `None` for everything
/// else (plain math).
fn ode_show_detail(value: &Value, mode: &str) -> Option<OutputDetail> {
    let Value::OdeProblem(problem) = value else {
        return None;
    };
    match mode {
        "classification" => crate::ode::classification_info(problem).ok().map(|info| {
            OutputDetail::OdeClassification {
                kind: info.kind,
                order: info.order,
                linear: info.linear,
                homogeneous: info.homogeneous,
            }
        }),
        "boundary" => Some(OutputDetail::OdeBoundary {
            boundary: crate::ode::boundary_info(problem),
        }),
        "methods" => crate::ode::methods_info(problem)
            .ok()
            .map(|info| OutputDetail::OdeMethods {
                available: info.available,
                default: info.default,
            }),
        _ => None,
    }
}

fn show_mode(args: &[Expr], kwargs: &[(String, Expr)]) -> Result<String, Error> {
    if !kwargs.is_empty() {
        return Err(Error::msg(
            "`.show(...)` takes an optional positional mode; use `A.show(matrix)`, not `mode=`",
        ));
    }
    match args {
        [] => Ok("symbol".to_string()),
        [Expr::Ident(mode)] | [Expr::Str(mode)] => Ok(mode.clone()),
        [_] => Err(Error::msg(
            "`.show(...)` mode must be a bare identifier such as `matrix` or `classification`",
        )),
        _ => Err(Error::msg(
            "`.show(...)` takes at most one mode, such as `symbol`, `matrix`, `classification`, or `methods`",
        )),
    }
}

fn show_header(subject: &str, mode: &str, implicit_symbol: bool) -> String {
    if implicit_symbol {
        format!("{subject}.show()")
    } else {
        format!("{subject}.show({mode})")
    }
}

fn ode_method_header(
    subject: &str,
    method: &str,
    args: &[Expr],
    kwargs: &[(String, Expr)],
) -> String {
    if args.is_empty() && kwargs.is_empty() {
        return format!("{subject}.{method}()");
    }
    if method == "solve" && args.is_empty() {
        let mut parts = Vec::new();
        for (key, value) in kwargs {
            match (key.as_str(), value) {
                ("details", Expr::Bool(value)) => parts.push(format!("details={value}")),
                ("method", Expr::Ident(name) | Expr::Str(name)) => {
                    parts.push(format!("method={name}"))
                }
                ("about", Expr::Num(value)) => parts.push(format!("about={value}")),
                ("terms", Expr::Num(value)) if value.fract() == 0.0 => {
                    parts.push(format!("terms={}", *value as usize))
                }
                _ => return format!("{subject}.{method}(...)"),
            }
        }
        if !parts.is_empty() {
            return format!("{subject}.solve({})", parts.join(", "));
        }
    }
    format!("{subject}.{method}(...)")
}

fn plot_header(target: &Expr, fallback: &str) -> String {
    match target {
        Expr::List(_) => "[...]",
        _ => fallback,
    }
    .to_string()
}

/// Is this the unknown function itself (e.g. `y(x)`, no derivative applied)?
/// Used to decide whether an ODE solution is explicit and therefore plottable.
fn is_bare_unknown(expr: &ScalarExpr) -> bool {
    matches!(
        expr,
        ScalarExpr::UnknownFunc {
            derivative_orders, ..
        } if derivative_orders.iter().all(|&order| order == 0)
    )
}

fn is_numeric_plot_candidate(expr: &ScalarExpr) -> bool {
    match expr {
        ScalarExpr::Num(_) | ScalarExpr::Sym { .. } => true,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => is_numeric_plot_candidate(a) && is_numeric_plot_candidate(b),
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) => is_numeric_plot_candidate(a),
        ScalarExpr::Func { arg, .. } => is_numeric_plot_candidate(arg),
        ScalarExpr::UnknownFunc { .. } | ScalarExpr::Integral { .. } => false,
        ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _)
        | ScalarExpr::SpecSum { .. }
        | ScalarExpr::SetElem { .. } => false,
    }
}

fn renamed_builtin_error(old: &str, new: &str) -> Error {
    Error::msg(format!("`{old}(...)` was renamed; use `{new}(...)`"))
}

fn sample_plot_series(
    expr: &Rc<ScalarExpr>,
    abscissa: &str,
    from: f64,
    to: f64,
) -> Result<PlotSeries, Error> {
    const SAMPLES: usize = 512;
    let mut segments = Vec::new();
    let mut current = Vec::new();
    for i in 0..SAMPLES {
        let t = i as f64 / (SAMPLES - 1) as f64;
        let x = from + (to - from) * t;
        if let Some(y) = crate::numeric::eval_at(expr, abscissa, x) {
            current.push([x, y]);
        } else if !current.is_empty() {
            if current.len() > 1 {
                segments.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
    }
    if current.len() > 1 {
        segments.push(current);
    }
    if segments.is_empty() {
        return Err(Error::msg("nothing to plot over this range"));
    }
    Ok(PlotSeries {
        label_latex: scalar_to_latex(expr),
        segments,
    })
}

fn robust_y_range(values: &[f64]) -> Result<[f64; 2], Error> {
    if values.is_empty() {
        return Err(Error::msg("nothing to plot over this range"));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let last = sorted.len() - 1;
    let low = sorted[last * 2 / 100];
    let high = sorted[last * 98 / 100];
    if (high - low).abs() < 1e-12 {
        return Ok([low - 1.0, high + 1.0]);
    }
    let pad = (high - low).abs() * 0.06;
    Ok([low - pad, high + pad])
}

fn split_asymptote_jumps(segments: Vec<Vec<[f64; 2]>>, y_range: [f64; 2]) -> Vec<Vec<[f64; 2]>> {
    let [y_min, y_max] = y_range;
    let mut split_segments = Vec::new();
    for segment in segments {
        let mut current: Vec<[f64; 2]> = Vec::new();
        for point in segment {
            if let Some(prev) = current.last().copied() {
                let jumps_across_band = ((prev[1] < y_min && point[1] > y_max)
                    || (prev[1] > y_max && point[1] < y_min))
                    && prev[1].signum() != point[1].signum();
                if jumps_across_band {
                    if current.len() > 1 {
                        split_segments.push(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                }
            }
            current.push(point);
        }
        if current.len() > 1 {
            split_segments.push(current);
        }
    }
    split_segments
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

fn is_raw_transpose_product_definition(expr: &Expr) -> bool {
    let Expr::Binary {
        op: BinOp::Mul,
        lhs,
        rhs,
    } = expr
    else {
        return false;
    };
    is_transpose_of(lhs, rhs) || is_transpose_of(rhs, lhs)
}

fn is_transpose_of(candidate: &Expr, base: &Expr) -> bool {
    matches!(
        candidate,
        Expr::Field { target, name } if name == "T" && target.as_ref() == base
    )
}

fn flatten_component_subject(expr: &Expr) -> Option<(&str, Vec<&Expr>)> {
    let mut indices = Vec::new();
    let mut base = expr;
    while let Expr::Index { target, index } = base {
        indices.push(index.as_ref());
        base = target;
    }
    indices.reverse();
    let Expr::Ident(name) = base else {
        return None;
    };
    (!indices.is_empty()).then_some((name.as_str(), indices))
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
        ScalarExpr::UnknownFunc { args, .. } => {
            args.iter().find_map(|arg| scalar_index_range(arg, idx))
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => scalar_index_range(integrand, idx).or_else(|| scalar_index_range(variable, idx)),
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

fn scalar_symbol_name(expr: &ScalarExpr) -> Option<&str> {
    match expr {
        ScalarExpr::Sym { name, .. } => Some(name),
        _ => None,
    }
}

fn scalar_contains_unknown_function(expr: &ScalarExpr) -> bool {
    match expr {
        ScalarExpr::UnknownFunc { .. } => true,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Pow(a, b) => {
            scalar_contains_unknown_function(a) || scalar_contains_unknown_function(b)
        }
        ScalarExpr::Neg(a) | ScalarExpr::Log(a) | ScalarExpr::Func { arg: a, .. } => {
            scalar_contains_unknown_function(a)
        }
        ScalarExpr::Integral {
            integrand,
            variable,
        } => {
            scalar_contains_unknown_function(integrand)
                || scalar_contains_unknown_function(variable)
        }
        ScalarExpr::SpecSum { body, .. } => scalar_contains_unknown_function(body),
        ScalarExpr::Sym { .. }
        | ScalarExpr::Num(_)
        | ScalarExpr::Det(_)
        | ScalarExpr::Tr(_)
        | ScalarExpr::Ddot(_, _)
        | ScalarExpr::SetElem { .. } => false,
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Scalar(_) => "Scalar",
        Value::Tensor(_) => "Tensor",
        Value::Equation(_) => "Equation",
        Value::BoundaryCondition(_) => "BoundaryCondition",
        Value::OdeProblem(_) => "ODE",
        Value::OdeClassification(_) => "OdeClassification",
        Value::OdeSolution(_) => "OdeSolution",
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

fn expect_solve_method(expr: &Expr) -> Result<SolveMethod, Error> {
    let name = match expr {
        Expr::Ident(name) | Expr::Str(name) => name.as_str(),
        _ => {
            return Err(Error::msg(
                "`method` must be one of auto, linear, separable, exact, characteristic, undetermined, variation, power_series, or frobenius",
            ))
        }
    };
    SolveMethod::parse(name).ok_or_else(|| {
        Error::msg(format!(
            "unknown ODE solve method `{name}`; expected auto, linear, separable, exact, characteristic, undetermined, variation, power_series, or frobenius"
        ))
    })
}
