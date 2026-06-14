//! User-visible symbol metadata and display capabilities.
//!
//! The evaluator already owns the mathematical truth in [`Value`] and
//! [`TensorExpr`]. This module is the compact, UI-facing summary of that truth:
//! what kind of value a name denotes, what tensor characteristics are known,
//! and which display modes are meaningful for it.

use crate::interpreter::Value;
use crate::tensor::TensorExpr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    Scalar,
    ScalarFunctionLike,
    Tensor { order: usize, dim: usize },
    ScalarSet { dim: usize },
    VectorSet { dim: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCharacteristic {
    pub order: usize,
    pub dim: usize,
    pub symmetric: bool,
    pub identity: bool,
    pub antisymmetric: bool,
    pub orthogonal: bool,
    pub isotropic: bool,
    pub filled: bool,
    pub derived: bool,
    pub derivative: bool,
    pub spectral: bool,
    pub sum: bool,
    pub outer_like: bool,
    pub boxtimes_like: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayCapabilityState {
    Available,
    UnsupportedRenderer,
    InvalidForType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayCapability {
    pub mode: String,
    pub state: DisplayCapabilityState,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolInfo {
    pub name: String,
    pub latex: String,
    pub kind: ValueKind,
    pub characteristic: Option<TensorCharacteristic>,
    pub display_modes: Vec<DisplayCapability>,
}

impl DisplayCapability {
    pub fn available(mode: &str) -> Self {
        Self {
            mode: mode.to_string(),
            state: DisplayCapabilityState::Available,
            message: None,
        }
    }

    pub fn invalid(mode: &str, message: impl Into<String>) -> Self {
        Self {
            mode: mode.to_string(),
            state: DisplayCapabilityState::InvalidForType,
            message: Some(message.into()),
        }
    }

    pub fn unsupported(mode: &str, message: impl Into<String>) -> Self {
        Self {
            mode: mode.to_string(),
            state: DisplayCapabilityState::UnsupportedRenderer,
            message: Some(message.into()),
        }
    }
}

pub fn value_kind(value: &Value, function_like: bool) -> ValueKind {
    match value {
        Value::Scalar(_) if function_like => ValueKind::ScalarFunctionLike,
        Value::Scalar(_) => ValueKind::Scalar,
        Value::Tensor(t) => ValueKind::Tensor {
            order: t.order(),
            dim: t.dim(),
        },
    }
}

pub fn tensor_characteristic(t: &TensorExpr, derived: bool) -> TensorCharacteristic {
    TensorCharacteristic {
        order: t.order(),
        dim: t.dim(),
        symmetric: t.is_symmetric(),
        identity: t.is_identity(),
        antisymmetric: tensor_declared_prop(t, |p| p.antisymmetric),
        orthogonal: tensor_declared_prop(t, |p| p.orthogonal),
        isotropic: tensor_declared_prop(t, |p| p.isotropic),
        filled: contains_filled(t),
        derived,
        derivative: contains_diff(t),
        spectral: contains_set_elem(t),
        sum: contains_sum(t),
        outer_like: contains_outer(t),
        boxtimes_like: contains_boxtimes(t),
    }
}

pub fn display_capabilities_for_kind(kind: &ValueKind) -> Vec<DisplayCapability> {
    const MODES: [&str; 4] = ["symbol", "components", "matrix", "block_components"];
    MODES
        .into_iter()
        .map(|mode| display_capability_for_kind(kind, mode))
        .collect()
}

pub fn display_capability_for_kind(kind: &ValueKind, mode: &str) -> DisplayCapability {
    match (kind, mode) {
        (ValueKind::Scalar | ValueKind::ScalarFunctionLike, "symbol")
        | (ValueKind::ScalarSet { .. } | ValueKind::VectorSet { .. }, "symbol")
        | (ValueKind::Tensor { .. }, "symbol") => DisplayCapability::available(mode),
        (ValueKind::Tensor { order: 1 | 2, .. }, "components" | "matrix") => {
            DisplayCapability::available(mode)
        }
        (ValueKind::Tensor { order: 4, .. }, "block_components") => {
            DisplayCapability::available(mode)
        }
        (ValueKind::Tensor { order, .. }, "components" | "matrix") => {
            DisplayCapability::invalid(
                mode,
                format!("mode={mode} is only available for order-1 or order-2 tensors; this tensor has order {order}"),
            )
        }
        (ValueKind::Tensor { order, .. }, "block_components") => {
            DisplayCapability::invalid(
                mode,
                format!("mode=block_components is only available for order-4 tensors; this tensor has order {order}"),
            )
        }
        (ValueKind::Scalar | ValueKind::ScalarFunctionLike, other) => {
            DisplayCapability::invalid(
                other,
                format!("mode={other} is not available for scalar values (supported: symbol)"),
            )
        }
        (ValueKind::ScalarSet { .. } | ValueKind::VectorSet { .. }, other) => {
            DisplayCapability::invalid(other, "set display only supports mode=symbol")
        }
        (_, other) => DisplayCapability::invalid(other, format!("unknown display mode `{other}`")),
    }
}

fn tensor_declared_prop(
    t: &TensorExpr,
    f: impl Fn(&crate::tensor::TensorProperties) -> bool,
) -> bool {
    match t {
        TensorExpr::Var { props, .. } => f(props),
        _ => false,
    }
}

fn contains_diff(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::Diff { .. } => true,
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } | TensorExpr::SetElem { .. } => false,
        TensorExpr::Filled { entries, .. } => entries.iter().any(|e| contains_diff_scalar(e)),
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::Power { base: a, .. } => contains_diff(a),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => contains_diff(a) || contains_diff(b),
        TensorExpr::SumIdx { body, .. } => contains_diff(body),
    }
}

fn contains_diff_scalar(s: &crate::symbolic::ScalarExpr) -> bool {
    match s {
        crate::symbolic::ScalarExpr::Det(t) | crate::symbolic::ScalarExpr::Tr(t) => {
            contains_diff(t)
        }
        crate::symbolic::ScalarExpr::Ddot(a, b) => contains_diff(a) || contains_diff(b),
        crate::symbolic::ScalarExpr::Add(a, b)
        | crate::symbolic::ScalarExpr::Sub(a, b)
        | crate::symbolic::ScalarExpr::Mul(a, b)
        | crate::symbolic::ScalarExpr::Div(a, b)
        | crate::symbolic::ScalarExpr::Pow(a, b) => {
            contains_diff_scalar(a) || contains_diff_scalar(b)
        }
        crate::symbolic::ScalarExpr::Neg(a)
        | crate::symbolic::ScalarExpr::Log(a)
        | crate::symbolic::ScalarExpr::Func { arg: a, .. }
        | crate::symbolic::ScalarExpr::SpecSum { body: a, .. } => contains_diff_scalar(a),
        crate::symbolic::ScalarExpr::Sym { .. }
        | crate::symbolic::ScalarExpr::Num(_)
        | crate::symbolic::ScalarExpr::SetElem { .. } => false,
    }
}

fn contains_set_elem(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::SetElem { .. } => true,
        TensorExpr::Var { .. } | TensorExpr::Identity4 { .. } => false,
        TensorExpr::Filled { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::Power { base: a, .. } => contains_set_elem(a),
        TensorExpr::ScalarMul(s, a) => contains_set_elem_scalar(s) || contains_set_elem(a),
        TensorExpr::Diff { num, den, .. } => contains_set_elem(num) || contains_set_elem(den),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => contains_set_elem(a) || contains_set_elem(b),
        TensorExpr::SumIdx { body, .. } => contains_set_elem(body),
    }
}

fn contains_set_elem_scalar(s: &crate::symbolic::ScalarExpr) -> bool {
    match s {
        crate::symbolic::ScalarExpr::SetElem { .. } => true,
        crate::symbolic::ScalarExpr::Sym { .. } | crate::symbolic::ScalarExpr::Num(_) => false,
        crate::symbolic::ScalarExpr::Add(a, b)
        | crate::symbolic::ScalarExpr::Sub(a, b)
        | crate::symbolic::ScalarExpr::Mul(a, b)
        | crate::symbolic::ScalarExpr::Div(a, b)
        | crate::symbolic::ScalarExpr::Pow(a, b) => {
            contains_set_elem_scalar(a) || contains_set_elem_scalar(b)
        }
        crate::symbolic::ScalarExpr::Neg(a)
        | crate::symbolic::ScalarExpr::Log(a)
        | crate::symbolic::ScalarExpr::Func { arg: a, .. }
        | crate::symbolic::ScalarExpr::SpecSum { body: a, .. } => contains_set_elem_scalar(a),
        crate::symbolic::ScalarExpr::Det(t) | crate::symbolic::ScalarExpr::Tr(t) => {
            contains_set_elem(t)
        }
        crate::symbolic::ScalarExpr::Ddot(a, b) => contains_set_elem(a) || contains_set_elem(b),
    }
}

fn contains_sum(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::SumIdx { .. } => true,
        TensorExpr::Var { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. }
        | TensorExpr::Filled { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::Power { base: a, .. } => contains_sum(a),
        TensorExpr::Diff { num, den, .. } => contains_sum(num) || contains_sum(den),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b)
        | TensorExpr::BoxTimes(a, b) => contains_sum(a) || contains_sum(b),
    }
}

fn contains_outer(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::Outer(..) => true,
        TensorExpr::Var { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. }
        | TensorExpr::Filled { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::Power { base: a, .. } => contains_outer(a),
        TensorExpr::Diff { num, den, .. } => contains_outer(num) || contains_outer(den),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::BoxTimes(a, b) => contains_outer(a) || contains_outer(b),
        TensorExpr::SumIdx { body, .. } => contains_outer(body),
    }
}

fn contains_boxtimes(t: &TensorExpr) -> bool {
    match t {
        TensorExpr::BoxTimes(..) => true,
        TensorExpr::Var { .. }
        | TensorExpr::Identity4 { .. }
        | TensorExpr::SetElem { .. }
        | TensorExpr::Filled { .. } => false,
        TensorExpr::Transpose(a)
        | TensorExpr::Inverse(a)
        | TensorExpr::InverseTranspose(a)
        | TensorExpr::Neg(a)
        | TensorExpr::ScalarMul(_, a)
        | TensorExpr::Power { base: a, .. } => contains_boxtimes(a),
        TensorExpr::Diff { num, den, .. } => contains_boxtimes(num) || contains_boxtimes(den),
        TensorExpr::MatMul(a, b)
        | TensorExpr::Add(a, b)
        | TensorExpr::Sub(a, b)
        | TensorExpr::Outer(a, b) => contains_boxtimes(a) || contains_boxtimes(b),
        TensorExpr::SumIdx { body, .. } => contains_boxtimes(body),
    }
}

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
