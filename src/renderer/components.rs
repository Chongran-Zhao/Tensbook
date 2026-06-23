//! Component-mode expansion: turn a second-order tensor expression into a
//! dim × dim matrix of scalar component expressions, rendered as a LaTeX
//! `bmatrix`.
//!
//! The MVP expands sums explicitly for a fixed dim (no abstract index
//! system yet): `(F^T F)_{ij} = Σ_k F_{ki} F_{kj}`.

use crate::error::Error;
use crate::renderer::latex::scalar_to_latex;
use crate::symbolic::ScalarExpr;
use crate::tensor::TensorExpr;
use std::rc::Rc;

/// Expand a second-order tensor expression into its components and render
/// them as a LaTeX `bmatrix`.
pub fn tensor_to_component_matrix(expr: &TensorExpr) -> Result<String, Error> {
    let entries = expand(expr)?;
    let mut rows = Vec::with_capacity(entries.len());
    for row_entries in &entries {
        let row: Vec<String> = row_entries.iter().map(scalar_to_latex).collect();
        rows.push(row.join(" & "));
    }
    Ok(format!(
        "\\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
        rows.join(" \\\\\n")
    ))
}

/// Render an order-4 tensor as a dim × dim matrix of second-order blocks.
/// The block columns/rows fix the last two indices `(m,n)`; each block leaves
/// `(i,j)` free, matching the existing derivative block display convention.
pub fn tensor_to_block_component_matrix(expr: &TensorExpr) -> Result<String, Error> {
    if expr.order() != 4 {
        return Err(Error::msg(format!(
            "block component expansion requires an order-4 tensor, got order {}",
            expr.order()
        )));
    }
    let dim = expr.dim();
    let mut rows = Vec::with_capacity(dim);
    for m in 1..=dim {
        let mut row = Vec::with_capacity(dim);
        for n in 1..=dim {
            let fixed = [("m", m), ("n", n)];
            let entry = fourth_component_latex(expr, "i", "j", "m", "n", &fixed)?;
            row.push(format!("\\left[ {entry} \\right]_{{ij}}"));
        }
        rows.push(row.join(" & "));
    }
    Ok(format!(
        "\\begin{{bmatrix}}\n{}\n\\end{{bmatrix}}",
        rows.join(" \\\\\n")
    ))
}

/// Component expression for entry (i, j) of a second-order tensor expression.
pub fn component(expr: &TensorExpr, i: usize, j: usize) -> Result<ScalarExpr, Error> {
    if expr.order() != 2 {
        return Err(Error::msg(format!(
            "component expansion requires a second-order tensor, got order {}",
            expr.order()
        )));
    }
    expandable(expr)?;
    Ok(entry(expr, i, j))
}

fn expand(expr: &TensorExpr) -> Result<Vec<Vec<ScalarExpr>>, Error> {
    if expr.order() != 2 {
        return Err(Error::msg(format!(
            "component expansion requires a second-order tensor, got order {}; \
             higher-order display (block_components) is not part of the MVP",
            expr.order()
        )));
    }
    expandable(expr)?;
    let dim = expr.dim();
    let mut entries = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut row = Vec::with_capacity(dim);
        for j in 0..dim {
            row.push(entry(expr, i, j));
        }
        entries.push(row);
    }
    Ok(entries)
}

/// Entry (i, j), 0-based, of a second-order tensor expression.
/// Inverse/Diff nodes cannot be expanded entrywise; callers route Diff
/// through the abstract-index engine instead, and `expand` rejects them
/// up front via [`expandable`].
fn entry(expr: &TensorExpr, i: usize, j: usize) -> ScalarExpr {
    match expr {
        TensorExpr::Var { latex, props, .. } => {
            if props.identity {
                ScalarExpr::Num(if i == j { 1.0 } else { 0.0 })
            } else {
                indexed_sym(latex, i, j)
            }
        }
        TensorExpr::Filled { dim, entries, .. } => (*entries[i * dim + j]).clone(),
        // (u ⊗ v)_{ij} = u_i v_j for component-filled vectors.
        TensorExpr::Outer(a, b) if a.order() == 1 && b.order() == 1 => {
            mul(entry1(a, i), entry1(b, j))
        }
        TensorExpr::Transpose(t) => entry(t, j, i),
        TensorExpr::Power { base, exp } => entry(&TensorExpr::expand_power(base, *exp), i, j),
        TensorExpr::MatMul(a, b) => {
            let dim = expr.dim();
            let mut sum: Option<ScalarExpr> = None;
            for k in 0..dim {
                let term = mul(entry(a, i, k), entry(b, k, j));
                sum = Some(match sum {
                    None => term,
                    Some(acc) => add(acc, term),
                });
            }
            sum.unwrap_or(ScalarExpr::Num(0.0))
        }
        TensorExpr::Add(a, b) => add(entry(a, i, j), entry(b, i, j)),
        TensorExpr::Sub(a, b) => {
            let lhs = entry(a, i, j);
            let rhs = entry(b, i, j);
            match (&lhs, &rhs) {
                (_, ScalarExpr::Num(n)) if *n == 0.0 => lhs,
                (ScalarExpr::Num(n), _) if *n == 0.0 => ScalarExpr::Neg(Rc::new(rhs)),
                _ => ScalarExpr::Sub(Rc::new(lhs), Rc::new(rhs)),
            }
        }
        TensorExpr::ScalarMul(s, t) => mul((**s).clone(), entry(t, i, j)),
        TensorExpr::Neg(t) => {
            let inner = entry(t, i, j);
            match inner {
                ScalarExpr::Num(n) => ScalarExpr::Num(-n),
                _ => ScalarExpr::Neg(Rc::new(inner)),
            }
        }
        // Rejected by `expandable` before this point.
        TensorExpr::Inverse(_)
        | TensorExpr::InverseTranspose(_)
        | TensorExpr::Identity4 { .. }
        | TensorExpr::Diff { .. }
        | TensorExpr::Outer(..)
        | TensorExpr::BoxTimes(..)
        | TensorExpr::SetElem { .. }
        | TensorExpr::SumIdx { .. } => {
            unreachable!("non-expandable node reached entry(); expandable() must screen first")
        }
    }
}

/// Check the whole tree for nodes that have no entrywise expansion.
fn expandable(expr: &TensorExpr) -> Result<(), Error> {
    match expr {
        TensorExpr::Inverse(_) | TensorExpr::InverseTranspose(_) => Err(Error::msg(
            "component expansion of tensor inverses is not supported in this phase; \
             use mode=symbol",
        )),
        TensorExpr::Diff { .. } => Err(Error::msg(
            "use dCdF.show(components) on a derivative variable; \
             nested derivative nodes inside expressions cannot be expanded",
        )),
        TensorExpr::Identity4 { .. } => Err(Error::msg(
            "component expansion of fourth-order identity tensors is not \
             supported in matrix mode; use mode=symbol",
        )),
        TensorExpr::Outer(a, b) if a.order() == 1 && b.order() == 1 => {
            expandable1(a).and_then(|_| expandable1(b))
        }
        TensorExpr::Outer(..) => Err(Error::msg(
            "component expansion of outer products is not supported yet; use mode=symbol",
        )),
        TensorExpr::BoxTimes(..) => Err(Error::msg(
            "component expansion of fourth-order ⊠ products is not supported \
             in matrix mode; use mode=symbol",
        )),
        TensorExpr::SetElem { .. } | TensorExpr::SumIdx { .. } => Err(Error::msg(
            "component expansion of set elements and spectral sums is not \
             supported in matrix mode; use mode=symbol",
        )),
        TensorExpr::Var { .. } => Ok(()),
        TensorExpr::Power { base, .. } => expandable(base),
        TensorExpr::Filled { order: 2, .. } => Ok(()),
        TensorExpr::Filled { .. } => Err(Error::msg(
            "matrix mode displays second-order tensors; this is not order 2",
        )),
        TensorExpr::Transpose(t) | TensorExpr::ScalarMul(_, t) | TensorExpr::Neg(t) => {
            expandable(t)
        }
        TensorExpr::MatMul(a, b) | TensorExpr::Add(a, b) | TensorExpr::Sub(a, b) => {
            expandable(a).and_then(|_| expandable(b))
        }
    }
}

/// Component `i` (0-based) of an order-1 component-filled vector.
fn entry1(expr: &TensorExpr, i: usize) -> ScalarExpr {
    match expr {
        TensorExpr::Filled {
            order: 1, entries, ..
        } => (*entries[i]).clone(),
        _ => unreachable!("expandable1() must screen first"),
    }
}

fn expandable1(expr: &TensorExpr) -> Result<(), Error> {
    match expr {
        TensorExpr::Filled { order: 1, .. } => Ok(()),
        _ => Err(Error::msg(
            "component expansion of vectors requires component-filled vectors \
             (e.g. from Spec_Decomp); use mode=symbol",
        )),
    }
}

/// Build the component symbol `X_{i+1,j+1}` from the tensor's LaTeX name,
/// stripping a leading `\bm ` so `\bm F` yields `F_{11}` rather than
/// `\bm F_{11}`.
fn indexed_sym(latex: &str, i: usize, j: usize) -> ScalarExpr {
    let base = latex
        .trim()
        .strip_prefix("\\bm ")
        .or_else(|| latex.trim().strip_prefix("\\bm"))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| latex.trim());
    ScalarExpr::Sym {
        name: format!("{base}_{}{}", i + 1, j + 1),
        latex: format!("{base}_{{{}{}}}", i + 1, j + 1),
    }
}

// Light constant folding so identity factors don't litter the output with
// `0 \, F_{11}` or `1 \, F_{11}` terms.
fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    match (&a, &b) {
        (ScalarExpr::Num(x), _) if *x == 0.0 => ScalarExpr::Num(0.0),
        (_, ScalarExpr::Num(x)) if *x == 0.0 => ScalarExpr::Num(0.0),
        (ScalarExpr::Num(x), _) if *x == 1.0 => b,
        (_, ScalarExpr::Num(x)) if *x == 1.0 => a,
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => ScalarExpr::Num(x * y),
        _ => ScalarExpr::Mul(Rc::new(a), Rc::new(b)),
    }
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    match (&a, &b) {
        (ScalarExpr::Num(x), _) if *x == 0.0 => b,
        (_, ScalarExpr::Num(x)) if *x == 0.0 => a,
        (ScalarExpr::Num(x), ScalarExpr::Num(y)) => ScalarExpr::Num(x + y),
        _ => ScalarExpr::Add(Rc::new(a), Rc::new(b)),
    }
}

fn fourth_component_latex(
    expr: &TensorExpr,
    i: &str,
    j: &str,
    m: &str,
    n: &str,
    fixed: &[(&str, usize)],
) -> Result<String, Error> {
    match expr {
        TensorExpr::Identity4 { .. } => Ok(format!(
            "{} \\, {}",
            delta_latex(i, m, fixed),
            delta_latex(j, n, fixed)
        )),
        TensorExpr::Diff {
            num,
            den,
            num_label,
        } => {
            let num_tex = num_label
                .clone()
                .unwrap_or_else(|| crate::renderer::latex::tensor_to_latex(num));
            Ok(format!(
                "\\frac{{\\partial {}_{{{i}{j}}}}}{{\\partial {}_{{{}{}}}}}",
                component_base_latex(&num_tex),
                component_base_latex(&crate::renderer::latex::tensor_to_latex(den)),
                instantiate_index(m, fixed),
                instantiate_index(n, fixed)
            ))
        }
        TensorExpr::Outer(a, b) if a.order() == 2 && b.order() == 2 => Ok(join_product([
            second_component_latex(a, i, j, fixed)?,
            second_component_latex(b, m, n, fixed)?,
        ])),
        TensorExpr::BoxTimes(a, b) => Ok(join_product([
            second_component_latex(a, i, m, fixed)?,
            second_component_latex(b, j, n, fixed)?,
        ])),
        TensorExpr::ScalarMul(s, t) => Ok(join_product([
            scalar_to_latex(s),
            paren_if_sum(fourth_component_latex(t, i, j, m, n, fixed)?),
        ])),
        TensorExpr::Add(a, b) => Ok(format!(
            "{} + {}",
            fourth_component_latex(a, i, j, m, n, fixed)?,
            fourth_component_latex(b, i, j, m, n, fixed)?
        )),
        TensorExpr::Sub(a, b) => Ok(format!(
            "{} - {}",
            fourth_component_latex(a, i, j, m, n, fixed)?,
            paren_if_sum(fourth_component_latex(b, i, j, m, n, fixed)?)
        )),
        TensorExpr::Neg(t) => Ok(format!(
            "-{}",
            paren_if_sum(fourth_component_latex(t, i, j, m, n, fixed)?)
        )),
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
            Ok(format!(
                "\\sum_{{{lower}}}^{{{range}}} {}",
                fourth_component_latex(body, i, j, m, n, fixed)?
            ))
        }
        TensorExpr::Var { latex, .. } | TensorExpr::Filled { latex, .. } => {
            Ok(component_symbol_latex(latex, &[i, j, m, n], fixed))
        }
        TensorExpr::SetElem { latex, index, .. } => {
            Ok(format!("{{{latex}}}_{{{}{i}{j}{m}{n}}}", index.latex()))
        }
        other => Err(Error::msg(format!(
            "mode=block_components is valid for order-4 tensors, but this \
             expression structure is not supported yet: {}",
            crate::renderer::latex::tensor_to_latex(other)
        ))),
    }
}

fn second_component_latex(
    expr: &TensorExpr,
    i: &str,
    j: &str,
    fixed: &[(&str, usize)],
) -> Result<String, Error> {
    match expr {
        TensorExpr::Var { latex, props, .. } => {
            if props.identity {
                Ok(delta_latex(i, j, fixed))
            } else {
                Ok(component_symbol_latex(latex, &[i, j], fixed))
            }
        }
        TensorExpr::Filled {
            latex,
            entries,
            dim,
            ..
        } => {
            let ii = fixed_value(i, fixed);
            let jj = fixed_value(j, fixed);
            if let (Some(ii), Some(jj)) = (ii, jj) {
                Ok(scalar_to_latex(&entries[(ii - 1) * dim + (jj - 1)]))
            } else {
                Ok(component_symbol_latex(latex, &[i, j], fixed))
            }
        }
        TensorExpr::Outer(a, b) if a.order() == 1 && b.order() == 1 => Ok(join_product([
            first_component_latex(a, i, fixed)?,
            first_component_latex(b, j, fixed)?,
        ])),
        TensorExpr::Transpose(t) => second_component_latex(t, j, i, fixed),
        TensorExpr::ScalarMul(s, t) => Ok(join_product([
            scalar_to_latex(s),
            paren_if_sum(second_component_latex(t, i, j, fixed)?),
        ])),
        TensorExpr::Add(a, b) => Ok(format!(
            "{} + {}",
            second_component_latex(a, i, j, fixed)?,
            second_component_latex(b, i, j, fixed)?
        )),
        TensorExpr::Sub(a, b) => Ok(format!(
            "{} - {}",
            second_component_latex(a, i, j, fixed)?,
            paren_if_sum(second_component_latex(b, i, j, fixed)?)
        )),
        TensorExpr::Neg(t) => Ok(format!(
            "-{}",
            paren_if_sum(second_component_latex(t, i, j, fixed)?)
        )),
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
            Ok(format!(
                "\\sum_{{{lower}}}^{{{range}}} {}",
                second_component_latex(body, i, j, fixed)?
            ))
        }
        TensorExpr::SetElem { latex, index, .. } => Ok(format!(
            "{{{latex}}}_{{{}{}}}",
            index.latex(),
            join_indices(&[i, j], fixed)
        )),
        other => Ok(component_symbol_latex(
            &crate::renderer::latex::tensor_to_latex(other),
            &[i, j],
            fixed,
        )),
    }
}

fn first_component_latex(
    expr: &TensorExpr,
    i: &str,
    fixed: &[(&str, usize)],
) -> Result<String, Error> {
    match expr {
        TensorExpr::SetElem { latex, index, .. } => Ok(format!(
            "{{{latex}}}_{{{}{}}}",
            index.latex(),
            instantiate_index(i, fixed)
        )),
        TensorExpr::Filled { latex, entries, .. } => match fixed_value(i, fixed) {
            Some(ii) => Ok(scalar_to_latex(&entries[ii - 1])),
            None => Ok(component_symbol_latex(latex, &[i], fixed)),
        },
        TensorExpr::Var { latex, .. } => Ok(component_symbol_latex(latex, &[i], fixed)),
        other => Ok(component_symbol_latex(
            &crate::renderer::latex::tensor_to_latex(other),
            &[i],
            fixed,
        )),
    }
}

fn join_product<const N: usize>(parts: [String; N]) -> String {
    parts
        .into_iter()
        .filter(|p| p != "1")
        .collect::<Vec<_>>()
        .join(" \\, ")
}

fn paren_if_sum(tex: String) -> String {
    if tex.contains(" + ") || tex.contains(" - ") {
        format!("\\left( {tex} \\right)")
    } else {
        tex
    }
}

fn delta_latex(a: &str, b: &str, fixed: &[(&str, usize)]) -> String {
    format!(
        "\\delta_{{{}{}}}",
        instantiate_index(a, fixed),
        instantiate_index(b, fixed)
    )
}

fn component_symbol_latex(latex: &str, indices: &[&str], fixed: &[(&str, usize)]) -> String {
    format!(
        "{{{}}}_{{{}}}",
        component_base_latex(latex),
        join_indices(indices, fixed)
    )
}

fn component_base_latex(latex: &str) -> String {
    latex
        .trim()
        .strip_prefix("\\bm ")
        .or_else(|| latex.trim().strip_prefix("\\bm"))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| latex.trim())
        .to_string()
}

fn join_indices(indices: &[&str], fixed: &[(&str, usize)]) -> String {
    indices
        .iter()
        .map(|idx| instantiate_index(idx, fixed))
        .collect::<Vec<_>>()
        .join("")
}

fn instantiate_index(idx: &str, fixed: &[(&str, usize)]) -> String {
    fixed_value(idx, fixed)
        .map(|v| v.to_string())
        .unwrap_or_else(|| idx.to_string())
}

fn fixed_value(idx: &str, fixed: &[(&str, usize)]) -> Option<usize> {
    fixed
        .iter()
        .find_map(|(name, value)| (*name == idx).then_some(*value))
}
