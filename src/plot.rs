//! Plot sampling for `.plot(from, to)`.
//!
//! The interpreter resolves *what* to plot (curves, abscissa, range); this
//! module turns one scalar expression into drawable polyline segments and
//! picks a robust y-range. Gaps (unbound values, NaN/±Inf) split a curve into
//! segments so the front end never draws through an asymptote.

use crate::ast::Expr;
use crate::error::Error;
use crate::renderer::latex::scalar_to_latex;
use crate::symbolic::ScalarExpr;
use std::rc::Rc;

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

pub(crate) fn plot_header(target: &Expr, fallback: &str) -> String {
    match target {
        Expr::List(_) => "[...]",
        _ => fallback,
    }
    .to_string()
}

/// Is this the unknown function itself (e.g. `y(x)`, no derivative applied)?
/// Used to decide whether an ODE solution is explicit and therefore plottable.
pub(crate) fn is_bare_unknown(expr: &ScalarExpr) -> bool {
    matches!(
        expr,
        ScalarExpr::UnknownFunc {
            derivative_orders, ..
        } if derivative_orders.iter().all(|&order| order == 0)
    )
}

pub(crate) fn is_numeric_plot_candidate(expr: &ScalarExpr) -> bool {
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

pub(crate) fn sample_plot_series(
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

/// 2nd–98th percentile band plus 6% padding, so a few asymptotic spikes do not
/// flatten the whole plot; a degenerate (constant) band widens to ±1.
pub(crate) fn robust_y_range(values: &[f64]) -> Result<[f64; 2], Error> {
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

/// Split a segment when consecutive samples jump across the whole clipped band
/// with a sign change — an asymptote crossing that would otherwise draw a
/// near-vertical connector.
pub(crate) fn split_asymptote_jumps(
    segments: Vec<Vec<[f64; 2]>>,
    y_range: [f64; 2],
) -> Vec<Vec<[f64; 2]>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(name: &str) -> Rc<ScalarExpr> {
        Rc::new(ScalarExpr::sym(name, name))
    }

    #[test]
    fn robust_range_pads_and_survives_spikes() {
        // 100 samples of y=x plus two huge asymptotic spikes: the band must
        // stay near [0, 99], not stretch to ±1e9.
        let mut values: Vec<f64> = (0..100).map(f64::from).collect();
        values.push(1e9);
        values.push(-1e9);
        let [lo, hi] = robust_y_range(&values).unwrap();
        assert!(lo < 0.0 && lo > -20.0, "lo = {lo}");
        assert!(hi > 99.0 && hi < 120.0, "hi = {hi}");
    }

    #[test]
    fn robust_range_widens_constant_functions() {
        let [lo, hi] = robust_y_range(&[2.0; 32]).unwrap();
        assert_eq!([lo, hi], [1.0, 3.0]);
    }

    #[test]
    fn asymptote_jump_splits_segment() {
        // tan-like: rises to +big, reappears at -big → must split into two.
        let segment = vec![[0.0, 1.0], [0.1, 50.0], [0.2, -50.0], [0.3, -1.0]];
        let out = split_asymptote_jumps(vec![segment], [-10.0, 10.0]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[1].len(), 2);
    }

    #[test]
    fn same_sign_swing_is_not_split() {
        // A steep but same-sign swing inside one branch stays connected.
        let segment = vec![[0.0, 12.0], [0.1, 11.0], [0.2, 12.5]];
        let out = split_asymptote_jumps(vec![segment], [-10.0, 10.0]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn sampling_splits_at_gaps() {
        // sqrt(x) over [-1, 1]: the negative half is a gap, one segment remains.
        let expr = Rc::new(ScalarExpr::Func {
            name: "sqrt".to_string(),
            arg: sym("x"),
        });
        let series = sample_plot_series(&expr, "x", -1.0, 1.0).unwrap();
        assert_eq!(series.segments.len(), 1);
        assert!(series.segments[0].iter().all(|p| p[0] >= 0.0));
    }

    #[test]
    fn bare_unknown_detects_explicit_solutions() {
        let bare = ScalarExpr::UnknownFunc {
            name: "y".to_string(),
            args: vec![sym("x")],
            derivative_orders: vec![0],
        };
        let derived = ScalarExpr::UnknownFunc {
            name: "y".to_string(),
            args: vec![sym("x")],
            derivative_orders: vec![1],
        };
        assert!(is_bare_unknown(&bare));
        assert!(!is_bare_unknown(&derived));
    }
}
