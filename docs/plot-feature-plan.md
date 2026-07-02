# Plot feature for `.tens` (1-D) — implementation spec for codex

Add a 1-D plotting feature to TensorForge. The engine is **purely symbolic** today —
there is no numeric evaluation of `ScalarExpr`, and no `pi`/`e`. This feature adds a
numeric channel: sample the expression in Rust → ship points through the existing
`Output.detail` payload → draw a themed SVG in the front end.

## Decisions (already made — do not re-litigate)
- **API**: a `.plot(from, to)` **output method** (like `.show()` / `.solve()`), not a
  `Plot(...)` builtin. The abscissa is **inferred** from the single free variable.
- **Range**: explicit `from, to` required. **Add `pi` and `e` constants** so
  `sin(x).plot(-pi, pi)` works.
- **Rendering**: hand-rolled **static themed SVG** (zero dependency, light/dark via
  CSS vars). No charting library, no hover/zoom.
- **Scope (do all of it)**: single curve **+** multiple overlaid curves **+** plotting a
  solved ODE's explicit `y(x)`.
- **Copy button**: plot blocks show **no "copy latex" button**.
- Do **not** break the 157 passing tests; keep every existing LaTeX string identical;
  `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` clean for both the
  core crate and `src-tauri`; `node --check ui/*.js` clean.

## Data flow
```
sin(x).plot(-pi, pi)
  → interpreter: infer abscissa, eval range, sample N points (numeric eval)
  → Output { latex: <formula>, detail: Some(Plot{ series, x_range, y_range, x_label }) }
  → src-tauri RunDetail::Plot   (serde, "type":"plot")
  → ui renderPlot(detail): themed SVG  (reuse the renderOutputBlock detail dispatch)
```
The live preview uses `detail` (SVG); the `latex` field stays the formula so
export/PDF degrades to showing the formula (charts in export are out of scope for now).

---

## A. `pi` / `e` constants
- Represent each as a plain symbol: `pi` → `ScalarExpr::Sym { name: "pi", latex: "\\pi" }`,
  `e` → `Sym { name: "e", latex: "e" }`. They are **not** `Var`s (must not enter `fn_vars`),
  so they never count as the plot abscissa.
- Resolve them in the interpreter's identifier lookup: when a bare identifier `pi`/`e`
  is evaluated and the user has not shadowed it (env / scalars / vars), return the symbol
  above. Keep it minimal — do not add a new `ScalarExpr` variant; the numeric evaluator
  (Section B) special-cases the two names.
- They render as `\pi` / `e` and stay symbolic everywhere else (no folding).

## B. Numeric evaluator — new file `src/numeric.rs`
`ScalarExpr` variants are in [`src/ast.rs`](../src/ast.rs); a `Var("x")` is
`Sym { name: "x", latex: "x" }` (see [`builtin_var`](../src/interpreter.rs#L1454),
tracked in `fn_vars`).

```rust
/// Evaluate `expr` with the abscissa `var` bound to `x`. `None` = not numerically
/// evaluable at this point (unknown symbol, unsupported node, or NaN/±Inf result —
/// the caller treats `None` as a gap in the curve).
pub fn eval_at(expr: &ScalarExpr, var: &str, x: f64) -> Option<f64>;
```
Match arms:
- `Num(n)` → `n`. `Neg(a)` → `-eval`. `Add/Sub/Mul/Div(a,b)` → obvious. `Pow(a,b)` →
  `a.powf(b)`. `Log(a)` → `ln` (the DSL `log` is natural log; cf. `Integrate(1/x)=\log x`).
- `Sym { name, .. }`: `name == var` → `x`; `name == "pi"` → `PI`; `name == "e"` → `E`;
  otherwise `None` (unbound).
- `Func { name, arg }`: `sin/cos/tan/exp/sqrt/sinh/cosh/tanh` → the matching `f64` fn of
  `eval_at(arg)`; any other name → `None`.
- `UnknownFunc | Integral | Det | Tr | Ddot | SpecSum | SetElem` → `None`.
- After computing, `if !v.is_finite() { return None }` (asymptotes, `sqrt(neg)`, `log(≤0)`,
  `0^neg`, etc. become gaps).

Helpers (same file):
```rust
/// All distinct Sym names referenced by `expr`.
pub fn symbol_names(expr: &ScalarExpr) -> Vec<String>;
/// Names that are neither the abscissa nor a known constant (pi/e). Non-empty ⇒
/// the expression is not a concrete function of one variable.
pub fn unbound_symbols(expr: &ScalarExpr, abscissa: &str) -> Vec<String>;
```
Unit tests: `eval_at(sin(x),0)=0`, `x^2` at 3 = 9, `exp(0)=1`, `1/x` at 0 → `None`,
`pi`/`e` resolve, `sqrt(-1)` → `None`, `unbound_symbols(a*x, "x") == ["a"]`.

## C. Output payload (purely additive)
Core enum [`OutputDetail`](../src/interpreter.rs#L38) — add a variant:
```rust
OutputDetail::Plot(PlotData)

pub struct PlotData {
    pub x_label: String,                 // abscissa latex, e.g. "x"
    pub x_range: [f64; 2],
    pub y_range: [f64; 2],
    pub series: Vec<PlotSeries>,
}
pub struct PlotSeries {
    pub label_latex: String,             // the curve's expression latex (legend / title)
    pub segments: Vec<Vec<[f64; 2]>>,    // each segment is a gap-free run of (x,y)
}
```
Serde mirror in [`src-tauri/src/main.rs`](../src-tauri/src/main.rs) `RunDetail` (follow the
existing pattern — explicit `#[serde(rename = "plot")]` so the JSON tag is `"plot"`; the
Rust variant name must avoid the `enum_variant_names` clippy lint that bit `RunDetail`
before). Mirror `PlotData`/`PlotSeries` as serde structs and extend `convert_detail`.
Front-end shape:
```json
{ "type":"plot", "x_label":"x", "x_range":[-3.14,3.14], "y_range":[-1,1],
  "series":[ { "label_latex":"\\sin x", "segments":[ [[x,y],...], ... ] } ] }
```

## D. Interpreter — `.plot()` as an output statement
Hook into the statement dispatch next to `.show()` / `.solve()`
([interpreter.rs#L293-L312](../src/interpreter.rs#L301)): add
`... if method == "plot" => self.exec_plot(target, args, *line, *block, None)`.
Also add `"plot"` to the expression-context guard at
[interpreter.rs#L651](../src/interpreter.rs#L651) so `y = f.plot(...)` errors like `.show`.
And add a `plot` arm to the output-row path so `[a.plot(...), b.plot(...)]` works like the
`.show()` row (optional but cheap; keep it consistent).

`exec_plot(target, args, line, block, row)`:
1. **Curves** from `target` (an `Expr`):
   - `Expr::List(elems)` (Section E) → evaluate each element to `Value::Scalar` → one
     series per element.
   - `Value::OdeSolution` → take `solution.solution` (an `Option<Equation>`); error if
     `None` ("no closed-form solution to plot"); use the **rhs**; error if the rhs contains
     an `Integral`/`UnknownFunc` ("solution is not in closed form; cannot plot").
   - otherwise evaluate to `Value::Scalar` → one series.
2. **Abscissa**: `free_fn_vars` over all curve expressions
   ([interpreter.rs#L1908](../src/interpreter.rs#L1908)). Exactly one shared free `Var` ⇒
   abscissa. Zero ⇒ error ("plot needs a variable; declare one with `Var(\"x\")`").
   More than one ⇒ error listing them. Also run `unbound_symbols` per curve and error
   naming the first offender (e.g. "`a` is unbound; plot needs a concrete function").
3. **Range**: exactly two positional args `from`, `to`; evaluate each with
   `numeric::eval_at(arg_expr, "", _)` against a 0-length var (constants only — `-pi`,
   `3.14`, `pi/2` are fine; anything referencing the abscissa errors "range bounds must be
   constant"). Require `from < to` and both finite.
4. **Sample** (Section G) → `PlotSeries.segments` + `y_range`. `x_range = [from, to]`.
5. Push an `Output` with `header = ode/show-style header` (e.g. `sin(x).plot`),
   `latex = <joined curve latex>` (so export shows the formula), and
   `detail = Some(OutputDetail::Plot(..))`.

## E. Parser — list receiver for multiple curves
There is **no list-literal rvalue** today: a statement-leading `[` is an output row
([parser/mod.rs#L321](../src/parser/mod.rs#L321)) or destructuring
([#L267](../src/parser/mod.rs#L267)); `[ ]` is also postfix indexing
([#L387](../src/parser/mod.rs#L387)). So `[sin(x), cos(x)].plot(...)` currently mis-parses
as an output row.

Add a **lookahead disambiguation** at statement start: a `[ … ]` that is **immediately
followed by `.`** (a method call on the bracketed group) is parsed as an
`Expr::List(Vec<Expr>)` primary, then normal postfix `.plot(...)` applies. Keep the
existing behaviour when `]` is followed by `=` (destructuring) or end-of-row (output row).
Add `Expr::List(Vec<Expr>)` to [`ast.rs`](../src/ast.rs); only `.plot()` consumes it (any
other use of a bare list can error "list literals are only valid as a `.plot()` target").
Add a parser test for each of the three `[`-forms so they stay unambiguous.

## F. Front end — SVG renderer
In [`ui/main.js`](../ui/main.js):
- **Hide copy on plots**: in `renderOutputBlock`, when `detail?.type === "plot"`, build the
  head with the `[header]` label only — **do not** append the copy-bar — then append
  `renderPlot(detail)` and add class `plot-output` to the block; return.
- `renderPlot(detail)` returns an `<svg>` (namespaced, `viewBox="0 0 W H"`,
  `width:100%`, height ~260px). Compute pixel transforms from `x_range`/`y_range` with an
  inner margin for ticks. Draw, in order:
  1. plot-area background + 1px frame (`var(--border)`);
  2. ~5 gridlines each axis (`var(--border)`, faint) with tick labels
     (`var(--muted)`, ~10.5px mono, rounded to ~3 sig figs);
  3. zero axes (x=0 / y=0) if within range, slightly stronger than gridlines;
  4. each series: one `<path>` per segment (no `M` across gaps), `fill:none`,
     `stroke: var(--accent)` for the first curve; for ≥2 curves use a small fixed palette
     (accent + e.g. `--note-accent` + one more) and draw a top-right legend keyed by the
     series `label_latex` (render with KaTeX, inline mode, `KATEX_MACROS`).
- All colors via CSS vars so light/dark just work. Keep the existing `[header]` row.

CSS in [`ui/index.html`](../ui/index.html) next to the `.ode-*` block:
`.block.plot-output { width: auto; }`,
`.tens-render-pane > .block.plot-output { width: auto; max-width: 100%; }`,
`.tf-plot { width: 100%; }`, plus tick/legend text helpers (reuse `--muted`/mono).

## G. Sampling, asymptotes, y-range
- Sample `N = 512` x in `[from, to]`. For each, `eval_at` → push `[x,y]` to the current
  segment, or close the segment on `None` (start a new one on the next finite value). Drop
  empty/singleton segments.
- If **all** samples are `None` → error ("nothing to plot over this range").
- **y-range (robust)**: collect finite `y`; take the 2nd–98th percentile to ignore a few
  asymptotic spikes (so `tan(x)` over `[-pi,pi]` is readable), then pad by ~6%. If the
  robust band is degenerate (constant function), use `±1` around the value. Also split a
  segment when consecutive points jump across the clipped band on a sign change (so an
  asymptote does not draw a near-vertical connector).

## H. Edge cases / errors (clear messages)
- abscissa: 0 free vars → declare one; >1 → name them.
- unbound non-abscissa symbol (`a*x`) → name it.
- range not constant / `from >= to` / non-finite → specific message.
- ODE solution: no closed form, or contains `∫` / unknown function → "cannot plot".
- multiple curves with different free variables → error.

## I. Tests + acceptance
- **Rust unit** (`src/numeric.rs`): the evaluator cases in Section B.
- **Rust integration** (`tests/`): a `plot_*` test asserting the structured detail —
  `sin(x).plot(-pi, pi)` yields one series, `x_range≈[-π,π]`, a sample near `(0,0)`, a
  reasonable `y_range≈[-1,1]`; a multi-curve `[sin(x),cos(x)].plot(...)` yields two series;
  a solved-ODE `.plot()` yields a series; an unbound-symbol plot returns `Err`.
- **No regressions**: full `cargo test` still green (157 + new); existing LaTeX assertions
  untouched.
- **Front-end**: verify in the static preview by injecting a real `type:"plot"` JSON into
  the output path (see how the ODE detail was verified) — confirm axes/curve/legend render
  and there is **no copy button** on plot blocks, in light and dark.
- clippy `-D warnings` (core **and** `src-tauri`), `cargo fmt --check`, `node --check`
  all clean.

## Suggested commit slicing
1. `pi`/`e` + `src/numeric.rs` (+ unit tests).
2. `OutputDetail::Plot` + `RunDetail` serde + `exec_plot` single-curve + sampling.
3. Front-end `renderPlot` + CSS + hide-copy (+ preview verification).
4. Parser `Expr::List` + multi-curve.
5. ODE-solution `.plot()`.
