# TensorForge

A symbolic tensor algebra system for continuum mechanics, driven by a small
declarative DSL (`.tens` files). The long-term goal is rigorous, verifiable
symbolic derivation for finite-deformation mechanics — second/fourth-order
tensors, tensor-by-tensor differentiation, index contraction checking, and
LaTeX/Markdown output — packaged as a desktop app distributed via Homebrew.

**Current status: Phase 6** — the symbolic engine (library + CLI) is
feature-complete for the core spec: `.tens` parsing, conservative property
inference, symbolic differentiation up to the fourth-order material tangent,
exact simplification rules, tensor products, spectral decomposition, and
LaTeX/Markdown rendering including `block_components`. A Tauri desktop shell
(DSL editor + KaTeX preview) lives in `src-tauri/` + `ui/`.

## Quick start

Requires a Rust toolchain (`brew install rustup && rustup default stable`).

Run the CLI on a `.tens` file:

```sh
cargo run -p tensorforge -- run examples/neo_hookean.tens
```

Run the test suite:

```sh
cargo test -p tensorforge
```

Run the desktop app (requires the [Tauri CLI](https://v2.tauri.app/reference/cli/):
`cargo install tauri-cli --version '^2'`):

```sh
cargo tauri dev    # development window
cargo tauri build  # produces TensorForge.app + .dmg under src-tauri/target
```

The app shows a DSL editor on the left and KaTeX-rendered mathematics on the
right (KaTeX is bundled — no network needed); press ⌘⏎ / Ctrl+Enter to run,
⌘O to open and ⌘S to save `.tens` files.

## Example

`examples/neo_hookean.tens`:

```text
mu = Scalar("\mu")
lambda = Scalar("\lambda")

F = Tensor("\bm F", order=2, dim=3)
I = Tensor("\bm I", order=2, dim=3, identity=true)

C = F.T * F
J = det(F)
I1 = tr(C)

W = mu/2 * (I1 - 3) - mu * log(J) + lambda/2 * log(J)^2

dCdF = diff(C, F)
dJdF = diff(J, F)
P = diff(W, F)
A = diff(P, F)

display(C, mode=symbol)
display(C, mode=components)
display(dCdF, mode=components)
display(dJdF, mode=symbol)
display(P, mode=symbol)
display(A, mode=components)
display(A, mode=block_components)
export(C, format=latex)
```

Output (LaTeX, printed to stdout) includes:

```text
[display C, mode=symbol]
\bm C = \bm F^{\mathsf{T}} \bm F

[display dCdF, mode=components]
\frac{\partial C_{ij}}{\partial F_{mn}} = \delta_{in} F_{mj} + \delta_{jn} F_{mi}

[display dJdF, mode=symbol]
dJdF = \det \bm F \, \bm F^{-\mathsf{T}}

[display P, mode=symbol]
\bm P = \mu \, \bm F - \mu \, \bm F^{-\mathsf{T}} + \lambda \, \log \left( \det \bm F \right) \, \bm F^{-\mathsf{T}}

[display A, mode=components]
\frac{\partial P_{ij}}{\partial F_{mn}} = \mu \delta_{im} \delta_{jn} + \mu F^{-1}_{jm} F^{-1}_{ni} - \lambda \, \log \left( \det \bm F \right) F^{-1}_{jm} F^{-1}_{ni} + \lambda F^{-1}_{ji} F^{-1}_{nm}
```

plus a 3 × 3 `bmatrix` of second-order blocks for
`display(A, mode=block_components)`.

`C = F.T * F` is automatically inferred to be a **symmetric** second-order
tensor — inference is strictly conservative: only properties that are
provable from the expression structure are inferred (e.g. the product of two
symmetric tensors is *not* assumed symmetric).

## Feature set

- `.tens` parsing: assignments, `Scalar(...)` / `Tensor(...)` declarations
  with `order`, `dim`, `identity`, `symmetric`, ... properties
- Expression evaluation with strict Scalar/Tensor typing and shape checking
- `F.T` transpose, `*` (tensor product / scalar multiplication), `+`, `-`,
  `/`, `^`
- `det(F)`, `tr(C)`, `log(J)` as opaque symbolic nodes
- Conservative symmetry inference (`F.T * F` ⇒ symmetric)
- **Differentiation** (`diff`):
  - scalar-by-tensor, evaluated to closed form: `∂(det F)/∂F = det(F) F^{-T}`
    (Jacobi), `∂ log(det F)/∂F = F^{-T}`, `∂ tr(FᵀF)/∂F = 2F`,
    `∂tr(AX)/∂X = Aᵀ`, sum/product/quotient/power/chain rules — so
    `P = diff(W, F)` yields the first Piola–Kirchhoff stress directly;
  - tensor-by-tensor: `diff(C, F)` is a fourth-order tensor (index
    convention `A_{iJkL} = ∂P_{iJ}/∂F_{kL}`); component display generates
    Kronecker-delta formulas via an abstract-index engine, including
    inverse-component rules (`∂(F⁻¹)_{ab}/∂F_{mn} = −F⁻¹_{am} F⁻¹_{nb}`),
    so the material tangent `A = diff(P, F)` works end to end
- **Simplification** (`simplify(expr)` / `simplify(expr, rules=...)`):
  cumulative rule sets `algebra ⊂ tensor ⊂ continuum`, every rule an exact
  identity — `(Aᵀ)ᵀ → A`, `F⁻¹F → I`, `F^{-T}Fᵀ → I`, `A I → A`,
  `Cᵀ → C` for provably symmetric `C`, `det(Fᵀ) → det F`,
  `det(AB) → det A det B`, `tr(I) → dim`, `log(xⁿ) → n log x`, and cyclic
  canonicalization so `tr(AB)` and `tr(BA)` compare equal
- `inv(F)` — symbolic tensor inverse
- **Tensor products**: `outer(A, B)` → `A ⊗ B` (order = sum of orders),
  `dot(A, B)` → single contraction `AB`, `ddot(A, B)` → double contraction
  `A : B` (a scalar)
- **Spectral decomposition**: `spectral(C)` →
  `Σ_{a=1}^{3} c_a N_a ⊗ N_a`, accepted only for *provably* symmetric
  tensors (declared `symmetric=true` or inferred, e.g. `C = F.T * F`)
- **Isotropic tensor functions**: `sqrt(C)` / `log(C)` / `exp(C)` through
  the spectral form, `f(C) = Σ f(c_a) N_a ⊗ N_a` — e.g. `U = sqrt(C)` is
  the right stretch tensor; same strict symmetry requirement
- `display(X, mode=symbol)` — bold symbolic LaTeX (`\bm C = ...`,
  `\frac{\partial \bm C}{\partial \bm F}`)
- `display(X, mode=components)` — explicit `dim × dim` component matrix, or
  the delta formula for derivative nodes
- `display(A, mode=block_components)` — fourth-order derivative expanded
  over its last two indices `(k, L)` into a `dim × dim` grid of
  second-order blocks (no premature Voigt compression)
- `export(X, format=latex)` and `export(X, format=markdown)` ($$-wrapped)

## Architecture

```text
.tens source ──parser──▶ syntactic AST ──interpreter──▶ semantic values
                                            │
                             display/export └──renderer──▶ LaTeX
```

| Module            | Role                                                       |
|-------------------|------------------------------------------------------------|
| `src/parser/`     | Hand-written lexer + Pratt parser for the DSL              |
| `src/ast.rs`      | Syntactic AST — no mathematical meaning                    |
| `src/symbolic/`   | Symbolic scalar expressions (`Add`, `Log`, `Det`, `Tr`, …) |
| `src/tensor/`     | Tensor object system + conservative property inference     |
| `src/differentiation.rs` | Symbolic derivative rules (scalar/tensor, tensor/tensor) |
| `src/indices.rs`  | Abstract-index engine: expansion, δ generation, contraction |
| `src/simplifier.rs` | Exact rewrite rules: algebra / tensor / continuum sets   |
| `src/interpreter.rs` | Evaluation, environment, Scalar/Tensor type checking    |
| `src/renderer/`   | LaTeX symbol-mode and component-mode rendering             |
| `src-tauri/`      | Tauri desktop shell (`run_tens` IPC command)               |
| `ui/`             | Static frontend: editor + KaTeX rendering (no Node build)  |
| `packaging/`      | Homebrew formula template                                  |

The two-layer design (syntactic AST vs. semantic `ScalarExpr`/`TensorExpr`)
is deliberate: differentiation and simplification rules operate on the
semantic layer without touching the parser.

## Roadmap

1. **Richer simplification** — `det(F) F^{-T} → cof F`, user-declared
   identities entering the rule system as known facts
2. **Desktop polish** — proper app icon (current one is a placeholder)
3. **Homebrew distribution** — publish the formula in `packaging/` once the
   repo has a tagged release; add a cask for the app bundle
