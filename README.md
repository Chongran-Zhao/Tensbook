# TensorForge

**Rigorous symbolic tensor algebra for finite-deformation continuum
mechanics**, driven by a small declarative DSL (`.tens` files). TensorForge
derives stresses and tangents from energy functions the way you would on
paper — generalized strains, spectral representations, tensor-by-tensor
derivatives with Kronecker-delta index formulas — and renders everything as
LaTeX. It ships as a Rust library, a CLI, and a Tauri desktop app with live
preview.

The core principle is **strictness**: every rewrite and every inferred
property is an exact mathematical fact. Symmetry is inferred only when
provable (`C = FᵀF` ⇒ symmetric; a product of two symmetric tensors is
*not* assumed symmetric). Derivatives that would silently drop a hidden
dependence are rejected with an explanatory error instead of returning a
wrong zero. There is no numerical guessing anywhere in the engine.

---

## Quick start

Requires a Rust toolchain (`brew install rustup && rustup default stable`).

Install the CLI with Homebrew:

```sh
brew install Chongran-Zhao/tensorforge/tensorforge
```

Or run from source:

```sh
# run the flagship example
cargo run -p tensorforge -- run examples/hill_cr.tens

# run the test suite (74 tests)
cargo test -p tensorforge
```

Desktop app (requires `cargo install tauri-cli --version '^2'` once):

```sh
cargo tauri dev      # development window
cargo tauri build    # produces TensorForge.app + .dmg
```

The app shows a DSL editor on the left and KaTeX-rendered mathematics on
the right. It renders **live as you type** (statement-level error recovery:
a broken line shows one error card, everything else keeps rendering), has
ghost-text autocompletion with function-signature hints, copy-as-LaTeX /
copy-as-Markdown buttons on every output, light/dark themes, and native
file open/save. KaTeX is bundled — no network needed.

---

## The flagship example: Hill's model with the CR strain

`examples/hill_cr.tens` derives the constitutive law of Hill's class
hyperelasticity with the coercive Curnier–Rakotomanana strain — the
calibratable-strain-parameter setup used in finite-strain viscoelasticity
research:

```text
mu    = Scalar("\mu")
kappa = Scalar("\kappa")
m     = Scalar("m")
n     = Scalar("n")

F = Tensor("\bm F", order=2, dim=3)
C = F.T * F

# generalized Lagrangian strain  E(C) = Σ_a E(λ_a) M_a,
# CR scale function  E(λ) = (λ^m − λ^{−n})/(m + n)
E = gstrain(C, scale=CR, m=m, n=n)

# Hill's quadratic energy
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

T = diff(W, E)        # thermodynamic force
S = 2 * diff(W, C)    # second Piola–Kirchhoff stress

display(E, mode=spectral)
display(T, mode=symbol)
display(S, mode=symbol)
display(S, mode=spectral)
```

TensorForge derives, exactly:

```latex
% E in its spectral form (CR scale function applied to the stretches)
\bm E = \sum_{a=1}^{3} \frac{\lambda_a^m - \lambda_a^{-n}}{m + n} \, \bm M_a

% the thermodynamic force from the quadratic energy
\bm T = 2 \mu \, \bm E + \kappa \, \operatorname{tr}\bm E \, \bm I

% the stress in the T : Q structure (Q = 2 ∂E/∂C kept opaque)
\bm S = \bm T : \mathbb{Q}

% and fully expanded along the principal axes
\bm S = \sum_{a=1}^{3} \left[
  \frac{ \frac{m \lambda_a^{m-1} + n \lambda_a^{-n-1}}{m+n} }{\lambda_a}
  \left( 2\mu \frac{\lambda_a^m - \lambda_a^{-n}}{m+n}
       + \kappa \sum_{b=1}^{3} \frac{\lambda_b^m - \lambda_b^{-n}}{m+n} \right)
\right] \bm M_a
```

The chain rule `∂W/∂C = ∂W/∂E : ∂E/∂C` is applied automatically when the
energy depends on `C` only through a generalized strain; the spectral
expansion uses the coaxiality of `T` with `C` (guaranteed because `T` is an
isotropic function of `E`), under which the off-diagonal part of `Q` drops
out.

The classical material-frame derivation is also fully supported — see
`examples/neo_hookean.tens` for `P = ∂W/∂F`, the material tangent
`A = ∂P/∂F` with Kronecker-delta component formulas, and `block_components`
display of fourth-order tensors.

---

## DSL reference

### Declarations

| Syntax | Meaning |
|---|---|
| `mu = Scalar("\mu")` | symbolic scalar with LaTeX display |
| `F = Tensor("\bm F", order=2, dim=3)` | tensor variable; kwargs: `order`, `dim`, `identity`, `symmetric`, `antisymmetric`, `orthogonal`, `isotropic` |

Derived quantities need no declaration — `C = F.T * F` defines `C` with
inferred order, dimension, and (provable) symmetry. A label declared via
`Scalar`/`Tensor` survives reassignment: `I1 = Scalar("I_1")` followed by
`I1 = tr(C)` displays as `I_1 = …`.

### Operators

| Syntax | Meaning |
|---|---|
| `A * B` | tensor product (single contraction) / scalar multiplication |
| `A : B` | double contraction — 2:2 gives a scalar, 2:4 / 4:2 give order 2 |
| `A.T` | transpose |
| `+  -  /  ^` | usual arithmetic (Pratt-parsed precedence, `^` right-assoc.) |

### Functions

| Function | Result |
|---|---|
| `det(A)`, `tr(A)` | symbolic invariants |
| `inv(A)` | symbolic inverse (`A^{-1}`, `A^{-T}` canonicalized) |
| `log/sqrt/exp/sinh/cosh(x)` | scalar functions with exact derivative rules |
| `log/sqrt/exp(C)` | isotropic tensor functions via the spectral form (provably symmetric `C` only) |
| `outer(A,B)` / `otimes(A,B)` | tensor product `A ⊗ B` (orders add) |
| `dot(A,B)`, `ddot(A,B)` | single / double contraction |
| `spectral(C)` | `Σ_a c_a N_a ⊗ N_a` (provably symmetric only) |
| `gstrain(C, scale=…, …)` | generalized strain; `scale = CR (m,n) \| SethHill (m) \| Hencky` |
| `diff(expr, X)` | symbolic derivative — see below |
| `simplify(expr, rules=…)` | exact rewriting; `rules = algebra ⊂ tensor ⊂ continuum` |
| `display(expr, mode=…)` | `symbol \| components \| matrix \| block_components \| spectral` |
| `export(expr, format=…)` | `latex \| markdown` |

### Differentiation

`diff(expr, X)` accepts as denominator:

- a **scalar symbol**: `diff(W, mu)` — full scalar calculus;
- a **tensor variable**: `diff(W, F)` — e.g. the first Piola–Kirchhoff
  stress `P = ∂W/∂F` in closed form (Jacobi `∂det F/∂F = det F F^{-T}`,
  `∂log det F/∂F = F^{-T}`, `∂tr(FᵀF)/∂F = 2F`, chain/product/quotient
  rules);
- a **compound tensor**: `diff(W, C)` with `C = FᵀF` — `C` is treated as the
  independent variable and matched structurally; `det F` is rewritten
  exactly through `det C = (det F)²`; any dependence that cannot be
  expressed through `C` is an **error**, never a silent zero;
- a **generalized strain**: `diff(W, E)` — yields the thermodynamic force;
  `diff(W, C)` with `W = W(E(C))` chains automatically into `½ T : Q`.

Tensor-by-tensor derivatives (`diff(C, F)`, the tangent `A = diff(P, F)`)
are order-4 objects with the index convention `A_{iJkL} = ∂P_{iJ}/∂F_{kL}`;
`mode=components` prints the exact Kronecker-delta formula
(`∂C_{ij}/∂F_{mn} = δ_{in}F_{mj} + δ_{jn}F_{mi}`, inverse-component rule
`∂(F^{-1})_{ab}/∂F_{mn} = −F^{-1}_{am}F^{-1}_{nb}` included), and
`mode=block_components` expands the last two indices into a 3×3 grid of
second-order blocks — no premature Voigt compression.

### Display

`display` back-substitutes your definitions: once `C = F.T*F`, `J = det(F)`,
`I1 = tr(C)` are defined, later results render as `\bm C`, `J`, `I_1`
rather than expanded trees (presentation only — internals stay exact).
`mode=spectral` renders principal-axis sums and is exempt from
substitution.

---

## Architecture

```text
.tens source ──parser──▶ syntactic AST ──interpreter──▶ semantic values
                                            │
                             display/export └──renderer──▶ LaTeX / Markdown
```

| Module | Role |
|---|---|
| `src/parser/` | hand-written lexer + Pratt parser (zero dependencies) |
| `src/ast.rs` | syntactic AST — carries no mathematical meaning |
| `src/symbolic/` | scalar expressions: arithmetic, `det/tr/log/…`, named functions, eigenvalue symbols, spectral sums |
| `src/tensor/` | tensor object system, scale functions, conservative property inference |
| `src/differentiation.rs` | derivative rules: scalar/tensor, tensor/tensor, strain chain rule |
| `src/indices.rs` | abstract-index engine: expansion, δ generation, contraction, concrete instantiation |
| `src/simplifier.rs` | exact rewrite rules in cumulative sets algebra/tensor/continuum |
| `src/substitute.rs` | display-time back-substitution of definitions |
| `src/renderer/` | symbol / components / block / spectral LaTeX rendering |
| `src/interpreter.rs` | evaluation, environment, strict Scalar/Tensor typing |
| `src-tauri/` + `ui/` | Tauri desktop shell; static frontend, bundled KaTeX |
| `packaging/` | Homebrew formula template |

Two deliberate design choices: the syntactic and semantic layers are
separate (new mathematics never touches the parser), and every expression
node is a Rust enum variant — adding a node makes the compiler enumerate
every site that must decide how to handle it. There are no silent fallback
branches.

## Scope and non-goals (current)

Supported today: the full derivation pipeline above. Not yet in scope:
time derivatives / evolution equations / hereditary integrals (rate-form
viscoelasticity), curvilinear coordinates, FE code generation. The
Miehe–Lambrecht closed-form components of `Q` and sixth-order `L` are
represented opaquely, not expanded.

## Roadmap

1. Rate forms: `Ė`, evolution equations, exponential integrator algebra
2. Richer simplification: `det(F)F^{-T} → cof F`, user-declared identities
3. Desktop polish: proper app icon
4. Homebrew release: formula in `packaging/` once the repo is tagged

---

**Author**: Chongran Zhao ([chongran-zhao.site](https://chongran-zhao.site),
[chongran_zhao@brown.edu](mailto:chongran_zhao@brown.edu)) — Ph.D. student
in Engineering, Brown University; M.Eng. in Mechanics, SUSTech.
License: MIT.
