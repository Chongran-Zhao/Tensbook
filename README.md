# TensorForge

Rigorous symbolic tensor algebra for finite-deformation continuum mechanics,
driven by a small declarative `.tens` DSL.

![TensorForge feature tour](docs/tensorforge-default.png)

## Install

```sh
brew install --cask Chongran-Zhao/tensorforge/tensorforge
```

Homebrew requires the full `owner/tap/cask` form for untapped casks.
`brew install --cask Chongran-Zhao/tensorforge` is only a tap reference, not an
installable app cask.

## Update

```sh
brew update
brew upgrade --cask tensorforge
```

## Functions

| Function | Use |
|---|---|
| `Scalar("\mu")` | Declare a symbolic scalar. |
| `Tensor("\bm F", order=2, dim=3, ...)` | Declare a tensor. Keyword args: `order`, `dim`, `identity`, `symmetric`, `antisymmetric`, `orthogonal`, `isotropic`. |
| `det(A)` | Symbolic determinant. |
| `tr(A)` | Symbolic trace. |
| `inv(A)` | Symbolic tensor inverse. |
| `log(x)`, `sqrt(x)`, `exp(x)` | Symbolic scalar functions. |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Symbolic scalar hyperbolic functions. |
| `A & B` | Tensor product `A \otimes B`. |
| `dot(A, B)` | Single contraction / matrix product. |
| `ddot(A, B)` | Double contraction `A : B`. |
| `ScalarSet("\lambda", dim=3)`, `VectorSet("\bm N", dim=3)` | Indexed families; elements `lambda[a]`, `N[1]`. |
| `sum(expr, a)` | Symbolic sum over a set index. |
| `F[1][1] = expr` | Assign tensor components (unset entries are zero). |
| `[c, N] = Spec_Decomp(C)` | Symbolic eigendecomposition of a diagonal component-filled tensor. |
| `diff(expr, X)` | Symbolic derivative with respect to a scalar, tensor, compound tensor, or generalized strain. |
| `simplify(expr, rules=...)` | Exact rewriting. Rule sets: `algebra`, `tensor`, `continuum`. |
| `display(expr, mode=...)` | Render in the app. Modes: `symbol`, `components`, `matrix`, `block_components`. |
| `export(expr, format=...)` | Export `latex` or `markdown`. |

Operators: `+`, `-`, `*`, `/`, `^`, `A & B`, `A : B`, and `A.T`.

## Example

```text
mu = Scalar("\mu")
kappa = Scalar("\kappa")
m = Scalar("m")
n = Scalar("n")

F = Tensor("\bm F", order=2, dim=3)
lambda = ScalarSet("\lambda", dim=3)
N = VectorSet("\bm N", dim=3)

lam = Var("\lambda")
Ecr = (lam^m - lam^(-n))/(m + n)

C = sum(lambda[a]^2 * N[a] & N[a], a)
E = sum(Ecr(lambda[a]) * N[a] & N[a], a)
Q = 2 * diff(E, C)
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

T = diff(W, E)
S = T : Q

display(E, mode=symbol)
display(Q, mode=symbol)
display(S, mode=symbol)
```

License: MIT.
