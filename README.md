# TensorForge

Rigorous symbolic tensor algebra for finite-deformation continuum mechanics,
driven by a small declarative `.tens` DSL.

![TensorForge default Hill-CR example](docs/tensorforge-default.png)

## Install

```sh
brew install Chongran-Zhao/tensorforge/tensorforge
```

## Update

```sh
brew update
brew upgrade tensorforge
```

## Functions

| Function | Use |
|---|---|
| `Scalar("\mu")` | Declare a symbolic scalar. |
| `Tensor("\bm F", order=2, dim=3, ...)` | Declare a tensor. Keyword args: `order`, `dim`, `identity`, `symmetric`, `antisymmetric`, `orthogonal`, `isotropic`. |
| `det(A)` | Symbolic determinant. |
| `tr(A)` | Symbolic trace. |
| `inv(A)` | Symbolic tensor inverse. |
| `log(x)`, `sqrt(x)`, `exp(x)` | Scalar functions; also tensor spectral functions for provably symmetric tensors. |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Symbolic scalar hyperbolic functions. |
| `outer(A, B)`, `otimes(A, B)` | Tensor product `A \otimes B`. |
| `dot(A, B)` | Single contraction / matrix product. |
| `ddot(A, B)` | Double contraction `A : B`. |
| `spectral(C)` | Spectral decomposition of a provably symmetric tensor. |
| `gstrain(C, scale=...)` | Generalized strain. Scales: `CR` with `m`, `n`; `SethHill` with `m`; `Hencky`. |
| `diff(expr, X)` | Symbolic derivative with respect to a scalar, tensor, compound tensor, or generalized strain. |
| `simplify(expr, rules=...)` | Exact rewriting. Rule sets: `algebra`, `tensor`, `continuum`. |
| `display(expr, mode=...)` | Render in the app/CLI. Modes: `symbol`, `components`, `matrix`, `block_components`, `spectral`. |
| `export(expr, format=...)` | Export `latex` or `markdown`. |

Operators: `+`, `-`, `*`, `/`, `^`, `A : B`, and `A.T`.

## Example

```text
mu = Scalar("\mu")
kappa = Scalar("\kappa")
m = Scalar("m")
n = Scalar("n")

F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
E = gstrain(C, scale=CR, m=m, n=n)
W = mu * ddot(E, E) + kappa/2 * tr(E)^2

T = diff(W, E)
S = 2 * diff(W, C)

display(E, mode=spectral)
display(S, mode=symbol)
display(S, mode=spectral)
```

License: MIT.
