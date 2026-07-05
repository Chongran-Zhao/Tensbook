# The Tensbook language (1.1)

The complete reference for the `.tens` DSL: declarations, operations,
display modes, ODE/plot syntax, and the precise scope of the current
release. The bundled feature tour (`examples/start.tens`) covers the same
ground as a runnable notebook.

## Notebook format

The same notebook can mix prose, equations, and executable derivations:

````markdown
The right Cauchy-Green tensor is computed from the deformation gradient:

<!-- tensorforge:tens -->
F = Tensor("\bm F", order=2, dim=3)
C = F.T * F
C.show()
<!-- /tensorforge:tens -->
````

The sentinel comments are the saved `.tens` file format. The editor presents
them as Tensbook code blocks.

## ODE Example

```text
x = Var("x")
y = Function("y", x)

eq = Equation(Derivative(y, x) + 2*y, exp(x))
problem = ODE(eq, y, x, BoundaryCondition(y(0), 1))

problem.show(classification)
problem.show(methods)
problem.solve(details=true)
```

When more than one method applies, choose the path explicitly:

```text
problem.solve(method=linear)
problem.solve(method=separable, details=true)
problem.solve(method=exact, details=true)
```

Second-order teaching examples are supported for the Lecture 03-05 path:

```text
ode = ODE(Equation(Derivative(y, x, order=2) + 5*Derivative(y, x) + 6*y, 0), y, x)
ode.solve(method=characteristic, details=true)

series = ODE(Equation(Derivative(y, x, order=2) + x*Derivative(y, x) + y, 0), y, x)
series.solve(method=power_series, about=0, terms=6, details=true)
```

## Plot Example

```text
x = Var("x")

sin(x).plot(-pi, pi)
[sin(x), cos(x)].plot(-pi, pi)
```

Explicit ODE solutions can also be plotted when the right-hand side is closed
form and numeric over the requested range.

## The language at a glance

### Declarations

| Syntax | Meaning |
|---|---|
| `Scalar("\mu")` | Symbolic scalar. |
| `Var("x")` | Independent scalar variable. |
| `Function("y", x)` | Unknown scalar function `y(x)`. |
| `Function("u", x, y, t)` | Multi-variable unknown function for formal partials/PDE classification. |
| `Tensor("\bm F", order=2, dim=3)` | Tensor declaration. Keyword args include `identity`, `symmetric`, `antisymmetric`, `orthogonal`, `isotropic`. |
| `ScalarSet("\lambda", dim=3)` | Indexed scalar family. |
| `VectorSet("\bm N", dim=3)` | Indexed vector family. |
| `[lambda, N] = Spectral(C, "\lambda", "\bm N")` | Paired symbolic eigenvalue/eigenvector sets. |
| `[c, N] = Spec_Decomp(C)` | Component-based spectral decomposition for diagonal second-order tensors. |
| `F[1][1] = expr` | Tensor component assignment. Unset entries are zero. |

### Algebra And Tensor Operations

| Syntax | Meaning |
|---|---|
| `A * B` | Scalar multiplication, tensor scaling, or second-order single contraction. |
| `A : B` | Standard double contraction. |
| `A & B` | Tensor product `A \otimes B`. |
| `A.T` | Transpose. |
| `Det(A)`, `Tr(A)`, `Inv(A)` | Determinant, trace, inverse. |
| `Sum(expr, a)` | Sum over an indexed set element. |
| `Simplify(expr, rules=algebra)` | Conservative rewriting. Rule sets: `algebra`, `tensor`, `continuum`. |

### Scalar Functions And Calculus

| Syntax | Meaning |
|---|---|
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric scalar functions. |
| `exp(x)`, `log(x)`, `sqrt(x)` | Elementary scalar functions. |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | Hyperbolic scalar functions. |
| `Diff(expr, X, order=1)` | Evaluated derivative of explicit scalar/tensor expressions. |
| `Derivative(f, x, order=1)` | Formal derivative or partial derivative of unknown `Function(...)` objects. |
| `Integrate(expr, x)` | Rule-based scalar integration. |
| `Integral(expr, x)` | Unevaluated formal integral. |

### Equations, ODEs, And Plots

| Syntax | Meaning |
|---|---|
| `Equation(lhs, rhs)` | Scalar equation object. |
| `BoundaryCondition(y(x0), y0)` | Boundary or initial condition. |
| `BoundaryCondition(Derivative(y, x), x0, y0)` | Derivative boundary condition such as `y'(x0)=y0`. |
| `ODE(eq, y, x, ...)` | ODE/PDE problem object. Boundary conditions are optional and repeatable. |
| `ode.show()` / `ode.show(equation)` | Display the equation. |
| `ode.show(boundary)` | Display boundary-condition status. |
| `ode.show(classification)` | Display type, order, linearity, and homogeneity. |
| `ode.show(methods)` | Display available symbolic solve paths. |
| `ode.solve(details=true)` | Display worked symbolic solution steps. |
| `ode.solve(method=...)` | Select `auto`, `linear`, `separable`, `exact`, `characteristic`, `undetermined`, `variation`, `power_series`, or `frobenius`. |
| `expr.plot(from, to)` | Plot a scalar expression. |
| `[expr1, expr2].plot(from, to)` | Plot multiple scalar curves together. |

### Display

| Syntax | Meaning |
|---|---|
| `expr.show()` | Symbol display mode. |
| `expr.show(matrix)` | Matrix display where supported. |
| `expr.show(components)` | Component display where supported. |
| `expr.show(block_components)` | Order-4 block component display for supported derivative tensors. |
| `[A.show(), B.show()]` | Side-by-side preview row. |
| `ViewSource on` / `ViewSource off` | Notebook directive for source/preview comparison blocks. |

## Current Scope

Tensbook is deliberately narrower than a general-purpose CAS. It is strict about
the mathematical objects it knows how to manipulate, and explicit about what it
does not support.

Current 1.1 boundaries:

- The DSL is the source of truth. Tensbook does not parse arbitrary LaTeX,
  Python, Mathematica, or Maple expressions.
- `Matrix` and `Vector` are not separate public constructors yet. Use typed
  tensors and indexed sets.
- General `contract(A, B, indices=...)` is not implemented. Use `*`, `:`, `&`,
  and indexed `Sum(...)` for supported contractions.
- `.show(matrix|components)` focuses on order-1/order-2 tensors and explicit
  diagonal/spectral cases.
- `.show(block_components)` is for supported order-4 derivative structures.
- Integration is rule-based. Unsupported antiderivatives remain as
  `Integral(...)` instead of being guessed.
- ODE support covers first-order linear, separable, and exact equations, plus a
  teaching-oriented set of second-order linear methods: characteristic roots,
  undetermined coefficients, variation of parameters, power series, and
  Frobenius/Bessel-style indicial output.
- PDEs can be represented and classified, but not solved.
- Plotting is one-dimensional and scalar-valued. Tensor-valued expressions,
  implicit ODE solutions, and formal integrals are rejected with diagnostics.
- Spectral decomposition support is intentionally conservative. `Spec_Decomp(C)`
  currently requires a diagonal component-filled second-order tensor in the
  working basis.
- Continuum-mechanics differentiation covers the tested paths for scalar/tensor
  derivatives, spectral strain derivatives, compound tensor denominators, and
  fourth-order tangents. Fully general tensor-chain rules and eigenvector
  derivatives are still rejected when unsupported.
