# Applied Math ODE Support

This document describes the ODE/PDE functionality currently implemented in
Tensbook. It is a capability reference, not a roadmap.

The runnable notebook example is `examples/applied-math-ode.tens`.

## Public DSL

| Command | Current behavior |
|---|---|
| `Var("x")` | Declares an independent scalar variable. |
| `Function("y", x)` | Declares a one-variable unknown function `y(x)`. |
| `Function("u", x, z)` | Declares a multi-variable unknown function for formal partial derivatives and PDE classification. |
| `Derivative(f, x, order=1)` | Constructs formal derivatives of unknown `Function(...)` objects. |
| `Diff(expr, x, order=1)` | Evaluates explicit scalar/tensor derivatives. It intentionally rejects unknown functions and tells the user to use `Derivative(...)`. |
| `Equation(lhs, rhs)` | Builds a scalar equation object and stores the residual `lhs - rhs`. |
| `BoundaryCondition(y(x0), y0)` | Declares a boundary/initial condition for supported solvers. |
| `ODE(eq, y, x)` | Builds an ODE/PDE problem object from an equation, target function, and independent variable. |
| `ODE(eq, y, x, BoundaryCondition(...))` | Builds the same problem with one supported boundary condition. |
| `ode.show()` / `ode.show(equation)` | Outputs the equation. |
| `ode.show(boundary)` | Outputs the boundary condition or `no explicit boundary condition`. |
| `ode.show(classification)` | Outputs ODE/PDE kind, order, linearity, and homogeneity. |
| `ode.show(methods)` | Outputs available symbolic solver methods and the default solve path. |
| `ode.solve()` | Outputs the final solution or an unsupported diagnostic. |
| `ode.solve(details=true)` | Outputs worked solution steps, warnings, and the final solution. |
| `ode.solve(method=auto)` | Uses the same default solver path as `ode.solve()`. |
| `ode.solve(method=linear\|separable\|exact)` | Requests a specific solver path and returns a clear diagnostic if unavailable. |
| `Integrate(expr, x)` | Applies conservative rule-based indefinite integration. |
| `Integral(expr, x)` | Builds an unevaluated formal integral. |

Old entry points are intentionally removed from the public DSL:

```text
IC(...)
ClassifyODE(eq, y, x)
SolveODE(eq, y, x, ic=...)
```

Use `BoundaryCondition(...)`, `ODE(...).show(classification)`, and `ODE(...).solve(...)`
instead.

## Formal Derivatives

`Derivative(...)` is for unknown functions:

```text
x = Var("x")
z = Var("z")
y = Function("y", x)
u = Function("u", x, z)

Derivative(y, x)
Derivative(y, x, order=2)
Derivative(u, x)
Derivative(u, z, order=2)
Derivative(Derivative(u, x), z, order=2)
```

One-variable derivatives render in Leibniz notation:

```text
dy/dx
d^2y/dx^2
```

Multi-variable derivatives render as partial derivatives. `Derivative(x^2, x)`
is rejected because explicit expressions should use `Diff(x^2, x)`.

## Classification

`ode.show(classification)` inspects the target unknown function and outputs:

- `ODE` vs `PDE`
- highest derivative order
- linear vs nonlinear
- homogeneous vs non-homogeneous by the zero-function test

Use `ode.show(methods)` for supported first-order solver methods:

- `linear`
- `separable`
- `exact`

Example:

```text
x = Var("x")
y = Function("y", x)

eq = Equation(Derivative(y, x) + sin(x), 0)
ode = ODE(eq, y, x)

ode.show(classification)
```

## Solvers

`ode.solve()` currently solves supported first-order ODEs only.

When multiple methods apply, `ode.solve()` uses the same default path shown by
`ode.show(methods)`. You can request a method explicitly:

```text
ode.solve(method=auto)
ode.solve(method=linear)
ode.solve(method=separable, details=true)
ode.solve(method=exact, details=true)
```

If the requested method is not available, Tensbook returns a diagnostic such
as `requested method exact is not available` and lists available methods.

### First-Order Linear

Equations equivalent to:

```text
dy/dx + p(x)y = q(x)
```

are solved by the integrating-factor method:

```text
mu = exp(Integrate(p(x), x))
y = (Integrate(mu*q(x), x) + C_1) / mu
```

Example:

```text
x = Var("x")
y = Function("y", x)

eq = Equation(Derivative(y, x) + 2*y, exp(x))
ode = ODE(eq, y, x, BoundaryCondition(y(0), 1))

ode.solve(details=true)
```

If a required integral is unsupported, the solution keeps a formal
`Integral(...)`.

### Separable

Tensbook recognizes forms such as:

```text
A(y) * dy/dx = B(x)
dy/dx = g(x) * h(y)
```

and returns an implicit equation.

Example:

```text
eq = Equation(3*y^2*Derivative(y, x), cos(x))
ode = ODE(eq, y, x)

ode.solve(method=separable, details=true)
ode.solve(method=separable)
```

This produces an implicit solution equivalent to:

```text
y^3 = sin(x) + C_1
```

### Exact

Tensbook recognizes first-order differential form:

```text
M(x,y) + N(x,y) * dy/dx = 0
```

It checks exactness by comparing:

```text
Diff(M, y) == Diff(N, x)
```

When exact, it constructs a potential function `phi(x,y) = C_1`.

Example:

```text
eq = Equation((2 + x^2*y)*Derivative(y, x) + x*y^2, 0)
ode = ODE(eq, y, x, BoundaryCondition(y(1), 2))
ode.solve(method=exact, details=true)
```

The tested solution includes the potential:

```text
2*y + 1/2*x^2*y^2 = 6
```

## Integration

`Integrate(expr, x)` is intentionally rule-based. When no rule applies,
Tensbook keeps a formal `Integral(...)` node instead of guessing.

Implemented rules include:

- constants independent of `x`: `Integrate(c, x) = c*x`
- powers: `Integrate(x^n, x) = x^(n+1)/(n+1)` for `n != -1`
- reciprocal: `Integrate(1/x, x) = log(x)`
- linearity over addition and subtraction
- constant-factor extraction
- affine elementary functions:
  - `exp(a*x + b)`
  - `sin(a*x + b)`
  - `cos(a*x + b)`
  - `sinh(a*x + b)`
  - `cosh(a*x + b)`
  - `tan(a*x + b)`, rendered through `-log(cos(...))`
- simple chain-rule products such as:
  - `Integrate(Diff(u, x) * exp(u), x)`
  - `Integrate(Diff(u, x) * sin(u), x)`
  - `Integrate(Diff(u, x) * cos(u), x)`
  - `Integrate(Diff(u, x) / u, x)`
  - `Integrate(Diff(u, x) * u^n, x)` for `n != -1`

Unsupported examples remain formal:

```text
Integrate(exp(-2/x)/x, x)
```

## Boundary Conditions

One boundary condition is supported when the constant can be eliminated through
simple symbolic substitution and linear extraction.

For explicit linear solutions, Tensbook substitutes `x = x0`, evaluates
`y(x0) = y0`, and solves for `C_1` when the expression is linear in `C_1`.

For implicit separable/exact solutions, Tensbook substitutes both `x0` and
`y0` into the implicit equation to determine the constant when possible.

If the constant cannot be eliminated safely, the solver keeps `C_1` and reports
a warning in `ode.solve(details=true)`.

## Unsupported Boundaries

Tensbook does not currently support:

- PDE solving
- second-order or higher-order ODE solving
- multiple boundary conditions
- general integrating-factor search for nonlinear non-exact equations
- numerical ODE solving
- broad CAS integration or special functions such as `Ei`
- arbitrary algebraic equation solving for boundary conditions

PDEs and higher-order ODEs can still be classified. Unsupported solver calls
return a diagnostic object instead of guessing.
