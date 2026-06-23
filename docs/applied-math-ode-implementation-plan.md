# Applied Math ODE Implementation Plan

This plan is based on the notes in:

`/Users/chongran/Library/Mobile Documents/com~apple~CloudDocs/Literature/Paper-Notes/Stanford-Engineering-Applied-Mathmatics`

The current note coverage is:

| Note | Implementable math content |
|---|---|
| `Lecture01.tens` | ODE/PDE classification, order, linear/nonlinear, homogeneous/nonhomogeneous, linear first-order ODEs, integrating factors, separable ODE roadmap. |
| `Lecture02.tens` | First-order nonlinear ODEs, separable vs not separable, exact ODEs, potential functions, initial conditions. |
| `Lecture03.tens` | Linear second-order ODEs, constant-coefficient homogeneous cases, characteristic equation, Wronskian, nonhomogeneous roadmap. |
| `Lecture04.tens` | Title/reference only for power series. Treat as future scope until the note content is filled in. |

## Goal

Add applied-math symbolic ODE support without turning TensorForge into a broad CAS.

The first milestone should let a user write, classify, solve, and display the same first-order ODE workflows that appear in Lecture 01-02:

```text
x = Var("x")
y = Function("y", x)

eq = Equation(Derivative(y, x) + p(x) * y, q(x))
ClassifyODE(eq, y, x).show()
SolveODE(eq, y, x).show()
```

The solver must show exact symbolic steps when it knows the method. When it reaches an unsupported integral or unsupported ODE class, it should keep a formal object instead of guessing.

## Public DSL

Add these public commands:

| Command | Purpose |
|---|---|
| `Equation(lhs, rhs)` | Full equation object. This should be preferred over encoding equations as `lhs - rhs = 0`. |
| `ClassifyODE(eq, y, x)` | Return a structured classification result. |
| `SolveODE(eq, y, x, ic=...)` | Solve an ODE when a supported method applies. |
| `Integrate(expr, x)` | Symbolic indefinite integral with formal fallback. |
| `Integral(expr, x)` | Unevaluated integral object used when `Integrate` cannot evaluate. |
| `IC(y(x0), y0)` | Initial condition object for ODE solvers. |

Keep existing commands:

| Command | Role in ODE work |
|---|---|
| `Var("x")` | Independent variable. |
| `Function("y", x)` | Unknown dependent function. |
| `Derivative(f, x, order=1)` | Formal derivative and partial derivative constructor for unknown functions. |
| `Diff(expr, x, order=1)` | Evaluated derivative for explicit scalar/tensor expressions. |
| `Simplify(expr, rules=...)` | Algebraic normalization used by classification and solver checks. |
| `expr.show(...)` | Display classification, derivation, and solution objects. |

Use `Derivative(...)` for ODE/PDE unknown functions and `Diff(...)` for explicit expressions. This keeps formal dependency notation separate from evaluated symbolic differentiation.

## Internal Model

Add equation and ODE-specific symbolic types:

| Type | Needed fields |
|---|---|
| `Equation` | `lhs`, `rhs`, source span, normalized residual `lhs - rhs`. |
| `FormalDerivative` | base function, variable list, order per variable, display form. |
| `FormalIntegral` | integrand, variable, optional assumptions, optional method tag. |
| `OdeClassification` | kind, order, linearity, homogeneity, first-order subtype, reason strings. |
| `OdeSolution` | equation, method, steps, final expression, constants, warnings. |

Formal derivatives should support:

```text
Derivative(y, x)
Derivative(y, x, order=2)
Derivative(u, x)
Derivative(u, y)
Derivative(Derivative(u, x), y)
```

For explicit expressions, `Diff` should continue evaluating normally. For unknown functions, `Derivative` should produce formal derivative objects that classification can inspect.

## Milestone 1: Classification

Implement `ClassifyODE(eq, y, x)` first. This gives the rest of the system a reliable method dispatcher.

Classification checks:

| Check | Rule |
|---|---|
| ODE vs PDE | ODE if the unknown function depends on one independent variable; PDE if it depends on multiple independent variables or the equation contains partial derivatives in multiple variables. |
| Order | Highest derivative order of the target unknown function. |
| Linear vs nonlinear | Linear if `y`, `Derivative(y,x)`, ..., `Derivative(y,x,order=n)` appear only to first power and are not multiplied together or placed inside nonlinear scalar functions. |
| Homogeneous vs nonhomogeneous | For linear ODEs, homogeneous if substituting `y=0` and all derivatives of `y` to zero makes the residual zero. |
| First-order linear | Normalize to `a1(x) y' + a0(x) y = b(x)`, with `a1 != 0`. |
| Separable | Detect `y' = g(x) h(y)` or equivalent product-separated forms. |
| Exact | Detect differential form `M(x,y) dx + N(x,y) dy = 0` or normalized `M(x,y) + N(x,y) y' = 0`, then check `Diff(M,y) == Diff(N,x)`. |

Return a structured object, not just a string:

```text
ClassifyODE(eq, y, x).show()
ClassifyODE(eq, y, x).show(details)
```

Example classification output should match the Lecture 01 vocabulary:

```text
ODE, 2nd-order, nonlinear, homogeneous
```

## Milestone 2: Integrator Foundation

Implement `Integrate(expr, x)` before the first solver, because even the linear first-order method depends on it.

The integrator should be intentionally rule-based. If no rule applies, return `Integral(expr, x)`.

Required rules:

| Pattern | Result |
|---|---|
| `Integrate(c, x)` where `c` is independent of `x` | `c * x` |
| `Integrate(x^n, x)`, `n != -1` | `x^(n+1)/(n+1)` |
| `Integrate(1/x, x)` | `log(x)` |
| `Integrate(exp(a*x + b), x)` | `exp(a*x + b)/a` |
| `Integrate(sin(a*x + b), x)` | `-cos(a*x + b)/a` |
| `Integrate(cos(a*x + b), x)` | `sin(a*x + b)/a` |
| `Integrate(sinh(a*x + b), x)` | `cosh(a*x + b)/a` |
| `Integrate(cosh(a*x + b), x)` | `sinh(a*x + b)/a` |
| `Integrate(a*f + b*g, x)` | Linearity when `a`, `b` are independent of `x`. |

Add simple chain-rule/substitution recognition:

| Pattern | Result |
|---|---|
| `Integrate(Diff(u,x) * exp(u), x)` | `exp(u)` |
| `Integrate(Diff(u,x) * sin(u), x)` | `-cos(u)` |
| `Integrate(Diff(u,x) * cos(u), x)` | `sin(u)` |
| `Integrate(Diff(u,x) / u, x)` | `log(u)` |
| `Integrate(Diff(u,x) * u^n, x)`, `n != -1` | `u^(n+1)/(n+1)` |

Support low-risk algebraic preprocessing:

- Pull out factors independent of the integration variable.
- Expand sums only when this helps linearity.
- Simplify after each successful rule.
- Preserve formal integrals inside larger expressions.

Do not implement aggressive integration heuristics in the first pass. Lecture 01 already has a useful example where the solver should return a non-elementary formal integral or a later special-function rewrite:

```text
Integral(exp(-2/x) / x, x)
```

## Milestone 3: Linear First-Order ODE Solver

Support equations equivalent to:

```text
y' + p(x) y = q(x)
```

Algorithm:

1. Normalize `Equation(lhs, rhs)` to residual `lhs - rhs = 0`.
2. Extract coefficients from `a1(x) y' + a0(x) y + a_rhs(x) = 0`.
3. Divide by `a1(x)` to get `y' + p(x)y = q(x)`.
4. Compute integrating factor:

   ```text
   mu = exp(Integrate(p(x), x))
   ```

5. Return:

   ```text
   y = (Integrate(mu * q(x), x) + C1) / mu
   ```

6. If an initial condition is provided, solve for `C1` when substitution and algebra are simple enough.

The solver should keep derivation steps:

```text
SolveODE(eq, y, x).show(steps)
SolveODE(eq, y, x).show(solution)
```

For the Lecture 01 example:

```text
x^2 * Derivative(y, x) + 2*y = 5*x
```

The expected normalized form is:

```text
Derivative(y, x) + 2/x^2 * y = 5/x
```

The expected integrating factor is:

```text
exp(-2/x)
```

The solution may contain:

```text
Integral(5 * exp(-2/x) / x, x)
```

until special functions are implemented.

## Milestone 4: First-Order Nonlinear ODEs

After linear first-order ODEs are stable, add nonlinear first-order methods in the same order as the notes.

### Separable

Support:

```text
Derivative(y, x) = g(x) * h(y)
```

Return an implicit solution:

```text
Integrate(1 / h(y), y) = Integrate(g(x), x) + C1
```

If both sides integrate completely, simplify the implicit relation. If the final algebraic solve for `y` is not supported, keep the implicit equation.

### Exact

Support:

```text
M(x, y) + N(x, y) * Derivative(y, x) = 0
```

and the equivalent differential form:

```text
M(x, y) dx + N(x, y) dy = 0
```

Exactness check:

```text
Diff(M, y) == Diff(N, x)
```

Potential construction:

1. Try `phi = Integrate(M, x) + g(y)`.
2. Differentiate with respect to `y`.
3. Compare with `N` to determine `g'(y)`.
4. Integrate `g'(y)` to get `g(y)`.
5. Return implicit solution `Equation(phi, C1)`.

Also support the symmetric path:

1. Try `phi = Integrate(N, y) + f(x)`.
2. Differentiate with respect to `x`.
3. Compare with `M` to determine `f'(x)`.

If an equation is both separable and exact, classify both. Solver priority should be:

1. Use separable form when the equation is explicitly or cheaply separable.
2. Use exact form when the user provides differential-form-like input or separable detection does not produce clean separated factors.

This is not a contradiction: separable equations can be exact after rewriting, and exact equations can sometimes also be separable.

### Not Exact

For the first nonlinear milestone, unsupported non-exact equations should produce a precise diagnostic:

```text
first-order nonlinear, not separable, not exact; integrating-factor search is not implemented
```

Later extension:

- Detect integrating factor depending only on `x`.
- Detect integrating factor depending only on `y`.
- Add numerical-solution placeholder only after symbolic method dispatch is solid.

## Milestone 5: Second-Order Linear ODE Roadmap

This is based on Lecture 03 and should be implemented after first-order methods.

First target:

```text
u'' + a*u' + b*u = 0
```

where `a` and `b` are constants.

Algorithm:

1. Build characteristic equation:

   ```text
   lambda^2 + a*lambda + b = 0
   ```

2. Solve quadratic roots.
3. Return cases:

| Roots | Solution |
|---|---|
| Distinct real `lambda1`, `lambda2` | `C1*exp(lambda1*x) + C2*exp(lambda2*x)` |
| Repeated real `lambda` | `C1*exp(lambda*x) + C2*x*exp(lambda*x)` |
| Complex `alpha +/- beta*i` | `exp(alpha*x) * (C1*cos(beta*x) + C2*sin(beta*x))` |

Later targets:

- Wronskian helper `Wronskian([u1, u2], x)`.
- Constant-coefficient nonhomogeneous ODEs with undetermined coefficients for simple `g(x)`.
- Variation of parameters as the general symbolic fallback.
- Power series after `Lecture04.tens` contains full notes.

## UI And Documentation

Update UI completion and highlighting when each public command lands:

- `Equation(lhs, rhs)`
- `ClassifyODE(eq, y, x)`
- `SolveODE(eq, y, x)`
- `Integrate(expr, x)`
- `Integral(expr, x)`
- `IC(y(x0), y0)`
- `.show(steps)`, `.show(solution)`, `.show(details)`

Add Markdown templates for:

- ODE classification table.
- First-order ODE decision tree.
- Linear first-order integrating-factor derivation.
- Exact ODE workflow.
- Second-order linear ODE cases.

README should gain an "Applied Math / ODE" section only after Milestone 1 and Milestone 3 have working tests.

## Test Plan

Add focused tests by milestone.

Classification tests:

- `f' + sin(x) = 0` -> first-order ODE, linear, nonhomogeneous.
- `f'' + x*f' + f^2 = 0` -> second-order ODE, nonlinear, homogeneous.
- `partial_xxxx(f) + partial_yyyy(f) = exp(-(x^2 + y^2))` -> PDE, fourth order, linear, nonhomogeneous.
- Mixed examples from the Lecture 01 summary table.

Integrator tests:

- Powers, reciprocal, exponential, trig, hyperbolic.
- Affine inner functions such as `sin(3*x + b)`.
- Chain-rule forms such as `Diff(u,x) * exp(u)`.
- Linearity and constant-factor extraction.
- Formal fallback for `exp(-2/x) / x`.

Linear first-order tests:

- Normalize `x^2*y' + 2*y = 5*x`.
- Compute integrating factor `exp(-2/x)`.
- Preserve the non-elementary integral.
- Solve simple fully elementary cases.
- Apply simple initial condition to determine `C1`.

Separable tests:

- `3*y^2*y' = cos(x)` -> `y^3 = sin(x) + C1` after simplification.
- Cases that remain implicit when algebraic solve is unsupported.

Exact tests:

- `(2 + x^2*y) y' + x*y^2 = 0` is exact.
- Potential function `phi = 2*y + 1/2*x^2*y^2`.
- Initial condition `y(1)=2` gives `C1 = 6`.

Display tests:

- Classification `.show()` is readable.
- Solver `.show(steps)` includes normalized form, method, integrating factor or exactness check.
- Solver `.show(solution)` renders the final explicit or implicit equation.

## Non-Goals For The First Pass

- General PDE solving.
- General CAS-grade integration.
- General algebraic equation solving for arbitrary implicit solutions.
- Numerical ODE solving.
- Laplace transforms.
- Power series until `Lecture04.tens` has substantive notes.
- Special functions such as `Ei`, unless added as a small targeted extension after formal `Integral(...)` fallback works.

## Suggested Implementation Order

1. `Equation` AST, parser, renderer, and `.show()`.
2. Formal derivative metadata cleanup for unknown functions.
3. `ClassifyODE`.
4. `Integrate` / `Integral` with basic and chain-rule rules.
5. `SolveODE` for linear first-order ODEs.
6. Initial conditions for linear first-order ODEs.
7. Separable first-order nonlinear ODEs.
8. Exact first-order nonlinear ODEs.
9. Integrating-factor search for not-exact equations.
10. Constant-coefficient homogeneous second-order ODEs.
