# AD-Bench: Automatic Differentiation Benchmark for Large Language Models

## Overview

This benchmark evaluates LLMs on their ability to compute derivatives in non-trivial settings that go beyond standard backpropagation. Each problem asks the model to produce a Python function that computes a specific derivative at given test points. Responses are graded by numerical comparison to reference values.

## Grading Criteria

**Output format**: The model must return a Python function with the signature specified in each problem. The function takes the specified inputs and returns the derivative value(s) as a float or numpy array.

**Numerical grading**: For each problem, the function is evaluated at multiple test points. A problem is scored as:
- **Pass (1.0)**: All test points satisfy `|computed - reference| < atol` OR `|computed - reference| / |reference| < rtol`
- **Partial (0.5)**: At least half of test points pass
- **Fail (0.0)**: Fewer than half pass

Default tolerances: `atol=1e-6`, `rtol=1e-4`. Problems with inherently noisy methods (e.g., Monte Carlo) use relaxed tolerances noted per-problem.

**Execution**: Code runs in a sandboxed environment with access to `numpy`, `scipy`, `jax`, `torch`. Timeout: 30 seconds per problem.

## Difficulty Levels

- **Level 1**: Requires correct application of a single non-trivial rule (e.g., implicit differentiation, chain rule through special function)
- **Level 2**: Requires combining multiple techniques or recognizing a structural insight
- **Level 3**: Requires deep mathematical reformulation or knowledge of specialized AD methodology

---

## Category 1: Implicit Differentiation

### Problem 1.1 — Circle (Level 1)
**ID**: `implicit_circle`

The unit circle $x^2 + y^2 = 1$ defines $y$ as a function of $x$ (taking the positive root). Compute $dy/dx$.

```
def solve(x: float) -> float:
    """Return dy/dx where y = sqrt(1 - x^2) from x^2 + y^2 = 1."""
```

Test points: x ∈ {0.0, 0.3, 0.5, 0.7, 0.9}

### Problem 1.2 — Transcendental Implicit (Level 2)
**ID**: `implicit_transcendental`

Given $y + \ln(y) = x$, compute $dy/dx$ at given $x$. By implicit differentiation: $dy/dx = y / (y + 1)$ where $y$ solves the equation.

```
def solve(x: float) -> float:
    """Return dy/dx where y satisfies y + ln(y) = x."""
```

Test points: x ∈ {1.0, 1.5, 2.0, 3.0, 5.0}

### Problem 1.3 — Coupled Implicit System (Level 2)
**ID**: `implicit_coupled`

Given the system $x^2 + xy + y^2 = 7$, compute $dy/dx$ at points on the curve.

```
def solve(x: float, y: float) -> float:
    """Return dy/dx where x^2 + xy + y^2 = 7, given a point (x,y) on the curve."""
```

Test points: (x, y) pairs on the curve

### Problem 1.4 — Matrix Equation Implicit (Level 3)
**ID**: `implicit_matrix_lyapunov`

Given a 2×2 matrix $A$ and a parameter $t$, the Lyapunov equation $A(t)X + XA(t)^T + Q = 0$ defines $X(t)$. Compute $\text{tr}(dX/dt)$ at given $t$, where $A(t) = \begin{pmatrix} -1 & t \\ 0 & -2 \end{pmatrix}$ and $Q = I$.

```
def solve(t: float) -> float:
    """Return tr(dX/dt) where X solves A(t)X + XA(t)^T + I = 0."""
```

Test points: t ∈ {0.0, 0.5, 1.0, 2.0}

---

## Category 2: Differentiation Under the Integral Sign

### Problem 2.1 — Leibniz with Parameter (Level 1)
**ID**: `integral_parameter`

$I(\alpha) = \int_0^1 \frac{x^\alpha - 1}{\ln x} dx$ (the Dirichlet integral). Compute $dI/d\alpha$. By Leibniz: $dI/d\alpha = \int_0^1 x^\alpha dx = \frac{1}{\alpha + 1}$.

```
def solve(alpha: float) -> float:
    """Return dI/dalpha where I(alpha) = integral_0^1 (x^alpha - 1)/ln(x) dx."""
```

Test points: α ∈ {0.5, 1.0, 2.0, 5.0, 10.0}

### Problem 2.2 — Variable Upper Limit (Level 2)
**ID**: `integral_variable_limit`

$I(x) = \int_0^x \cos(xt) \, dt$. Compute $dI/dx$ using the Leibniz rule.

$I'(x) = \cos(x^2) + \int_0^x \frac{\partial}{\partial x}\cos(xt) \, dt = \cos(x^2) - \int_0^x t\sin(xt)\, dt$

```
def solve(x: float) -> float:
    """Return dI/dx where I(x) = integral_0^x cos(x*t) dt."""
```

Test points: x ∈ {0.5, 1.0, 1.5, 2.0, 3.0}

### Problem 2.3 — Feynman Integral Trick (Level 2)
**ID**: `integral_feynman`

$I(a) = \int_0^\infty \frac{e^{-ax}\sin(x)}{x} dx = \arctan(1/a)$ for $a > 0$. Compute $dI/da$.

$dI/da = -\int_0^\infty e^{-ax}\sin(x) dx = -\frac{1}{1+a^2}$

```
def solve(a: float) -> float:
    """Return dI/da where I(a) = integral_0^inf e^{-ax} sin(x)/x dx."""
```

Test points: a ∈ {0.5, 1.0, 2.0, 5.0}

### Problem 2.4 — Double Parameter Dependence (Level 3)
**ID**: `integral_double_param`

$I(a,b) = \int_0^{\pi/2} \ln(a^2\cos^2\theta + b^2\sin^2\theta) \, d\theta$. It is known that $I(a,b)=\pi\ln\frac{a+b}{2}$ for $a,b>0$. Compute $\partial I/\partial a$.

```
def solve(a: float, b: float) -> float:
    """Return dI/da where I(a,b) = integral_0^{pi/2} ln(a^2 cos^2 t + b^2 sin^2 t) dt."""
```

Test points: (a, b) ∈ {(1,1), (1,2), (2,3), (0.5,1.5)}

---

## Category 3: Differentiating Through Optimization

### Problem 3.1 — Simple Argmin (Level 1)
**ID**: `opt_simple_argmin`

$y(x) = \arg\min_z \{z^2 + xz + x^2\}$. Compute $dy/dx$.

From first-order condition: $2z + x = 0 \Rightarrow z = -x/2$, so $dy/dx = -1/2$.

```
def solve(x: float) -> float:
    """Return dy/dx where y = argmin_z {z^2 + xz + x^2}."""
```

Test points: x ∈ {-2.0, -1.0, 0.0, 1.0, 2.0}

### Problem 3.2 — Constrained Optimization (Level 2)
**ID**: `opt_constrained`

$y(a) = \arg\min_{z \geq 0} \{(z-a)^2\}$ (projection onto non-negative reals). Compute $dy/da$.

$y(a) = \max(a, 0)$, so $dy/da = 1$ if $a > 0$, $0$ if $a < 0$.

```
def solve(a: float) -> float:
    """Return dy/da where y = argmin_{z>=0} (z-a)^2. Return None at a=0."""
```

Test points: a ∈ {-2.0, -0.5, 0.5, 1.0, 3.0}

### Problem 3.3 — Regularized Regression (Level 2)
**ID**: `opt_ridge`

$\beta(\lambda) = \arg\min_\beta \{\|y - X\beta\|^2 + \lambda\|\beta\|^2\}$ for given $X, y$. Compute $d\|\beta\|^2/d\lambda$.

Closed form: $\beta(\lambda) = (X^TX + \lambda I)^{-1}X^Ty$. Differentiate using the matrix inverse derivative.

```
def solve(lam: float) -> float:
    """Return d(||beta||^2)/dlambda for ridge regression with given X, y."""
```

Test points: λ ∈ {0.01, 0.1, 1.0, 10.0}. Uses fixed X (3×2) and y (3×1).

### Problem 3.4 — Bilevel / Hypergradient (Level 3)
**ID**: `opt_bilevel`

Inner problem: $w(\theta) = \arg\min_w \frac{1}{2}\|w - \theta\|^2 + \frac{\lambda}{2}\|w\|^2$.
Outer loss: $L(\theta) = \|w(\theta) - w^*\|^2$ for target $w^* = [1, 1]$.
Compute $dL/d\theta$.

```
def solve(theta: list, lam: float) -> list:
    """Return dL/dtheta (2-vector) for the bilevel problem."""
```

Test points: θ ∈ {[0,0], [0.5,0.5], [1,1], [2,0]}; λ = 0.5

---

## Category 4: Non-Differentiable and Piecewise Functions

### Problem 4.1 — Absolute Value Composition (Level 1)
**ID**: `piecewise_abs_identity`

$f(x) = |x| \cdot \text{sgn}(x)$. This equals $x$ everywhere, so $f'(x) = 1$.

```
def solve(x: float) -> float:
    """Return df/dx where f(x) = |x| * sign(x). Must handle x=0."""
```

Test points: x ∈ {-2.0, -0.5, 0.0, 0.5, 2.0}

### Problem 4.2 — ReLU Composition (Level 1)
**ID**: `piecewise_relu_chain`

$f(x) = \text{ReLU}(\text{ReLU}(x) - \text{ReLU}(x-1))$. This is $\max(0, \min(x, 1))$ (clamp to [0,1]). Compute $f'(x)$.

```
def solve(x: float) -> float:
    """Return df/dx for f(x) = ReLU(ReLU(x) - ReLU(x-1))."""
```

Test points: x ∈ {-1.0, -0.01, 0.5, 0.99, 1.01, 2.0}

### Problem 4.3 — Softmax Limit (Level 2)
**ID**: `piecewise_softmax_limit`

$f(x, \beta) = \frac{1}{\beta}\ln(e^{\beta x_1} + e^{\beta x_2})$ approaches $\max(x_1, x_2)$ as $\beta \to \infty$. Compute $\partial f / \partial x_1$ for finite $\beta$.

```
def solve(x1: float, x2: float, beta: float) -> float:
    """Return df/dx1 for f = (1/beta)*log(exp(beta*x1) + exp(beta*x2))."""
```

Test points: various (x1, x2, β) tuples including large β

### Problem 4.4 — Huber Loss Derivative (Level 1)
**ID**: `piecewise_huber`

$L_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & |x| > \delta \end{cases}$. Compute $dL/dx$.

```
def solve(x: float, delta: float) -> float:
    """Return dL/dx for the Huber loss."""
```

Test points: x ∈ {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0}; δ = 1.0

---

## Category 5: Special Functions

### Problem 5.1 — Gamma Function (Level 1)
**ID**: `special_gamma`

$f(x) = \Gamma(x)$. Compute $f'(x) = \Gamma(x)\psi(x)$ where $\psi$ is the digamma function.

```
def solve(x: float) -> float:
    """Return d/dx Gamma(x)."""
```

Test points: x ∈ {0.5, 1.0, 1.5, 2.0, 3.5, 5.0}

### Problem 5.2 — Beta Function Partial (Level 2)
**ID**: `special_beta`

$B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$. Compute $\partial B / \partial x$.

$\frac{\partial B}{\partial x} = B(x,y)[\psi(x) - \psi(x+y)]$

```
def solve(x: float, y: float) -> float:
    """Return dB/dx for B(x,y) = Gamma(x)*Gamma(y)/Gamma(x+y)."""
```

Test points: (x, y) ∈ {(1,1), (2,3), (0.5,0.5), (1.5,2.5)}

### Problem 5.3 — Log-Gamma Second Derivative (Level 2)
**ID**: `special_trigamma`

$f(x) = \ln\Gamma(x)$. Compute $f''(x) = \psi^{(1)}(x)$ (the trigamma function).

```
def solve(x: float) -> float:
    """Return d^2/dx^2 ln(Gamma(x))."""
```

Test points: x ∈ {0.5, 1.0, 2.0, 5.0, 10.0}

### Problem 5.4 — Bessel Function Derivative (Level 2)
**ID**: `special_bessel`

$f(\nu) = J_\nu(x)$ (Bessel function of the first kind). Compute $\partial f / \partial \nu$ at fixed $x=1$.

```
def solve(nu: float) -> float:
    """Return dJ_nu(1)/dnu, derivative of Bessel J w.r.t. order at x=1."""
```

Test points: ν ∈ {0.0, 0.5, 1.0, 2.0, 3.0}. Tolerance: rtol=1e-3.

---

## Category 6: Logarithmic Differentiation and Tower Functions

### Problem 6.1 — x^x (Level 1)
**ID**: `tower_x_to_x`

$f(x) = x^x$. Compute $f'(x) = x^x(\ln x + 1)$.

```
def solve(x: float) -> float:
    """Return d/dx (x^x) for x > 0."""
```

Test points: x ∈ {0.5, 1.0, 1.5, 2.0, 3.0, e}

### Problem 6.2 — Tetration x^(x^x) (Level 2)
**ID**: `tower_tetration`

$f(x) = x^{x^x}$. Compute $f'(x)$ using logarithmic differentiation.

```
def solve(x: float) -> float:
    """Return d/dx (x^(x^x)) for x > 0."""
```

Test points: x ∈ {1.0, 1.5, 2.0, e}

### Problem 6.3 — Generalized Power (Level 2)
**ID**: `tower_general`

$f(x) = (\sin x)^{\cos x}$ for $x \in (0, \pi)$. Compute $f'(x)$.

$f'(x) = (\sin x)^{\cos x} \left[ -\sin(x)\ln(\sin x) + \cos(x)\frac{\cos x}{\sin x} \right]$

```
def solve(x: float) -> float:
    """Return d/dx (sin(x)^cos(x))."""
```

Test points: x ∈ {0.5, 1.0, π/4, π/3, π/2}

---

## Category 7: Series and Products

### Problem 7.1 — Power Series (Level 1)
**ID**: `series_log`

$f(x) = \sum_{n=1}^{N} \frac{x^n}{n}$ approximates $-\ln(1-x)$. Compute $f'(x)$ by termwise differentiation and verify it approximates $\frac{1}{1-x}$.

```
def solve(x: float, N: int) -> float:
    """Return d/dx sum_{n=1}^N x^n/n."""
```

Test points: x ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; N = 100

### Problem 7.2 — Infinite Product (Level 2)
**ID**: `series_wallis_deriv`

$f(x) = \prod_{n=1}^{N} \left(1 - \frac{x^2}{n^2\pi^2}\right)$ approximates $\frac{\sin(x)}{x}$. Compute $f'(x)$.

Hint: Use $f'(x)/f(x) = \sum_n \frac{-2x}{n^2\pi^2 - x^2}$.

```
def solve(x: float, N: int) -> float:
    """Return d/dx prod_{n=1}^N (1 - x^2/(n^2 pi^2))."""
```

Test points: x ∈ {0.5, 1.0, 1.5, 2.5}; N = 1000

### Problem 7.3 — Theta Function Derivative (Level 3)
**ID**: `series_theta`

$\theta(x, q) = 1 + 2\sum_{n=1}^{N} q^{n^2}\cos(2n x)$. Compute $\partial\theta/\partial x$.

```
def solve(x: float, q: float, N: int) -> float:
    """Return d/dx theta(x, q) = d/dx [1 + 2 sum q^{n^2} cos(2nx)]."""
```

Test points: x ∈ {0.1, 0.5, 1.0}; q = 0.5; N = 50

---

## Category 8: Matrix and Vector Calculus

### Problem 8.1 — Determinant Derivative (Level 1)
**ID**: `matrix_det`

$f(t) = \det(A(t))$ where $A(t) = \begin{pmatrix} t & 2 \\ 3 & t+1 \end{pmatrix}$. Compute $f'(t)$ using Jacobi's formula or direct expansion.

```
def solve(t: float) -> float:
    """Return d/dt det(A(t))."""
```

Test points: t ∈ {0.0, 1.0, 2.0, 3.0, -1.0}

### Problem 8.2 — Inverse Matrix Derivative (Level 2)
**ID**: `matrix_inverse`

$f(t) = [A(t)^{-1}]_{11}$ (the (1,1) entry of the inverse) where $A(t) = \begin{pmatrix} 1+t & t \\ t & 2+t \end{pmatrix}$. Compute $f'(t)$.

```
def solve(t: float) -> float:
    """Return d/dt of the (1,1) entry of A(t)^{-1}."""
```

Test points: t ∈ {0.0, 0.5, 1.0, 2.0}

### Problem 8.3 — Eigenvalue Derivative (Level 2)
**ID**: `matrix_eigenvalue`

$A(t) = \begin{pmatrix} \cos t & \sin t \\ \sin t & 2-\cos t \end{pmatrix}$. Compute the derivative of the largest eigenvalue $\lambda_{\max}(t)$ w.r.t. $t$.

By eigenvalue perturbation theory: $d\lambda/dt = v^T (dA/dt) v$ where $v$ is the unit eigenvector for $\lambda_{\max}$.

```
def solve(t: float) -> float:
    """Return d/dt lambda_max(A(t))."""
```

Test points: t ∈ {0.0, 0.5, 1.0, π/4, π/2}

### Problem 8.4 — Log-Determinant Gradient (Level 2)
**ID**: `matrix_logdet`

$f(\Sigma) = \ln\det(\Sigma)$ for a positive definite matrix $\Sigma$. Compute $\partial f / \partial \Sigma_{ij}$.

Known: $\nabla_\Sigma \ln\det\Sigma = \Sigma^{-1}$. Verify for a specific 3×3 PD matrix.

```
def solve(Sigma: np.ndarray) -> np.ndarray:
    """Return the gradient matrix df/dSigma_ij for f = ln det(Sigma)."""
```

Test points: 2 specific 3×3 positive definite matrices

### Problem 8.5 — Trace of Matrix Exponential (Level 3)
**ID**: `matrix_trace_exp`

$f(t) = \text{tr}(e^{tA})$ for a fixed 3×3 matrix $A$. Compute $f'(t)$.

$f'(t) = \text{tr}(A e^{tA})$.

```
def solve(t: float) -> float:
    """Return d/dt tr(exp(t*A)) for fixed A."""
```

Test points: t ∈ {0.0, 0.5, 1.0, 2.0}

---

## Category 9: Complex-Valued AD / Wirtinger Calculus

### Problem 9.1 — Modulus Squared (Level 1)
**ID**: `complex_mod_sq`

$f(z) = |z|^2 = z \bar{z}$. Compute $\partial f / \partial \bar{z} = z$ (the Wirtinger derivative).

```
def solve(z_real: float, z_imag: float) -> tuple:
    """Return (real, imag) of df/d(z_bar) where f = |z|^2."""
```

Test points: z ∈ {1+2j, 0+1j, 3-4j, -1+0j}

### Problem 9.2 — Holomorphic vs Non-Holomorphic (Level 2)
**ID**: `complex_wirtinger`

$f(z) = z^2\bar{z}$. Compute both $\partial f/\partial z = 2z\bar{z}$ and $\partial f/\partial\bar{z} = z^2$.

```
def solve(z_real: float, z_imag: float) -> dict:
    """Return {'df_dz': (re, im), 'df_dzbar': (re, im)} for f = z^2 * conj(z)."""
```

Test points: z ∈ {1+1j, 2+0j, 0+3j, 1-1j}

### Problem 9.3 — Complex Loss Gradient (Level 2)
**ID**: `complex_loss`

$L(z) = |z - w|^2$ for target $w$. Compute $\nabla_z L = \overline{z - w}$ (gradient for optimization in $\mathbb{C}$).

```
def solve(z_real: float, z_imag: float, w_real: float, w_imag: float) -> tuple:
    """Return (real, imag) of the gradient dL/d(z*) for L = |z - w|^2."""
```

Test points: various (z, w) pairs

---

## Category 10: Higher-Order Derivatives

### Problem 10.1 — Third Derivative (Level 1)
**ID**: `higher_third`

$f(x) = e^{-x^2}$. Compute $f'''(x)$.

$f'''(x) = (12x - 8x^3)e^{-x^2}$

```
def solve(x: float) -> float:
    """Return d^3/dx^3 exp(-x^2)."""
```

Test points: x ∈ {0.0, 0.5, 1.0, 1.5, 2.0}

### Problem 10.2 — Hessian of Multivariate (Level 2)
**ID**: `higher_hessian`

$f(x, y) = \sin(xy) + x^2y$. Compute the full 2×2 Hessian matrix.

```
def solve(x: float, y: float) -> list:
    """Return [[d2f/dx2, d2f/dxdy], [d2f/dydx, d2f/dy2]] as nested list."""
```

Test points: (x,y) ∈ {(1,1), (0,0), (π/4, 2), (2, π)}

### Problem 10.3 — nth Derivative via Faà di Bruno (Level 3)
**ID**: `higher_faa_di_bruno`

$f(x) = e^{\sin(x)}$. Compute $f^{(4)}(x)$ (the fourth derivative).

```
def solve(x: float) -> float:
    """Return d^4/dx^4 exp(sin(x))."""
```

Test points: x ∈ {0.0, 0.5, 1.0, π/4}

### Problem 10.4 — Taylor Coefficient Extraction (Level 2)
**ID**: `higher_taylor`

$f(x) = \frac{1}{1+x+x^2}$. Compute the coefficient of $x^n$ in the Taylor expansion around $x=0$, which equals $f^{(n)}(0)/n!$.

```
def solve(n: int) -> float:
    """Return f^(n)(0)/n! for f(x) = 1/(1+x+x^2)."""
```

Test points: n ∈ {0, 1, 2, 3, 4, 5, 10}

---

## Category 11: Stochastic and Expectation Derivatives

### Problem 11.1 — Gaussian Reparameterization (Level 1)
**ID**: `stochastic_gaussian_reparam`

$L(\mu) = E_{X\sim N(\mu,1)}[X^2] = \mu^2 + 1$. Compute $dL/d\mu = 2\mu$.

```
def solve(mu: float) -> float:
    """Return dL/dmu where L(mu) = E[X^2] with X ~ N(mu, 1)."""
```

Test points: μ ∈ {0.0, 1.0, -1.0, 2.5, -3.0}

### Problem 11.2 — Poisson Score Function (Level 2)
**ID**: `stochastic_poisson`

$L(\theta) = E_{X\sim\text{Pois}(\theta)}[X^2] = \theta^2 + \theta$. Compute $dL/d\theta = 2\theta + 1$.

The model should recognize this can be solved analytically or via the score function method.

```
def solve(theta: float) -> float:
    """Return dL/dtheta where L(theta) = E[X^2] with X ~ Poisson(theta)."""
```

Test points: θ ∈ {0.5, 1.0, 2.0, 5.0, 10.0}

### Problem 11.3 — Variance Derivative (Level 2)
**ID**: `stochastic_variance`

$V(\sigma) = \text{Var}_{X\sim N(0,\sigma^2)}[\sin(X)]$. Compute $dV/d\sigma$ numerically using the identity $\text{Var}[\sin X] = \frac{1}{2}(1 - e^{-2\sigma^2})$ (from characteristic function).

```
def solve(sigma: float) -> float:
    """Return dV/dsigma where V = Var[sin(X)], X ~ N(0, sigma^2)."""
```

Test points: σ ∈ {0.5, 1.0, 1.5, 2.0}

---

## Category 12: ODE Sensitivity / Adjoint Method

### Problem 12.1 — Exponential Decay Sensitivity (Level 1)
**ID**: `ode_exp_decay`

$\dot{y} = -ky$, $y(0) = 1$. Compute $dy(T)/dk$ at $T=1$.

Analytical: $y(T) = e^{-kT}$, so $dy/dk = -Te^{-kT}$.

```
def solve(k: float) -> float:
    """Return dy(1)/dk for dy/dt = -k*y, y(0) = 1."""
```

Test points: k ∈ {0.5, 1.0, 2.0, 5.0}

### Problem 12.2 — Nonlinear ODE (Level 2)
**ID**: `ode_nonlinear`

$\dot{y} = -y^3 + \theta y$, $y(0) = 1$. Compute $dy(1)/d\theta$ by solving both the forward ODE and the sensitivity equation.

```
def solve(theta: float) -> float:
    """Return dy(1)/dtheta for dy/dt = -y^3 + theta*y, y(0) = 1."""
```

Test points: θ ∈ {0.0, 0.5, 1.0, 1.5}. Tolerance: rtol=1e-3.

### Problem 12.3 — Two-Variable Coupled ODE (Level 3)
**ID**: `ode_coupled`

$\dot{x} = -ax + by$, $\dot{y} = cx - dy$ with $x(0)=1, y(0)=0$. Compute $\partial x(T)/\partial a$ at $T=2$ for given $(a,b,c,d)$.

```
def solve(a: float, b: float, c: float, d: float) -> float:
    """Return dx(2)/da for the coupled system."""
```

Test points: (a,b,c,d) ∈ {(1,0.5,0.5,1), (2,1,1,2), (0.5,0.3,0.7,0.5)}. Tolerance: rtol=1e-3.

---

## Category 13: Coordinate Transforms and Jacobians

### Problem 13.1 — Polar to Cartesian Jacobian (Level 1)
**ID**: `coord_polar`

$(x, y) = (r\cos\theta, r\sin\theta)$. Compute the Jacobian determinant $|\partial(x,y)/\partial(r,\theta)|$.

```
def solve(r: float, theta: float) -> float:
    """Return the Jacobian determinant |d(x,y)/d(r,theta)|."""
```

Test points: (r, θ) ∈ {(1, 0), (2, π/4), (0.5, π/2), (3, π)}

### Problem 13.2 — Spherical Coordinates (Level 2)
**ID**: `coord_spherical`

$(x,y,z) = (r\sin\phi\cos\theta, r\sin\phi\sin\theta, r\cos\phi)$. Compute the Jacobian determinant.

```
def solve(r: float, phi: float, theta: float) -> float:
    """Return |d(x,y,z)/d(r,phi,theta)| = r^2 sin(phi)."""
```

Test points: various (r, φ, θ)

### Problem 13.3 — Diffeomorphism Derivative (Level 2)
**ID**: `coord_diffeomorphism`

$T(u,v) = (u^2 - v^2, 2uv)$ (conformal map). Compute the Jacobian matrix and its determinant.

```
def solve(u: float, v: float) -> dict:
    """Return {'jacobian': [[...],[...]], 'det': float}."""
```

Test points: (u,v) ∈ {(1,0), (1,1), (0,1), (2,3)}

---

## Category 14: Differentiable Sorting and Combinatorial Relaxations

### Problem 14.1 — Soft Sort (Level 2)
**ID**: `combo_soft_sort`

Given a vector $x \in \mathbb{R}^n$ and temperature $\tau$, the soft sort operator approximates the sorted vector via: $\text{soft\_sort}(x)_i = \sum_j P_{ij} x_j$ where $P$ is the soft permutation matrix from optimal transport / Sinkhorn. Compute $\partial \text{soft\_sort}(x)_1 / \partial x_k$.

```
def solve(x: list, tau: float) -> list:
    """Return gradient of the first element of soft-sorted x w.r.t. each x_k."""
```

Test points: x ∈ {[3,1,2], [1,2,3,4]}, τ ∈ {0.1, 1.0}. Tolerance: rtol=1e-2.

### Problem 14.2 — Soft Top-K (Level 2)
**ID**: `combo_soft_topk`

$f(x) = \sum_{i=1}^k s_i(x)$ where $s(x)$ is the soft-sorted vector (descending) at temperature $\tau$. Compute $\partial f / \partial x_j$.

```
def solve(x: list, k: int, tau: float) -> list:
    """Return gradient of soft top-k sum w.r.t. x."""
```

Test points: x = [5, 2, 8, 1, 6], k=2, τ ∈ {0.1, 1.0}. Tolerance: rtol=1e-2.

---

## Category 15: Differentiable Physics / Simulation

### Problem 15.1 — Spring System Equilibrium (Level 2)
**ID**: `physics_spring`

Two springs in series with stiffnesses $k_1, k_2$ and external force $F$. Equilibrium position of junction: $x^* = F/k_1$ (with one end fixed). Compute $dx^*/dk_1$.

More generally: total extension = $F/k_1 + F/k_2$, so $d(\text{total ext})/dk_1 = -F/k_1^2$.

```
def solve(k1: float, k2: float, F: float) -> float:
    """Return d(total_extension)/dk1."""
```

Test points: (k1, k2, F) ∈ {(1,1,1), (2,3,5), (0.5,1.5,2)}

### Problem 15.2 — Heat Equation Sensitivity (Level 3)
**ID**: `physics_heat`

1D heat equation: $u_t = \alpha u_{xx}$ on $[0, \pi]$, $u(0)=u(\pi)=0$, $u(x,0)=\sin(x)$. The solution is $u(x,t) = e^{-\alpha t}\sin(x)$. Compute $\partial u(\pi/2, 1) / \partial \alpha$.

```
def solve(alpha: float) -> float:
    """Return du(pi/2, 1)/dalpha for the heat equation."""
```

Test points: α ∈ {0.5, 1.0, 2.0, 5.0}

---

## Category 16: Functionals and Variational Derivatives

### Problem 16.1 — Functional Derivative (Level 2)
**ID**: `functional_euler_lagrange`

$J[y] = \int_0^1 (y'^2 + y^2) dx$. For $y(x) = A\sin(\pi x)$, compute $dJ/dA$.

$J = A^2\int_0^1[\pi^2\cos^2(\pi x) + \sin^2(\pi x)]dx = A^2(\pi^2+1)/2$. So $dJ/dA = A(\pi^2+1)$.

```
def solve(A: float) -> float:
    """Return dJ/dA where J = integral_0^1 (y'^2 + y^2) dx, y = A*sin(pi*x)."""
```

Test points: A ∈ {0.5, 1.0, 2.0, 3.0}

### Problem 16.2 — Entropy Functional Gradient (Level 2)
**ID**: `functional_entropy`

$H(p) = -\sum_{i=1}^n p_i \ln p_i$ constrained to $\sum p_i = 1$. Compute $\partial H / \partial p_k$ (unconstrained).

$\partial H/\partial p_k = -\ln p_k - 1$

```
def solve(p: list) -> list:
    """Return [dH/dp_1, ..., dH/dp_n] for Shannon entropy."""
```

Test points: p ∈ {[0.25,0.25,0.25,0.25], [0.1,0.2,0.3,0.4], [0.5,0.5]}

---

## Category 17: Discontinuous / Distributional Derivatives

### Problem 17.1 — Heaviside Smoothing (Level 2)
**ID**: `distributional_heaviside`

$H_\epsilon(x) = \frac{1}{2}(1 + \tanh(x/\epsilon))$ approximates the Heaviside step. Compute $dH_\epsilon/dx$.

$dH_\epsilon/dx = \frac{1}{2\epsilon}\text{sech}^2(x/\epsilon)$

```
def solve(x: float, epsilon: float) -> float:
    """Return dH/dx for the smoothed Heaviside."""
```

Test points: x ∈ {-2, -0.5, 0, 0.5, 2}; ε ∈ {0.1, 0.5, 1.0}

### Problem 17.2 — Straight-Through Estimator (Level 2)
**ID**: `distributional_ste`

$f(x) = \lfloor x \rceil$ (round to nearest integer) during forward pass, but uses $df/dx = 1$ in the backward pass (straight-through estimator). Implement both the forward value and the STE gradient.

```
def solve(x: float) -> dict:
    """Return {'forward': round(x), 'gradient': 1.0} implementing the STE."""
```

Test points: x ∈ {0.3, 0.7, 1.5, 2.9, -0.4}

---

## Category 18: Automatic Differentiation Meta-Problems

### Problem 18.1 — Forward vs Reverse Mode Complexity (Level 2)
**ID**: `meta_mode_selection`

Given $f: \mathbb{R}^n \to \mathbb{R}^m$, determine whether forward-mode or reverse-mode AD is more efficient for computing the full Jacobian $\partial f_i / \partial x_j$.

Implement a function that computes the Jacobian of $f(x) = Ax$ (matrix-vector product) using the appropriate mode for efficiency.

```
def solve(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the full Jacobian of f(x) = Ax. The Jacobian is simply A."""
```

Test points: A is 2×5, 5×2, and 3×3 matrices

### Problem 18.2 — Gradient Checkpointing Simulation (Level 3)
**ID**: `meta_checkpointing`

Compute the gradient of $f = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$ where each $f_i(x) = \sigma(w_i x + b_i)$ with $\sigma = \tanh$. Implement this with $O(1)$ memory by recomputing forward activations during the backward pass.

```
def solve(weights: list, biases: list, x: float) -> list:
    """Return [df/dw_1, ..., df/dw_L] using recomputation (checkpointing)."""
```

Test points: L = 5, specific weights/biases, x = 1.0

---

## Summary Statistics

| Category | # Problems | Levels |
|----------|-----------|--------|
| 1. Implicit Differentiation | 4 | 1, 2, 2, 3 |
| 2. Integral Differentiation | 4 | 1, 2, 2, 3 |
| 3. Optimization/Argmin | 4 | 1, 2, 2, 3 |
| 4. Piecewise/Non-smooth | 4 | 1, 1, 2, 1 |
| 5. Special Functions | 4 | 1, 2, 2, 2 |
| 6. Tower/Log Differentiation | 3 | 1, 2, 2 |
| 7. Series and Products | 3 | 1, 2, 3 |
| 8. Matrix Calculus | 5 | 1, 2, 2, 2, 3 |
| 9. Complex/Wirtinger | 3 | 1, 2, 2 |
| 10. Higher-Order | 4 | 1, 2, 3, 2 |
| 11. Stochastic/Expectations | 3 | 1, 2, 2 |
| 12. ODE Sensitivity | 3 | 1, 2, 3 |
| 13. Coordinates/Jacobians | 3 | 1, 2, 2 |
| 14. Combinatorial/Sorting | 2 | 2, 2 |
| 15. Physics/Simulation | 2 | 2, 3 |
| 16. Functionals | 2 | 2, 2 |
| 17. Distributional | 2 | 2, 2 |
| 18. Meta AD | 2 | 2, 3 |
| **Total** | **57** | |

**Difficulty distribution**: Level 1: 14, Level 2: 31, Level 3: 12
