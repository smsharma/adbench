"""Category 7: Higher-Order and Meta-AD — hard cases.

Hessian-vector products, double backprop, and problems requiring
higher-order reasoning about the differentiation process itself.
"""

import numpy as np
from scipy.linalg import expm

from adbench.problem import Problem


# --- 7.1: Gradient of gradient norm (double backprop) ---
# f(x) = sin(x₁x₂) + x₁²x₂. Let g(x) = ||∇f(x)||².
# Compute ∇g(x) (gradient of the squared gradient norm).
# This requires "double backprop" — differentiating through the gradient computation.

def _ref_grad_grad_norm(x1: float, x2: float) -> list:
    """∇g where g = ||∇f||², f = sin(x1*x2) + x1²x2."""
    h = 1e-5
    def grad_norm_sq(a, b):
        # f = sin(a*b) + a²b
        # df/da = b*cos(a*b) + 2*a*b
        # df/db = a*cos(a*b) + a²
        da = b * np.cos(a*b) + 2*a*b
        db = a * np.cos(a*b) + a**2
        return da**2 + db**2

    dg_dx1 = (grad_norm_sq(x1 + h, x2) - grad_norm_sq(x1 - h, x2)) / (2*h)
    dg_dx2 = (grad_norm_sq(x1, x2 + h) - grad_norm_sq(x1, x2 - h)) / (2*h)
    return [dg_dx1, dg_dx2]

P_HIGHER_GRAD_NORM = Problem(
    id="higher_grad_of_grad_norm",
    category="higher_order",
    difficulty=3,
    description="Compute the gradient of ||∇f||² (double backpropagation).",
    prompt="""f(x₁, x₂) = sin(x₁x₂) + x₁²x₂

Let g(x₁, x₂) = ||∇f(x₁,x₂)||² = (∂f/∂x₁)² + (∂f/∂x₂)².

Compute ∇g = [∂g/∂x₁, ∂g/∂x₂].

```python
def solve(x1: float, x2: float) -> list:
    \"\"\"Return [dg/dx1, dg/dx2] where g = ||grad f||^2.\"\"\"
```""",
    reference=_ref_grad_grad_norm,
    test_cases=[
        {"inputs": {"x1": x1, "x2": x2}, "expected": _ref_grad_grad_norm(x1, x2)}
        for x1, x2 in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5), (0.0, 1.0), (1.0, 0.0)]
    ],
    rtol=1e-2,
)


# --- 7.2: Hessian of log-partition function ---
# Z(θ₁,θ₂) = Σ_x exp(θ₁ x₁ + θ₂ x₂ + θ₁θ₂ x₁x₂)
# summed over (x₁,x₂) ∈ {0,1}².
# The Hessian of log Z gives the covariance structure.

def _log_Z(t1, t2):
    terms = []
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            terms.append(t1*x1 + t2*x2 + t1*t2*x1*x2)
    terms = np.array(terms)
    m = np.max(terms)
    return m + np.log(np.sum(np.exp(terms - m)))

def _ref_higher_hessian_logZ(t1: float, t2: float) -> list:
    """Full 2x2 Hessian of log Z(t1, t2)."""
    h = 1e-5
    f = _log_Z
    # d²f/dt1² via central difference
    d2_dt1 = (f(t1+h, t2) - 2*f(t1, t2) + f(t1-h, t2)) / h**2
    d2_dt2 = (f(t1, t2+h) - 2*f(t1, t2) + f(t1, t2-h)) / h**2
    d2_dt1dt2 = (f(t1+h, t2+h) - f(t1+h, t2-h) - f(t1-h, t2+h) + f(t1-h, t2-h)) / (4*h**2)
    return [[d2_dt1, d2_dt1dt2], [d2_dt1dt2, d2_dt2]]

P_HIGHER_HESSIAN_LOGZ = Problem(
    id="higher_hessian_log_partition",
    category="higher_order",
    difficulty=3,
    description="Full Hessian of log-partition function for a 2D binary model.",
    prompt="""A binary model over (x₁,x₂) ∈ {0,1}² has partition function:
  Z(θ₁,θ₂) = Σ_{(x₁,x₂)} exp(θ₁x₁ + θ₂x₂ + θ₁θ₂x₁x₂)

Compute the full 2×2 Hessian matrix of log Z(θ₁,θ₂).

```python
def solve(t1: float, t2: float) -> list:
    \"\"\"Return [[d2logZ/dt1^2, d2logZ/dt1dt2], [d2logZ/dt2dt1, d2logZ/dt2^2]].\"\"\"
```""",
    reference=_ref_higher_hessian_logZ,
    test_cases=[
        {"inputs": {"t1": t1, "t2": t2}, "expected": _ref_higher_hessian_logZ(t1, t2)}
        for t1, t2 in [(0.0, 0.0), (1.0, 0.5), (-0.5, 1.0), (2.0, -1.0)]
    ],
    rtol=1e-2,
)


# --- 7.3: Fifth derivative via Taylor arithmetic ---
# f(x) = exp(x) / (1 + x²). Compute f⁽⁵⁾(0) / 5!
# The Taylor coefficient of order 5. This requires careful computation
# because the 5th derivative of this composition is complex symbolically.

def _ref_higher_taylor_5th(x0: float) -> float:
    """f^(5)(x0)/5! for f(x) = exp(x)/(1+x^2) via high-order finite diff."""
    h = 1e-3
    def f(x):
        return np.exp(x) / (1 + x**2)

    # 5th derivative via 7-point stencil
    # f^(5) ≈ [-0.5 f(x-3h) + 2 f(x-2h) - 2.5 f(x-h) + 0 f(x)
    #           + 2.5 f(x+h) - 2 f(x+2h) + 0.5 f(x+3h)] / h^5
    coeffs = [-0.5, 2.0, -2.5, 0.0, 2.5, -2.0, 0.5]
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    d5 = sum(c * f(x0 + o * h) for c, o in zip(coeffs, offsets)) / h**5
    return d5 / 120  # /5!

P_HIGHER_TAYLOR_5TH = Problem(
    id="higher_taylor_5th",
    category="higher_order",
    difficulty=3,
    description="Compute the 5th Taylor coefficient of exp(x)/(1+x²) at x=0.",
    prompt="""f(x) = exp(x) / (1 + x²)

Compute f⁽⁵⁾(x₀) / 5! (the 5th Taylor coefficient around x₀).

```python
def solve(x0: float) -> float:
    \"\"\"Return f^(5)(x0)/5! where f(x) = exp(x)/(1+x^2).\"\"\"
```""",
    reference=_ref_higher_taylor_5th,
    test_cases=[
        {"inputs": {"x0": v}, "expected": _ref_higher_taylor_5th(v)}
        for v in [0.0, 0.5, 1.0]
    ],
    rtol=5e-2,
)


# --- 7.4: Jacobian of softmax (full matrix) ---
# The Jacobian of softmax(x) is diag(p) - pp^T where p = softmax(x).
# But compute it for a specific 4-dimensional input and verify.
# Seems easy but the numerical stability aspect makes it non-trivial.

def _ref_higher_softmax_jacobian(x: list) -> list:
    x = np.array(x, dtype=float)
    # Stable softmax
    e = np.exp(x - np.max(x))
    p = e / np.sum(e)
    J = np.diag(p) - np.outer(p, p)
    return J.tolist()

P_HIGHER_SOFTMAX_JAC = Problem(
    id="higher_softmax_jacobian",
    category="higher_order",
    difficulty=2,
    description="Full Jacobian matrix of the softmax function.",
    prompt="""Compute the full n×n Jacobian matrix J_{ij} = ∂softmax(x)_i/∂x_j.

The softmax function is p_i = exp(x_i) / Σ_k exp(x_k).

```python
def solve(x: list) -> list:
    \"\"\"Return the n×n Jacobian matrix of softmax(x) as nested list.\"\"\"
```""",
    reference=_ref_higher_softmax_jacobian,
    test_cases=[
        {"inputs": {"x": [1, 2, 3, 4]}, "expected": _ref_higher_softmax_jacobian([1, 2, 3, 4])},
        {"inputs": {"x": [0, 0, 0]}, "expected": _ref_higher_softmax_jacobian([0, 0, 0])},
        {"inputs": {"x": [100, 101, 102]}, "expected": _ref_higher_softmax_jacobian([100, 101, 102])},
        {"inputs": {"x": [-500, 0, 500]}, "expected": _ref_higher_softmax_jacobian([-500, 0, 500])},
    ],
)


ALL = [P_HIGHER_GRAD_NORM, P_HIGHER_HESSIAN_LOGZ, P_HIGHER_TAYLOR_5TH, P_HIGHER_SOFTMAX_JAC]
