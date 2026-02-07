"""Category: Numerical Traps — problems where naive implementations fail.

These test whether the model understands numerical stability and can
avoid catastrophic cancellation, overflow, and other pitfalls.
"""

import numpy as np
from scipy.special import logsumexp as _logsumexp

from adbench.problem import Problem


# --- 1: Gradient of log-sum-exp at extreme values ---
# naive: exp overflows for large x. Must use the stable form.
# d/dx_k log(Σ exp(x_i)) = exp(x_k) / Σ exp(x_i) = softmax(x)_k
# But test at x = [1000, 1001, 999] where naive exp overflows.

def _ref_logsumexp_grad(x: list, k: int) -> float:
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e[k] / np.sum(e)

P_TRAP_LOGSUMEXP = Problem(
    id="trap_logsumexp_grad",
    category="numerical_traps",
    difficulty=2,
    description="Gradient of log-sum-exp at extreme values (overflow trap).",
    prompt="""Compute ∂/∂x_k log(Σᵢ exp(xᵢ)) for a given vector x and index k.

```python
def solve(x: list, k: int) -> float:
    \"\"\"Return d/dx_k log(sum(exp(x_i))) — must be numerically stable.\"\"\"
```""",
    reference=_ref_logsumexp_grad,
    test_cases=[
        {"inputs": {"x": [1, 2, 3], "k": 2}, "expected": _ref_logsumexp_grad([1, 2, 3], 2)},
        {"inputs": {"x": [1000, 1001, 999], "k": 1}, "expected": _ref_logsumexp_grad([1000, 1001, 999], 1)},
        {"inputs": {"x": [-1000, -999, -1001], "k": 1}, "expected": _ref_logsumexp_grad([-1000, -999, -1001], 1)},
        {"inputs": {"x": [0, 0, 0, 0], "k": 0}, "expected": 0.25},
    ],
)


# --- 2: Derivative of (exp(x)-1-x)/x² at x near 0 ---
# True value at x=0: 1/2 (by L'Hôpital or Taylor).
# f(x) = (exp(x) - 1 - x) / x² for x ≠ 0.
# f'(x) = [(x-2)exp(x) + 2 + x] / x³ for x ≠ 0.
# At x→0: f'(0) = 1/6 by Taylor expansion.
# Trap: catastrophic cancellation in numerator for small x.

def _ref_trap_cancellation(x: float) -> float:
    if abs(x) < 1e-4:
        # Taylor: f(x) = 1/2 + x/6 + x²/24 + ...
        # f'(x) = 1/6 + x/12 + x²/40 + ...
        return 1/6 + x/12 + x**2/40 + x**3/180
    return ((x - 2) * np.exp(x) + 2 + x) / x**3

P_TRAP_CANCELLATION = Problem(
    id="trap_catastrophic_cancellation",
    category="numerical_traps",
    difficulty=3,
    description="Derivative of (exp(x)-1-x)/x² near x=0 (catastrophic cancellation).",
    prompt="""f(x) = (exp(x) - 1 - x) / x²  for x ≠ 0, and f(0) = 1/2.

Compute f'(x). Your implementation must be accurate for x near 0 (e.g., x = 1e-8).

```python
def solve(x: float) -> float:
    \"\"\"Return df/dx for f(x) = (exp(x)-1-x)/x^2. Must handle x near 0.\"\"\"
```""",
    reference=_ref_trap_cancellation,
    test_cases=[
        {"inputs": {"x": v}, "expected": _ref_trap_cancellation(v)}
        for v in [1.0, 0.1, 0.001, 1e-8, 0.0, -0.5, -1e-6]
    ],
    rtol=1e-3,
)


# --- 3: Gradient of softmax cross-entropy (numerically stable combined form) ---
# L = -log(softmax(x)_k) = -x_k + log(Σ exp(x_i))
# dL/dx_j = softmax(x)_j - δ_{jk}
# Test at extreme logits where naive implementation fails.

def _ref_trap_xent_grad(x: list, k: int) -> list:
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    p = e / np.sum(e)
    grad = p.copy()
    grad[k] -= 1.0
    return grad.tolist()

P_TRAP_XENT = Problem(
    id="trap_softmax_xent_grad",
    category="numerical_traps",
    difficulty=2,
    description="Gradient of softmax cross-entropy at extreme logits.",
    prompt="""Cross-entropy loss: L(x, k) = -log(softmax(x)_k) where k is the target class.

Compute ∇_x L (gradient w.r.t. all logits x).

```python
def solve(x: list, k: int) -> list:
    \"\"\"Return gradient dL/dx_j for all j. Must handle extreme logit values.\"\"\"
```""",
    reference=_ref_trap_xent_grad,
    test_cases=[
        {"inputs": {"x": [1, 2, 3], "k": 0}, "expected": _ref_trap_xent_grad([1, 2, 3], 0)},
        {"inputs": {"x": [100, 200, 300], "k": 2}, "expected": _ref_trap_xent_grad([100, 200, 300], 2)},
        {"inputs": {"x": [-500, 0, 500], "k": 2}, "expected": _ref_trap_xent_grad([-500, 0, 500], 2)},
        {"inputs": {"x": [0, 0, 0, 0, 0], "k": 3}, "expected": _ref_trap_xent_grad([0, 0, 0, 0, 0], 3)},
    ],
)


# --- 4: Fisher information of a mixture model ---
# p(x|θ) = θ N(x;0,1) + (1-θ) N(x;3,1)
# Fisher info I(θ) = E[(d log p / dθ)²]
# = ∫ [N(x;0,1) - N(x;3,1)]² / [θ N(x;0,1) + (1-θ) N(x;3,1)] dx
# This must be computed numerically and has no closed form.

def _ref_fisher_info(theta: float) -> float:
    from scipy.integrate import quad
    from scipy.stats import norm

    def integrand(x):
        p0 = norm.pdf(x, 0, 1)
        p1 = norm.pdf(x, 3, 1)
        p_mix = theta * p0 + (1 - theta) * p1
        if p_mix < 1e-300:
            return 0.0
        return (p0 - p1)**2 / p_mix

    result, _ = quad(integrand, -10, 15, limit=200)
    return result

P_TRAP_FISHER = Problem(
    id="trap_fisher_mixture",
    category="numerical_traps",
    difficulty=3,
    description="Fisher information of a Gaussian mixture model (no closed form).",
    prompt="""A Gaussian mixture model:
  p(x|θ) = θ N(x; 0, 1) + (1-θ) N(x; 3, 1)

Compute the Fisher information I(θ) = E[(∂ log p(X|θ) / ∂θ)²].

This equals ∫ [∂p/∂θ]² / p(x|θ) dx = ∫ [N(x;0,1) - N(x;3,1)]² / p(x|θ) dx.

```python
def solve(theta: float) -> float:
    \"\"\"Return Fisher information I(theta) for the Gaussian mixture.\"\"\"
```""",
    reference=_ref_fisher_info,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_fisher_info(v)}
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    rtol=1e-2,
)


ALL = [P_TRAP_LOGSUMEXP, P_TRAP_CANCELLATION, P_TRAP_XENT, P_TRAP_FISHER]
