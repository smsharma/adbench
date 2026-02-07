"""Category 3: Differentiating Through Optimization — hard cases.

Problems requiring differentiation through KKT conditions, envelope theorem,
or bilevel optimization with no closed-form inner solution.
"""

import numpy as np
from scipy.optimize import minimize, linprog

from adbench.problem import Problem


# --- 3.1: QP with active constraint ---
# min_x 0.5*x^T Q x + c(θ)^T x  s.t. Ax ≤ b
# As θ changes, active set may change. Compute dx*/dθ at a point
# where the active set is stable.

_Q_qp = np.array([[2.0, 0.5], [0.5, 1.0]])
_A_qp = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
_b_qp = np.array([1.0, 0.0, 0.0])

def _ref_opt_qp(theta: float) -> list:
    """dx*/dtheta for QP: min 0.5 x'Qx + [theta, 1-theta]'x s.t. Ax ≤ b."""
    dt = 1e-7
    def solve_qp(th):
        c = np.array([th, 1 - th])
        res = minimize(lambda x: 0.5 * x @ _Q_qp @ x + c @ x, [0.3, 0.3],
                       method='SLSQP',
                       constraints={'type': 'ineq', 'fun': lambda x: _b_qp - _A_qp @ x})
        return res.x
    x_plus = solve_qp(theta + dt)
    x_minus = solve_qp(theta - dt)
    return ((x_plus - x_minus) / (2 * dt)).tolist()

P_OPT_QP = Problem(
    id="opt_qp_kkt",
    category="optimization",
    difficulty=3,
    description="Differentiate solution of a QP through KKT conditions.",
    prompt="""Consider the quadratic program:
  min_x  0.5 x^T Q x + c(θ)^T x
  s.t.   Ax ≤ b

where Q = [[2, 0.5], [0.5, 1]], c(θ) = [θ, 1-θ],
A = [[1,1],[-1,0],[0,-1]], b = [1,0,0].

Compute dx*/dθ (a 2-element vector) where x* is the optimal solution.

```python
def solve(theta: float) -> list:
    \"\"\"Return [dx1*/dtheta, dx2*/dtheta] for the parameterized QP.\"\"\"
```""",
    reference=_ref_opt_qp,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_opt_qp(v)}
        for v in [-0.5, 0.0, 0.3, 0.7, 1.5]
    ],
    rtol=1e-2,
)


# --- 3.2: Envelope theorem ---
# V(θ) = min_x f(x, θ). By envelope theorem, dV/dθ = ∂f/∂θ|_{x=x*(θ)}.
# The model should NOT differentiate through the argmin.
# f(x, θ) = (x - θ)⁴ + θ²x²
# x* must be found numerically. Then dV/dθ = ∂f/∂θ = 2θ x*²

def _ref_opt_envelope(theta: float) -> float:
    res = minimize(lambda x: (x[0] - theta)**4 + theta**2 * x[0]**2, [0.0], method='Nelder-Mead')
    x_star = res.x[0]
    # dV/dθ = ∂f/∂θ at x* = -4(x*-θ)³ + 2θ x*²
    return -4 * (x_star - theta)**3 + 2 * theta * x_star**2

P_OPT_ENVELOPE = Problem(
    id="opt_envelope",
    category="optimization",
    difficulty=2,
    description="Apply envelope theorem: dV/dθ = ∂f/∂θ evaluated at the optimum.",
    prompt="""Define V(θ) = min_x [(x - θ)⁴ + θ²x²].

Compute dV/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return dV/dtheta where V(theta) = min_x [(x-theta)^4 + theta^2 * x^2].\"\"\"
```""",
    reference=_ref_opt_envelope,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_opt_envelope(v)}
        for v in [0.0, 0.5, 1.0, 2.0, -1.0]
    ],
    rtol=1e-2,
)


# --- 3.3: Bilevel where inner has no closed form ---
# Outer: L(θ) = ||w*(θ)||²
# Inner: w*(θ) = argmin_w [ sum_i log(1 + exp(-y_i (X_i · w))) + θ ||w||² ]
# (Regularized logistic regression)
# Compute dL/dθ using implicit differentiation through the optimality conditions.

_X_logistic = np.array([[1.0, 0.5], [0.3, -1.0], [-0.5, 0.8], [1.2, 0.1], [-0.3, -0.7]])
_y_logistic = np.array([1.0, -1.0, 1.0, 1.0, -1.0])

def _solve_logistic(theta):
    def loss(w):
        z = _y_logistic * (_X_logistic @ w)
        return np.sum(np.logaddexp(0, -z)) + theta * np.sum(w**2)
    res = minimize(loss, np.zeros(2), method='L-BFGS-B')
    return res.x

def _ref_opt_bilevel_logistic(theta: float) -> float:
    dt = 1e-6
    w_plus = _solve_logistic(theta + dt)
    w_minus = _solve_logistic(theta - dt)
    # L = ||w||²
    L_plus = np.sum(w_plus**2)
    L_minus = np.sum(w_minus**2)
    return (L_plus - L_minus) / (2 * dt)

P_OPT_BILEVEL = Problem(
    id="opt_bilevel_logistic",
    category="optimization",
    difficulty=3,
    description="Differentiate ||w*||² w.r.t. regularization θ in logistic regression.",
    prompt="""Regularized logistic regression:
  w*(θ) = argmin_w [Σᵢ log(1 + exp(-yᵢ (Xᵢ·w))) + θ ||w||²]

with X = [[1,0.5],[0.3,-1],[-0.5,0.8],[1.2,0.1],[-0.3,-0.7]]
and  y = [1, -1, 1, 1, -1].

Compute d(||w*(θ)||²)/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return d(||w*||^2)/dtheta for regularized logistic regression.\"\"\"
```""",
    reference=_ref_opt_bilevel_logistic,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_opt_bilevel_logistic(v)}
        for v in [0.01, 0.1, 0.5, 1.0, 5.0]
    ],
    rtol=5e-2,
)


# --- 3.4: LP sensitivity / parametric LP ---
# min c(θ)^T x  s.t. Ax ≤ b, x ≥ 0
# The optimal value V(θ) is piecewise linear in θ. Compute dV/dθ
# when the basis is non-degenerate (derivative exists).

def _ref_opt_lp_sensitivity(theta: float) -> float:
    c = np.array([1.0 + theta, 2.0 - theta])
    A_ub = np.array([[-1, -1], [-2, -1]])
    b_ub = np.array([-1, -1.5])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])
    x_star = res.x
    # dV/dθ = (dc/dθ)^T x* = x*[0] - x*[1]
    return x_star[0] - x_star[1]

P_OPT_LP = Problem(
    id="opt_lp_sensitivity",
    category="optimization",
    difficulty=2,
    description="Sensitivity of LP optimal value to cost vector perturbation.",
    prompt="""Linear program:
  min  [1+θ, 2-θ]^T x
  s.t.  x₁ + x₂ ≥ 1
        2x₁ + x₂ ≥ 1.5
        x ≥ 0

Compute dV/dθ where V(θ) is the optimal value.

```python
def solve(theta: float) -> float:
    \"\"\"Return dV/dtheta for the parameterized LP.\"\"\"
```""",
    reference=_ref_opt_lp_sensitivity,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_opt_lp_sensitivity(v)}
        for v in [-0.5, 0.0, 0.3, 0.8]
    ],
    rtol=1e-2,
)


ALL = [P_OPT_QP, P_OPT_ENVELOPE, P_OPT_BILEVEL, P_OPT_LP]
