"""Category 1: Implicit Differentiation — hard cases.

These problems require recognizing implicit structure and applying IFT
without being told to do so. No hints about method.
"""

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.linalg import solve_sylvester

from adbench.problem import Problem


# --- 1.1: Fixed point of a contraction map ---
# y* satisfies y* = θ sin(y*) + 0.5 for given θ.
# The model must: find y* numerically, then apply IFT:
#   dy*/dθ = sin(y*) / (1 - θ cos(y*))
# Trap: unrolling many iterations of y_{n+1} = θ sin(y_n) + 0.5
# and differentiating through the loop is fragile and biased
# for θ near the edge of contraction.

def _find_fixed_point(theta):
    return brentq(lambda y: theta * np.sin(y) + 0.5 - y, -10, 10)

def _ref_implicit_fixed_point(theta: float) -> float:
    y_star = _find_fixed_point(theta)
    return np.sin(y_star) / (1 - theta * np.cos(y_star))

P_IMPLICIT_FIXED_POINT = Problem(
    id="implicit_fixed_point",
    category="implicit",
    difficulty=2,
    description="Derivative of fixed point y* = θ sin(y*) + 0.5 w.r.t. θ.",
    prompt="""The equation y = θ sin(y) + 0.5 defines y as an implicit function of θ.

Compute dy/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return dy/dtheta where y satisfies y = theta * sin(y) + 0.5.\"\"\"
```""",
    reference=_ref_implicit_fixed_point,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_implicit_fixed_point(v)}
        for v in [0.1, 0.5, 0.8, 0.95, -0.5]
    ],
)


# --- 1.2: Stationary distribution of parameterized Markov chain ---
# P(θ) is a 3x3 row-stochastic matrix parameterized by θ.
# π(θ) P(θ) = π(θ), sum π = 1.
# Compute dπ_1/dθ.
# The model must: solve for π, then implicitly differentiate the
# eigenvalue equation. No closed form.

def _make_transition_matrix(theta):
    """3x3 transition matrix parameterized by theta."""
    # Ensure row-stochastic
    p = 0.5 + 0.3 * np.tanh(theta)  # varies in (0.2, 0.8)
    q = 0.3 + 0.2 * np.tanh(theta)
    P = np.array([
        [1 - p,     p/2,       p/2],
        [q/3,       1 - q,     2*q/3],
        [0.2,       0.3,       0.5],
    ])
    return P

def _stationary_dist(P):
    """Compute stationary distribution of row-stochastic matrix P."""
    n = P.shape[0]
    # Solve π P = π, sum π = 1
    # Equivalent to π (P - I) = 0, sum π = 1
    A = (P.T - np.eye(n))
    A[-1, :] = 1.0  # Replace last equation with normalization
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)

def _ref_implicit_markov(theta: float) -> float:
    dt = 1e-7
    P_plus = _make_transition_matrix(theta + dt)
    P_minus = _make_transition_matrix(theta - dt)
    pi_plus = _stationary_dist(P_plus)
    pi_minus = _stationary_dist(P_minus)
    return (pi_plus[0] - pi_minus[0]) / (2 * dt)

P_IMPLICIT_MARKOV = Problem(
    id="implicit_markov_stationary",
    category="implicit",
    difficulty=3,
    description="Derivative of stationary distribution π₁ of a parameterized Markov chain.",
    prompt="""A 3-state Markov chain has transition matrix P(θ) where:

P(θ) = [[1-p,   p/2,   p/2  ],
         [q/3,   1-q,   2q/3 ],
         [0.2,   0.3,   0.5  ]]

with p = 0.5 + 0.3*tanh(θ) and q = 0.3 + 0.2*tanh(θ).

The stationary distribution π satisfies π P = π, Σπ_i = 1.

Compute dπ₁/dθ (derivative of the first component of the stationary distribution).

```python
def solve(theta: float) -> float:
    \"\"\"Return d(pi_1)/d(theta) for the parameterized Markov chain.\"\"\"
```""",
    reference=_ref_implicit_markov,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_implicit_markov(v)}
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]
    ],
    rtol=1e-3,
)


# --- 1.3: Implicit through coupled nonlinear system ---
# x, y satisfy: x³ + y³ = 6α and x²y + xy² = 6α
# Both parameterized by α. Compute dx/dα at α=1.
# These two equations combined give x³ + y³ = x²y + xy², i.e., (x-y)(x²+2xy+y²-xy) = 0
# which simplifies. The model needs to work with the full system.

def _ref_implicit_nonlinear_system(alpha: float) -> float:
    dt = 1e-7
    def solve_sys(a):
        def eqs(vars):
            x, y = vars
            return [x**3 + y**3 - 6*a, x**2*y + x*y**2 - 6*a]
        sol = fsolve(eqs, [1.0, 1.0], full_output=False)
        return sol[0]  # return x
    return (solve_sys(alpha + dt) - solve_sys(alpha - dt)) / (2 * dt)

P_IMPLICIT_NONLINEAR = Problem(
    id="implicit_nonlinear_system",
    category="implicit",
    difficulty=3,
    description="Derivative dx/dα where (x,y) solve a coupled nonlinear system parameterized by α.",
    prompt="""The system of equations:
  x³ + y³ = 6α
  x²y + xy² = 6α

defines x and y as functions of α. Starting from the solution near x=y=1 at α=1,
compute dx/dα.

```python
def solve(alpha: float) -> float:
    \"\"\"Return dx/dalpha where (x,y) solve x^3+y^3=6*alpha, x^2*y+x*y^2=6*alpha.\"\"\"
```""",
    reference=_ref_implicit_nonlinear_system,
    test_cases=[
        {"inputs": {"alpha": v}, "expected": _ref_implicit_nonlinear_system(v)}
        for v in [0.5, 0.8, 1.0, 1.5, 2.0]
    ],
    rtol=1e-3,
)


# --- 1.4: Lyapunov equation sensitivity (harder version) ---
# A(t) is 3x3, Q is non-identity. Compute d(det(X))/dt.
# This combines Lyapunov solve + Jacobi's formula + implicit differentiation.

def _make_A_lyap(t):
    return np.array([
        [-1 - t**2,  0.5*t,     0.1],
        [0.3,       -2 + 0.5*t, 0.2*t],
        [0.1*t,      0.2,      -3 + 0.1*t**2],
    ])

def _make_Q_lyap():
    return np.array([[2, 0.5, 0], [0.5, 1, 0.3], [0, 0.3, 1.5]])

def _ref_implicit_lyapunov_det(t: float) -> float:
    dt = 1e-7
    Q = _make_Q_lyap()
    def solve_lyap(tv):
        A = _make_A_lyap(tv)
        X = solve_sylvester(A, A.T, -Q)
        return np.linalg.det(X)
    return (solve_lyap(t + dt) - solve_lyap(t - dt)) / (2 * dt)

P_IMPLICIT_LYAPUNOV = Problem(
    id="implicit_lyapunov_det",
    category="implicit",
    difficulty=3,
    description="Derivative of det(X(t)) where X solves a 3x3 Lyapunov equation.",
    prompt="""The continuous Lyapunov equation A(t)X + XA(t)ᵀ + Q = 0 defines X(t), where:

A(t) = [[-1-t²,   0.5t,     0.1    ],
         [0.3,     -2+0.5t,  0.2t   ],
         [0.1t,    0.2,      -3+0.1t²]]

Q = [[2, 0.5, 0], [0.5, 1, 0.3], [0, 0.3, 1.5]]

Compute d(det(X))/dt.

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt det(X(t)) where X solves A(t)X + XA(t)^T + Q = 0.\"\"\"
```""",
    reference=_ref_implicit_lyapunov_det,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_implicit_lyapunov_det(v)}
        for v in [0.0, 0.5, 1.0, 1.5]
    ],
    rtol=1e-3,
)


ALL = [P_IMPLICIT_FIXED_POINT, P_IMPLICIT_MARKOV, P_IMPLICIT_NONLINEAR, P_IMPLICIT_LYAPUNOV]
