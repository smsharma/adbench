"""Category: ODE Sensitivity — hard cases.

Adjoint method, stiff systems, chaotic dynamics, and
sensitivity through non-trivial dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp

from adbench.problem import Problem


# --- Adjoint method for nonlinear ODE ---
def _ref_ode_adjoint(theta: float) -> float:
    dt = 1e-6
    def solve_ode(th):
        sol = solve_ivp(lambda t, y: [-y[0]**3 + np.sin(th * t)],
                        [0, 2], [1.0], rtol=1e-10, atol=1e-12, max_step=0.01)
        return sol.y[0, -1]
    return (solve_ode(theta + dt) - solve_ode(theta - dt)) / (2 * dt)

P_ODE_ADJOINT = Problem(
    id="ode_adjoint_nonlinear",
    category="ode_sensitivity",
    difficulty=3,
    description="Sensitivity of nonlinear ODE solution w.r.t. parameter.",
    prompt="""dy/dt = -y³ + sin(θt),  y(0) = 1.

Compute dy(2)/dθ (sensitivity of the solution at T=2 to the parameter θ).

```python
def solve(theta: float) -> float:
    \"\"\"Return dy(2)/dtheta for dy/dt = -y^3 + sin(theta*t), y(0)=1.\"\"\"
```""",
    reference=_ref_ode_adjoint,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_ode_adjoint(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=1e-2,
)


# --- Stiff ODE sensitivity ---
def _ref_ode_stiff(k: float) -> float:
    dt = 1e-6
    def solve_ode(kv):
        sol = solve_ivp(lambda t, y: [-kv * (y[0] - np.sin(t)) + np.cos(t)],
                        [0, 1], [0.0], method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[0, -1]
    return (solve_ode(k + dt) - solve_ode(k - dt)) / (2 * dt)

P_ODE_STIFF = Problem(
    id="ode_stiff_sensitivity",
    category="ode_sensitivity",
    difficulty=2,
    description="Sensitivity of stiff ODE solution requiring implicit integration.",
    prompt="""dy/dt = -k(y - sin(t)) + cos(t),  y(0) = 0.

Compute dy(1)/dk at T=1 (sensitivity to stiffness parameter k).

Note: For large k, the sensitivity equation is also stiff.

```python
def solve(k: float) -> float:
    \"\"\"Return dy(1)/dk for the stiff ODE.\"\"\"
```""",
    reference=_ref_ode_stiff,
    test_cases=[
        {"inputs": {"k": v}, "expected": _ref_ode_stiff(v)}
        for v in [10.0, 50.0, 100.0, 500.0]
    ],
    rtol=1e-2,
)


# --- BVP sensitivity ---
def _ref_bvp_sensitivity(alpha: float) -> float:
    s = np.sin(alpha)
    c = np.cos(alpha)
    s05 = np.sin(0.5 * alpha)
    c05 = np.cos(0.5 * alpha)
    return (0.5 * c05 * s - s05 * c) / s**2

P_ODE_BVP = Problem(
    id="ode_bvp_sensitivity",
    category="ode_sensitivity",
    difficulty=3,
    description="Sensitivity of boundary value problem solution to parameter.",
    prompt="""Boundary value problem:
  y'' + α²y = 0,  y(0) = 0,  y(1) = 1.

Compute dy(0.5)/dα (sensitivity of the solution at x=0.5 to the parameter α).

```python
def solve(alpha: float) -> float:
    \"\"\"Return dy(0.5)/dalpha for the BVP y'' + alpha^2 y = 0, y(0)=0, y(1)=1.\"\"\"
```""",
    reference=_ref_bvp_sensitivity,
    test_cases=[
        {"inputs": {"alpha": v}, "expected": _ref_bvp_sensitivity(v)}
        for v in [0.5, 1.0, 1.5, 2.0, 2.5]
    ],
    rtol=1e-3,
)


# --- NEW: Coupled ODE system with matrix exponential sensitivity ---
# dx/dt = A(θ)x, x(0) = [1,0,0]
# A depends on θ. Compute d(x₁(T))/dθ.
# Requires: either augmented sensitivity system or matrix exponential derivative.

def _make_A_coupled(theta):
    return np.array([
        [-1 - theta, 0.5, 0.2*theta],
        [0.3*theta, -2, 0.1],
        [0.1, 0.4*theta, -1.5 - 0.5*theta]
    ])

def _ref_ode_matrix_exp_sensitivity(theta: float) -> float:
    dt = 1e-6
    T = 3.0
    x0 = np.array([1.0, 0.0, 0.0])
    def solve_sys(th):
        A = _make_A_coupled(th)
        sol = solve_ivp(lambda t, x: A @ x, [0, T], x0, rtol=1e-10, atol=1e-12)
        return sol.y[0, -1]
    return (solve_sys(theta + dt) - solve_sys(theta - dt)) / (2 * dt)

P_ODE_MATRIX_EXP = Problem(
    id="ode_matrix_exp_sensitivity",
    category="ode_sensitivity",
    difficulty=3,
    description="Sensitivity of linear ODE system with parameter-dependent matrix.",
    prompt="""dx/dt = A(θ)x,  x(0) = [1, 0, 0],  T = 3

where A(θ) = [[-1-θ,  0.5,    0.2θ ],
               [0.3θ,  -2,     0.1  ],
               [0.1,   0.4θ,  -1.5-0.5θ]]

Compute dx₁(3)/dθ (sensitivity of first component at T=3).

```python
def solve(theta: float) -> float:
    \"\"\"Return dx1(3)/dtheta for the parameterized linear ODE system.\"\"\"
```""",
    reference=_ref_ode_matrix_exp_sensitivity,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_ode_matrix_exp_sensitivity(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=1e-2,
)


# --- NEW: Periodic orbit sensitivity ---
# Van der Pol oscillator: x'' - μ(1-x²)x' + x = 0
# Compute d(period)/dμ near μ=1.
# The period depends on μ in a non-trivial way.
# Requires: finding the period numerically (zero-crossing), then differentiating.

def _find_period_vdp(mu, x0=2.0):
    """Find the period of Van der Pol oscillator by detecting zero crossings."""
    def rhs(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
    # Integrate long enough and find period from zero crossings
    sol = solve_ivp(rhs, [0, 50], [x0, 0], max_step=0.01, rtol=1e-10, atol=1e-12)
    t = sol.t
    x = sol.y[0]
    # Find zero crossings (positive going)
    crossings = []
    for i in range(1, len(x)):
        if x[i-1] < 0 and x[i] >= 0:
            # Linear interpolation for crossing time
            tc = t[i-1] + (t[i] - t[i-1]) * (-x[i-1]) / (x[i] - x[i-1])
            crossings.append(tc)
    if len(crossings) >= 3:
        # Use last few crossings for period estimate (skip transient)
        periods = [crossings[j+1] - crossings[j] for j in range(len(crossings)-1)]
        return np.mean(periods[-3:])
    return None

def _ref_ode_period_sensitivity(mu: float) -> float:
    dm = 1e-5
    T_plus = _find_period_vdp(mu + dm)
    T_minus = _find_period_vdp(mu - dm)
    if T_plus is None or T_minus is None:
        return 0.0
    return (T_plus - T_minus) / (2 * dm)

P_ODE_PERIOD = Problem(
    id="ode_period_sensitivity",
    category="ode_sensitivity",
    difficulty=3,
    description="Sensitivity of limit cycle period of Van der Pol oscillator to damping parameter.",
    prompt="""Van der Pol oscillator:
  x'' - μ(1-x²)x' + x = 0

with initial conditions x(0)=2, x'(0)=0.

After transients die out, the system reaches a limit cycle with period T(μ).

Compute dT/dμ.

```python
def solve(mu: float) -> float:
    \"\"\"Return d(period)/d(mu) for the Van der Pol oscillator.\"\"\"
```""",
    reference=_ref_ode_period_sensitivity,
    test_cases=[
        {"inputs": {"mu": v}, "expected": _ref_ode_period_sensitivity(v)}
        for v in [0.5, 1.0, 1.5, 2.0, 3.0]
    ],
    rtol=5e-2,
)


ALL = [P_ODE_ADJOINT, P_ODE_STIFF, P_ODE_BVP, P_ODE_MATRIX_EXP, P_ODE_PERIOD]
