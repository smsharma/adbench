"""Category 4: Matrix Calculus — hard cases.

Fréchet derivatives of matrix functions, differentiation through
decompositions, and cases where standard formulas break down.
"""

import numpy as np
from scipy.linalg import sqrtm, expm, logm, cho_factor, cho_solve

from adbench.problem import Problem


# --- 4.1: Matrix square root derivative ---
# d/dt tr(sqrtm(A(t))) — no simple formula like for exp or inverse.
# Requires solving a Sylvester equation or using the integral representation
# of the Fréchet derivative.

def _make_A_sqrt(t):
    return np.array([
        [3 + t,   0.5,     0.2],
        [0.5,     2 + 0.5*t, 0.3],
        [0.2,     0.3,     1 + 0.3*t],
    ])

def _ref_matrix_sqrt_deriv(t: float) -> float:
    dt = 1e-7
    f_plus = np.trace(sqrtm(_make_A_sqrt(t + dt)).real)
    f_minus = np.trace(sqrtm(_make_A_sqrt(t - dt)).real)
    return (f_plus - f_minus) / (2 * dt)

P_MATRIX_SQRT = Problem(
    id="matrix_sqrt_deriv",
    category="matrix_calculus",
    difficulty=3,
    description="Derivative of tr(sqrtm(A(t))) for a parameterized PD matrix.",
    prompt="""Compute d/dt tr(A(t)^{1/2}) where:

A(t) = [[3+t,    0.5,    0.2  ],
         [0.5,    2+0.5t, 0.3  ],
         [0.2,    0.3,    1+0.3t]]

A(t)^{1/2} is the unique positive definite square root.

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt tr(sqrtm(A(t))).\"\"\"
```""",
    reference=_ref_matrix_sqrt_deriv,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_matrix_sqrt_deriv(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 5.0]
    ],
    rtol=1e-3,
)


# --- 4.2: Derivative through Cholesky ---
# Given SPD matrix A(t), A = LL^T. Compute d/dt L_{21}.
# The backward pass through Cholesky is non-trivial — requires
# a specific recursive formula.

def _make_A_chol(t):
    return np.array([
        [4 + t,    1 + 0.3*t],
        [1 + 0.3*t, 3 + 0.5*t],
    ])

def _ref_cholesky_entry(t: float) -> float:
    dt = 1e-7
    L_plus = np.linalg.cholesky(_make_A_chol(t + dt))
    L_minus = np.linalg.cholesky(_make_A_chol(t - dt))
    return (L_plus[1, 0] - L_minus[1, 0]) / (2 * dt)

P_MATRIX_CHOLESKY = Problem(
    id="matrix_cholesky_deriv",
    category="matrix_calculus",
    difficulty=2,
    description="Derivative of a specific entry of the Cholesky factor L w.r.t. parameter t.",
    prompt="""A(t) = [[4+t,      1+0.3t],
         [1+0.3t,  3+0.5t]]

is SPD for the test range. Let L(t) be its lower Cholesky factor (A = LL^T).
Compute dL₂₁/dt (derivative of the (2,1) entry of L).

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt L[1,0] where L is Cholesky factor of A(t).\"\"\"
```""",
    reference=_ref_cholesky_entry,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_cholesky_entry(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=1e-3,
)


# --- 4.3: Nuclear norm gradient ---
# ||A||_* = sum of singular values. Gradient w.r.t. A is U V^T
# where A = U Σ V^T is the SVD. Test with a specific parameterized matrix.

def _make_A_nuc(t):
    return np.array([
        [1 + t,   0.5*t,    0.1],
        [0.2,     2 - t,    0.3*t],
    ])

def _ref_nuclear_norm_deriv(t: float) -> float:
    """d/dt ||A(t)||_* = d/dt sum(sigma_i)."""
    dt = 1e-7
    def nuc_norm(tv):
        return np.sum(np.linalg.svd(_make_A_nuc(tv), compute_uv=False))
    return (nuc_norm(t + dt) - nuc_norm(t - dt)) / (2 * dt)

P_MATRIX_NUCLEAR = Problem(
    id="matrix_nuclear_norm",
    category="matrix_calculus",
    difficulty=2,
    description="Derivative of nuclear norm ||A(t)||_* for a parameterized matrix.",
    prompt="""A(t) = [[1+t,  0.5t,  0.1 ],
         [0.2,  2-t,   0.3t]]

Compute d/dt ||A(t)||_* where ||·||_* is the nuclear norm (sum of singular values).

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt of the nuclear norm of A(t).\"\"\"
```""",
    reference=_ref_nuclear_norm_deriv,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_nuclear_norm_deriv(v)}
        for v in [0.0, 0.5, 1.0, 2.0]
    ],
    rtol=1e-3,
)


# --- 4.4: Fréchet derivative of matrix exponential ---
# Compute d/dt tr(B expm(tA)) where A and B are fixed matrices.
# Note: tr(B expm(tA))' = tr(B A expm(tA)) only when A and B commute.
# In general, need the full Fréchet derivative.

_A_exp = np.array([[1, 0.5, 0.2], [-0.3, 2, 0.1], [0.1, -0.2, 0.8]])
_B_exp = np.array([[1, 0, 0.5], [0, 2, 0], [0.5, 0, 1]])

def _ref_matrix_frechet_exp(t: float) -> float:
    dt = 1e-7
    f_plus = np.trace(_B_exp @ expm(t * _A_exp))
    f_minus = np.trace(_B_exp @ expm((t - dt) * _A_exp))
    return (f_plus - f_minus) / dt

P_MATRIX_FRECHET_EXP = Problem(
    id="matrix_frechet_exp",
    category="matrix_calculus",
    difficulty=3,
    description="Derivative of tr(B expm(tA)) where A and B don't commute.",
    prompt="""Let A = [[1, 0.5, 0.2], [-0.3, 2, 0.1], [0.1, -0.2, 0.8]]
and B = [[1, 0, 0.5], [0, 2, 0], [0.5, 0, 1]].

Compute d/dt tr(B exp(tA)).

Note: A and B do not commute, so tr(B A exp(tA)) is not the correct formula.

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt tr(B @ expm(t*A)).\"\"\"
```""",
    reference=_ref_matrix_frechet_exp,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_matrix_frechet_exp(v)}
        for v in [0.0, 0.5, 1.0, 2.0]
    ],
    rtol=1e-3,
)


# --- 4.5: Matrix log derivative ---
# d/dt det(logm(A(t))) where A(t) is parameterized.
# Requires: computing matrix log, then its determinant, then differentiating.

def _make_A_log(t):
    # Eigenvalues must be positive for real log
    return np.array([
        [3 + t, 0.5],
        [0.5,   2 + 0.3*t],
    ])

def _ref_matrix_log_det(t: float) -> float:
    dt = 1e-7
    def f(tv):
        return np.linalg.det(logm(_make_A_log(tv)).real)
    return (f(t + dt) - f(t - dt)) / (2 * dt)

P_MATRIX_LOG_DET = Problem(
    id="matrix_log_det_deriv",
    category="matrix_calculus",
    difficulty=3,
    description="Derivative of det(logm(A(t))) for a parameterized matrix.",
    prompt="""A(t) = [[3+t, 0.5], [0.5, 2+0.3t]]

Compute d/dt det(log(A(t))) where log is the matrix logarithm.

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt det(logm(A(t))).\"\"\"
```""",
    reference=_ref_matrix_log_det,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_matrix_log_det(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=1e-3,
)


ALL = [P_MATRIX_SQRT, P_MATRIX_CHOLESKY, P_MATRIX_NUCLEAR, P_MATRIX_FRECHET_EXP, P_MATRIX_LOG_DET]
