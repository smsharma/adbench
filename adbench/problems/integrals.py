"""Category 2: Differentiation Under the Integral Sign — hard cases.

Problems where naive numerical differentiation or simple Leibniz
application fails due to singularities, oscillation, or
parameter-dependent domains.
"""

import numpy as np
from scipy import integrate
from scipy.special import ellipk, ellipe

from adbench.problem import Problem


# --- 2.1: Oscillatory integral with parameter-dependent decay ---
# I(ω) = ∫₀^∞ sin(ωt)/(1+t²) dt = (π/2) * e^{-ω} for ω > 0
# dI/dω = ∫₀^∞ t cos(ωt)/(1+t²) dt = -(π/2) e^{-ω}
# Trap: numerical integration of oscillatory integrand is hard;
# naive quad will be inaccurate for large ω.

def _ref_integral_oscillatory(omega: float) -> float:
    return -(np.pi / 2) * np.exp(-omega)

P_INTEGRAL_OSCILLATORY = Problem(
    id="integral_oscillatory",
    category="integrals",
    difficulty=2,
    description="Derivative of ∫₀^∞ sin(ωt)/(1+t²) dt w.r.t. ω.",
    prompt="""Compute dI/dω where I(ω) = ∫₀^∞ sin(ωt)/(1+t²) dt for ω > 0.

```python
def solve(omega: float) -> float:
    \"\"\"Return dI/domega where I(omega) = integral_0^inf sin(omega*t)/(1+t^2) dt.\"\"\"
```""",
    reference=_ref_integral_oscillatory,
    test_cases=[
        {"inputs": {"omega": v}, "expected": _ref_integral_oscillatory(v)}
        for v in [0.1, 0.5, 1.0, 2.0, 5.0]
    ],
)


# --- 2.2: Complete elliptic integral derivative ---
# K(k) = ∫₀^{π/2} 1/√(1-k²sin²θ) dθ
# dK/dk = E(k)/(k(1-k²)) - K(k)/k
# where E is the complete elliptic integral of the second kind.
# The model must know or derive this identity.

def _ref_integral_elliptic(k: float) -> float:
    return ellipe(k**2) / (k * (1 - k**2)) - ellipk(k**2) / k

P_INTEGRAL_ELLIPTIC = Problem(
    id="integral_elliptic",
    category="integrals",
    difficulty=3,
    description="Derivative of complete elliptic integral K(k) w.r.t. modulus k.",
    prompt="""The complete elliptic integral of the first kind is:

K(k) = ∫₀^{π/2} 1/√(1 - k²sin²θ) dθ

Compute dK/dk for 0 < k < 1.

```python
def solve(k: float) -> float:
    \"\"\"Return dK/dk for the complete elliptic integral of the first kind.\"\"\"
```""",
    reference=_ref_integral_elliptic,
    test_cases=[
        {"inputs": {"k": v}, "expected": _ref_integral_elliptic(v)}
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    rtol=1e-3,
)


# --- 2.3: Integral with parameter-dependent singularity ---
# I(a) = ∫₀¹ x^a ln(x) dx for a > -1
# = d/da [∫₀¹ x^a dx] = d/da [1/(a+1)] = -1/(a+1)²
# But also: directly I(a) = ∫₀¹ x^a ln(x) dx = -1/(a+1)²
# Trap: for a near -1, the integrand is nearly singular.
# Also: the model might try numerical integration + finite differences,
# which is noisy near a=-0.9.

def _ref_integral_singular(a: float) -> float:
    # d/da I(a) where I(a) = ∫₀¹ x^a ln(x) dx = -1/(a+1)²
    # dI/da = ∫₀¹ x^a (ln x)² dx = 2/(a+1)³
    return 2.0 / (a + 1)**3

P_INTEGRAL_SINGULAR = Problem(
    id="integral_param_singular",
    category="integrals",
    difficulty=2,
    description="Derivative of ∫₀¹ x^a ln(x) dx w.r.t. a (near-singular integrand).",
    prompt="""Compute dI/da where I(a) = ∫₀¹ x^a ln(x) dx for a > -1.

```python
def solve(a: float) -> float:
    \"\"\"Return dI/da where I(a) = integral_0^1 x^a * ln(x) dx.\"\"\"
```""",
    reference=_ref_integral_singular,
    test_cases=[
        {"inputs": {"a": v}, "expected": _ref_integral_singular(v)}
        for v in [-0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    ],
)


# --- 2.4: Laplace-type integral derivative ---
# I(λ) = ∫_{-∞}^{∞} exp(-λ(x⁴ - x²)) dx
# No closed form. Must differentiate under the integral:
# dI/dλ = -∫ (x⁴ - x²) exp(-λ(x⁴-x²)) dx
# Reference computed by numerical integration.

def _ref_integral_laplace(lam: float) -> float:
    def integrand(x):
        return -(x**4 - x**2) * np.exp(-lam * (x**4 - x**2))
    result, _ = integrate.quad(integrand, -10, 10, limit=200)
    return result

P_INTEGRAL_LAPLACE = Problem(
    id="integral_laplace",
    category="integrals",
    difficulty=3,
    description="Derivative of a Laplace-type integral ∫exp(-λ(x⁴-x²))dx w.r.t. λ.",
    prompt="""Compute dI/dλ where I(λ) = ∫_{-∞}^{∞} exp(-λ(x⁴ - x²)) dx for λ > 0.

```python
def solve(lam: float) -> float:
    \"\"\"Return dI/dlambda where I(lambda) = integral exp(-lambda*(x^4-x^2)) dx.\"\"\"
```""",
    reference=_ref_integral_laplace,
    test_cases=[
        {"inputs": {"lam": v}, "expected": _ref_integral_laplace(v)}
        for v in [0.5, 1.0, 2.0, 5.0, 10.0]
    ],
    rtol=1e-3,
)


ALL = [P_INTEGRAL_OSCILLATORY, P_INTEGRAL_ELLIPTIC, P_INTEGRAL_SINGULAR, P_INTEGRAL_LAPLACE]
