"""Category: Special Functions — genuinely hard cases.

Derivatives that DON'T have simple scipy one-liners, requiring
the model to derive non-trivial identities or implement series.
"""

import numpy as np
from scipy.special import gamma, digamma, polygamma, gammainc
from scipy.integrate import quad

from adbench.problem import Problem


# --- Derivative of regularized incomplete gamma w.r.t. a ---
# ∂P(a,x)/∂a is non-trivial — no scipy function.

def _ref_special_inc_gamma(a: float, x: float) -> float:
    da = 1e-7
    return (gammainc(a + da, x) - gammainc(a - da, x)) / (2 * da)

P_SPECIAL_INC_GAMMA = Problem(
    id="special_inc_gamma_deriv",
    category="special_functions",
    difficulty=3,
    description="Derivative of the regularized incomplete gamma function P(a,x) w.r.t. a.",
    prompt="""The regularized lower incomplete gamma function is:
  P(a, x) = γ(a, x) / Γ(a) = (1/Γ(a)) ∫₀ˣ t^{a-1} e^{-t} dt

Compute ∂P/∂a.

```python
def solve(a: float, x: float) -> float:
    \"\"\"Return dP(a,x)/da for the regularized incomplete gamma.\"\"\"
```""",
    reference=_ref_special_inc_gamma,
    test_cases=[
        {"inputs": {"a": a, "x": x}, "expected": _ref_special_inc_gamma(a, x)}
        for a, x in [(1.0, 1.0), (2.0, 1.0), (0.5, 2.0), (3.0, 3.0), (1.5, 0.5)]
    ],
    rtol=1e-2,
)


# --- Derivative of the polylogarithm Li_s(z) w.r.t. s ---
# Li_s(z) = Σ_{k=1}^∞ z^k / k^s
# ∂Li_s/∂s = -Σ_{k=1}^∞ z^k ln(k) / k^s
# No scipy function; must implement the series.

def _polylog(s, z, N=5000):
    k = np.arange(1, N + 1)
    return np.sum(z**k / k**s)

def _ref_special_polylog_deriv(s: float, z: float) -> float:
    """∂Li_s(z)/∂s = -Σ z^k ln(k)/k^s."""
    N = 5000
    k = np.arange(1, N + 1)
    # k=1 has ln(1)=0 so no issue
    return -np.sum(z**k * np.log(k) / k**s)

P_SPECIAL_POLYLOG = Problem(
    id="special_polylog_deriv",
    category="special_functions",
    difficulty=3,
    description="Derivative of the polylogarithm Li_s(z) w.r.t. the order s.",
    prompt="""The polylogarithm is:
  Li_s(z) = Σ_{k=1}^∞ z^k / k^s  for |z| < 1.

Compute ∂Li_s/∂s.

```python
def solve(s: float, z: float) -> float:
    \"\"\"Return d/ds Li_s(z), derivative of polylogarithm w.r.t. order.\"\"\"
```""",
    reference=_ref_special_polylog_deriv,
    test_cases=[
        {"inputs": {"s": s, "z": z}, "expected": _ref_special_polylog_deriv(s, z)}
        for s, z in [(2.0, 0.5), (3.0, 0.5), (1.5, 0.9), (2.0, 0.3), (2.5, 0.7)]
    ],
    rtol=1e-3,
)


# --- Mixed partial of the Hurwitz zeta function ---
def _hurwitz_zeta(s, a, N=10000):
    n = np.arange(0, N)
    return np.sum(1.0 / (n + a)**s)

def _ref_special_hurwitz_mixed(s: float, a: float) -> float:
    ds = 1e-5
    da = 1e-5
    f_pp = _hurwitz_zeta(s + ds, a + da)
    f_pm = _hurwitz_zeta(s + ds, a - da)
    f_mp = _hurwitz_zeta(s - ds, a + da)
    f_mm = _hurwitz_zeta(s - ds, a - da)
    return (f_pp - f_pm - f_mp + f_mm) / (4 * ds * da)

P_SPECIAL_HURWITZ = Problem(
    id="special_hurwitz_mixed",
    category="special_functions",
    difficulty=3,
    description="Mixed partial ∂²ζ/∂s∂a of the Hurwitz zeta function.",
    prompt="""The Hurwitz zeta function is:
  ζ(s, a) = Σ_{n=0}^∞ 1/(n+a)^s  for s > 1, a > 0.

Compute ∂²ζ/∂s∂a (the mixed partial derivative).

```python
def solve(s: float, a: float) -> float:
    \"\"\"Return d^2 zeta / (ds da) for the Hurwitz zeta function.\"\"\"
```""",
    reference=_ref_special_hurwitz_mixed,
    test_cases=[
        {"inputs": {"s": s, "a": a}, "expected": _ref_special_hurwitz_mixed(s, a)}
        for s, a in [(2.0, 1.0), (3.0, 1.0), (2.0, 2.0), (2.5, 0.5)]
    ],
    rtol=1e-2,
)


ALL = [P_SPECIAL_INC_GAMMA, P_SPECIAL_POLYLOG, P_SPECIAL_HURWITZ]
