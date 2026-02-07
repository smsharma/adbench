"""Category: Complex-Valued AD and Wirtinger Calculus — hard cases."""

import numpy as np

from adbench.problem import Problem


# --- Phase retrieval gradient (kept, was hard enough) ---
_a_phase = [np.array([1+0j, 0.5j]), np.array([0.3-0.2j, 1+0j]), np.array([0.5+0.5j, -0.5+0.3j])]
_b_phase = [1.0, 0.5, 0.8]

def _ref_complex_phase_retrieval(z_re: list, z_im: list) -> dict:
    z = np.array(z_re) + 1j * np.array(z_im)
    grad = np.zeros_like(z)
    for a, b in zip(_a_phase, _b_phase):
        inner = np.dot(a, z)
        grad += 2 * (abs(inner)**2 - b) * a * np.conj(inner)
    return {"grad_re": grad.real.tolist(), "grad_im": grad.imag.tolist()}

P_COMPLEX_PHASE = Problem(
    id="complex_phase_retrieval",
    category="complex_ad",
    difficulty=3,
    description="Wirtinger gradient of phase retrieval loss.",
    prompt="""Phase retrieval loss:
  L(z) = Σᵢ (|⟨aᵢ, z⟩|² - bᵢ)²

where z ∈ ℂ², and:
  a₁ = [1, 0.5j], a₂ = [0.3-0.2j, 1], a₃ = [0.5+0.5j, -0.5+0.3j]
  b = [1.0, 0.5, 0.8]

Compute ∇_{z*} L (the Wirtinger gradient for optimization).

```python
def solve(z_re: list, z_im: list) -> dict:
    \"\"\"Return {'grad_re': [...], 'grad_im': [...]} for the Wirtinger gradient.\"\"\"
```""",
    reference=_ref_complex_phase_retrieval,
    test_cases=[
        {"inputs": {"z_re": [1.0, 0.0], "z_im": [0.0, 1.0]},
         "expected": _ref_complex_phase_retrieval([1.0, 0.0], [0.0, 1.0])},
        {"inputs": {"z_re": [0.5, 0.5], "z_im": [0.5, -0.5]},
         "expected": _ref_complex_phase_retrieval([0.5, 0.5], [0.5, -0.5])},
        {"inputs": {"z_re": [1.0, 1.0], "z_im": [0.0, 0.0]},
         "expected": _ref_complex_phase_retrieval([1.0, 1.0], [0.0, 0.0])},
    ],
    rtol=1e-3,
)


# --- NEW: Gradient of complex-valued neural network loss ---
# f(z) = |σ(wz + b)|² where σ is the complex sigmoid σ(z) = 1/(1+exp(-z))
# w, b, z all complex. Compute d|σ(wz+b)|²/dw* (Wirtinger w.r.t. w conjugate).
# This is genuinely tricky because σ is holomorphic but ||² is not.

def _complex_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _ref_complex_nn_grad(w_re: float, w_im: float, z_re: float, z_im: float,
                          b_re: float, b_im: float) -> tuple:
    """d|σ(wz+b)|²/dw* via numerical differentiation."""
    dh = 1e-7
    w = complex(w_re, w_im)
    z = complex(z_re, z_im)
    b = complex(b_re, b_im)

    def f(wr, wi):
        wc = complex(wr, wi)
        return abs(_complex_sigmoid(wc * z + b))**2

    # Wirtinger: df/dw* = (1/2)(df/du + i df/dv) where w = u + iv
    df_du = (f(w_re + dh, w_im) - f(w_re - dh, w_im)) / (2 * dh)
    df_dv = (f(w_re, w_im + dh) - f(w_re, w_im - dh)) / (2 * dh)
    # df/dw* = (1/2)(df/du + i df/dv)
    grad_conj = 0.5 * (df_du + 1j * df_dv)
    return (grad_conj.real, grad_conj.imag)

P_COMPLEX_NN = Problem(
    id="complex_nn_wirtinger",
    category="complex_ad",
    difficulty=3,
    description="Wirtinger gradient through complex sigmoid neural network.",
    prompt="""Complex neural network with one unit:
  f(w) = |σ(wz + b)|²

where σ(z) = 1/(1+exp(-z)) is the complex sigmoid, and w, z, b ∈ ℂ.

Given z = 1+0.5j, b = 0.1-0.2j, compute ∂f/∂w* (Wirtinger derivative w.r.t. conjugate of w).

Recall: for f: ℂ→ℝ, the Wirtinger derivative ∂f/∂w* = (1/2)(∂f/∂u + i ∂f/∂v) where w = u + iv.

```python
def solve(w_re: float, w_im: float, z_re: float, z_im: float,
          b_re: float, b_im: float) -> tuple:
    \"\"\"Return (real, imag) of df/dw* for f = |sigmoid(w*z + b)|^2.\"\"\"
```""",
    reference=_ref_complex_nn_grad,
    test_cases=[
        {"inputs": {"w_re": wr, "w_im": wi, "z_re": 1.0, "z_im": 0.5,
                     "b_re": 0.1, "b_im": -0.2},
         "expected": _ref_complex_nn_grad(wr, wi, 1.0, 0.5, 0.1, -0.2)}
        for wr, wi in [(0.5, 0.3), (1.0, 0.0), (0.0, 1.0), (-0.5, 0.5)]
    ],
    rtol=1e-2,
)


ALL = [P_COMPLEX_PHASE, P_COMPLEX_NN]
