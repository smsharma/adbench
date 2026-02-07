"""Category: Brutal — research-level problems designed to defeat frontier models.

Each problem requires either:
- Multi-step algorithmic reasoning with no standard formula
- Knowledge of obscure mathematical identities
- Combining 3+ non-trivial concepts
- Implementing a non-trivial algorithm from scratch
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, brentq
from scipy.linalg import expm, sqrtm, solve_sylvester
from scipy.special import gamma, digamma

from adbench.problem import Problem


# --- B1: Derivative of Wasserstein-2 distance ---
# W_2^2(μ,ν) between two 1D Gaussians N(m1,s1^2) and N(m2,s2^2):
# W_2^2 = (m1-m2)^2 + (s1-s2)^2
# Compute d(W_2^2)/dm1 = 2(m1-m2) — but for discrete distributions.
# For discrete 1D: W_2^2 = integral |F^{-1}(t) - G^{-1}(t)|^2 dt
# where F, G are CDFs. Differentiate w.r.t. shift of first distribution.
# With atoms at x_i with weights p_i, shifted by θ:
# W_2^2(θ) = Σ_i p_i (x_i + θ - y_{σ(i)})^2 where σ is optimal coupling
# d/dθ = 2 Σ_i p_i (x_i + θ - y_{σ(i)})

_x_wass = np.array([0.0, 1.0, 3.0, 5.0])
_y_wass = np.array([0.5, 2.0, 4.0, 6.0])
_p_wass = np.array([0.25, 0.25, 0.25, 0.25])

def _ref_brutal_wasserstein(theta: float) -> float:
    """d/dtheta W_2^2 between uniform on (x+theta) and uniform on y."""
    # For equal-weight 1D distributions, optimal coupling is sorting
    x_shifted = np.sort(_x_wass + theta)
    y_sorted = np.sort(_y_wass)
    # W_2^2 = mean((x_shifted - y_sorted)^2)
    # dW_2^2/dtheta = 2 * mean(x_shifted - y_sorted)
    return 2 * np.mean(x_shifted - y_sorted)

P_BRUTAL_WASSERSTEIN = Problem(
    id="brutal_wasserstein_grad",
    category="brutal",
    difficulty=3,
    description="Gradient of Wasserstein-2 distance w.r.t. location shift.",
    prompt="""Two discrete 1D distributions with equal weights 1/4:
  μ: atoms at {0, 1, 3, 5} shifted by θ (so atoms at {θ, 1+θ, 3+θ, 5+θ})
  ν: atoms at {0.5, 2, 4, 6}

The squared Wasserstein-2 distance is W₂²(μ,ν) = min_coupling E[|X-Y|²].

Compute d(W₂²)/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return d/dtheta of the squared Wasserstein-2 distance.\"\"\"
```""",
    reference=_ref_brutal_wasserstein,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_brutal_wasserstein(v)}
        for v in [-1.0, 0.0, 0.5, 1.0, 2.0]
    ],
    rtol=1e-3,
)


# --- B2: Sensitivity of Lyapunov exponent ---
# For the map x_{n+1} = r x_n (1 - x_n) (logistic map),
# the Lyapunov exponent is λ(r) = lim (1/N) Σ ln|f'(x_n)| = lim (1/N) Σ ln|r(1-2x_n)|
# Compute dλ/dr numerically via long orbit average.
# This is hard because:
# 1) Must compute the Lyapunov exponent correctly
# 2) Must differentiate it w.r.t. r
# 3) Chaotic sensitivity makes this delicate

def _lyapunov_exp(r, N=100000, discard=10000):
    x = 0.4  # initial condition
    for _ in range(discard):
        x = r * x * (1 - x)
    lyap_sum = 0.0
    for _ in range(N):
        x = r * x * (1 - x)
        deriv = abs(r * (1 - 2*x))
        if deriv > 0:
            lyap_sum += np.log(deriv)
    return lyap_sum / N

def _ref_brutal_lyapunov(r: float) -> float:
    dr = 1e-5
    return (_lyapunov_exp(r + dr) - _lyapunov_exp(r - dr)) / (2 * dr)

P_BRUTAL_LYAPUNOV = Problem(
    id="brutal_lyapunov_sensitivity",
    category="brutal",
    difficulty=3,
    description="Sensitivity of Lyapunov exponent of logistic map to parameter r.",
    prompt="""The logistic map x_{n+1} = r x_n (1 - x_n) has Lyapunov exponent:
  λ(r) = lim_{N→∞} (1/N) Σ_{n=1}^N ln|r(1 - 2x_n)|

starting from x_0 = 0.4 (discard first 10000 iterates as transient).

Compute dλ/dr.

```python
def solve(r: float) -> float:
    \"\"\"Return d(lambda)/dr for the Lyapunov exponent of the logistic map.\"\"\"
```""",
    reference=_ref_brutal_lyapunov,
    test_cases=[
        {"inputs": {"r": v}, "expected": _ref_brutal_lyapunov(v)}
        for v in [3.6, 3.7, 3.8, 3.9, 3.99]
    ],
    rtol=0.1,  # Very relaxed — chaotic sensitivity is inherently noisy
)


# --- B3: Riemannian gradient on the Stiefel manifold ---
# Given f(Q) = tr(Q^T A Q B) where Q is orthogonal (Q^T Q = I),
# the Riemannian gradient is: grad_R f = (I - QQ^T) euclidean_grad + Q skew(Q^T euclidean_grad)
# Actually for Stiefel: grad = G - Q G^T Q where G is Euclidean grad
# if using canonical metric. This is a projection.

_A_stiefel = np.array([[3, 1, 0.5], [1, 2, 0.3], [0.5, 0.3, 1]])
_B_stiefel = np.array([[1, 0.2], [0.2, 2]])

def _ref_brutal_stiefel(Q_flat: list) -> list:
    """Riemannian gradient of tr(Q^T A Q B) on Stiefel manifold St(3,2)."""
    Q = np.array(Q_flat).reshape(3, 2)
    # Euclidean gradient
    G = _A_stiefel @ Q @ _B_stiefel + _A_stiefel.T @ Q @ _B_stiefel.T
    # Project to tangent space of Stiefel: P_Q(G) = G - Q sym(Q^T G)
    sym_part = 0.5 * (Q.T @ G + G.T @ Q)
    riem_grad = G - Q @ sym_part
    return riem_grad.flatten().tolist()

P_BRUTAL_STIEFEL = Problem(
    id="brutal_stiefel_gradient",
    category="brutal",
    difficulty=3,
    description="Riemannian gradient on Stiefel manifold for tr(Q^T A Q B).",
    prompt="""f(Q) = tr(Q^T A Q B) where Q ∈ St(3,2) (3×2 matrices with Q^T Q = I₂).

A = [[3,1,0.5],[1,2,0.3],[0.5,0.3,1]], B = [[1,0.2],[0.2,2]].

Compute the Riemannian gradient of f on the Stiefel manifold (project Euclidean gradient
onto the tangent space at Q).

The tangent space at Q is {Z : Q^T Z + Z^T Q = 0} and the projection is:
  proj_Q(G) = G - Q * sym(Q^T G)

where sym(M) = (M + M^T)/2.

Q is given as a flat list of 6 numbers (row-major 3×2).

```python
def solve(Q_flat: list) -> list:
    \"\"\"Return the Riemannian gradient as a flat list of 6 numbers.\"\"\"
```""",
    reference=_ref_brutal_stiefel,
    test_cases=[
        # Q = first two columns of a 3x3 rotation
        {"inputs": {"Q_flat": [1, 0, 0, 1, 0, 0]},
         "expected": _ref_brutal_stiefel([1, 0, 0, 1, 0, 0])},
        {"inputs": {"Q_flat": [1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0, 1]},
         "expected": _ref_brutal_stiefel([1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0, 1])},
    ],
    rtol=1e-3,
)


# --- B4: Gradient of the spectral gap of a graph Laplacian ---
# L(θ) = D - A(θ) where A(θ)_{ij} = exp(-θ||x_i - x_j||^2)
# Spectral gap = λ_2(L). Compute dλ_2/dθ.
# Requires: eigenvalue derivative through non-trivial matrix.

_points_sg = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0.5, 0.5]])

def _make_laplacian(theta):
    n = len(_points_sg)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d2 = np.sum((_points_sg[i] - _points_sg[j])**2)
                A[i, j] = np.exp(-theta * d2)
    D = np.diag(A.sum(axis=1))
    return D - A

def _ref_brutal_spectral_gap(theta: float) -> float:
    dt = 1e-6
    def gap(th):
        L = _make_laplacian(th)
        eigvals = np.sort(np.linalg.eigvalsh(L))
        return eigvals[1]  # second smallest eigenvalue
    return (gap(theta + dt) - gap(theta - dt)) / (2 * dt)

P_BRUTAL_SPECTRAL_GAP = Problem(
    id="brutal_spectral_gap",
    category="brutal",
    difficulty=3,
    description="Derivative of spectral gap of parameterized graph Laplacian.",
    prompt="""A weighted graph on 4 points in ℝ²:
  points = [[0,0], [1,0], [0.5,√3/2], [0.5,0.5]]

Edge weights: w_{ij}(θ) = exp(-θ ||xᵢ - xⱼ||²).
Graph Laplacian: L(θ) = D(θ) - A(θ) where D is degree matrix, A is adjacency.
Spectral gap: λ₂(θ) = second smallest eigenvalue of L.

Compute dλ₂/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return d(lambda_2)/d(theta) for the graph Laplacian spectral gap.\"\"\"
```""",
    reference=_ref_brutal_spectral_gap,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_brutal_spectral_gap(v)}
        for v in [0.1, 0.5, 1.0, 2.0, 5.0]
    ],
    rtol=1e-2,
)


# --- B5: Derivative of the condition number of a matrix ---
# κ(A(t)) = σ_max / σ_min. Compute dκ/dt.
# Requires: SVD sensitivity for BOTH extremal singular values.

def _make_A_cond(t):
    return np.array([
        [2 + t, 0.5*t, 0.1],
        [0.3, 1 + 0.3*t, 0.2*t],
        [0.1*t, 0.2, 3 - 0.5*t]
    ])

def _ref_brutal_condition_number(t: float) -> float:
    dt = 1e-7
    def cond(tv):
        sv = np.linalg.svd(_make_A_cond(tv), compute_uv=False)
        return sv[0] / sv[-1]
    return (cond(t + dt) - cond(t - dt)) / (2 * dt)

P_BRUTAL_CONDITION = Problem(
    id="brutal_condition_number",
    category="brutal",
    difficulty=3,
    description="Derivative of condition number κ(A(t)) = σ_max/σ_min.",
    prompt="""A(t) = [[2+t,   0.5t,   0.1  ],
         [0.3,   1+0.3t, 0.2t ],
         [0.1t,  0.2,    3-0.5t]]

Compute d/dt κ(A(t)) where κ = σ_max / σ_min (ratio of largest to smallest singular value).

```python
def solve(t: float) -> float:
    \"\"\"Return d/dt of the condition number of A(t).\"\"\"
```""",
    reference=_ref_brutal_condition_number,
    test_cases=[
        {"inputs": {"t": v}, "expected": _ref_brutal_condition_number(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=1e-2,
)


# --- B6: GP marginal likelihood gradient ---
# log p(y|θ) = -0.5 y^T K^{-1} y - 0.5 log det K - n/2 log 2π
# K = σ² exp(-||x_i-x_j||² / (2ℓ²)) + σ_n² I
# Compute d(log p)/dℓ. Requires Cholesky-based gradient.

_x_gp = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])[:, None]
_y_gp = np.array([0.1, 0.4, 0.3, 0.8, 0.5, 0.9, 0.7])
_sigma_f = 1.0
_sigma_n = 0.1

def _make_K_gp(length_scale):
    sq_dists = np.sum((_x_gp[:, None] - _x_gp[None, :])**2, axis=-1)
    K = _sigma_f**2 * np.exp(-sq_dists / (2 * length_scale**2))
    K += _sigma_n**2 * np.eye(len(_x_gp))
    return K

def _gp_log_marginal(length_scale):
    K = _make_K_gp(length_scale)
    n = len(_y_gp)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, _y_gp))
    return -0.5 * _y_gp @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

def _ref_brutal_gp_gradient(length_scale: float) -> float:
    dl = 1e-6
    return (_gp_log_marginal(length_scale + dl) - _gp_log_marginal(length_scale - dl)) / (2 * dl)

P_BRUTAL_GP = Problem(
    id="brutal_gp_marginal_likelihood",
    category="brutal",
    difficulty=3,
    description="Gradient of GP log marginal likelihood w.r.t. length scale.",
    prompt="""Gaussian process with squared-exponential kernel:
  K(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²)) + σ_n² δ(x,x')

Data: x = [0, 0.5, 1, 1.5, 2, 2.5, 3], y = [0.1, 0.4, 0.3, 0.8, 0.5, 0.9, 0.7]
σ_f = 1.0, σ_n = 0.1.

Log marginal likelihood: log p(y|ℓ) = -½ y^T K⁻¹ y - ½ log|K| - (n/2)log(2π)

Compute d(log p)/dℓ.

```python
def solve(length_scale: float) -> float:
    \"\"\"Return d/d(length_scale) of GP log marginal likelihood.\"\"\"
```""",
    reference=_ref_brutal_gp_gradient,
    test_cases=[
        {"inputs": {"length_scale": v}, "expected": _ref_brutal_gp_gradient(v)}
        for v in [0.3, 0.5, 1.0, 2.0, 5.0]
    ],
    rtol=1e-2,
)


# --- B7: Sensitivity of a DAE (differential algebraic equation) ---
# DAE: x' = -x + y, 0 = x² + y² - r(t)²
# where r(t) = 1 + 0.5*sin(θ*t)
# Compute dx(1)/dθ. Requires solving an index-1 DAE and differentiating.

def _ref_brutal_dae(theta: float) -> float:
    dt = 1e-6
    def solve_dae(th):
        # Convert DAE to ODE by solving algebraic constraint:
        # y = ±sqrt(r(t)^2 - x^2). Take positive branch.
        def rhs(t, state):
            x = state[0]
            r = 1 + 0.5 * np.sin(th * t)
            y2 = r**2 - x**2
            if y2 < 0:
                y2 = 1e-10
            y = np.sqrt(y2)
            return [-x + y]
        sol = solve_ivp(rhs, [0, 1], [0.5], rtol=1e-10, atol=1e-12, max_step=0.01)
        return sol.y[0, -1]
    return (solve_dae(theta + dt) - solve_dae(theta - dt)) / (2 * dt)

P_BRUTAL_DAE = Problem(
    id="brutal_dae_sensitivity",
    category="brutal",
    difficulty=3,
    description="Sensitivity of DAE solution where algebraic constraint must be maintained.",
    prompt="""Differential-algebraic equation:
  x' = -x + y
  0  = x² + y² - r(t)²

where r(t) = 1 + 0.5 sin(θt), and y > 0 (positive branch).
Initial condition: x(0) = 0.5.

Compute dx(1)/dθ.

```python
def solve(theta: float) -> float:
    \"\"\"Return dx(1)/dtheta for the DAE system.\"\"\"
```""",
    reference=_ref_brutal_dae,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_brutal_dae(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 3.0]
    ],
    rtol=5e-2,
)


# --- B8: Derivative of the Sinkhorn divergence ---
# S(a,b) = OT_ε(a,b) - 0.5*OT_ε(a,a) - 0.5*OT_ε(b,b)
# where OT_ε is the entropy-regularized optimal transport.
# Compute dS/da_1.

def _sinkhorn_cost(a, b, C, eps, num_iter=100):
    """Entropy-regularized OT cost."""
    n, m = len(a), len(b)
    K = np.exp(-C / eps)
    u = np.ones(n)
    for _ in range(num_iter):
        v = b / (K.T @ u)
        u = a / (K @ v)
    return np.sum(u[:, None] * K * v[None, :] * C)

_C_sink = np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]], dtype=float)
_b_sink = np.array([0.3, 0.4, 0.3])
_eps_sink = 0.1

def _ref_brutal_sinkhorn(a1: float) -> float:
    da = 1e-6
    def sinkhorn_div(av):
        a = np.array([av, (1-av)/2, (1-av)/2])
        s_ab = _sinkhorn_cost(a, _b_sink, _C_sink, _eps_sink)
        s_aa = _sinkhorn_cost(a, a, _C_sink, _eps_sink)
        s_bb = _sinkhorn_cost(_b_sink, _b_sink, _C_sink, _eps_sink)
        return s_ab - 0.5*s_aa - 0.5*s_bb
    return (sinkhorn_div(a1 + da) - sinkhorn_div(a1 - da)) / (2 * da)

P_BRUTAL_SINKHORN = Problem(
    id="brutal_sinkhorn_divergence",
    category="brutal",
    difficulty=3,
    description="Gradient of Sinkhorn divergence w.r.t. source distribution weight.",
    prompt="""The Sinkhorn divergence between discrete distributions a and b is:
  S(a,b) = OT_ε(a,b) - 0.5 OT_ε(a,a) - 0.5 OT_ε(b,b)

where OT_ε is the entropy-regularized OT cost with regularization ε = 0.1.

Cost matrix C = [[0,1,4],[1,0,1],[4,1,0]].
b = [0.3, 0.4, 0.3].
a = [a₁, (1-a₁)/2, (1-a₁)/2] (parameterized by a₁).

Compute dS/da₁.

```python
def solve(a1: float) -> float:
    \"\"\"Return d(Sinkhorn divergence)/d(a1).\"\"\"
```""",
    reference=_ref_brutal_sinkhorn,
    test_cases=[
        {"inputs": {"a1": v}, "expected": _ref_brutal_sinkhorn(v)}
        for v in [0.2, 0.3, 0.4, 0.5, 0.6]
    ],
    rtol=5e-2,
)


# --- B9: Sensitivity of a continued fraction ---
# f(a) = a / (1 + a / (2 + a / (3 + a / (4 + ...))))
# Truncated to depth N. Compute df/da.
# The continued fraction converges to a ratio of Bessel functions.

def _continued_fraction(a, N=50):
    """Evaluate continued fraction a/(1 + a/(2 + a/(3 + ...))) truncated at depth N."""
    result = 0.0
    for k in range(N, 0, -1):
        result = a / (k + result)
    return result

def _ref_brutal_continued_fraction(a: float) -> float:
    da = 1e-7
    return (_continued_fraction(a + da) - _continued_fraction(a - da)) / (2 * da)

P_BRUTAL_CONTFRAC = Problem(
    id="brutal_continued_fraction",
    category="brutal",
    difficulty=3,
    description="Derivative of a generalized continued fraction.",
    prompt="""The continued fraction:
  f(a) = a/(1 + a/(2 + a/(3 + a/(4 + ... ))))

truncated at depth N=50.

Compute df/da.

```python
def solve(a: float) -> float:
    \"\"\"Return df/da for the continued fraction truncated at depth 50.\"\"\"
```""",
    reference=_ref_brutal_continued_fraction,
    test_cases=[
        {"inputs": {"a": v}, "expected": _ref_brutal_continued_fraction(v)}
        for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ],
    rtol=1e-3,
)


# --- B10: Jacobian of the matrix Riccati equation solution ---
# X satisfies A^T X + X A - X B R^{-1} B^T X + Q = 0 (algebraic Riccati)
# Compute d(tr X)/d(A_{11}).

_A_ric = np.array([[-1.0, 0.5], [0.3, -2.0]])
_B_ric = np.array([[1.0], [0.5]])
_Q_ric = np.array([[1.0, 0.0], [0.0, 1.0]])
_R_ric = np.array([[1.0]])

def _solve_riccati(A):
    from scipy.linalg import solve_continuous_are
    return solve_continuous_are(A, _B_ric, _Q_ric, _R_ric)

def _ref_brutal_riccati(a11: float) -> float:
    da = 1e-7
    A_plus = _A_ric.copy(); A_plus[0, 0] = a11 + da
    A_minus = _A_ric.copy(); A_minus[0, 0] = a11 - da
    X_plus = _solve_riccati(A_plus)
    X_minus = _solve_riccati(A_minus)
    return (np.trace(X_plus) - np.trace(X_minus)) / (2 * da)

P_BRUTAL_RICCATI = Problem(
    id="brutal_riccati_sensitivity",
    category="brutal",
    difficulty=3,
    description="Sensitivity of algebraic Riccati equation solution to system matrix entry.",
    prompt="""The continuous algebraic Riccati equation:
  A^T X + X A - X B R^{-1} B^T X + Q = 0

with A = [[-1, 0.5], [0.3, -2]], B = [[1], [0.5]], Q = I₂, R = [1].

The entry A₁₁ is varied around -1. Compute d(tr X)/d(A₁₁).

```python
def solve(a11: float) -> float:
    \"\"\"Return d(tr X)/d(A_11) for the algebraic Riccati equation.\"\"\"
```""",
    reference=_ref_brutal_riccati,
    test_cases=[
        {"inputs": {"a11": v}, "expected": _ref_brutal_riccati(v)}
        for v in [-2.0, -1.5, -1.0, -0.5, -0.1]
    ],
    rtol=1e-2,
)


# --- B11: KL divergence gradient for normalizing flow ---
# q(x) = N(x; μ, 1) pushed through f(x) = x + ε tanh(x).
# KL(q_pushed || p) where p = N(0, 1.5²).
# Compute d(KL)/dμ via the change-of-variables formula.

def _ref_brutal_flow_kl(mu: float, eps: float) -> float:
    dmu = 1e-6
    def kl(m):
        # Sample-based approximation with quadrature
        # q(x) = N(x; m, 1), pushed through y = x + eps*tanh(x)
        # KL = E_q[log q(y) - log p(y)] = E_q[-log|det J| + log N(x;m,1) - log p(y)]
        # Use quadrature over x
        from scipy.stats import norm
        xs = np.linspace(m - 5, m + 5, 1000)
        dx = xs[1] - xs[0]
        qx = norm.pdf(xs, m, 1)
        ys = xs + eps * np.tanh(xs)
        # |dy/dx| = 1 + eps * (1 - tanh(x)^2)
        jac = 1 + eps * (1 - np.tanh(xs)**2)
        # p(y) = N(y; 0, 1.5)
        py = norm.pdf(ys, 0, 1.5)
        # KL integrand = q(x) * [log(q(x)/|jac|) - log p(y)]
        log_q_pushed = np.log(qx + 1e-300) - np.log(jac + 1e-300)
        log_p = np.log(py + 1e-300)
        integrand = qx * (log_q_pushed - log_p)
        return np.sum(integrand) * dx
    return (kl(mu + dmu) - kl(mu - dmu)) / (2 * dmu)

P_BRUTAL_FLOW_KL = Problem(
    id="brutal_normalizing_flow_kl",
    category="brutal",
    difficulty=3,
    description="KL divergence gradient for a simple normalizing flow.",
    prompt="""A normalizing flow: x ~ N(μ, 1), pushed through y = x + ε tanh(x) with ε = 0.5.
Target distribution: p(y) = N(y; 0, 1.5²).

The KL divergence KL(q_flow || p) is computed via the change-of-variables formula.

Compute d(KL)/dμ.

```python
def solve(mu: float, eps: float) -> float:
    \"\"\"Return d(KL divergence)/d(mu) for the normalizing flow.\"\"\"
```""",
    reference=_ref_brutal_flow_kl,
    test_cases=[
        {"inputs": {"mu": m, "eps": 0.5}, "expected": _ref_brutal_flow_kl(m, 0.5)}
        for m in [-1.0, 0.0, 0.5, 1.0, 2.0]
    ],
    rtol=5e-2,
)


# --- B12: Second derivative of the Fredholm determinant ---
# det(I + θK) where K is an integral operator discretized as a matrix.
# K_{ij} = h * k(x_i, x_j) with k(x,y) = exp(-|x-y|) on [0,1].
# Compute d²/dθ² log det(I + θK).

_N_fred = 50
_h_fred = 1.0 / _N_fred
_x_fred = np.linspace(_h_fred/2, 1 - _h_fred/2, _N_fred)
_K_fred = _h_fred * np.exp(-np.abs(_x_fred[:, None] - _x_fred[None, :]))

def _ref_brutal_fredholm(theta: float) -> float:
    dt = 1e-5
    def log_det(th):
        M = np.eye(_N_fred) + th * _K_fred
        sign, logdet = np.linalg.slogdet(M)
        return logdet

    # Second derivative via central differences
    f_plus = log_det(theta + dt)
    f_0 = log_det(theta)
    f_minus = log_det(theta - dt)
    return (f_plus - 2*f_0 + f_minus) / dt**2

P_BRUTAL_FREDHOLM = Problem(
    id="brutal_fredholm_second_deriv",
    category="brutal",
    difficulty=3,
    description="Second derivative of log Fredholm determinant.",
    prompt="""The Fredholm determinant of the operator I + θK, where K is discretized on [0,1]
with N=50 quadrature points and kernel k(x,y) = exp(-|x-y|):

  f(θ) = log det(I + θK)

where K_{ij} = (1/N) exp(-|x_i - x_j|) with x_i uniformly spaced.

Compute d²f/dθ².

```python
def solve(theta: float) -> float:
    \"\"\"Return d^2/dtheta^2 log det(I + theta*K).\"\"\"
```""",
    reference=_ref_brutal_fredholm,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_brutal_fredholm(v)}
        for v in [0.0, 0.5, 1.0, 2.0, 5.0]
    ],
    rtol=5e-2,
)


ALL = [
    P_BRUTAL_WASSERSTEIN, P_BRUTAL_LYAPUNOV, P_BRUTAL_STIEFEL,
    P_BRUTAL_SPECTRAL_GAP, P_BRUTAL_CONDITION, P_BRUTAL_GP,
    P_BRUTAL_DAE, P_BRUTAL_SINKHORN, P_BRUTAL_CONTFRAC,
    P_BRUTAL_RICCATI, P_BRUTAL_FLOW_KL, P_BRUTAL_FREDHOLM,
]
