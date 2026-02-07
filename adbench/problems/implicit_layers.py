"""Implicit differentiation layer problems.

These problems require differentiating through fixed-point equations, optimization
layers, and other implicit functions where the forward pass involves solving a
(possibly nonlinear) equation and the backward pass requires the implicit function
theorem or adjoint methods.
"""

import numpy as np
from scipy import linalg as sla
from adbench.problem import Problem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FD_H = 1e-6  # step size for finite differences


def _fd_scalar(f, x0, h=_FD_H):
    """Central finite-difference derivative of scalar f w.r.t. scalar x0."""
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def _fd_gradient(f, x, h=_FD_H):
    """Central finite-difference gradient of scalar f w.r.t. vector x."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.flat[i] += h
        xm.flat[i] -= h
        grad.flat[i] = (f(xp) - f(xm)) / (2 * h)
    return grad


def _fd_jacobian_vec(f, x0, m, h=_FD_H):
    """Central finite-difference Jacobian of f: R^n -> R^m w.r.t. vector x0."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    jac = np.zeros((m, n))
    for j in range(n):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += h
        xm[j] -= h
        fp = np.asarray(f(xp)).ravel()
        fm = np.asarray(f(xm)).ravel()
        jac[:, j] = (fp - fm) / (2 * h)
    return jac


# ===================================================================
# Problem 21: deq_backward -- Deep Equilibrium Model Backward Pass
# ===================================================================

_DEQ_W = np.array([[-0.3, 0.2, 0.1],
                    [0.1, -0.4, 0.2],
                    [0.15, 0.1, -0.35]])
_DEQ_B = np.array([0.1, -0.2, 0.3])
_DEQ_TARGET = np.array([0.5, -0.3, 0.1])


def _deq_find_fixed_point(W, b, n_iter=2000, tol=1e-12):
    """Find z* = tanh(W @ z* + b) by fixed-point iteration."""
    z = np.zeros(W.shape[0])
    for _ in range(n_iter):
        z_new = np.tanh(W @ z + b)
        if np.max(np.abs(z_new - z)) < tol:
            break
        z = z_new
    return z


def _deq_loss(W, b, target):
    """L = ||z* - target||^2 where z* is the fixed point."""
    z = _deq_find_fixed_point(W, b)
    return np.sum((z - target) ** 2)


def _ref_deq_backward():
    """Compute dL/dW via finite differences over the full loss."""
    W = _DEQ_W.copy()
    b = _DEQ_B.copy()
    target = _DEQ_TARGET.copy()
    h = 1e-5
    grad_W = np.zeros_like(W)
    for i in range(3):
        for j in range(3):
            Wp = W.copy()
            Wm = W.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            Lp = _deq_loss(Wp, b, target)
            Lm = _deq_loss(Wm, b, target)
            grad_W[i, j] = (Lp - Lm) / (2 * h)
    return grad_W.ravel().tolist()


def solve_deq_backward():
    return _ref_deq_backward()


_deq_cases = [{
    "inputs": {},
    "expected": _ref_deq_backward(),
}]

deq_backward = Problem(
    id="deq_backward",
    category="implicit_layers",
    difficulty=3,
    description=(
        "Deep equilibrium model backward pass. Differentiate a loss through "
        "the fixed-point equation z* = tanh(W @ z* + b) to compute dL/dW."
    ),
    prompt=r"""Implement a function `solve()` that computes the gradient dL/dW for a deep
equilibrium model, returning a flat list of 9 floats (row-major order for the 3x3 matrix).

The model defines a fixed-point equation:
  z* = tanh(W @ z* + b)

where W is 3x3 and b is 3x1. The loss is:
  L = ||z* - target||^2

with target = [0.5, -0.3, 0.1].

Use:
  W = [[-0.3, 0.2, 0.1], [0.1, -0.4, 0.2], [0.15, 0.1, -0.35]]
  b = [0.1, -0.2, 0.3]

Steps:
1. Find the fixed point z* by iterating z <- tanh(W @ z + b) until convergence
   (start from z=0, iterate until change < 1e-10 or up to 2000 steps).
2. Compute dL/dW. Note that z* depends implicitly on W through the fixed-point equation.
   You must account for how perturbing W changes the fixed point z*.

Return dL/dW as a flat list of 9 floats [dL/dW_{00}, dL/dW_{01}, ..., dL/dW_{22}].

Available: numpy (as np), scipy, math.
""",
    reference=solve_deq_backward,
    test_cases=_deq_cases,
    atol=1e-4,
    rtol=1e-2,
)


# ===================================================================
# Problem 22: optnet_layer -- Differentiable QP Layer
# ===================================================================

_OPTNET_Q = np.array([[2.0, 0.5], [0.5, 1.0]])
_OPTNET_G = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
_OPTNET_h = np.array([0.0, 0.0, 1.5])


def _solve_qp(Q, p, G, h_vec):
    """Solve QP: min 0.5 y^T Q y + p^T y  s.t. G y <= h via active-set on small problem.

    For this small problem, we enumerate all subsets of constraints to find the optimum.
    """
    from itertools import combinations
    n = Q.shape[0]
    m = G.shape[0]

    best_y = None
    best_obj = np.inf

    # Try unconstrained solution first
    try:
        y_unc = np.linalg.solve(Q, -p)
        if np.all(G @ y_unc <= h_vec + 1e-10):
            obj = 0.5 * y_unc @ Q @ y_unc + p @ y_unc
            if obj < best_obj:
                best_obj = obj
                best_y = y_unc
    except np.linalg.LinAlgError:
        pass

    # Try each subset of active constraints
    for k in range(1, min(n, m) + 1):
        for active in combinations(range(m), k):
            active = list(active)
            Ga = G[active]
            ha = h_vec[active]
            # KKT: Q y + p + Ga^T lam = 0, Ga y = ha
            # [Q, Ga^T; Ga, 0] [y; lam] = [-p; ha]
            na = len(active)
            KKT = np.zeros((n + na, n + na))
            KKT[:n, :n] = Q
            KKT[:n, n:] = Ga.T
            KKT[n:, :n] = Ga
            rhs = np.zeros(n + na)
            rhs[:n] = -p
            rhs[n:] = ha
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                continue
            y_cand = sol[:n]
            lam_cand = sol[n:]
            # Check feasibility and complementarity
            if np.all(G @ y_cand <= h_vec + 1e-8) and np.all(lam_cand >= -1e-8):
                obj = 0.5 * y_cand @ Q @ y_cand + p @ y_cand
                if obj < best_obj:
                    best_obj = obj
                    best_y = y_cand
    return best_y


def _optnet_forward(theta):
    """Solve QP for given theta, return y*.

    p(theta) = [-2 + 3*theta, -1 + theta] so that dp/dtheta = [3, 1].
    This parameterization ensures the QP solution transitions through
    different active-set configurations as theta varies.
    """
    p = np.array([-2.0 + 3.0 * theta, -1.0 + theta])
    y_star = _solve_qp(_OPTNET_Q, p, _OPTNET_G, _OPTNET_h)
    return y_star


def _ref_optnet_layer(theta):
    """Compute dy*/dtheta via finite differences."""
    h = 1e-6
    yp = _optnet_forward(theta + h)
    ym = _optnet_forward(theta - h)
    return ((yp - ym) / (2 * h)).tolist()


def solve_optnet_layer(theta):
    return _ref_optnet_layer(theta)


_optnet_cases = []
for _theta in [0.0, 0.25, 0.5, 0.7, 0.9]:
    _optnet_cases.append({
        "inputs": {"theta": _theta},
        "expected": _ref_optnet_layer(_theta),
    })

optnet_layer = Problem(
    id="optnet_layer",
    category="implicit_layers",
    difficulty=3,
    description=(
        "Differentiable QP layer (OptNet). Differentiate the solution of a "
        "quadratic program with respect to a parameter in the linear term."
    ),
    prompt=r"""Implement a function `solve(theta)` that computes dy*/dtheta, where y* is the
solution to a quadratic program parameterized by theta. Return a list of 2 floats.

The QP is:
  y* = argmin  0.5 * y^T Q y + p^T y
       subject to  G y <= h

where:
  Q = [[2, 0.5], [0.5, 1]]
  p = [-2 + 3*theta, -1 + theta]
  G = [[-1, 0], [0, -1], [1, 1]]
  h = [0, 0, 1.5]

So p depends on theta linearly: dp/dtheta = [3, 1].

The constraints mean: y_0 >= 0, y_1 >= 0, y_0 + y_1 <= 1.5.

Steps:
1. Solve the QP for the given theta to get y* and identify which inequality
   constraints are active (satisfied with equality: G_i y* = h_i, within
   a tolerance of ~1e-8).
2. Let A be the matrix of active constraint rows of G, and b_a the corresponding
   rows of h. At the optimum, the KKT conditions give:
     Q y* + p + A^T lambda* = 0
     A y* = b_a
   where lambda* >= 0 are the active multipliers.
3. Differentiate the KKT system implicitly with respect to theta to find dy*/dtheta.
   The KKT matrix is:
     [[Q, A^T], [A, 0]] @ [dy/dtheta; dlambda/dtheta] = [-dp/dtheta; 0]
   If no constraints are active, simply: Q @ dy/dtheta = -dp/dtheta.

Return dy*/dtheta as a list of 2 floats.

Parameters:
  theta : float

Available: numpy (as np), scipy, math.
""",
    reference=solve_optnet_layer,
    test_cases=_optnet_cases,
    atol=1e-4,
    rtol=1e-2,
)


# ===================================================================
# Problem 23: sinkhorn_implicit -- Sinkhorn as Implicit Layer
# ===================================================================

_SINK_C_BASE = np.array([[0.0, 1.0, 2.0],
                          [1.0, 0.0, 1.0],
                          [2.0, 1.0, 0.0]], dtype=float)
_SINK_EPS = 0.1
_SINK_A = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
_SINK_B = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])


def _sinkhorn_transport_cost(C, a, b, eps, n_iter=500):
    """Compute optimal transport cost via Sinkhorn algorithm.

    Returns transport cost = sum(P * C) where P is the optimal transport plan.
    """
    C = np.asarray(C, dtype=float)
    n, m = C.shape
    K = np.exp(-C / eps)

    u = np.ones(n)
    v = np.ones(m)
    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = np.diag(u) @ K @ np.diag(v)
    return np.sum(P * C)


def _ref_sinkhorn_implicit(C_flat):
    """Compute d(transport_cost)/dC_00 for the given flattened cost perturbation.

    C_flat is a flat array representing the full C matrix. We compute the gradient
    w.r.t. C_flat[0] (which is C_{00}).
    """
    C_flat = np.asarray(C_flat, dtype=float)
    C = C_flat.reshape(3, 3)

    def cost_fn(c00):
        C_mod = C.copy()
        C_mod[0, 0] = c00
        return _sinkhorn_transport_cost(C_mod, _SINK_A, _SINK_B, _SINK_EPS)

    return _fd_scalar(cost_fn, C[0, 0], h=1e-5)


def _ref_sinkhorn_gradient_full(scale):
    """Compute d(transport_cost)/dC_{00} when C = scale * C_base."""
    C = scale * _SINK_C_BASE

    def cost_fn(c00):
        C_mod = C.copy()
        C_mod[0, 0] = c00
        return _sinkhorn_transport_cost(C_mod, _SINK_A, _SINK_B, _SINK_EPS)

    return _fd_scalar(cost_fn, C[0, 0], h=1e-5)


def solve_sinkhorn_implicit(scale):
    return _ref_sinkhorn_gradient_full(scale)


_sinkhorn_cases = []
for _scale in [1.0, 0.5, 1.5, 2.0]:
    _sinkhorn_cases.append({
        "inputs": {"scale": _scale},
        "expected": _ref_sinkhorn_gradient_full(_scale),
    })

sinkhorn_implicit = Problem(
    id="sinkhorn_implicit",
    category="implicit_layers",
    difficulty=3,
    description=(
        "Sinkhorn optimal transport as an implicit layer. Differentiate the "
        "transport cost with respect to an entry of the cost matrix."
    ),
    prompt=r"""Implement a function `solve(scale)` that computes the derivative of the
Sinkhorn optimal transport cost with respect to the (0,0) entry of the cost matrix.
Return a single float.

Setup:
  Base cost matrix C_base = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
  C = scale * C_base
  Marginals: a = b = [1/3, 1/3, 1/3]
  Regularization: eps = 0.1

The Sinkhorn algorithm finds the optimal transport plan:
1. Compute kernel K = exp(-C / eps).
2. Initialize u = ones(3), v = ones(3).
3. Iterate 500 times:
     u = a / (K @ v)
     v = b / (K^T @ u)
4. Transport plan: P = diag(u) @ K @ diag(v).
5. Transport cost: L = sum(P * C)  (element-wise product, then sum).

Compute dL/dC_{00} -- the derivative of the transport cost with respect to the
(0,0) entry of C, while keeping all other entries of C fixed.

Note: C_{00} = scale * 0 = 0 for all scale values since C_base[0,0] = 0, but the
gradient is still well-defined via the Sinkhorn iterations (K depends on all entries
of C including C_{00}).

Parameters:
  scale : float

Available: numpy (as np), scipy, math.
""",
    reference=solve_sinkhorn_implicit,
    test_cases=_sinkhorn_cases,
    atol=1e-4,
    rtol=1e-2,
)


# ===================================================================
# Problem 24: fixed_point_composition -- Fixed Point of Composed Map
# ===================================================================

_FPC_A0 = np.array([[0.1, 0.2, 0.0],
                     [0.0, 0.1, 0.2],
                     [0.2, 0.0, 0.1]])
_FPC_A1 = np.array([[0.2, 0.0, 0.1],
                     [0.1, 0.2, 0.0],
                     [0.0, 0.1, 0.2]])
_FPC_B = np.array([0.5, -0.3, 0.1])


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _fpc_forward(z, theta):
    """f(z, theta) = sigmoid(A(theta) @ z + b)."""
    A = theta * _FPC_A0 + (1 - theta) * _FPC_A1
    return _sigmoid(A @ z + _FPC_B)


def _fpc_find_fixed_point(theta, n_iter=5000, tol=1e-12):
    """Find z* = f(z*, theta) by iteration."""
    z = np.zeros(3)
    for _ in range(n_iter):
        z_new = _fpc_forward(z, theta)
        if np.max(np.abs(z_new - z)) < tol:
            break
        z = z_new
    return z


def _fpc_z0(theta):
    """First component of the fixed point."""
    return _fpc_find_fixed_point(theta)[0]


def _ref_fixed_point_composition(theta):
    """Compute dz*_0/dtheta via finite differences."""
    return _fd_scalar(_fpc_z0, theta, h=1e-6)


def solve_fixed_point_composition(theta):
    return _ref_fixed_point_composition(theta)


_fpc_cases = []
for _theta in [0.0, 0.3, 0.5, 0.7, 1.0]:
    _fpc_cases.append({
        "inputs": {"theta": _theta},
        "expected": _ref_fixed_point_composition(_theta),
    })

fixed_point_composition = Problem(
    id="fixed_point_composition",
    category="implicit_layers",
    difficulty=3,
    description=(
        "Fixed point of a parameterized composed nonlinear map. "
        "Differentiate the first component of the fixed point w.r.t. the mixing parameter."
    ),
    prompt=r"""Implement a function `solve(theta)` that computes dz*_0/dtheta, where z* is
the fixed point of a parameterized map. Return a single float.

The map is:
  f(z, theta) = sigmoid(A(theta) @ z + b)

where sigmoid is the element-wise logistic function 1/(1+exp(-x)), and:
  A(theta) = theta * A0 + (1 - theta) * A1

with:
  A0 = [[0.1, 0.2, 0.0], [0.0, 0.1, 0.2], [0.2, 0.0, 0.1]]
  A1 = [[0.2, 0.0, 0.1], [0.1, 0.2, 0.0], [0.0, 0.1, 0.2]]
  b = [0.5, -0.3, 0.1]

Steps:
1. Find the fixed point z* satisfying z* = f(z*, theta) by iterating
   z <- f(z, theta) starting from z = [0, 0, 0] until convergence
   (change < 1e-10 or up to 5000 iterations).

2. Compute dz*_0/dtheta (derivative of the first component of z* with respect to theta).
   Since z* is defined implicitly by F(z*, theta) = z* - f(z*, theta) = 0,
   the implicit function theorem gives:
     dz*/dtheta = (I - df/dz)^{-1} @ df/dtheta
   evaluated at (z*, theta).

Return just dz*_0/dtheta as a float.

Parameters:
  theta : float

Available: numpy (as np), scipy, math.
""",
    reference=solve_fixed_point_composition,
    test_cases=_fpc_cases,
    atol=1e-4,
    rtol=1e-2,
)


# ===================================================================
# Problem 25: power_iteration_backward -- Spectral Norm Gradient
# ===================================================================

_PI_W0 = np.array([[1.0, 0.3, -0.2],
                    [0.5, -0.8, 0.4],
                    [-0.3, 0.6, 0.1],
                    [0.2, -0.1, 0.7]])

_PI_DW = np.array([[0.1, -0.2, 0.3],
                    [-0.4, 0.5, -0.1],
                    [0.2, -0.3, 0.4],
                    [-0.1, 0.2, -0.5]])


def _spectral_norm_power(W, n_iter=500):
    """Compute the largest singular value of W via power iteration.

    Returns sigma_1, u (left singular vector), v (right singular vector).
    """
    W = np.asarray(W, dtype=float)
    m, n = W.shape
    # Initialize v randomly but deterministically
    np.random.seed(42)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(n_iter):
        u = W @ v
        sigma = np.linalg.norm(u)
        if sigma < 1e-15:
            return 0.0, np.zeros(m), np.zeros(n)
        u = u / sigma
        v = W.T @ u
        sigma = np.linalg.norm(v)
        v = v / sigma

    u = W @ v
    sigma = np.linalg.norm(u)
    u = u / sigma
    return sigma, u, v


def _spectral_norm_at_t(t):
    """Compute spectral norm of W(t) = W0 + t * dW."""
    W = _PI_W0 + t * _PI_DW
    sigma, _, _ = _spectral_norm_power(W)
    return sigma


def _ref_power_iteration_backward(t):
    """Compute dsigma_1/dt via finite differences."""
    return _fd_scalar(_spectral_norm_at_t, t, h=1e-6)


def solve_power_iteration_backward(t):
    return _ref_power_iteration_backward(t)


_pi_cases = []
for _t in [0.0, 0.5, 1.0, 2.0]:
    _pi_cases.append({
        "inputs": {"t": _t},
        "expected": _ref_power_iteration_backward(_t),
    })

power_iteration_backward = Problem(
    id="power_iteration_backward",
    category="implicit_layers",
    difficulty=3,
    description=(
        "Spectral norm gradient via implicit differentiation of power iteration. "
        "Differentiate the largest singular value with respect to a matrix perturbation parameter."
    ),
    prompt=r"""Implement a function `solve(t)` that computes dsigma_1/dt, where sigma_1 is the
largest singular value of a parameterized matrix W(t). Return a single float.

Setup:
  W(t) = W0 + t * dW

where W0 (4x3) and dW (4x3) are:
  W0 = [[1.0, 0.3, -0.2],
        [0.5, -0.8, 0.4],
        [-0.3, 0.6, 0.1],
        [0.2, -0.1, 0.7]]

  dW = [[0.1, -0.2, 0.3],
        [-0.4, 0.5, -0.1],
        [0.2, -0.3, 0.4],
        [-0.1, 0.2, -0.5]]

The spectral norm sigma_1 is the largest singular value of W(t), which can be
computed via power iteration:
1. Initialize v as a random unit vector of size 3 (use np.random.seed(42) then
   v = np.random.randn(3); v = v / ||v||).
2. Repeat 500 times:
     u = W v;  u = u / ||u||
     v = W^T u;  sigma = ||v||;  v = v / sigma
3. sigma_1 = ||W v|| (or equivalently sigma from the last iteration).

Then compute dsigma_1/dt. Hint: if W = U Sigma V^T is the SVD, then
dsigma_1/dt = u_1^T (dW/dt) v_1, where u_1 and v_1 are the leading left and
right singular vectors (which the power iteration converges to).

Parameters:
  t : float

Available: numpy (as np), scipy, math.
""",
    reference=solve_power_iteration_backward,
    test_cases=_pi_cases,
    atol=1e-4,
    rtol=1e-2,
)


# ===================================================================
# Export
# ===================================================================

ALL = [deq_backward, optnet_layer, sinkhorn_implicit, fixed_point_composition,
       power_iteration_backward]
