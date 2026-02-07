"""Advanced compositions of non-differentiable operations.

These problems require composing multiple layers of relaxation for inherently
non-differentiable operations (branching, hard assignment, combinatorial distances,
hard attention) and correctly differentiating through the resulting smooth
approximations.
"""

import numpy as np
from adbench.problem import Problem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FD_H = 1e-6  # step size for finite differences


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


def _fd_gradient_scalar(f, x, h=_FD_H):
    """Central finite-difference derivative of scalar f w.r.t. scalar x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ===================================================================
# Problem 26: branch_smooth
# ===================================================================

def _sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _branch_smooth_forward(x, c=1.0, tau=0.1):
    """Smoothed if-else branch:
    f_tau(x) = sigma((x-c)/tau) * sin(x^2) + (1 - sigma((x-c)/tau)) * cos(x)
    """
    x = float(x)
    z = (x - c) / tau
    s = float(_sigmoid(z))
    return s * np.sin(x ** 2) + (1.0 - s) * np.cos(x)


def _ref_branch_smooth(x, c=1.0, tau=0.1):
    x = float(x)
    return _fd_gradient_scalar(lambda t: _branch_smooth_forward(t, c, tau), x)


def solve_branch_smooth(x, c=1.0, tau=0.1):
    return _ref_branch_smooth(x, c, tau)


_branch_smooth_cases = []
for _x in [0.5, 0.9, 1.0, 1.1, 1.5, 2.0]:
    _branch_smooth_cases.append({
        "inputs": {"x": _x, "c": 1.0, "tau": 0.1},
        "expected": _ref_branch_smooth(_x, 1.0, 0.1),
    })

branch_smooth = Problem(
    id="branch_smooth",
    category="compositions",
    difficulty=3,
    description=(
        "Gradient through a smoothed if-else branch. The relaxation replaces the "
        "hard branch with a sigmoid blend, and the gradient has three terms: "
        "the switching term (sigmoid derivative times branch difference), plus the "
        "gradients of each branch weighted by their sigmoid weights."
    ),
    prompt=r"""Implement a function `solve(x, c=1.0, tau=0.1)` that returns the derivative
df_tau/dx as a float, where:

  f_tau(x) = sigma((x - c) / tau) * sin(x^2) + (1 - sigma((x - c) / tau)) * cos(x)

and sigma(z) = 1 / (1 + exp(-z)) is the sigmoid function.

This is a smooth relaxation of the piecewise function:
  f(x) = sin(x^2)  if x > c
  f(x) = cos(x)    otherwise

Compute the EXACT analytical derivative df_tau/dx.

Hint: There are three contributions to the derivative:
  - The derivative of the sigmoid weight times the difference of the two branches
  - The derivative of sin(x^2) weighted by sigma
  - The derivative of cos(x) weighted by (1 - sigma)

Do NOT use finite differences. Return the derivative as a single float.

Parameters:
  x : float
  c : float (default 1.0), the branch point
  tau : float (default 0.1), the smoothing temperature

Available: numpy (as np), scipy, math.
""",
    reference=solve_branch_smooth,
    test_cases=_branch_smooth_cases,
    atol=1e-6,
    rtol=1e-2,
)

# ===================================================================
# Problem 27: soft_kmeans_gradient
# ===================================================================

_KMEANS_POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                            [5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])


def _soft_kmeans_loss(mu1, mu2, points, tau=0.5):
    """Soft k-means loss with K=2 centroids.

    p_{ik} = softmax(-||x_i - mu_k||^2 / tau) over k
    loss = sum_i sum_k p_{ik} * ||x_i - mu_k||^2
    """
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    points = np.asarray(points, dtype=float)

    loss = 0.0
    for i in range(len(points)):
        d1 = np.sum((points[i] - mu1) ** 2)
        d2 = np.sum((points[i] - mu2) ** 2)
        # softmax(-d/tau) for numerical stability
        logits = np.array([-d1 / tau, -d2 / tau])
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        p = exp_logits / np.sum(exp_logits)
        loss += p[0] * d1 + p[1] * d2
    return loss


def _ref_soft_kmeans(mu1, mu2, tau=0.5):
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    grad = _fd_gradient(
        lambda m: _soft_kmeans_loss(m, mu2, _KMEANS_POINTS, tau),
        mu1
    )
    return grad.tolist()


def solve_soft_kmeans(mu1, mu2, tau=0.5):
    return _ref_soft_kmeans(mu1, mu2, tau)


_soft_kmeans_cases = []
for _mu1, _mu2 in [([0.5, 0.5], [5.5, 5.5]),
                     ([0.0, 0.0], [5.0, 5.0]),
                     ([1.0, 0.0], [5.0, 6.0]),
                     ([0.3, 0.3], [5.5, 5.3]),
                     ([2.0, 2.0], [4.0, 4.0])]:
    _soft_kmeans_cases.append({
        "inputs": {"mu1": _mu1, "mu2": _mu2, "tau": 0.5},
        "expected": _ref_soft_kmeans(_mu1, _mu2, 0.5),
    })

soft_kmeans_gradient = Problem(
    id="soft_kmeans_gradient",
    category="compositions",
    difficulty=3,
    description=(
        "Gradient of soft k-means loss with respect to the first centroid. "
        "The soft assignment probabilities couple all centroids to all points."
    ),
    prompt=r"""Implement a function `solve(mu1, mu2, tau=0.5)` that returns the gradient
of the soft k-means loss with respect to mu1 (the first centroid), as a list of 2 floats.

Data points (fixed): x = [[0,0],[1,0],[0,1],[5,5],[6,5],[5,6]]
Number of centroids K = 2: mu1 (2D), mu2 (2D).

Soft k-means loss:
  For each point x_i and centroid mu_k, compute squared distance:
    d_{ik} = ||x_i - mu_k||^2
  Compute soft assignments via softmax over k (with temperature tau):
    p_{ik} = exp(-d_{ik}/tau) / sum_k' exp(-d_{ik'}/tau)
  (Use numerically stable softmax: subtract max logit before exp.)
  Loss = sum_i sum_k p_{ik} * d_{ik}

Compute d(Loss)/d(mu1) analytically. Return as a list of 2 floats [dL/dmu1_x, dL/dmu1_y].

Note: The gradient has TWO contributions for each point:
  1. The direct gradient from the weighted distance term: p_{i1} * (-2)(x_i - mu1)
  2. The indirect gradient through the softmax probabilities dp_{ik}/dmu1,
     since moving mu1 changes the assignment probabilities for ALL clusters.

Do NOT use finite differences.

Parameters:
  mu1 : list of 2 floats, first centroid
  mu2 : list of 2 floats, second centroid
  tau : float, softmax temperature (default 0.5)

Available: numpy (as np), scipy, math.
""",
    reference=solve_soft_kmeans,
    test_cases=_soft_kmeans_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Problem 28: diff_earth_movers
# ===================================================================

def _emd_1d(x_a, w_a, x_b, w_b):
    """Compute 1D Earth Mover's Distance exactly.

    EMD = integral |F_a(t) - F_b(t)| dt
    For discrete distributions this is a sum over intervals between sorted atoms.
    """
    x_a = np.asarray(x_a, dtype=float)
    w_a = np.asarray(w_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    w_b = np.asarray(w_b, dtype=float)

    # Collect all atom positions with their CDF contributions
    events = []
    for i in range(len(x_a)):
        events.append((x_a[i], w_a[i], 0.0))  # (position, delta_cdf_a, delta_cdf_b)
    for j in range(len(x_b)):
        events.append((x_b[j], 0.0, w_b[j]))

    # Sort by position (stable sort to handle ties)
    events.sort(key=lambda e: e[0])

    # Walk through sorted events, accumulating CDF difference
    cdf_a = 0.0
    cdf_b = 0.0
    emd = 0.0
    prev_pos = events[0][0]

    for pos, da, db in events:
        gap = pos - prev_pos
        if gap > 0:
            emd += abs(cdf_a - cdf_b) * gap
        cdf_a += da
        cdf_b += db
        prev_pos = pos

    return emd


def _ref_diff_emd(x_a, w_a, x_b, w_b):
    x_a = np.asarray(x_a, dtype=float)
    w_a = np.asarray(w_a, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    w_b = np.asarray(w_b, dtype=float)
    grad = _fd_gradient(
        lambda xa: _emd_1d(xa, w_a, x_b, w_b),
        x_a,
        h=1e-7,
    )
    return grad.tolist()


def solve_diff_emd(x_a, w_a, x_b, w_b):
    return _ref_diff_emd(x_a, w_a, x_b, w_b)


_emd_w = [0.25, 0.25, 0.25, 0.25]
_emd_xb = [0.5, 2.0, 4.0, 6.0]

_emd_cases = []
for _xa in [[0.0, 1.0, 3.0, 5.0],
            [0.5, 2.0, 4.0, 6.0],
            [0.0, 0.5, 1.0, 1.5],
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 3.0, 3.5, 7.0]]:
    _emd_cases.append({
        "inputs": {"x_a": _xa, "w_a": _emd_w, "x_b": _emd_xb, "w_b": _emd_w},
        "expected": _ref_diff_emd(_xa, _emd_w, _emd_xb, _emd_w),
    })

diff_earth_movers = Problem(
    id="diff_earth_movers",
    category="compositions",
    difficulty=3,
    description=(
        "Gradient of the 1D Earth Mover's Distance with respect to atom positions "
        "of the first distribution. Moving an atom changes the CDF step location, "
        "which changes the integral of the absolute CDF difference."
    ),
    prompt=r"""Implement a function `solve(x_a, w_a, x_b, w_b)` that returns the gradient
d(EMD)/d(x_a) as a list of floats (one per atom in distribution a).

The 1D Earth Mover's Distance between two discrete distributions is:
  EMD = integral from -inf to +inf of |F_a(t) - F_b(t)| dt

where F_a and F_b are the cumulative distribution functions:
  F_a(t) = sum_{i: x_a[i] <= t} w_a[i]
  F_b(t) = sum_{j: x_b[j] <= t} w_b[j]

For discrete distributions, this integral reduces to a finite sum over intervals
between consecutive sorted atom positions from both distributions:
  1. Merge all atom positions from x_a and x_b into a sorted list of events.
  2. Walk through the sorted positions, tracking CDF_a and CDF_b.
  3. EMD = sum over consecutive pairs of positions: |CDF_a - CDF_b| * gap_length.

Compute d(EMD)/d(x_a[i]) for each atom position in distribution a.

Think carefully: moving atom x_a[i] (which has weight w_a[i]) shifts where its CDF
step occurs. This changes the lengths of the intervals on either side of x_a[i],
and therefore changes the EMD. The sign depends on the CDF difference just before
and just after the step.

Do NOT use finite differences.

Parameters:
  x_a : list of floats, atom positions of distribution a
  w_a : list of floats, weights of distribution a (sum to 1)
  x_b : list of floats, atom positions of distribution b
  w_b : list of floats, weights of distribution b (sum to 1)

Available: numpy (as np), scipy, math.
""",
    reference=solve_diff_emd,
    test_cases=_emd_cases,
    atol=1e-5,
    rtol=5e-2,
)

# ===================================================================
# Problem 29: diff_hausdorff
# ===================================================================

def _soft_hausdorff(A, B, alpha=0.1, beta=0.1):
    """Differentiable Hausdorff distance using log-sum-exp relaxations.

    softmin_beta(x) = -beta * log(sum exp(-x/beta))
    softmax_alpha(x) = alpha * log(sum exp(x/alpha))

    H_AB = softmax_alpha over i of (softmin_beta over j of ||a_i - b_j||)
    H_BA = softmax_alpha over j of (softmin_beta over i of ||a_i - b_j||)
    soft_H = max(H_AB, H_BA)  -- we use soft max: alpha * log(exp(H_AB/alpha) + exp(H_BA/alpha))
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    n = len(A)
    m = len(B)

    # Compute distance matrix
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))

    # H_AB: for each point in A, find soft-min distance to B, then soft-max over A
    # softmin_beta over j: -beta * log(sum_j exp(-dist[i,j]/beta))
    softmin_AB = np.zeros(n)
    for i in range(n):
        vals = -dist[i, :] / beta
        vals_shifted = vals - np.max(vals)
        softmin_AB[i] = -beta * (np.log(np.sum(np.exp(vals_shifted))) + np.max(vals))
        # Undo: -beta * log(sum exp(-d/beta)) = -beta * (log(sum exp(v - max)) + max)
        # where v = -d/beta

    # softmax_alpha over i: alpha * log(sum_i exp(softmin_AB[i]/alpha))
    vals_ab = softmin_AB / alpha
    vals_ab_shifted = vals_ab - np.max(vals_ab)
    H_AB = alpha * (np.log(np.sum(np.exp(vals_ab_shifted))) + np.max(vals_ab))

    # H_BA: for each point in B, find soft-min distance to A, then soft-max over B
    softmin_BA = np.zeros(m)
    for j in range(m):
        vals = -dist[:, j] / beta
        vals_shifted = vals - np.max(vals)
        softmin_BA[j] = -beta * (np.log(np.sum(np.exp(vals_shifted))) + np.max(vals))

    vals_ba = softmin_BA / alpha
    vals_ba_shifted = vals_ba - np.max(vals_ba)
    H_BA = alpha * (np.log(np.sum(np.exp(vals_ba_shifted))) + np.max(vals_ba))

    # Symmetrized: soft max of H_AB, H_BA
    pair = np.array([H_AB, H_BA]) / alpha
    pair_shifted = pair - np.max(pair)
    soft_H = alpha * (np.log(np.sum(np.exp(pair_shifted))) + np.max(pair))

    return soft_H


def _ref_diff_hausdorff(A, B, alpha=0.1, beta=0.1):
    """Gradient of soft_H w.r.t. a_{0,0} (x-coordinate of first point in A)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    def f_scalar(val):
        A_mod = A.copy()
        A_mod[0, 0] = val
        return _soft_hausdorff(A_mod, B, alpha, beta)

    return _fd_gradient_scalar(f_scalar, A[0, 0], h=1e-7)


def solve_diff_hausdorff(A, B, alpha=0.1, beta=0.1):
    return _ref_diff_hausdorff(A, B, alpha, beta)


_haus_A_default = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
_haus_B_default = [[0.5, 0.5], [1.5, 0.5], [1.0, 1.5]]

_haus_cases = []
# Case 1: default configuration
_haus_cases.append({
    "inputs": {"A": _haus_A_default, "B": _haus_B_default, "alpha": 0.1, "beta": 0.1},
    "expected": _ref_diff_hausdorff(
        np.array(_haus_A_default), np.array(_haus_B_default), 0.1, 0.1
    ),
})
# Case 2: first point of A moved closer to B
_haus_A2 = [[0.3, 0.3], [1.0, 0.0], [0.5, 1.0]]
_haus_cases.append({
    "inputs": {"A": _haus_A2, "B": _haus_B_default, "alpha": 0.1, "beta": 0.1},
    "expected": _ref_diff_hausdorff(
        np.array(_haus_A2), np.array(_haus_B_default), 0.1, 0.1
    ),
})
# Case 3: first point of A far away
_haus_A3 = [[-1.0, -1.0], [1.0, 0.0], [0.5, 1.0]]
_haus_cases.append({
    "inputs": {"A": _haus_A3, "B": _haus_B_default, "alpha": 0.1, "beta": 0.1},
    "expected": _ref_diff_hausdorff(
        np.array(_haus_A3), np.array(_haus_B_default), 0.1, 0.1
    ),
})
# Case 4: different alpha/beta
_haus_cases.append({
    "inputs": {"A": _haus_A_default, "B": _haus_B_default, "alpha": 0.5, "beta": 0.5},
    "expected": _ref_diff_hausdorff(
        np.array(_haus_A_default), np.array(_haus_B_default), 0.5, 0.5
    ),
})
# Case 5: first point on top of a B point
_haus_A5 = [[0.5, 0.5], [1.0, 0.0], [0.5, 1.0]]
_haus_cases.append({
    "inputs": {"A": _haus_A5, "B": _haus_B_default, "alpha": 0.1, "beta": 0.1},
    "expected": _ref_diff_hausdorff(
        np.array(_haus_A5), np.array(_haus_B_default), 0.1, 0.1
    ),
})

diff_hausdorff = Problem(
    id="diff_hausdorff",
    category="compositions",
    difficulty=3,
    description=(
        "Gradient of a differentiable Hausdorff distance w.r.t. the x-coordinate "
        "of the first point in set A. Uses log-sum-exp relaxations of min and max."
    ),
    prompt=r"""Implement a function `solve(A, B, alpha=0.1, beta=0.1)` that returns the
derivative d(soft_H)/d(a_{0,x}) as a float, where a_{0,x} is the x-coordinate of
the first point in set A.

Differentiable Hausdorff distance:

Given point sets A = {a_0, ..., a_{n-1}} and B = {b_0, ..., b_{m-1}} in R^2:

1. Compute pairwise Euclidean distances: dist[i][j] = ||a_i - b_j||_2.

2. For each a_i, compute soft-minimum distance to B:
     softmin_B(i) = -beta * log( sum_j exp(-dist[i][j] / beta) )
   Use numerically stable log-sum-exp (subtract max before exp).

3. Compute soft-maximum over A of these soft-min distances:
     H_AB = alpha * log( sum_i exp(softmin_B(i) / alpha) )
   Again use stable log-sum-exp.

4. Similarly compute H_BA:
   For each b_j, compute softmin_A(j) = -beta * log( sum_i exp(-dist[i][j] / beta) )
   H_BA = alpha * log( sum_j exp(softmin_A(j) / alpha) )

5. Symmetrize with soft max:
     soft_H = alpha * log( exp(H_AB / alpha) + exp(H_BA / alpha) )
   Use stable log-sum-exp.

Compute d(soft_H) / d(A[0][0]) analytically. Return as a single float.

The chain rule must propagate through: symmetrized soft-max -> soft-max over A/B ->
soft-min over B/A -> Euclidean distances -> coordinates.

Do NOT use finite differences.

Parameters:
  A : list of lists, shape (n, 2), points in set A
  B : list of lists, shape (m, 2), points in set B
  alpha : float (default 0.1), softmax temperature
  beta : float (default 0.1), softmin temperature

Available: numpy (as np), scipy, math.
""",
    reference=solve_diff_hausdorff,
    test_cases=_haus_cases,
    atol=1e-5,
    rtol=5e-2,
)

# ===================================================================
# Problem 30: attention_temperature
# ===================================================================

_ATT_K = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
], dtype=float)  # (4, 3)

_ATT_V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
], dtype=float)  # (4, 2)

_ATT_Q = np.array([0.8, 0.1, 0.1], dtype=float)  # (3,)


def _soft_attention_output(q, K, V, tau=1.0):
    """Soft attention: output = softmax(K @ q / tau)^T @ V.

    Returns a vector of shape (v,).
    """
    q = np.asarray(q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)

    scores = K @ q / tau  # (n,)
    scores -= np.max(scores)  # numerical stability
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores)  # (n,)
    output = weights @ V  # (v,)
    return output


def _ref_attention_temperature(q, K, V, tau=1.0):
    """Analytical gradient d(output)/d(q): a (d, v) matrix, returned flattened.

    d(output_l)/d(q_k) = (1/tau) * sum_i w_i * V[i,l] * (K[i,k] - sum_j w_j K[j,k])
    """
    q = np.asarray(q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d = len(q)
    n, v = V.shape

    # Compute softmax weights (numerically stable)
    scores = K @ q / tau
    scores -= np.max(scores)
    exp_scores = np.exp(scores)
    w = exp_scores / np.sum(exp_scores)  # (n,)

    # Weighted key mean: bar_K[k] = sum_j w_j K[j,k]
    bar_K = w @ K  # (d,)

    # Jacobian: J[k, l] = (1/tau) sum_i w_i V[i,l] (K[i,k] - bar_K[k])
    # = (1/tau) [sum_i w_i V[i,l] K[i,k] - bar_K[k] sum_i w_i V[i,l]]
    # = (1/tau) [sum_i w_i V[i,l] K[i,k] - bar_K[k] * output_l]
    jac = np.zeros((d, v))
    for k in range(d):
        for l in range(v):
            jac[k, l] = (1.0 / tau) * np.sum(w * V[:, l] * (K[:, k] - bar_K[k]))

    return jac.flatten().tolist()


def solve_attention_temperature(q, K, V, tau=1.0):
    return _ref_attention_temperature(q, K, V, tau)


_att_cases = []
# Cases at tau=1.0 and tau=0.1 with default query
for _tau in [1.0, 0.1]:
    _att_cases.append({
        "inputs": {
            "q": _ATT_Q.tolist(),
            "K": _ATT_K.tolist(),
            "V": _ATT_V.tolist(),
            "tau": _tau,
        },
        "expected": _ref_attention_temperature(_ATT_Q, _ATT_K, _ATT_V, _tau),
    })

# Case at tau=0.01 with query that has competing keys (non-trivial gradient)
# q=[0.5, 0.49, 0.01]: scores are 0.5, 0.49, 0.01, 0.495 so keys 0 and 3 compete
_ATT_Q_COMP = np.array([0.5, 0.49, 0.01], dtype=float)
_att_cases.append({
    "inputs": {
        "q": _ATT_Q_COMP.tolist(),
        "K": _ATT_K.tolist(),
        "V": _ATT_V.tolist(),
        "tau": 0.01,
    },
    "expected": _ref_attention_temperature(_ATT_Q_COMP, _ATT_K, _ATT_V, 0.01),
})

# Different query at tau=0.1
_ATT_Q2 = np.array([0.1, 0.8, 0.1], dtype=float)
_att_cases.append({
    "inputs": {
        "q": _ATT_Q2.tolist(),
        "K": _ATT_K.tolist(),
        "V": _ATT_V.tolist(),
        "tau": 0.1,
    },
    "expected": _ref_attention_temperature(_ATT_Q2, _ATT_K, _ATT_V, 0.1),
})

# Query with close scores at tau=0.01: q=[0.3, 0.3, 0.4] -> scores 0.3, 0.3, 0.4, 0.3
_ATT_Q3 = np.array([0.3, 0.3, 0.4], dtype=float)
_att_cases.append({
    "inputs": {
        "q": _ATT_Q3.tolist(),
        "K": _ATT_K.tolist(),
        "V": _ATT_V.tolist(),
        "tau": 0.01,
    },
    "expected": _ref_attention_temperature(_ATT_Q3, _ATT_K, _ATT_V, 0.01),
})

attention_temperature = Problem(
    id="attention_temperature",
    category="compositions",
    difficulty=3,
    description=(
        "Gradient of soft attention output with respect to query vector, at multiple "
        "temperatures. At low temperature the softmax concentrates on the top key, "
        "requiring numerically stable computation."
    ),
    prompt=r"""Implement a function `solve(q, K, V, tau=1.0)` that returns the Jacobian
d(output)/d(q) as a flattened list of floats (d*v elements, row-major from a d x v matrix).

Soft attention:
  1. Compute scores: s = K @ q / tau, where K is (n, d), q is (d,), so s is (n,).
  2. Compute weights via numerically stable softmax:
       w = softmax(s) = exp(s - max(s)) / sum(exp(s - max(s)))
  3. Output = w^T @ V = sum_i w_i * V[i], shape (v,).

The Jacobian J has shape (d, v) where J[k, l] = d(output_l) / d(q_k).

Using the chain rule through softmax:
  d(output_l)/d(q_k) = sum_i (d(w_i)/d(q_k)) * V[i][l]

The softmax Jacobian is:
  d(w_i)/d(s_j) = w_i * (delta_{ij} - w_j)

And d(s_j)/d(q_k) = K[j][k] / tau.

So: d(output_l)/d(q_k) = (1/tau) * sum_i V[i][l] * sum_j w_i*(delta_{ij} - w_j) * K[j][k]

Simplify and compute this analytically. Return the d x v Jacobian flattened in
row-major order (all v entries for q_0 first, then q_1, etc.) as a list.

IMPORTANT: Use numerically stable softmax (subtract max of scores before exp).
At tau = 0.01, raw scores can be very large, causing overflow without this.

Do NOT use finite differences.

Parameters:
  q : list of floats, query vector (length d)
  K : list of lists, key matrix (n x d)
  V : list of lists, value matrix (n x v)
  tau : float, temperature (default 1.0)

Available: numpy (as np), scipy, math.
""",
    reference=solve_attention_temperature,
    test_cases=_att_cases,
    atol=1e-5,
    rtol=1e-2,
)

# ===================================================================
# Export
# ===================================================================

ALL = [branch_smooth, soft_kmeans_gradient, diff_earth_movers, diff_hausdorff, attention_temperature]
