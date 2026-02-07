"""Differentiable sorting and ranking problems.

These problems require making inherently non-differentiable discrete operations
(sorting, ranking, top-k, quantiles, NMS) differentiable via continuous relaxations,
then computing gradients through those relaxations.
"""

import numpy as np
from adbench.problem import Problem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FD_H = 1e-5  # step size for finite differences


def _fd_gradient(f, x, h=_FD_H):
    """Central finite-difference gradient of scalar f w.r.t. vector x."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)
    return grad


def _fd_jacobian(f, x, m, h=_FD_H):
    """Central finite-difference Jacobian of vector-valued f (length m) w.r.t. x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    jac = np.zeros((m, n))
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        fp = np.asarray(f(xp))
        fm = np.asarray(f(xm))
        jac[:, j] = (fp - fm) / (2 * h)
    return jac


# ===================================================================
# Problem 1: sort_sinkhorn
# ===================================================================

def _sinkhorn_soft_sort(x, eps=0.1, n_iter=100):
    """Compute soft-sorted vector via Sinkhorn optimal transport.

    Steps:
      1. Build cost matrix C_ij = (x_i - s_j)^2, s = [1, 2, ..., n] (sorted targets).
      2. K = exp(-C / eps).
      3. Sinkhorn iterations to get doubly-stochastic matrix P.
      4. soft_sort = P^T @ x.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.arange(1, n + 1, dtype=float)
    # Cost matrix
    C = (x[:, None] - s[None, :]) ** 2
    K = np.exp(-C / eps)
    # Sinkhorn
    u = np.ones(n)
    for _ in range(n_iter):
        v = 1.0 / (K.T @ u)
        u = 1.0 / (K @ v)
    P = np.diag(u) @ K @ np.diag(v)
    soft_sorted = P.T @ x
    return soft_sorted


def _forward_sort_sinkhorn(x, eps=0.1, n_iter=100):
    """First element of soft-sorted vector."""
    return _sinkhorn_soft_sort(x, eps, n_iter)[0]


def _ref_sort_sinkhorn(x, eps=0.1, n_iter=100):
    x = np.asarray(x, dtype=float)
    return _fd_gradient(lambda z: _forward_sort_sinkhorn(z, eps, n_iter), x).tolist()


def solve_sort_sinkhorn(x, eps=0.1, n_iter=100):
    return _ref_sort_sinkhorn(x, eps, n_iter)


_sort_sinkhorn_cases = []
for _x in [[3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]]:
    _sort_sinkhorn_cases.append({
        "inputs": {"x": _x, "eps": 0.1, "n_iter": 100},
        "expected": _ref_sort_sinkhorn(_x, 0.1, 100),
    })

sort_sinkhorn = Problem(
    id="sort_sinkhorn",
    category="sorting_ranking",
    difficulty=3,
    description=(
        "Differentiable sort via Sinkhorn optimal transport. "
        "Compute the gradient of the first element of the soft-sorted vector w.r.t. input x."
    ),
    prompt=r"""Implement a function `solve(x, eps=0.1, n_iter=100)` that returns the gradient
of soft_sort(x)[0] with respect to each element of x, as a list of floats.

The soft sort is defined via Sinkhorn optimal transport:
1. Let s = [1, 2, ..., n] be sorted target positions.
2. Build cost matrix C_{ij} = (x_i - s_j)^2.
3. Compute K = exp(-C / eps).
4. Run `n_iter` Sinkhorn iterations to obtain doubly-stochastic matrix P:
   - Initialize u = ones(n).
   - Repeat: v = 1 / (K^T u), u = 1 / (K v).
   - P = diag(u) K diag(v).
5. soft_sort(x) = P^T @ x.

Compute d(soft_sort(x)[0]) / dx_k for all k. Return as a list of length n.

Parameters:
  x : list of floats (length n)
  eps : float, Sinkhorn temperature (default 0.1)
  n_iter : int, number of Sinkhorn iterations (default 100)

Available: numpy (as np), scipy, math.
""",
    reference=solve_sort_sinkhorn,
    test_cases=_sort_sinkhorn_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Problem 2: rank_pairwise
# ===================================================================

def _pairwise_soft_rank(x, tau=0.1):
    """Compute soft ranks via pairwise sigmoid comparisons.

    soft_rank_i(x) = 1 + sum_{j != i} sigmoid((x_j - x_i) / tau)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    ranks = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                ranks[i] += 1.0 / (1.0 + np.exp(-(x[j] - x[i]) / tau))
        ranks[i] += 1.0
    return ranks


def _ref_rank_pairwise(x, tau=0.1):
    x = np.asarray(x, dtype=float)
    n = len(x)
    jac = _fd_jacobian(lambda z: _pairwise_soft_rank(z, tau), x, n)
    return jac.tolist()


def solve_rank_pairwise(x, tau=0.1):
    return _ref_rank_pairwise(x, tau)


_rank_pairwise_cases = []
for _x in [[3.0, 1.0, 4.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.1, 1.9, 2.05]]:
    _rank_pairwise_cases.append({
        "inputs": {"x": _x, "tau": 0.1},
        "expected": _ref_rank_pairwise(_x, 0.1),
    })

rank_pairwise = Problem(
    id="rank_pairwise",
    category="sorting_ranking",
    difficulty=3,
    description=(
        "Differentiable ranking via pairwise sigmoid. "
        "Compute the full Jacobian of soft ranks with respect to input x."
    ),
    prompt=r"""Implement a function `solve(x, tau=0.1)` that returns the n x n Jacobian
of the soft ranking function with respect to x, as a nested list (list of lists).

The soft rank is defined as:
  soft_rank_i(x) = 1 + sum_{j != i} sigmoid((x_j - x_i) / tau)

where sigmoid(z) = 1 / (1 + exp(-z)).

Return J where J[i][j] = d(soft_rank_i) / d(x_j), as a list of n lists each of length n.

Parameters:
  x : list of floats (length n)
  tau : float, temperature (default 0.1)

Available: numpy (as np), scipy, math.
""",
    reference=solve_rank_pairwise,
    test_cases=_rank_pairwise_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Problem 3: topk_successive
# ===================================================================

def _forward_topk_successive(x, tau=0.5, k=2):
    """Iterative soft top-k extraction and sum.

    For each of k rounds:
      1. w = softmax(x / tau)
      2. v = w . x  (dot product = extracted value)
      3. x <- x - w * v  (soft removal)
    Return sum of extracted values.
    """
    x = np.asarray(x, dtype=float)
    total = 0.0
    for _ in range(k):
        # softmax
        ex = np.exp((x - np.max(x)) / tau)
        w = ex / np.sum(ex)
        v = np.dot(w, x)
        total += v
        x = x - w * v
    return total


def _ref_topk_successive(x, tau=0.5, k=2):
    x = np.asarray(x, dtype=float)
    return _fd_gradient(lambda z: _forward_topk_successive(z, tau, k), x).tolist()


def solve_topk_successive(x, tau=0.5, k=2):
    return _ref_topk_successive(x, tau, k)


_topk_cases = []
for _x in [[3.0, 1.0, 4.0, 1.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 10.0],
            [2.5, 2.5, 2.5, 2.5, 2.5]]:
    _topk_cases.append({
        "inputs": {"x": _x, "tau": 0.5, "k": 2},
        "expected": _ref_topk_successive(_x, 0.5, 2),
    })

topk_successive = Problem(
    id="topk_successive",
    category="sorting_ranking",
    difficulty=3,
    description=(
        "Differentiable top-k via iterative soft-argmax extraction. "
        "Compute gradient of the sum of k extracted values w.r.t. input x."
    ),
    prompt=r"""Implement a function `solve(x, tau=0.5, k=2)` that returns the gradient
of the soft top-k sum with respect to x, as a list of floats.

The soft top-k sum is computed iteratively for k rounds. Starting with y = copy of x:
  For each round:
    1. Compute softmax weights: w = softmax(y / tau)
       (use the numerically stable form: subtract max before exp)
    2. Extract value: v = dot(w, y)
    3. Soft removal: y <- y - w * v
  The output is the sum of the k extracted values v.

Compute d(total) / d(x_i) for all i. Return as a list of length n.

Parameters:
  x : list of floats (length n)
  tau : float, softmax temperature (default 0.5)
  k : int, number of elements to extract (default 2)

Available: numpy (as np), scipy, math.
""",
    reference=solve_topk_successive,
    test_cases=_topk_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Problem 4: soft_quantile
# ===================================================================

def _forward_soft_quantile(x, q=0.5, eps=0.1, n_iter=100):
    """Soft quantile: soft-sort via Sinkhorn, then pick element at index floor(q*(n-1))."""
    x = np.asarray(x, dtype=float)
    ss = _sinkhorn_soft_sort(x, eps, n_iter)
    idx = int(np.floor(q * (len(x) - 1)))
    return ss[idx]


def _ref_soft_quantile(x, q=0.5, eps=0.1, n_iter=100):
    x = np.asarray(x, dtype=float)
    return _fd_gradient(
        lambda z: _forward_soft_quantile(z, q, eps, n_iter), x
    ).tolist()


def solve_soft_quantile(x, q=0.5, eps=0.1, n_iter=100):
    return _ref_soft_quantile(x, q, eps, n_iter)


_soft_quantile_cases = []
for _x in [[3.0, 1.0, 4.0, 1.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [10.0, 1.0, 5.0, 3.0, 7.0]]:
    _soft_quantile_cases.append({
        "inputs": {"x": _x, "q": 0.5, "eps": 0.1, "n_iter": 100},
        "expected": _ref_soft_quantile(_x, 0.5, 0.1, 100),
    })

soft_quantile = Problem(
    id="soft_quantile",
    category="sorting_ranking",
    difficulty=3,
    description=(
        "Differentiable quantile via Sinkhorn soft-sort. "
        "Compute gradient of the soft q-quantile w.r.t. input x."
    ),
    prompt=r"""Implement a function `solve(x, q=0.5, eps=0.1, n_iter=100)` that returns
the gradient of the soft q-quantile of x with respect to each element x_k, as a list.

The soft quantile is defined as:
1. Compute the Sinkhorn soft-sort of x:
   a. Let s = [1, 2, ..., n] be sorted target positions.
   b. Cost matrix: C_{ij} = (x_i - s_j)^2.
   c. K = exp(-C / eps).
   d. Sinkhorn iterations (n_iter rounds):
      - Initialize u = ones(n).
      - Repeat: v = 1 / (K^T u), u = 1 / (K v).
      - P = diag(u) K diag(v).
   e. soft_sorted = P^T @ x.
2. The soft q-quantile is soft_sorted[floor(q * (n-1))].

Compute d(soft_quantile) / d(x_k) for all k. Return as a list of length n.

Parameters:
  x : list of floats (length n)
  q : float, quantile level (default 0.5)
  eps : float, Sinkhorn temperature (default 0.1)
  n_iter : int, number of Sinkhorn iterations (default 100)

Available: numpy (as np), scipy, math.
""",
    reference=solve_soft_quantile,
    test_cases=_soft_quantile_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Problem 5: nms_soft
# ===================================================================

def _iou_1d(c1, w1, c2, w2):
    """IoU of two 1D intervals [c - w/2, c + w/2]."""
    l1, r1 = c1 - w1 / 2, c1 + w1 / 2
    l2, r2 = c2 - w2 / 2, c2 + w2 / 2
    inter = max(0.0, min(r1, r2) - max(l1, l2))
    union = (r1 - l1) + (r2 - l2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _forward_nms_soft(scores, centers, widths, tau=0.1, n_steps=None):
    """Soft NMS on 1D intervals.

    For n_steps rounds (default = N):
      1. Compute soft selection weight for each box: w_i = softmax(s / tau).
      2. Selected score contribution = dot(w, s).
      3. For each box j, compute IoU with every other box i, weighted by w_i.
         Suppression factor for j: prod_i (1 - IoU(i,j) * w_i)  for i != j
      4. Update scores: s_j <- s_j * suppression_factor_j.
    Return the sum of all selected score contributions across rounds.
    """
    scores = np.asarray(scores, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    N = len(scores)
    if n_steps is None:
        n_steps = N

    # Precompute IoU matrix
    iou_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                iou_mat[i, j] = _iou_1d(centers[i], widths[i], centers[j], widths[j])

    s = scores.copy()
    total = 0.0
    for _ in range(n_steps):
        # Soft selection via softmax
        ex = np.exp((s - np.max(s)) / tau)
        w = ex / np.sum(ex)
        # Selected score
        total += np.dot(w, s)
        # Suppression
        for j in range(N):
            suppress = 1.0
            for i in range(N):
                if i != j:
                    suppress *= (1.0 - iou_mat[i, j] * w[i])
            s[j] = s[j] * suppress
    return total


def _ref_nms_soft(scores, centers, widths, tau=0.1):
    scores = np.asarray(scores, dtype=float)
    centers = np.asarray(centers, dtype=float)
    widths = np.asarray(widths, dtype=float)
    return _fd_gradient(
        lambda sc: _forward_nms_soft(sc, centers, widths, tau), scores
    ).tolist()


def solve_nms_soft(scores, centers, widths, tau=0.1):
    return _ref_nms_soft(scores, centers, widths, tau)


# Test cases: 4 boxes with various overlap patterns
_nms_cases = []
# Case 1: distinct boxes
_nms_cases.append({
    "inputs": {
        "scores": [0.9, 0.8, 0.7, 0.3],
        "centers": [1.0, 3.0, 5.0, 7.0],
        "widths": [1.0, 1.0, 1.0, 1.0],
        "tau": 0.1,
    },
    "expected": _ref_nms_soft(
        [0.9, 0.8, 0.7, 0.3], [1.0, 3.0, 5.0, 7.0], [1.0, 1.0, 1.0, 1.0], 0.1
    ),
})
# Case 2: two overlapping pairs
_nms_cases.append({
    "inputs": {
        "scores": [0.9, 0.85, 0.5, 0.45],
        "centers": [1.0, 1.3, 5.0, 5.2],
        "widths": [1.0, 1.0, 1.0, 1.0],
        "tau": 0.1,
    },
    "expected": _ref_nms_soft(
        [0.9, 0.85, 0.5, 0.45], [1.0, 1.3, 5.0, 5.2], [1.0, 1.0, 1.0, 1.0], 0.1
    ),
})
# Case 3: all overlapping
_nms_cases.append({
    "inputs": {
        "scores": [0.6, 0.8, 0.7, 0.5],
        "centers": [2.0, 2.2, 2.4, 2.6],
        "widths": [2.0, 2.0, 2.0, 2.0],
        "tau": 0.1,
    },
    "expected": _ref_nms_soft(
        [0.6, 0.8, 0.7, 0.5], [2.0, 2.2, 2.4, 2.6], [2.0, 2.0, 2.0, 2.0], 0.1
    ),
})
# Case 4: equal scores
_nms_cases.append({
    "inputs": {
        "scores": [0.5, 0.5, 0.5, 0.5],
        "centers": [1.0, 2.0, 3.0, 4.0],
        "widths": [1.5, 1.5, 1.5, 1.5],
        "tau": 0.1,
    },
    "expected": _ref_nms_soft(
        [0.5, 0.5, 0.5, 0.5], [1.0, 2.0, 3.0, 4.0], [1.5, 1.5, 1.5, 1.5], 0.1
    ),
})

nms_soft = Problem(
    id="nms_soft",
    category="sorting_ranking",
    difficulty=3,
    description=(
        "Differentiable non-maximum suppression for 1D intervals. "
        "Compute gradient of total selected score w.r.t. input scores."
    ),
    prompt=r"""Implement a function `solve(scores, centers, widths, tau=0.1)` that returns
the gradient of the soft NMS total selected score with respect to the input scores,
as a list of floats.

Soft NMS on N 1D intervals [center_i - width_i/2, center_i + width_i/2]:

Precompute IoU matrix:
  IoU(i,j) = intersection(i,j) / union(i,j) for 1D intervals. IoU(i,i) = 0.

Starting with s = copy of scores, run N rounds:
  1. Compute softmax weights: w = softmax(s / tau) (numerically stable).
  2. Selected score contribution: dot(w, s). Accumulate this into total.
  3. For each box j, compute suppression factor:
       suppress_j = product over i != j of (1 - IoU(i,j) * w_i)
  4. Update: s_j <- s_j * suppress_j.

Return d(total) / d(scores_k) for all k, as a list of length N.

Parameters:
  scores : list of floats (length N)
  centers : list of floats (length N), interval centers
  widths : list of floats (length N), interval widths
  tau : float, softmax temperature (default 0.1)

Available: numpy (as np), scipy, math.
""",
    reference=solve_nms_soft,
    test_cases=_nms_cases,
    atol=1e-4,
    rtol=1e-2,
)

# ===================================================================
# Export
# ===================================================================

ALL = [sort_sinkhorn, rank_pairwise, topk_successive, soft_quantile, nms_soft]
