"""Differentiable algorithms: dynamic programming and graph algorithm relaxations.

Each problem asks the model to analytically differentiate through a relaxed
(smoothed) version of a classically non-differentiable algorithm. Reference
solutions use the forward relaxed computation + central finite differences.
"""

import numpy as np
from adbench.problem import Problem

# ---------------------------------------------------------------------------
# Shared utility: numerically-stable softmin / softmax helpers
# ---------------------------------------------------------------------------

def _softmin(vals, gamma):
    """Numerically stable softmin: -gamma * log(sum(exp(-v/gamma)))."""
    vals = np.asarray(vals, dtype=float)
    v_min = np.min(vals)
    return -gamma * np.log(np.sum(np.exp(-(vals - v_min) / gamma))) + v_min


def _softmax_op(vals, gamma):
    """Numerically stable softmax (log-sum-exp): gamma * log(sum(exp(v/gamma)))."""
    vals = np.asarray(vals, dtype=float)
    v_max = np.max(vals)
    return gamma * np.log(np.sum(np.exp((vals - v_max) / gamma))) + v_max


# ===================================================================
# Problem 6: Soft Dynamic Time Warping
# ===================================================================

def _soft_dtw_forward(a, b, gamma=1.0):
    """Forward pass of soft-DTW between sequences a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)
    INF = 1e30
    # D is (m+1) x (n+1), with D[0,:] = D[:,0] = INF except D[0,0] = 0
    D = np.full((m + 1, n + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + _softmin(
                [D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]], gamma
            )
    return D[m, n]


def _ref_dtw_soft(a, b, gamma=1.0):
    """Reference: gradient of soft-DTW w.r.t. a via finite differences."""
    a = np.asarray(a, dtype=float)
    h = 1e-5
    grad = np.zeros(len(a))
    for i in range(len(a)):
        a_plus = a.copy(); a_plus[i] += h
        a_minus = a.copy(); a_minus[i] -= h
        grad[i] = (_soft_dtw_forward(a_plus, b, gamma) -
                    _soft_dtw_forward(a_minus, b, gamma)) / (2 * h)
    return grad.tolist()


_dtw_soft_prompt = r"""Implement a function `solve(a, b, gamma)` that computes the gradient of the
Soft Dynamic Time Warping (soft-DTW) distance with respect to each element of
sequence `a`.

**Soft-DTW definition:**
Given two real-valued sequences a of length m and b of length n, define the
(m+1) x (n+1) DP table D where:
- D[0, 0] = 0
- D[i, 0] = D[0, j] = +infinity  for i, j > 0
- For i = 1..m, j = 1..n:
    cost_ij = (a[i-1] - b[j-1])^2
    D[i, j] = cost_ij + softmin_gamma(D[i-1, j], D[i, j-1], D[i-1, j-1])

where softmin_gamma(x1, x2, x3) = -gamma * log(exp(-x1/gamma) + exp(-x2/gamma) + exp(-x3/gamma)).

The soft-DTW value is D[m, n].

**Your task:** Return a list of length m containing d(soft-DTW) / d(a_i) for
i = 0, ..., m-1.

Parameters:
- a: list of floats (sequence 1)
- b: list of floats (sequence 2)
- gamma: float, smoothing temperature (e.g. 1.0)

Return: list of floats (gradient w.r.t. a)

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (backpropagation through the DP), not finite differences.
"""

_dtw_soft_cases = [
    {
        "inputs": {"a": [1.0, 3.0, 2.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0], "gamma": 1.0},
        "expected": _ref_dtw_soft([1.0, 3.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0], 1.0),
    },
    {
        "inputs": {"a": [0.5, 1.5, 2.5], "b": [0.0, 1.0, 2.0, 3.0], "gamma": 1.0},
        "expected": _ref_dtw_soft([0.5, 1.5, 2.5], [0.0, 1.0, 2.0, 3.0], 1.0),
    },
    {
        "inputs": {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [1.5, 2.5, 3.5], "gamma": 1.0},
        "expected": _ref_dtw_soft([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5], 1.0),
    },
]

dtw_soft = Problem(
    id="dtw_soft",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of Soft Dynamic Time Warping distance w.r.t. first sequence.",
    prompt=_dtw_soft_prompt,
    reference=lambda a, b, gamma=1.0: _ref_dtw_soft(a, b, gamma),
    test_cases=_dtw_soft_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 7: Differentiable Edit Distance
# ===================================================================

def _soft_edit_distance_forward(a, b, gamma=0.5):
    """Forward pass of soft edit distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)
    D = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        D[i, 0] = float(i)
    for j in range(1, n + 1):
        D[0, j] = float(j)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = _softmin(
                [D[i - 1, j - 1] + sub_cost, D[i - 1, j] + 1.0, D[i, j - 1] + 1.0],
                gamma,
            )
    return D[m, n]


def _ref_edit_distance_soft(a, b, gamma=0.5):
    """Reference: gradient of soft edit distance w.r.t. a via finite differences."""
    a = np.asarray(a, dtype=float)
    h = 1e-5
    grad = np.zeros(len(a))
    for i in range(len(a)):
        a_plus = a.copy(); a_plus[i] += h
        a_minus = a.copy(); a_minus[i] -= h
        grad[i] = (_soft_edit_distance_forward(a_plus, b, gamma) -
                    _soft_edit_distance_forward(a_minus, b, gamma)) / (2 * h)
    return grad.tolist()


_edit_distance_soft_prompt = r"""Implement a function `solve(a, b, gamma)` that computes the gradient of a
soft (differentiable) edit distance with respect to each element of sequence `a`.

**Soft edit distance definition:**
Given real-valued sequences a (length m) and b (length n), define the
(m+1) x (n+1) DP table D:
- D[0, 0] = 0
- D[i, 0] = i  for i = 1..m  (i deletions)
- D[0, j] = j  for j = 1..n  (j insertions)
- For i = 1..m, j = 1..n:
    sub_cost = (a[i-1] - b[j-1])^2
    D[i, j] = softmin_gamma(
        D[i-1, j-1] + sub_cost,   # substitution
        D[i-1, j] + 1,            # deletion
        D[i, j-1] + 1             # insertion
    )

where softmin_gamma(x1, x2, x3) = -gamma * log(exp(-x1/gamma) + exp(-x2/gamma) + exp(-x3/gamma)).

The soft edit distance is D[m, n].

**Your task:** Return a list of length m containing d(soft_edit_dist) / d(a_i)
for i = 0, ..., m-1.

Parameters:
- a: list of floats (sequence 1)
- b: list of floats (sequence 2)
- gamma: float, smoothing temperature (e.g. 0.5)

Return: list of floats (gradient w.r.t. a)

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (backpropagation through the DP), not finite differences.
"""

_edit_distance_soft_cases = [
    {
        "inputs": {"a": [1.0, 2.0, 3.0], "b": [1.1, 2.9, 3.1], "gamma": 0.5},
        "expected": _ref_edit_distance_soft([1.0, 2.0, 3.0], [1.1, 2.9, 3.1], 0.5),
    },
    {
        "inputs": {"a": [0.0, 1.0, 2.0, 3.0], "b": [0.5, 2.5], "gamma": 0.5},
        "expected": _ref_edit_distance_soft([0.0, 1.0, 2.0, 3.0], [0.5, 2.5], 0.5),
    },
    {
        "inputs": {"a": [1.0, 4.0, 2.0], "b": [1.0, 2.0, 4.0, 3.0], "gamma": 0.5},
        "expected": _ref_edit_distance_soft([1.0, 4.0, 2.0], [1.0, 2.0, 4.0, 3.0], 0.5),
    },
]

edit_distance_soft = Problem(
    id="edit_distance_soft",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of soft (differentiable) edit distance w.r.t. first sequence.",
    prompt=_edit_distance_soft_prompt,
    reference=lambda a, b, gamma=0.5: _ref_edit_distance_soft(a, b, gamma),
    test_cases=_edit_distance_soft_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 8: Differentiable Shortest Path (Bellman-Ford with softmin)
# ===================================================================

def _soft_bellman_ford_forward(n, edges, weights, source, target, gamma=0.5):
    """Bellman-Ford with softmin relaxation.

    edges: list of (u, v) pairs
    weights: list of floats (same length as edges)
    Returns soft shortest-path distance d[target].
    """
    INF = 1e30
    d = np.full(n, INF)
    d[source] = 0.0
    for _iteration in range(n - 1):
        d_new = d.copy()
        for v in range(n):
            # Gather all candidate relaxations for node v
            candidates = [d[v]]  # keep current distance as candidate
            for idx, (u, vv) in enumerate(edges):
                if vv == v:
                    candidates.append(d[u] + weights[idx])
            if len(candidates) > 1:
                d_new[v] = _softmin(candidates, gamma)
            # If only candidate is current d[v], no update
        d = d_new
    return d[target]


def _ref_shortest_path_softmin(n, edges, weights, source, target, gamma=0.5):
    """Reference: gradient of soft shortest-path distance w.r.t. each edge weight."""
    weights = np.asarray(weights, dtype=float)
    h = 1e-5
    grad = np.zeros(len(weights))
    for i in range(len(weights)):
        w_plus = weights.copy(); w_plus[i] += h
        w_minus = weights.copy(); w_minus[i] -= h
        grad[i] = (_soft_bellman_ford_forward(n, edges, w_plus.tolist(), source, target, gamma) -
                    _soft_bellman_ford_forward(n, edges, w_minus.tolist(), source, target, gamma)) / (2 * h)
    return grad.tolist()


# Define a specific graph: 5 nodes, 8 directed edges
_sp_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (0, 3)]
_sp_weights_1 = [1.0, 4.0, 2.0, 5.0, 1.0, 6.0, 2.0, 7.0]
_sp_weights_2 = [2.0, 1.0, 3.0, 2.0, 1.0, 5.0, 1.0, 10.0]
_sp_weights_3 = [3.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0, 8.0]

_shortest_path_prompt = r"""Implement a function `solve(n, edges, weights, source, target, gamma)` that
computes the gradient of a differentiable shortest-path distance with respect to
each edge weight.

**Algorithm: Bellman-Ford with softmin relaxation**
Given a directed graph with n nodes, a list of directed edges [(u, v), ...], and
corresponding edge weights, compute distances using a softmin-relaxed
Bellman-Ford algorithm:

1. Initialize d[source] = 0, d[v] = +infinity for all v != source.
2. For (n-1) iterations:
   For each node v, gather all candidate distances:
     - The current d[v]
     - d[u] + w(u,v) for each incoming edge (u, v)
   If there are at least two candidates, set:
     d[v] = softmin_gamma(candidates)
   where softmin_gamma(x1, ..., xk) = -gamma * log(sum_i exp(-x_i / gamma)).

The soft shortest-path distance is d[target] after n-1 iterations.

**Your task:** Return a list of length |edges| containing
d(d[target]) / d(w_e) for each edge e, in the same order as the input edge list.

Parameters:
- n: int, number of nodes
- edges: list of [u, v] pairs (directed edges, 0-indexed)
- weights: list of floats (edge weights, same length as edges)
- source: int (source node)
- target: int (target node)
- gamma: float, smoothing temperature (e.g. 0.5)

Return: list of floats (gradient w.r.t. each edge weight)

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (backpropagation through the relaxation iterations), not finite
differences.
"""

_shortest_path_cases = [
    {
        "inputs": {
            "n": 5, "edges": _sp_edges, "weights": _sp_weights_1,
            "source": 0, "target": 4, "gamma": 0.5,
        },
        "expected": _ref_shortest_path_softmin(5, _sp_edges, _sp_weights_1, 0, 4, 0.5),
    },
    {
        "inputs": {
            "n": 5, "edges": _sp_edges, "weights": _sp_weights_2,
            "source": 0, "target": 4, "gamma": 0.5,
        },
        "expected": _ref_shortest_path_softmin(5, _sp_edges, _sp_weights_2, 0, 4, 0.5),
    },
    {
        "inputs": {
            "n": 5, "edges": _sp_edges, "weights": _sp_weights_3,
            "source": 0, "target": 4, "gamma": 0.5,
        },
        "expected": _ref_shortest_path_softmin(5, _sp_edges, _sp_weights_3, 0, 4, 0.5),
    },
]

shortest_path_softmin = Problem(
    id="shortest_path_softmin",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of softmin-relaxed Bellman-Ford shortest path w.r.t. edge weights.",
    prompt=_shortest_path_prompt,
    reference=lambda n, edges, weights, source, target, gamma=0.5: _ref_shortest_path_softmin(
        n, edges, weights, source, target, gamma
    ),
    test_cases=_shortest_path_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 9: Differentiable Needleman-Wunsch Alignment
# ===================================================================

def _soft_nw_forward(a, b, g, gamma=1.0):
    """Forward pass of soft Needleman-Wunsch alignment score.

    Uses softmax (log-sum-exp) because we are *maximizing* alignment score.
    Match score = -(a_i - b_j)^2, gap penalty = g (subtracted).
    S[i,j] = softmax_gamma(S[i-1,j-1] + match, S[i-1,j] - g, S[i,j-1] - g)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)
    NEG_INF = -1e30
    S = np.full((m + 1, n + 1), NEG_INF)
    S[0, 0] = 0.0
    for i in range(1, m + 1):
        S[i, 0] = S[i - 1, 0] - g
    for j in range(1, n + 1):
        S[0, j] = S[0, j - 1] - g
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = -(a[i - 1] - b[j - 1]) ** 2
            S[i, j] = _softmax_op(
                [S[i - 1, j - 1] + match, S[i - 1, j] - g, S[i, j - 1] - g],
                gamma,
            )
    return S[m, n]


def _ref_nw_alignment(a, b, g, gamma=1.0):
    """Reference: gradient of soft NW alignment score w.r.t. gap penalty g."""
    h = 1e-5
    grad = (_soft_nw_forward(a, b, g + h, gamma) -
            _soft_nw_forward(a, b, g - h, gamma)) / (2 * h)
    return float(grad)


_nw_alignment_prompt = r"""Implement a function `solve(a, b, g, gamma)` that computes the gradient of a
differentiable Needleman-Wunsch alignment score with respect to the gap penalty g.

**Soft Needleman-Wunsch definition:**
Given real-valued sequences a (length m) and b (length n), a gap penalty g > 0,
and temperature gamma, define the (m+1) x (n+1) DP table S:
- S[0, 0] = 0
- S[i, 0] = S[i-1, 0] - g  for i = 1..m
- S[0, j] = S[0, j-1] - g  for j = 1..n
- For i = 1..m, j = 1..n:
    match_ij = -(a[i-1] - b[j-1])^2
    S[i, j] = softmax_gamma(
        S[i-1, j-1] + match_ij,   # align a_i with b_j
        S[i-1, j] - g,            # gap in b
        S[i, j-1] - g             # gap in a
    )

where softmax_gamma(x1, x2, x3) = gamma * log(exp(x1/gamma) + exp(x2/gamma) + exp(x3/gamma)).

The alignment score is S[m, n].

**Your task:** Return a single float: d(S[m,n]) / d(g).

Parameters:
- a: list of floats (sequence 1)
- b: list of floats (sequence 2)
- g: float (gap penalty, positive)
- gamma: float, smoothing temperature (e.g. 1.0)

Return: float (gradient of alignment score w.r.t. gap penalty g)

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (backpropagation through the DP), not finite differences.
"""

_nw_alignment_cases = [
    {
        "inputs": {"a": [1.0, 3.0, 2.0], "b": [1.0, 2.0, 3.0], "g": 1.0, "gamma": 1.0},
        "expected": _ref_nw_alignment([1.0, 3.0, 2.0], [1.0, 2.0, 3.0], 1.0, 1.0),
    },
    {
        "inputs": {"a": [1.0, 3.0, 2.0], "b": [1.0, 2.0, 3.0], "g": 2.0, "gamma": 1.0},
        "expected": _ref_nw_alignment([1.0, 3.0, 2.0], [1.0, 2.0, 3.0], 2.0, 1.0),
    },
    {
        "inputs": {"a": [0.0, 1.0, 2.0, 3.0], "b": [0.5, 1.5, 2.5], "g": 0.5, "gamma": 1.0},
        "expected": _ref_nw_alignment([0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5], 0.5, 1.0),
    },
    {
        "inputs": {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0, 4.0], "g": 1.5, "gamma": 1.0},
        "expected": _ref_nw_alignment([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], 1.5, 1.0),
    },
]

nw_alignment = Problem(
    id="nw_alignment",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of soft Needleman-Wunsch alignment score w.r.t. gap penalty.",
    prompt=_nw_alignment_prompt,
    reference=lambda a, b, g, gamma=1.0: _ref_nw_alignment(a, b, g, gamma),
    test_cases=_nw_alignment_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 10: Differentiable Minimum Spanning Tree Weight
# ===================================================================

def _soft_mst_weight_forward(n, edge_weights, gamma=1.0):
    """Compute a soft MST weight using the matrix-tree theorem approach.

    For an undirected graph on n nodes with edge weights w_ij, the
    weighted matrix tree theorem says: the sum of spanning tree weights
    (where weight of a tree = product of its edge weights) equals
    det(L_reduced), where L is the weighted Laplacian and L_reduced is
    obtained by deleting any one row and column.

    We define: soft_MST_weight = -gamma * log(det(L_reduced))
                                 + (n-1) * gamma * log(gamma)
    where L_ij = -w_ij for i != j, L_ii = sum_j w_ij, using
    Boltzmann weights exp(-c_ij / gamma) as effective edge weights
    where c_ij are the given edge costs.

    This gives a smooth approximation to the MST cost: as gamma -> 0,
    the dominant tree is the MST and the value approaches the MST cost.

    edge_weights: dict or flat structure. We use a flat list for edges of
    the complete graph (i < j), ordered lexicographically.
    """
    # Build effective Boltzmann edge weights
    boltz = np.exp(-np.asarray(edge_weights, dtype=float) / gamma)

    # Map flat index to (i, j) pairs for complete graph: (0,1),(0,2),...,(0,n-1),(1,2),...
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    assert len(boltz) == len(edges), f"Expected {len(edges)} edges, got {len(boltz)}"

    # Build weighted Laplacian
    L = np.zeros((n, n))
    for idx, (i, j) in enumerate(edges):
        w = boltz[idx]
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w

    # Reduced Laplacian (delete last row and column)
    L_red = L[:-1, :-1]
    det_val = np.linalg.det(L_red)
    if det_val <= 0:
        det_val = 1e-300  # safeguard

    # soft MST weight = -gamma * log(det(L_reduced)) + (n-1) * gamma * log(gamma)
    # The (n-1)*gamma*log(gamma) normalizes for the Boltzmann transform
    result = -gamma * np.log(det_val) + (n - 1) * gamma * np.log(gamma)
    return result


def _ref_mst_soft(n, edge_weights, gamma=1.0):
    """Reference: gradient of soft MST weight w.r.t. each edge weight."""
    edge_weights = np.asarray(edge_weights, dtype=float)
    h = 1e-5
    grad = np.zeros(len(edge_weights))
    for i in range(len(edge_weights)):
        w_plus = edge_weights.copy(); w_plus[i] += h
        w_minus = edge_weights.copy(); w_minus[i] -= h
        grad[i] = (_soft_mst_weight_forward(n, w_plus.tolist(), gamma) -
                    _soft_mst_weight_forward(n, w_minus.tolist(), gamma)) / (2 * h)
    return grad.tolist()


# Complete graph on 4 nodes: 6 edges (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_mst_weights_1 = [1.0, 3.0, 5.0, 2.0, 4.0, 1.5]
_mst_weights_2 = [2.0, 1.0, 4.0, 3.0, 2.0, 5.0]
_mst_weights_3 = [1.5, 2.5, 3.5, 1.0, 4.0, 2.0]

_mst_soft_prompt = r"""Implement a function `solve(n, edge_weights, gamma)` that computes the gradient
of a soft (differentiable) minimum spanning tree weight with respect to each
edge cost.

**Soft MST via the matrix-tree theorem:**
Given an undirected complete graph on n nodes with edge costs c_e for each edge,
we define effective Boltzmann edge weights:
    w_e = exp(-c_e / gamma)

Build the weighted Laplacian matrix L of size n x n:
    L[i, j] = -w_e   for i != j where e = (i, j)
    L[i, i] = sum of w_e for all edges incident to i

Let L_red be the (n-1) x (n-1) matrix obtained by deleting the last row and
column of L. The soft MST cost is defined as:

    soft_MST = -gamma * log(det(L_red)) + (n - 1) * gamma * log(gamma)

Edges are ordered lexicographically by (i, j) with i < j for the complete graph:
(0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1).

**Your task:** Return a list of length C(n,2) containing
d(soft_MST) / d(c_e) for each edge e in the lexicographic order above.

Parameters:
- n: int, number of nodes
- edge_weights: list of floats (edge costs c_e in lexicographic order)
- gamma: float, smoothing temperature (e.g. 1.0)

Return: list of floats (gradient w.r.t. each edge cost)

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (through the Laplacian determinant), not finite differences.
"""

_mst_soft_cases = [
    {
        "inputs": {"n": 4, "edge_weights": _mst_weights_1, "gamma": 1.0},
        "expected": _ref_mst_soft(4, _mst_weights_1, 1.0),
    },
    {
        "inputs": {"n": 4, "edge_weights": _mst_weights_2, "gamma": 1.0},
        "expected": _ref_mst_soft(4, _mst_weights_2, 1.0),
    },
    {
        "inputs": {"n": 4, "edge_weights": _mst_weights_3, "gamma": 1.0},
        "expected": _ref_mst_soft(4, _mst_weights_3, 1.0),
    },
]

mst_soft = Problem(
    id="mst_soft",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of soft MST weight (matrix-tree theorem) w.r.t. edge costs.",
    prompt=_mst_soft_prompt,
    reference=lambda n, edge_weights, gamma=1.0: _ref_mst_soft(n, edge_weights, gamma),
    test_cases=_mst_soft_cases,
    atol=1e-4,
    rtol=5e-2,
)


# ===================================================================
# Problem 11: CTC Loss Gradient
# ===================================================================

def _ctc_forward_log(logits, labels):
    """Compute CTC loss = -log P(labels | logits) in log-space.

    logits: T x K array of log-probabilities (already log-softmaxed).
    labels: list of label indices (no blanks, no repeats needed).
    Blank index = 0.

    Uses the standard CTC forward algorithm with extended label sequence
    (interleaved blanks).
    """
    logits = np.asarray(logits, dtype=float)
    T, K = logits.shape
    labels = list(labels)
    L = len(labels)

    # Build extended label with blanks: [blank, l1, blank, l2, blank, ...]
    ext = [0]
    for l in labels:
        ext.append(l)
        ext.append(0)
    S = len(ext)  # = 2*L + 1

    # log-softmax the logits to get log-probabilities
    log_probs = logits - np.log(np.sum(np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1, keepdims=True)) - np.max(logits, axis=1, keepdims=True) + np.max(logits, axis=1, keepdims=True)
    # Simpler: proper log-softmax
    mx = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - mx - np.log(np.sum(np.exp(logits - mx), axis=1, keepdims=True))

    NEG_INF = -1e30

    # alpha[t, s] = log-probability of emitting ext[:s+1] in time steps 0..t
    alpha = np.full((T, S), NEG_INF)

    # t = 0: can only start with blank or first label
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    def _log_add(a, b):
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        mx = max(a, b)
        return mx + np.log(np.exp(a - mx) + np.exp(b - mx))

    for t in range(1, T):
        for s in range(S):
            # Can come from same state
            val = alpha[t - 1, s]
            # Can come from previous state
            if s >= 1:
                val = _log_add(val, alpha[t - 1, s - 1])
            # Can skip a blank (if current is not blank and not same as s-2)
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]:
                val = _log_add(val, alpha[t - 1, s - 2])
            alpha[t, s] = val + log_probs[t, ext[s]]

    # Total log-probability: log-add of last two states at time T-1
    log_prob = _log_add(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    return -log_prob  # CTC loss = negative log-probability


def _ref_ctc_loss_grad(logits, labels):
    """Reference: gradient of CTC loss w.r.t. logits via finite differences."""
    logits = np.asarray(logits, dtype=float)
    T, K = logits.shape
    h = 1e-5
    grad = np.zeros((T, K))
    for t in range(T):
        for k in range(K):
            logits_plus = logits.copy(); logits_plus[t, k] += h
            logits_minus = logits.copy(); logits_minus[t, k] -= h
            grad[t, k] = (_ctc_forward_log(logits_plus, labels) -
                          _ctc_forward_log(logits_minus, labels)) / (2 * h)
    return grad.tolist()


# Test logits: T=4 timesteps, K=4 classes (blank=0, class 1, 2, 3)
# Using moderate logit values so gradients are not too extreme
_ctc_logits_1 = [
    [2.0, 1.0, 0.5, 0.1],
    [0.1, 2.5, 0.3, 0.2],
    [0.3, 0.2, 2.0, 0.5],
    [1.5, 0.1, 0.3, 0.2],
]

_ctc_logits_2 = [
    [1.0, 1.5, 0.5, 0.3],
    [0.5, 0.3, 1.8, 0.4],
    [0.2, 2.0, 0.3, 0.1],
    [1.0, 0.5, 0.5, 1.5],
]

_ctc_logits_3 = [
    [0.5, 1.0, 0.8, 0.2],
    [1.0, 0.5, 1.5, 0.3],
    [0.3, 1.2, 0.4, 1.0],
    [1.5, 0.2, 0.8, 0.5],
]

_ctc_loss_prompt = r"""Implement a function `solve(logits, labels)` that computes the gradient of the
CTC (Connectionist Temporal Classification) loss with respect to the input logits.

**CTC loss definition:**
Given:
- logits: a T x K matrix of raw logits (NOT log-probabilities). Class 0 is the
  blank symbol.
- labels: a list of L label indices (not containing blank, no consecutive
  repeats needed).

Steps:
1. Compute log-probabilities via log-softmax:
   log_probs[t, k] = logits[t, k] - log(sum_j exp(logits[t, j]))

2. Build the extended label sequence by interleaving blanks:
   ext = [0, labels[0], 0, labels[1], 0, ..., labels[L-1], 0]
   This has length S = 2*L + 1.

3. Forward DP in log-space:
   alpha[t, s] = log-probability that the output at times 0..t generates
   the prefix ext[0..s].

   Initialization (t=0):
     alpha[0, 0] = log_probs[0, ext[0]]
     alpha[0, 1] = log_probs[0, ext[1]]
     alpha[0, s] = -infinity for s >= 2

   Recurrence (t >= 1):
     For each s:
       val = alpha[t-1, s]                           (stay in same state)
       if s >= 1: val = log_add(val, alpha[t-1, s-1])  (from previous state)
       if s >= 2 and ext[s] != 0 and ext[s] != ext[s-2]:
           val = log_add(val, alpha[t-1, s-2])        (skip blank)
       alpha[t, s] = val + log_probs[t, ext[s]]

   where log_add(a, b) = log(exp(a) + exp(b)) computed in a numerically
   stable way.

4. Total log-probability:
   log_prob = log_add(alpha[T-1, S-1], alpha[T-1, S-2])

5. CTC loss = -log_prob

**Your task:** Return a T x K nested list (list of lists) containing
d(CTC_loss) / d(logits[t, k]) for all t, k.

Parameters:
- logits: list of lists (T x K), raw logit values
- labels: list of ints (label indices, not containing 0 = blank)

Return: list of lists (T x K), the gradient matrix

You have access to numpy (as np), scipy, and math. Implement analytic
differentiation (forward-backward algorithm + backpropagation through
log-softmax), not finite differences.
"""

_ctc_labels_1 = [1, 2]

_ctc_loss_cases = [
    {
        "inputs": {"logits": _ctc_logits_1, "labels": _ctc_labels_1},
        "expected": _ref_ctc_loss_grad(_ctc_logits_1, _ctc_labels_1),
    },
    {
        "inputs": {"logits": _ctc_logits_2, "labels": _ctc_labels_1},
        "expected": _ref_ctc_loss_grad(_ctc_logits_2, _ctc_labels_1),
    },
    {
        "inputs": {"logits": _ctc_logits_3, "labels": [1, 3]},
        "expected": _ref_ctc_loss_grad(_ctc_logits_3, [1, 3]),
    },
]

ctc_loss_grad = Problem(
    id="ctc_loss_grad",
    category="diff_algorithms",
    difficulty=3,
    description="Gradient of CTC loss w.r.t. logits (forward-backward through DP).",
    prompt=_ctc_loss_prompt,
    reference=lambda logits, labels: _ref_ctc_loss_grad(logits, labels),
    test_cases=_ctc_loss_cases,
    atol=1e-3,
    rtol=5e-2,
)


# ===================================================================
# ALL problems in this module
# ===================================================================

ALL = [
    dtw_soft,
    edit_distance_soft,
    shortest_path_softmin,
    nw_alignment,
    mst_soft,
    ctc_loss_grad,
]
