"""Category: Gradient Estimation — differentiating through discrete and stochastic programs.

Problems requiring exact computation of gradients of expected values
involving categorical distributions, Gumbel-Softmax relaxations,
compound stochastic programs, and perturbation-based methods.
"""

import numpy as np
from scipy.special import erf

from adbench.problem import Problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(theta):
    """Numerically stable softmax."""
    t = np.asarray(theta, dtype=float)
    e = np.exp(t - np.max(t))
    return e / np.sum(e)


def _folded_normal_mean(mu):
    """E[|Y|] where Y ~ N(mu, 1), i.e. the mean of a folded normal."""
    return mu * erf(mu / np.sqrt(2.0)) + np.sqrt(2.0 / np.pi) * np.exp(-mu**2 / 2.0)


# ---------------------------------------------------------------------------
# Problem 12: reinforce_exact
# ---------------------------------------------------------------------------

_R_12 = np.array([2.0, -1.0, 0.5, 1.5, -0.5])


def _ref_reinforce_exact(theta):
    """Exact gradient of E[r(Z)] w.r.t. logits theta, Z ~ Categorical(softmax(theta))."""
    t = np.asarray(theta, dtype=float)
    p = _softmax(t)
    grad = p * (_R_12 - np.dot(p, _R_12))
    return grad.tolist()


def _make_tc_12():
    cases = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.5, -0.5],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [3.0, -2.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 2.0, -2.0, 0.0],
    ]
    return [
        {"inputs": {"theta": c}, "expected": _ref_reinforce_exact(c)}
        for c in cases
    ]


P_REINFORCE_EXACT = Problem(
    id="reinforce_exact",
    category="gradient_estimation",
    difficulty=2,
    description="Exact REINFORCE gradient for a categorical reward.",
    prompt=(
        "Z is drawn from a Categorical distribution with probabilities softmax(theta) "
        "over K=5 states (0-indexed). The reward function is r(z) for the fixed vector "
        "r = [2.0, -1.0, 0.5, 1.5, -0.5].\n\n"
        "Compute the exact gradient dE[r(Z)]/dtheta (a vector of length 5). "
        "This is NOT a Monte Carlo estimate -- compute the exact expected gradient.\n\n"
        "```python\n"
        "def solve(theta: list) -> list:\n"
        '    """Return dE[r(Z)]/dtheta as a list of 5 floats."""\n'
        "```"
    ),
    reference=_ref_reinforce_exact,
    test_cases=_make_tc_12(),
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# Problem 13: gumbel_softmax_expected
# ---------------------------------------------------------------------------

_C_13 = np.array([1.0, 2.0, 3.0])


def _ref_gumbel_softmax_expected(theta, tau):
    """Gradient of E[L] w.r.t. logits, where L = c^T y and y ~ Gumbel-Softmax(theta, tau).

    The key identity: E[y_i] under Gumbel-Softmax equals softmax(theta)_i
    (independent of temperature tau), so dE[L]/dtheta = Jacobian_softmax^T @ c.
    """
    t = np.asarray(theta, dtype=float)
    p = _softmax(t)
    # Jacobian of softmax applied to c: diag(p) @ c - p * (p @ c)
    grad = p * (_C_13 - np.dot(p, _C_13))
    return grad.tolist()


def _make_tc_13():
    cases = [
        ([0.0, 0.0, 0.0], 0.5),
        ([1.0, -1.0, 0.0], 0.5),
        ([2.0, 0.0, -2.0], 1.0),
        ([0.5, 0.5, 0.5], 1.0),
        ([-1.0, 2.0, -0.5], 0.5),
        ([3.0, 1.0, -1.0], 1.0),
    ]
    return [
        {"inputs": {"theta": c[0], "tau": c[1]},
         "expected": _ref_gumbel_softmax_expected(c[0], c[1])}
        for c in cases
    ]


P_GUMBEL_SOFTMAX_EXPECTED = Problem(
    id="gumbel_softmax_expected",
    category="gradient_estimation",
    difficulty=2,
    description="Gradient of expected loss under Gumbel-Softmax relaxation.",
    prompt=(
        "The Gumbel-Softmax distribution generates a sample y in the K=3 simplex:\n"
        "  y_i = exp((theta_i + g_i)/tau) / sum_j exp((theta_j + g_j)/tau)\n"
        "where g_1, g_2, g_3 are i.i.d. Gumbel(0,1) random variables, theta are logits, "
        "and tau > 0 is the temperature.\n\n"
        "The loss is L = sum_i c_i * y_i with cost vector c = [1, 2, 3].\n\n"
        "Compute the exact gradient dE[L]/dtheta (a vector of length 3).\n\n"
        "```python\n"
        "def solve(theta: list, tau: float) -> list:\n"
        '    """Return dE[L]/dtheta as a list of 3 floats."""\n'
        "```"
    ),
    reference=_ref_gumbel_softmax_expected,
    test_cases=_make_tc_13(),
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# Problem 14: score_function_compound
# ---------------------------------------------------------------------------

_MU_14 = np.array([0.0, 2.0, -1.0])


def _ref_score_function_compound(theta):
    """Gradient of E[Y^2] w.r.t. theta in a compound stochastic program.

    X ~ Categorical(softmax(theta)) with K=3.
    Given X=k, Y ~ N(mu_k, 1).
    E[Y^2 | X=k] = mu_k^2 + 1.
    E[Y^2] = sum_k p_k * (mu_k^2 + 1).
    Gradient = softmax Jacobian^T applied to the vector (mu_k^2 + 1).
    """
    t = np.asarray(theta, dtype=float)
    p = _softmax(t)
    v = _MU_14**2 + 1.0  # [1, 5, 2]
    grad = p * (v - np.dot(p, v))
    return grad.tolist()


def _make_tc_14():
    cases = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [0.0, 2.0, -1.0],
        [-2.0, 1.0, 0.0],
        [1.5, 1.5, -1.0],
    ]
    return [
        {"inputs": {"theta": c}, "expected": _ref_score_function_compound(c)}
        for c in cases
    ]


P_SCORE_FUNCTION_COMPOUND = Problem(
    id="score_function_compound",
    category="gradient_estimation",
    difficulty=2,
    description="Score function gradient for a compound categorical-Gaussian program.",
    prompt=(
        "A two-stage stochastic program:\n"
        "  Stage 1: X ~ Categorical(softmax(theta)) with K=3 categories (0, 1, 2).\n"
        "  Stage 2: Given X=k, draw Y ~ Normal(mu_k, 1) where mu = [0, 2, -1].\n\n"
        "Compute the exact gradient dE[Y^2]/dtheta (a vector of length 3).\n\n"
        "```python\n"
        "def solve(theta: list) -> list:\n"
        '    """Return dE[Y^2]/dtheta as a list of 3 floats."""\n'
        "```"
    ),
    reference=_ref_score_function_compound,
    test_cases=_make_tc_14(),
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# Problem 15: rao_blackwell
# ---------------------------------------------------------------------------

_MU_15 = np.array([0.0, 2.0, -1.0])


def _ref_rao_blackwell(theta):
    """Gradient of E[|Y|] w.r.t. theta using Rao-Blackwellization.

    X ~ Categorical(softmax(theta)) with K=3.
    Given X=k, Y ~ N(mu_k, 1).
    E[|Y| | X=k] = mu_k * erf(mu_k/sqrt(2)) + sqrt(2/pi) * exp(-mu_k^2 / 2).
    E[|Y|] = sum_k p_k * E[|Y| | X=k].
    Gradient = softmax Jacobian^T applied to the vector of conditional means.
    """
    t = np.asarray(theta, dtype=float)
    p = _softmax(t)
    v = np.array([_folded_normal_mean(m) for m in _MU_15])
    grad = p * (v - np.dot(p, v))
    return grad.tolist()


def _make_tc_15():
    cases = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [0.0, 2.0, -1.0],
        [-1.0, 0.0, 1.0],
        [2.0, -2.0, 0.0],
        [0.5, 1.0, -0.5],
    ]
    return [
        {"inputs": {"theta": c}, "expected": _ref_rao_blackwell(c)}
        for c in cases
    ]


P_RAO_BLACKWELL = Problem(
    id="rao_blackwell",
    category="gradient_estimation",
    difficulty=3,
    description="Rao-Blackwellized gradient for nested expectation with absolute value.",
    prompt=(
        "A two-stage stochastic program:\n"
        "  Stage 1: X ~ Categorical(softmax(theta)) with K=3 categories (0, 1, 2).\n"
        "  Stage 2: Given X=k, draw Y ~ Normal(mu_k, 1) where mu = [0, 2, -1].\n\n"
        "Compute the exact gradient dE[|Y|]/dtheta (a vector of length 3), "
        "where |Y| denotes the absolute value of Y.\n\n"
        "```python\n"
        "def solve(theta: list) -> list:\n"
        '    """Return dE[|Y|]/dtheta as a list of 3 floats."""\n'
        "```"
    ),
    reference=_ref_rao_blackwell,
    test_cases=_make_tc_15(),
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# Problem 16: perturb_and_map
# ---------------------------------------------------------------------------

_ALPHA_16 = 0.3


def _ref_perturb_and_map(theta):
    """Gradient of V(theta) = max_z f(z, theta) where f has inter-component coupling.

    f(z, theta) = theta_z + alpha * sum_{j != z} theta_j^2
               = theta_z + alpha * (S - theta_z^2)
    where S = sum_j theta_j^2 and alpha = 0.3.

    V(theta) = max_z f(z, theta).
    At a non-degenerate optimum z*:
        dV/dtheta_i = df(z*, theta)/dtheta_i
                    = [i==z*] + alpha * (2*theta_i - 2*theta_{z*} * [i==z*])
                    = 2*alpha*theta_i + [i==z*] * (1 - 2*alpha*theta_{z*})
    """
    t = np.asarray(theta, dtype=float)
    K = len(t)
    alpha = _ALPHA_16

    # Compute f(z, theta) for each z
    S = np.sum(t**2)
    f_vals = t + alpha * (S - t**2)
    z_star = np.argmax(f_vals)

    # Gradient: df(z*, theta)/dtheta_i
    grad = 2.0 * alpha * t.copy()
    grad[z_star] += 1.0 - 2.0 * alpha * t[z_star]
    return grad.tolist()


def _make_tc_16():
    # Use cases with unique argmax to avoid subgradient issues
    cases = [
        [1.0, 0.0, -1.0, 0.5],
        [3.0, -1.0, 0.0, 2.0],
        [0.5, 1.5, -0.5, 0.0],
        [-1.0, 2.0, 1.0, -2.0],
        [0.1, 0.2, 0.3, 0.4],
        [-0.5, 3.0, -1.0, 1.0],
    ]
    return [
        {"inputs": {"theta": c}, "expected": _ref_perturb_and_map(c)}
        for c in cases
    ]


P_PERTURB_AND_MAP = Problem(
    id="perturb_and_map",
    category="gradient_estimation",
    difficulty=3,
    description="Gradient of a discrete optimization objective with inter-component coupling.",
    prompt=(
        "Consider a discrete optimization problem over z in {0, 1, 2, 3}:\n"
        "  f(z, theta) = theta_z + alpha * sum_{j != z} theta_j^2\n"
        "where alpha = 0.3 and theta is a real vector of length 4.\n\n"
        "The value function is V(theta) = max_z f(z, theta).\n\n"
        "Compute dV/dtheta (a vector of length 4). Assume the argmax is unique.\n\n"
        "```python\n"
        "def solve(theta: list) -> list:\n"
        '    """Return dV/dtheta as a list of 4 floats."""\n'
        "```"
    ),
    reference=_ref_perturb_and_map,
    test_cases=_make_tc_16(),
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# ALL problems in this module
# ---------------------------------------------------------------------------

ALL = [
    P_REINFORCE_EXACT,
    P_GUMBEL_SOFTMAX_EXPECTED,
    P_SCORE_FUNCTION_COMPOUND,
    P_RAO_BLACKWELL,
    P_PERTURB_AND_MAP,
]
