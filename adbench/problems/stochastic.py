"""Category 6: Stochastic Gradient Estimation — hard cases.

Score function estimators, pathwise gradients through non-trivial
distributions, and cases where standard reparameterization fails.
"""

import numpy as np
from scipy.stats import norm

from adbench.problem import Problem


# --- 6.1: Score function for discrete Bernoulli program ---
# f(p) = E[g(X)] where X ~ Bernoulli(p) and g(0) = a, g(1) = b.
# More complex: g involves a branching computation.
# f(p) = p * g(1) + (1-p) * g(0).
# Actually make it harder: X ~ Bernoulli(p), Y ~ Bernoulli(q(X)),
# where q(0) = 0.3, q(1) = 0.7. Compute dE[Y]/dp.
# E[Y] = p * 0.7 + (1-p) * 0.3 = 0.3 + 0.4p, so dE[Y]/dp = 0.4.
# But the model has to reason through the compound stochastic graph.

# Let's make it genuinely harder:
# X ~ Bernoulli(p), reward = X * (X + 1) + (1-X) * (-1) = X(X+1) - (1-X)
# = X² + X - 1 + X = X² + 2X - 1 (but X is 0 or 1)
# If X=1: reward = 1 + 2 - 1 = 2. If X=0: reward = 0 + 0 - 1 = -1.
# E[reward] = 2p + (-1)(1-p) = 3p - 1.
# d/dp = 3.

# Even harder: compound with E[f(X,Z)] where Z ~ N(X, 1) and f = Z².
# E[Z²|X=0] = 1, E[Z²|X=1] = 2. So E[Z²] = p*2 + (1-p)*1 = 1+p. d/dp = 1.

# Let's do something genuinely requiring insight:
# X ~ Geometric(p): P(X=k) = (1-p)^k p for k=0,1,2,...
# E[X] = (1-p)/p. Compute d/dp E[X²].
# E[X²] = (1-p)(2-p)/p². d/dp E[X²] = d/dp [(2-3p+p²)/p²]
# = d/dp [2/p² - 3/p + 1] = -4/p³ + 3/p² = (3p - 4)/p³

def _ref_stochastic_geometric(p: float) -> float:
    # E[X^2] for X ~ Geometric(p) (starting from 0): (1-p)(2-p)/p^2
    # d/dp = [(−1)(2−p)+(1−p)(−1)]p² − (1−p)(2−p)(2p)] / p⁴
    # Numerator: [−(2−p)−(1−p)]p² − 2p(1−p)(2−p)
    # = [−2+p−1+p]p² − 2p(1−p)(2−p)
    # = (2p−3)p² − 2p(2−3p+p²)
    # = 2p³ − 3p² − 4p + 6p² − 2p³
    # = 3p² − 4p = p(3p-4)
    # Then /p⁴ = (3p-4)/p³
    return (3*p - 4) / p**3

P_STOCHASTIC_GEOMETRIC = Problem(
    id="stochastic_geometric",
    category="stochastic",
    difficulty=2,
    description="Derivative of E[X²] w.r.t. p for X ~ Geometric(p).",
    prompt="""X ~ Geometric(p) with P(X=k) = (1-p)^k p for k = 0, 1, 2, ...

Compute d/dp E[X²].

```python
def solve(p: float) -> float:
    \"\"\"Return d/dp E[X^2] where X ~ Geometric(p).\"\"\"
```""",
    reference=_ref_stochastic_geometric,
    test_cases=[
        {"inputs": {"p": v}, "expected": _ref_stochastic_geometric(v)}
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
)


# --- 6.2: Gradient through max of Gaussians ---
# L(μ) = E[max(X₁, X₂)] where X₁ ~ N(μ, 1), X₂ ~ N(0, 1), independent.
# E[max(X₁,X₂)] = μΦ(μ/√2) + (1/√2)·φ(μ/√2)·√2 + ... (Mills ratio-like)
# Actually: for X ~ N(μ₁,σ²), Y ~ N(μ₂,σ²) indep:
# E[max(X,Y)] = μ₁Φ((μ₁-μ₂)/(σ√2)) + μ₂Φ((μ₂-μ₁)/(σ√2)) + (σ/√π)exp(-(μ₁-μ₂)²/(4σ²))
# With μ₁=μ, μ₂=0, σ=1:
# E[max] = μ Φ(μ/√2) + (1/√π) exp(-μ²/4)
# dE/dμ = Φ(μ/√2) + μ φ(μ/√2)/√2 + (1/√π)(-μ/2)exp(-μ²/4)
# = Φ(μ/√2) + (μ/(√(2π)·√2))exp(-μ²/4) - (μ/(2√π))exp(-μ²/4)
# = Φ(μ/√2) + (μ exp(-μ²/4))/(2√π) - (μ exp(-μ²/4))/(2√π)
# = Φ(μ/√2)

def _ref_stochastic_max_gaussian(mu: float) -> float:
    return norm.cdf(mu / np.sqrt(2))

P_STOCHASTIC_MAX_GAUSSIAN = Problem(
    id="stochastic_max_gaussian",
    category="stochastic",
    difficulty=3,
    description="Derivative of E[max(X₁,X₂)] where X₁~N(μ,1), X₂~N(0,1).",
    prompt="""X₁ ~ N(μ, 1) and X₂ ~ N(0, 1) are independent.

Compute d/dμ E[max(X₁, X₂)].

```python
def solve(mu: float) -> float:
    \"\"\"Return d/dmu E[max(X1, X2)] with X1 ~ N(mu,1), X2 ~ N(0,1).\"\"\"
```""",
    reference=_ref_stochastic_max_gaussian,
    test_cases=[
        {"inputs": {"mu": v}, "expected": _ref_stochastic_max_gaussian(v)}
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    ],
)


# --- 6.3: Gradient of log-partition function (Ising model) ---
# Z(θ) = Σ_{s∈{-1,+1}^3} exp(θ Σ_{i<j} s_i s_j)
# (3-spin fully-connected Ising model)
# log Z(θ) is the log-partition function.
# d/dθ log Z = E[Σ_{i<j} s_i s_j] = (1/Z) Σ_s (Σ_{i<j} s_i s_j) exp(θ Σ s_i s_j)

def _ref_stochastic_ising(theta: float) -> float:
    """d/dtheta log Z for 3-spin Ising model."""
    configs = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                configs.append((s1, s2, s3))

    energies = []
    interactions = []
    for s1, s2, s3 in configs:
        E = s1*s2 + s1*s3 + s2*s3  # sum over i<j
        energies.append(theta * E)
        interactions.append(E)

    energies = np.array(energies)
    interactions = np.array(interactions)
    # log-sum-exp for stability
    max_e = np.max(energies)
    log_Z = max_e + np.log(np.sum(np.exp(energies - max_e)))
    # d log Z / dtheta = E[interaction] = sum(interaction * exp(energy)) / Z
    weights = np.exp(energies - max_e)
    return np.sum(interactions * weights) / np.sum(weights)

P_STOCHASTIC_ISING = Problem(
    id="stochastic_ising_partition",
    category="stochastic",
    difficulty=3,
    description="Derivative of log-partition function for 3-spin Ising model.",
    prompt="""A 3-spin fully-connected Ising model has partition function:
  Z(θ) = Σ_{s∈{-1,+1}³} exp(θ Σ_{i<j} sᵢsⱼ)

where the sum over i<j includes pairs (1,2), (1,3), (2,3).

Compute d/dθ log Z(θ).

```python
def solve(theta: float) -> float:
    \"\"\"Return d/dtheta log Z(theta) for the 3-spin Ising model.\"\"\"
```""",
    reference=_ref_stochastic_ising,
    test_cases=[
        {"inputs": {"theta": v}, "expected": _ref_stochastic_ising(v)}
        for v in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
    ],
)


ALL = [P_STOCHASTIC_GEOMETRIC, P_STOCHASTIC_MAX_GAUSSIAN, P_STOCHASTIC_ISING]
