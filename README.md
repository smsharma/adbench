# AD-Bench

A benchmark for evaluating large language models on non-trivial automatic differentiation problems.

## Overview

AD-Bench tests whether LLMs can write correct code for computing derivatives in settings where standard backpropagation or textbook formulas are insufficient. Problems require genuine mathematical reasoning: implicit differentiation through fixed points, sensitivity analysis of ODEs and PDEs, Fréchet derivatives of matrix functions, stochastic gradient estimation, numerical stability, and more.

**50 problems** across **11 categories**, calibrated so that frontier models score 70–90%.

| Category | Problems | Description |
|---|---|---|
| Implicit Differentiation | 4 | Fixed points, Markov chains, Lyapunov equations |
| Integrals | 4 | Oscillatory, elliptic, singular, Laplace-type |
| Optimization | 4 | KKT, envelope theorem, bilevel, LP sensitivity |
| Matrix Calculus | 5 | Matrix sqrt/log/exp Fréchet derivatives, Cholesky, nuclear norm |
| ODE Sensitivity | 5 | Adjoint method, stiff systems, BVPs, limit cycles |
| Stochastic | 3 | Geometric distribution, max of Gaussians, Ising model |
| Higher-Order | 4 | Double backprop, Hessians, Taylor coefficients |
| Complex AD | 2 | Wirtinger calculus, phase retrieval |
| Special Functions | 3 | Incomplete gamma, polylogarithm, Hurwitz zeta |
| Numerical Traps | 4 | Log-sum-exp overflow, catastrophic cancellation, Fisher information |
| Brutal | 12 | Wasserstein, Lyapunov exponent, Stiefel manifold, Sinkhorn, Riccati, GP, DAE, Fredholm |

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

```bash
# Verify all reference solutions
python -m adbench.runner --dry-run

# List all problems
python -m adbench.runner --list

# Run against Anthropic (set ANTHROPIC_API_KEY)
python -m adbench.runner --provider anthropic --model claude-sonnet-4-20250514 -o results.json

# Run against OpenAI (set OPENAI_API_KEY)
python -m adbench.runner --provider openai --model gpt-4o -o results.json

# Filter by category or difficulty
python -m adbench.runner --category brutal matrix_calculus
python -m adbench.runner --level 3
```

## How It Works

1. Each problem has a **prompt** (math statement + function signature) and a **reference solution**
2. The LLM is asked to write a Python `solve()` function
3. Code is extracted from the response and executed in a sandboxed environment
4. Output is compared numerically to the reference at multiple test points
5. Scoring: **Pass** (1.0) if all test points match, **Partial** (0.5) if ≥ half match, **Fail** (0.0) otherwise

Default tolerances: `atol=1e-6`, `rtol=1e-4` (relaxed per-problem where appropriate).

## Design Principles

- **No hints**: Prompts state the mathematical problem, not the solution method
- **Traps**: Several problems have obvious-but-wrong approaches (e.g., naive finite differences on chaotic systems, unstable formulas near cancellation points)
- **Numerical verification**: All grading is automated — no LLM judge needed
- **Diverse difficulty**: Level 2 (combine known techniques) and Level 3 (research-level reasoning)
