# AD-Bench: Differentiable Almost Everything

A benchmark for evaluating LLMs on making non-differentiable computations differentiable — sorting, ranking, dynamic programming, discrete sampling, physics simulation, and implicit layers.

Inspired by the [ICML 2024 "Differentiable Almost Everything" workshop](https://differentiable.xyz): this benchmark tests cases where **vanilla automatic differentiation fails or does not yield meaningful gradients**, requiring relaxations, custom gradient estimators, or implicit differentiation.

## Overview

**30 problems** across **6 categories**. The model must write Python code that computes gradients through differentiable relaxations of non-differentiable operations.

| Category | Problems | Examples |
|---|---|---|
| Sorting & Ranking | 5 | Sinkhorn sort, pairwise ranking, soft top-k, soft quantile, soft NMS |
| Differentiable Algorithms | 6 | Soft-DTW, soft edit distance, soft Bellman-Ford, Needleman-Wunsch, MST (matrix-tree theorem), CTC loss |
| Gradient Estimation | 5 | REINFORCE, Gumbel-Softmax, score function, Rao-Blackwell, perturb-and-MAP |
| Differentiable Simulation | 4 | 1D rendering with occlusion, elastic collision, spring-contact, sphere tracing (SDF) |
| Implicit Layers | 5 | Deep equilibrium model, OptNet QP layer, Sinkhorn implicit, fixed-point IFT, power iteration backward |
| Advanced Compositions | 5 | Smooth branching, soft k-means, Earth mover's distance, Hausdorff distance, temperature-scaled attention |

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
python -m adbench.runner --provider anthropic --model claude-sonnet-4-5-20250929 -o results.json

# Run with parallelism (5 concurrent API calls)
python -m adbench.runner --provider anthropic --model claude-opus-4-6 -j 5 -o results.json

# Run against OpenAI (set OPENAI_API_KEY)
python -m adbench.runner --provider openai --model gpt-4o -o results.json

# Filter by category
python -m adbench.runner --category diff_algorithms implicit_layers
```

## How It Works

1. Each problem has a **prompt** describing a non-differentiable computation and asking for the gradient through a differentiable relaxation
2. The LLM writes a Python `solve()` function
3. Code is extracted and executed in a sandboxed environment (numpy/scipy available, no JAX/PyTorch)
4. Output is compared numerically to reference values at multiple test points
5. Scoring: **Pass** (1.0) all test points match, **Partial** (0.5) >= half match, **Fail** (0.0)

## Design Principles

- **No hints** about which relaxation or technique to use
- **Genuine differentiable programming**: sorting via Sinkhorn, DP with soft-min, implicit differentiation through fixed points — not textbook calculus
- **Numerical verification**: all grading is automated, no LLM judge
- **Prompts describe the forward computation** — the model must figure out how to differentiate through it
