"""LLM API client for querying models."""

import os
import json
from typing import Any


SYSTEM_PROMPT = """You are an expert in differentiable programming, automatic differentiation, and scientific computing.
You will be given a problem involving a non-differentiable computation (sorting, ranking, discrete sampling,
combinatorial algorithms, simulation with contacts, etc.) that needs to be made differentiable.
Your task is to write a Python function called `solve` that computes the gradient through the
appropriate differentiable relaxation or gradient estimator.

Rules:
- Your function MUST be named `solve` with the exact signature specified.
- You may use: numpy, scipy (scipy.special, scipy.integrate, scipy.optimize, scipy.linalg), math.
- Return numerical values (float, list of floats, numpy arrays, or dicts as specified).
- Do NOT use JAX or PyTorch.
- Put your code in a ```python code block.
- Include all necessary imports inside or above the function.
- You may need to implement both the forward pass and backward pass of the differentiable relaxation."""


def query_openai(prompt: str, model: str = "gpt-4o", temperature: float = 0.0, **kwargs) -> str:
    """Query OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return response.choices[0].message.content


def query_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0, **kwargs) -> str:
    """Query Anthropic API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.content[0].text


def query_model(prompt: str, provider: str = "anthropic", model: str | None = None, **kwargs) -> str:
    """Query an LLM and return the raw response text."""
    if provider == "openai":
        return query_openai(prompt, model=model or "gpt-4o", **kwargs)
    elif provider == "anthropic":
        return query_anthropic(prompt, model=model or "claude-sonnet-4-20250514", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
