"""LLM API client for querying models."""

import os
import json
from typing import Any


SYSTEM_PROMPT = """You are an expert in automatic differentiation, calculus, and scientific computing.
You will be given a mathematical differentiation problem. Your task is to write a Python function
called `solve` that computes the requested derivative.

Rules:
- Your function MUST be named `solve` with the exact signature specified.
- You may use: numpy, scipy (scipy.special, scipy.integrate, scipy.optimize, scipy.linalg), math.
- Return numerical values (float, list of floats, numpy arrays, or dicts as specified).
- Do NOT use JAX or PyTorch unless the problem specifically allows it.
- Put your code in a ```python code block.
- Include all necessary imports inside or above the function."""


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
