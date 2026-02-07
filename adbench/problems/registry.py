"""Central registry of all benchmark problems."""

from adbench.problems import (
    implicit,
    integrals,
    optimization,
    matrix_calculus,
    ode_sensitivity,
    stochastic,
    higher_order,
    complex_ad,
    special_functions,
    numerical_traps,
    brutal,
)

_ALL_MODULES = [
    implicit,
    integrals,
    optimization,
    matrix_calculus,
    ode_sensitivity,
    stochastic,
    higher_order,
    complex_ad,
    special_functions,
    numerical_traps,
    brutal,
]

PROBLEMS: dict = {}
for _mod in _ALL_MODULES:
    for _p in _mod.ALL:
        PROBLEMS[_p.id] = _p


def get_problem(problem_id: str):
    return PROBLEMS[problem_id]


def get_problems_by_category(category: str):
    return [p for p in PROBLEMS.values() if p.category == category]


def get_problems_by_level(level: int):
    return [p for p in PROBLEMS.values() if p.difficulty == level]
