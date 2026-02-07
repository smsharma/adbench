"""Central registry of all benchmark problems."""

from adbench.problems import (
    sorting_ranking,
    diff_algorithms,
    gradient_estimation,
    diff_simulation,
    implicit_layers,
    compositions,
)

_ALL_MODULES = [
    sorting_ranking,
    diff_algorithms,
    gradient_estimation,
    diff_simulation,
    implicit_layers,
    compositions,
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
