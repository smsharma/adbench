"""Problem dataclass definition."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Problem:
    """A single AD benchmark problem."""

    id: str
    category: str
    difficulty: int  # 1, 2, or 3
    description: str
    prompt: str  # The prompt sent to the LLM (includes function signature)
    reference: Callable  # Reference solution function
    test_cases: list[dict[str, Any]]  # Each dict has 'inputs' and 'expected'
    atol: float = 1e-6
    rtol: float = 1e-4

    def grade(self, fn: Callable) -> dict:
        """Grade a candidate function against reference on all test cases.

        Returns dict with 'score' (0.0, 0.5, or 1.0), 'details' per test case.
        """
        import numpy as np

        results = []
        for tc in self.test_cases:
            inputs = tc["inputs"]
            expected = tc["expected"]
            try:
                computed = fn(**inputs)
                # Handle dict/list/scalar comparison
                passed = _compare(computed, expected, self.atol, self.rtol)
                results.append({"inputs": inputs, "expected": expected, "computed": computed, "passed": passed, "error": None})
            except Exception as e:
                results.append({"inputs": inputs, "expected": expected, "computed": None, "passed": False, "error": str(e)})

        n_passed = sum(1 for r in results if r["passed"])
        n_total = len(results)

        if n_passed == n_total:
            score = 1.0
        elif n_passed >= n_total / 2:
            score = 0.5
        else:
            score = 0.0

        return {"score": score, "n_passed": n_passed, "n_total": n_total, "details": results}


def _compare(computed, expected, atol, rtol) -> bool:
    """Compare computed vs expected with tolerance."""
    import numpy as np

    if computed is None:
        return expected is None

    if isinstance(expected, dict):
        if not isinstance(computed, dict):
            return False
        return all(k in computed and _compare(computed[k], v, atol, rtol) for k, v in expected.items())

    if isinstance(expected, (list, tuple)):
        computed = np.asarray(computed, dtype=float)
        expected = np.asarray(expected, dtype=float)

    computed = np.asarray(computed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if computed.shape != expected.shape:
        return False

    return bool(np.all(np.isclose(computed, expected, atol=atol, rtol=rtol)))
