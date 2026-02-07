"""Execute extracted code in a restricted environment."""

import signal
import traceback
from typing import Any, Callable


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def execute_code(code: str, timeout: int = 30) -> Callable | None:
    """Execute code string and return the 'solve' function, or None on failure.

    Returns a tuple (solve_fn, error_msg). If execution succeeds, error_msg is None.
    """
    # Allowed modules for the sandbox
    import numpy as np
    import scipy
    import scipy.special
    import scipy.integrate
    import scipy.optimize
    import scipy.linalg
    import math

    namespace = {
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "math": math,
        "__builtins__": __builtins__,
    }

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        exec(code, namespace)
        signal.alarm(0)

        if "solve" not in namespace:
            return None, "No 'solve' function defined in code"

        return namespace["solve"], None

    except TimeoutError:
        return None, "Code execution timed out"
    except Exception as e:
        return None, f"Code execution error: {traceback.format_exc()}"
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
