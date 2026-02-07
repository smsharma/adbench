"""Execute extracted code in a restricted environment."""

import signal
import threading
import traceback
from typing import Any, Callable


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def _is_main_thread():
    return threading.current_thread() is threading.main_thread()


def execute_code(code: str, timeout: int = 60) -> Callable | None:
    """Execute code string and return the 'solve' function, or None on failure.

    Returns a tuple (solve_fn, error_msg). If execution succeeds, error_msg is None.
    Uses signal-based timeout on main thread, thread-based timeout otherwise.
    """
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

    if _is_main_thread():
        # Use signal-based timeout (only works in main thread)
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
    else:
        # Thread-based timeout for worker threads
        result_box = [None, None]  # [solve_fn, error_msg]
        exc_box = [None]

        def _run():
            try:
                exec(code, namespace)
                if "solve" not in namespace:
                    result_box[1] = "No 'solve' function defined in code"
                else:
                    result_box[0] = namespace["solve"]
            except Exception as e:
                result_box[1] = f"Code execution error: {traceback.format_exc()}"

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            # Thread still running — timeout
            return None, "Code execution timed out"

        return result_box[0], result_box[1]
