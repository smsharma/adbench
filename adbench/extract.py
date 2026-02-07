"""Extract executable Python code from LLM responses."""

import re


def extract_code(response: str) -> str | None:
    """Extract a Python function named 'solve' from the LLM response.

    Looks for code blocks (```python ... ```) first, then falls back
    to finding a bare 'def solve' block.
    """
    # Try fenced code blocks first
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
        if "def solve" in match:
            return match.strip()

    # If multiple code blocks, concatenate ones that have imports or def solve
    if matches:
        relevant = [m.strip() for m in matches if "import" in m or "def " in m]
        if relevant:
            combined = "\n\n".join(relevant)
            if "def solve" in combined:
                return combined

    # Fall back: find def solve in raw text
    lines = response.split("\n")
    in_function = False
    function_lines = []
    import_lines = []

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(stripped)
        elif stripped.startswith("def solve"):
            in_function = True
            function_lines.append(stripped)
        elif in_function:
            if stripped == "" or stripped[0] in (" ", "\t"):
                function_lines.append(stripped)
            elif stripped.startswith("def ") or stripped.startswith("class "):
                break
            else:
                # Could be end of function
                if function_lines and not stripped.startswith("#"):
                    break
                function_lines.append(stripped)

    if function_lines:
        code = "\n".join(import_lines + [""] + function_lines)
        return code.strip()

    return None
