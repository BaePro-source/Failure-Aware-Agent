import subprocess
import sys
import json
import re


def _build_test_script(code: str, task: dict) -> str:
    """Build a self-contained Python script that runs all test cases and exits 0 on pass."""
    function_name = _extract_function_name(task["function_signature"])
    test_blocks = []

    for i, tc in enumerate(task["test_cases"]):
        args = ", ".join(repr(v) for v in tc["input"].values())
        expected = repr(tc["expected"])
        test_blocks.append(
            f"try:\n"
            f"    _result = {function_name}({args})\n"
            f"    _expected = {expected}\n"
            f"    assert _result == _expected, "
            f"\"Test {i+1} FAILED: expected \" + repr(_expected) + \", got \" + repr(_result)\n"
            f"except AssertionError as _e:\n"
            f"    _failures.append(str(_e))\n"
            f"except Exception as _e:\n"
            f"    _failures.append(\"Test {i+1} RUNTIME ERROR: \" + type(_e).__name__ + \": \" + str(_e))"
        )

    test_section = "\n\n".join(test_blocks)

    # Build script with no leading indentation — avoid textwrap.dedent + f-string multiline pitfall
    parts = [
        "import sys",
        "",
        code,
        "",
        "_failures = []",
        "",
        test_section,
        "",
        "if _failures:",
        "    for _f in _failures:",
        "        print(_f, file=sys.stderr)",
        "    sys.exit(1)",
        "else:",
        "    print('ALL_PASS')",
        "    sys.exit(0)",
    ]
    return "\n".join(parts)


def _extract_function_name(signature: str) -> str:
    match = re.search(r"def (\w+)\(", signature)
    return match.group(1) if match else "solution"


def run_tests(code: str, task: dict, timeout: int = 10) -> dict:
    """
    Execute generated code against all test cases.
    Returns: {"passed": bool, "failure_detail": str, "clean_code": str}
    """
    clean_code = _clean_code(code)
    script = _build_test_script(clean_code, task)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "failure_detail": f"Execution timed out after {timeout}s (possible infinite loop)",
            "clean_code": clean_code,
        }

    if result.returncode == 0:
        return {"passed": True, "failure_detail": "", "clean_code": clean_code}

    stderr = result.stderr.strip() or result.stdout.strip()
    return {
        "passed": False,
        "failure_detail": stderr[:500] if stderr else "Unknown error (non-zero exit)",
        "clean_code": clean_code,
    }


def _clean_code(code: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    code = code.strip()
    # Remove ```python ... ``` or ``` ... ``` fences
    if code.startswith("```"):
        lines = code.split("\n")
        # Drop first line (```python or ```) and last ``` line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        code = "\n".join(inner)
    return code.strip()
