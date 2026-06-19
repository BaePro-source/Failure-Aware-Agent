import json
import os
from datetime import datetime, timezone

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "memory_store", "failures.json")


def _load() -> list[dict]:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(records: list[dict]) -> None:
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def store_failure(task_id: str, error_category: str, failed_code: str,
                  test_failure_detail: str, root_cause: str, strategy_to_avoid: str) -> str:
    records = _load()
    fail_id = f"fail_{len(records) + 1:04d}"
    record = {
        "id": fail_id,
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_category": error_category,
        "failed_code_snippet": failed_code[:300],
        "test_failure_detail": test_failure_detail[:300],
        "root_cause": root_cause,
        "strategy_to_avoid": strategy_to_avoid,
    }
    records.append(record)
    _save(records)
    return fail_id


def lookup_hints(task_id: str, max_hints: int = 3) -> list[str]:
    """
    Proactive lookup before task execution.

    Priority order:
      1. Same task_id (direct past failures)
      2. Different task_id but matching error_category keywords (cross-task transfer)

    Returns a list of hint strings: "[error_category]: strategy_to_avoid"
    """
    records = _load()
    if not records:
        return []

    # --- 1st pass: same task ---
    same_task = [r for r in records if r["task_id"] == task_id]

    # --- 2nd pass: cross-task by error_category keyword overlap ---
    # Only include cross-task hints whose error_category contains a shared keyword.
    # This ensures cross-task transfer is semantically relevant, not just "everything unique".
    seen_categories = {r["error_category"].lower() for r in same_task}

    cross_task = []
    for r in records:
        if r["task_id"] == task_id:
            continue
        cat = r["error_category"].lower()
        if cat in seen_categories:
            continue
        # Only transfer if the error category contains a known shared keyword
        if _has_shared_keyword(cat):
            cross_task.append(r)
            seen_categories.add(cat)

    # Merge: same-task first, then cross-task, deduplicate by strategy text
    combined = same_task + cross_task
    seen_strategies = set()
    hints = []
    for r in combined:
        key = r["strategy_to_avoid"][:80]
        if key not in seen_strategies:
            hints.append(f"[{r['error_category']}]: {r['strategy_to_avoid']}")
            seen_strategies.add(key)
        if len(hints) >= max_hints:
            break

    return hints


def _has_shared_keyword(error_category: str) -> bool:
    """
    Return True if error_category contains a specific, transferable error pattern.

    Only multi-word or highly specific phrases are used — single generic words like
    "index", "empty", "loop", "edge" are excluded to prevent spurious cross-task
    transfer between algorithmically unrelated problems.

    Genuine cross-task patterns:
      off-by-one        → binary_search, climbing_stairs, remove_duplicates, fibonacci
      boundary condition → binary_search, climbing_stairs
      base case         → fibonacci_memo, climbing_stairs
      stack underflow   → valid_parentheses (stack-based problems)
      in-place          → remove_duplicates
      initialization error → max_subarray
      recursion error   → fibonacci_memo
    """
    SHARED_KEYWORDS = {
        "off-by-one",
        "off by one",
        "boundary condition",
        "base case",
        "stack underflow",
        "in-place",
        "in place",
        "initialization error",
        "recursion error",
    }
    cat_lower = error_category.lower()
    return any(kw in cat_lower for kw in SHARED_KEYWORDS)


def get_all_failures() -> list[dict]:
    return _load()


def clear_memory() -> None:
    _save([])
