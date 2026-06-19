"""
Core pipeline for a single task execution.

Modes:
  - use_memory=True  → Failure-Aware Agent (proactive lookup + store on failure)
  - use_memory=False → Baseline Agent (no memory, plain retry)
"""
import time
from . import llm_client, executor, memory as mem


MAX_RETRIES = 3


def run_task(task: dict, use_memory: bool = True, verbose: bool = True) -> dict:
    task_id = task["task_id"]
    signature = task["function_signature"]
    description = task["description"]

    result = {
        "task_id": task_id,
        "mode": "failure_aware" if use_memory else "baseline",
        "attempts": [],
        "final_status": "failed",
        "first_attempt_passed": False,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"\n[{task_id}] Attempt {attempt}/{MAX_RETRIES} ({'memory ON' if use_memory else 'no memory'})")

        # --- Proactive Lookup (only in failure-aware mode) ---
        hints = []
        if use_memory:
            hints = mem.lookup_hints(task_id)
            if hints and verbose:
                print(f"  [Memory] {len(hints)} hint(s) injected:")
                for h in hints:
                    print(f"    • {h}")

        # --- Code Generation ---
        t0 = time.time()
        code = llm_client.generate_code(description, signature, hints if use_memory else None)
        gen_time = round(time.time() - t0, 2)

        if verbose:
            print(f"  [LLM] Code generated in {gen_time}s")

        # --- Test Execution ---
        test_result = executor.run_tests(code, task)

        attempt_record = {
            "attempt": attempt,
            "passed": test_result["passed"],
            "failure_detail": test_result["failure_detail"],
            "hints_used": hints,
            "gen_time_s": gen_time,
        }
        result["attempts"].append(attempt_record)

        if test_result["passed"]:
            result["final_status"] = "passed"
            if attempt == 1:
                result["first_attempt_passed"] = True
            if verbose:
                print(f"  [PASS] Task {task_id} passed on attempt {attempt}")
            break

        # --- Failure Path ---
        if verbose:
            print(f"  [FAIL] {test_result['failure_detail'][:120]}")

        if use_memory:
            analysis = llm_client.analyze_failure(
                description,
                signature,
                test_result["clean_code"],
                test_result["failure_detail"],
            )
            fail_id = mem.store_failure(
                task_id=task_id,
                error_category=analysis.get("error_category", "unknown"),
                failed_code=test_result["clean_code"],
                test_failure_detail=test_result["failure_detail"],
                root_cause=analysis.get("root_cause", ""),
                strategy_to_avoid=analysis.get("strategy_to_avoid", ""),
            )
            if verbose:
                print(f"  [Memory] Stored failure {fail_id}: [{analysis.get('error_category')}]")
        else:
            if verbose:
                print("  [Baseline] No memory stored, retrying without hints.")

    return result
