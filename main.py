"""
main.py — Run the Failure-Aware Agent on all tasks (single pass, memory ON).

Usage:
  python main.py
  python main.py --no-memory   # baseline mode
  python main.py --task two_sum  # single task
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from agent.pipeline import run_task

TASKS_PATH = os.path.join(os.path.dirname(__file__), "tasks", "tasks.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "results", "run_log.json")


def load_tasks(task_id: str = None) -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if task_id:
        tasks = [t for t in tasks if t["task_id"] == task_id]
    return tasks


def append_log(results: list[dict], mode: str) -> None:
    existing = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    entry = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "results": results,
    }
    existing.append(entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    first_pass = sum(1 for r in results if r["first_attempt_passed"])
    final_pass = sum(1 for r in results if r["final_status"] == "passed")
    avg_attempts = sum(len(r["attempts"]) for r in results) / total if total else 0

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Tasks run          : {total}")
    print(f"1st-attempt pass   : {first_pass}/{total} ({100*first_pass//total if total else 0}%)")
    print(f"Final pass (≤3 try): {final_pass}/{total} ({100*final_pass//total if total else 0}%)")
    print(f"Avg attempts       : {avg_attempts:.2f}")
    print("=" * 50)
    for r in results:
        status = "PASS" if r["final_status"] == "passed" else "FAIL"
        n = len(r["attempts"])
        print(f"  {r['task_id']:25s} [{status}] in {n} attempt(s)")


def main():
    parser = argparse.ArgumentParser(description="Failure-Aware Agent runner")
    parser.add_argument("--no-memory", action="store_true", help="Run in baseline mode (no memory)")
    parser.add_argument("--task", type=str, default=None, help="Run a single task by task_id")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    use_memory = not args.no_memory
    mode = "failure_aware" if use_memory else "baseline"

    tasks = load_tasks(args.task)
    if not tasks:
        print(f"No tasks found (filter: {args.task})")
        sys.exit(1)

    print(f"\nRunning {len(tasks)} task(s) in [{mode.upper()}] mode")
    print("-" * 50)

    all_results = []
    for task in tasks:
        result = run_task(task, use_memory=use_memory, verbose=not args.quiet)
        all_results.append(result)

    append_log(all_results, mode)
    print_summary(all_results)


if __name__ == "__main__":
    main()
