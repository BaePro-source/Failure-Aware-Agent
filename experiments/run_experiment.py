"""
run_experiment.py — Baseline vs Failure-Aware comparison over multiple rounds.

Each round runs all tasks in both modes.
Failure-Aware mode accumulates memory across rounds; Baseline always starts fresh.

Usage:
  python experiments/run_experiment.py
  python experiments/run_experiment.py --rounds 3
  python experiments/run_experiment.py --rounds 3 --reset
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.pipeline import run_task
from agent import memory as mem

TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_results.json")


def load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_round(tasks: list[dict], use_memory: bool, round_num: int) -> dict:
    mode = "failure_aware" if use_memory else "baseline"
    print(f"\n{'='*60}")
    print(f"Round {round_num} — {mode.upper()}")
    print(f"{'='*60}")

    results = []
    for task in tasks:
        r = run_task(task, use_memory=use_memory, verbose=True)
        results.append(r)

    total = len(results)
    first_pass = sum(1 for r in results if r["first_attempt_passed"])
    final_pass = sum(1 for r in results if r["final_status"] == "passed")
    avg_attempts = sum(len(r["attempts"]) for r in results) / total if total else 0

    summary = {
        "round": round_num,
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tasks": total,
        "first_attempt_pass_count": first_pass,
        "first_attempt_pass_rate": round(first_pass / total, 3) if total else 0,
        "final_pass_count": final_pass,
        "final_pass_rate": round(final_pass / total, 3) if total else 0,
        "avg_attempts": round(avg_attempts, 2),
        "task_results": results,
    }
    return summary


def print_comparison_table(experiment_data: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS — Round-by-Round Comparison")
    print("=" * 70)
    print(f"{'Round':<8} {'Baseline 1st%':<18} {'FA 1st%':<18} {'Baseline Final%':<18} {'FA Final%'}")
    print("-" * 70)

    rounds = sorted(set(d["round"] for d in experiment_data))
    for rnd in rounds:
        baseline = next((d for d in experiment_data if d["round"] == rnd and d["mode"] == "baseline"), None)
        fa = next((d for d in experiment_data if d["round"] == rnd and d["mode"] == "failure_aware"), None)

        b1 = f"{baseline['first_attempt_pass_rate']*100:.0f}%" if baseline else "N/A"
        f1 = f"{fa['first_attempt_pass_rate']*100:.0f}%" if fa else "N/A"
        bf = f"{baseline['final_pass_rate']*100:.0f}%" if baseline else "N/A"
        ff = f"{fa['final_pass_rate']*100:.0f}%" if fa else "N/A"

        print(f"{rnd:<8} {b1:<18} {f1:<18} {bf:<18} {ff}")

    print("=" * 70)
    print("\nKey insight: Failure-Aware 1st% should increase across rounds as")
    print("memory accumulates. Baseline 1st% stays flat (no learning).")


def main():
    parser = argparse.ArgumentParser(description="Run baseline vs failure-aware experiment")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--reset", action="store_true", help="Clear failure memory before starting")
    args = parser.parse_args()

    if args.reset:
        mem.clear_memory()
        # Also wipe previous experiment results so the table shows only this run
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "w") as f:
                json.dump([], f)
        print("[Reset] Failure memory and experiment results cleared.")

    tasks = load_tasks()
    all_summaries = []

    # Load existing results to append (only if not reset)
    if not args.reset and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_summaries = json.load(f)

    for rnd in range(1, args.rounds + 1):
        # Baseline: always runs without memory (no clear needed, memory not used)
        baseline_summary = run_round(tasks, use_memory=False, round_num=rnd)
        all_summaries.append(baseline_summary)

        # Failure-Aware: uses and accumulates memory across rounds
        fa_summary = run_round(tasks, use_memory=True, round_num=rnd)
        all_summaries.append(fa_summary)

        # Save after each round in case of interruption
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)

        print(f"\n[Round {rnd} complete] Results saved to {RESULTS_PATH}")

    print_comparison_table(all_summaries)


if __name__ == "__main__":
    main()
