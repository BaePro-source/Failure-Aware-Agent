"""
run_experiment_repeated.py — N independent 3-round runs for statistical robustness.

Each independent run:
  - Resets Failure Memory (no cross-run contamination)
  - Runs 3 rounds: within a run, FA memory ACCUMULATES across rounds
  - Saves full results per run to repeated_experiment_results.json

Usage:
  python experiments/run_experiment_repeated.py --n 8 --rounds 3
"""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.pipeline import run_task
from agent import memory as mem

TASKS_PATH  = os.path.join(os.path.dirname(__file__), "..", "tasks", "tasks.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "repeated_experiment_results.json")


def load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _round_stats(task_results: list[dict]) -> dict:
    total = len(task_results)
    first_pass  = sum(1 for r in task_results if r["first_attempt_passed"])
    final_pass  = sum(1 for r in task_results if r["final_status"] == "passed")
    avg_attempts = sum(len(r["attempts"]) for r in task_results) / total if total else 0
    failed_tasks_1st = [r["task_id"] for r in task_results if not r["first_attempt_passed"]]
    return {
        "first_attempt_pass_count": first_pass,
        "first_attempt_pass_rate":  round(first_pass / total, 4) if total else 0.0,
        "final_pass_count":  final_pass,
        "final_pass_rate":   round(final_pass / total, 4) if total else 0.0,
        "avg_attempts":      round(avg_attempts, 3),
        "failed_tasks_1st":  failed_tasks_1st,
        "task_results":      task_results,
    }


def run_single_independent_run(tasks: list[dict], rounds: int,
                                run_id: int, total_runs: int) -> dict:
    """One fully independent experiment: reset memory, then run `rounds` rounds."""
    mem.clear_memory()

    run_data = {
        "run_id":    run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rounds":    [],
    }

    for rnd in range(1, rounds + 1):
        round_entry = {"round": rnd}

        # ── Baseline ─────────────────────────────────────────────
        baseline_results = []
        for task in tasks:
            r = run_task(task, use_memory=False, verbose=False)
            baseline_results.append(r)
        round_entry["baseline"] = _round_stats(baseline_results)

        # ── Failure-Aware ─────────────────────────────────────────
        fa_results = []
        for task in tasks:
            r = run_task(task, use_memory=True, verbose=False)
            fa_results.append(r)
        round_entry["failure_aware"] = _round_stats(fa_results)

        run_data["rounds"].append(round_entry)

        b = round_entry["baseline"]
        f = round_entry["failure_aware"]
        b_fail = b["failed_tasks_1st"] or ["–"]
        f_fail = f["failed_tasks_1st"] or ["–"]
        print(
            f"  [Run {run_id:2d}/{total_runs}] Round {rnd}/{rounds} │ "
            f"Baseline 1st: {b['first_attempt_pass_count']}/{len(tasks)} "
            f"(fail: {', '.join(b_fail)}) │ "
            f"FA 1st: {f['first_attempt_pass_count']}/{len(tasks)} "
            f"(fail: {', '.join(f_fail)})"
        )

    return run_data


def print_live_summary(all_runs: list[dict], rounds: int) -> None:
    """Print a compact running tally after each completed run."""
    import statistics as st

    print()
    print(f"  {'Round':<6} {'B 1st% (mean±σ)':<22} {'FA 1st% (mean±σ)':<22} {'Δ FA–B'}")
    print(f"  {'─'*6} {'─'*22} {'─'*22} {'─'*8}")

    for rnd in range(1, rounds + 1):
        b_rates  = [r["rounds"][rnd-1]["baseline"]["first_attempt_pass_rate"]     * 100 for r in all_runs if len(r["rounds"]) >= rnd]
        fa_rates = [r["rounds"][rnd-1]["failure_aware"]["first_attempt_pass_rate"] * 100 for r in all_runs if len(r["rounds"]) >= rnd]
        if not b_rates:
            continue
        b_mean, b_std   = st.mean(b_rates),  (st.stdev(b_rates)  if len(b_rates)  > 1 else 0.0)
        fa_mean, fa_std = st.mean(fa_rates), (st.stdev(fa_rates) if len(fa_rates) > 1 else 0.0)
        delta = fa_mean - b_mean
        print(f"  {rnd:<6} {b_mean:5.1f} ± {b_std:4.1f}%          "
              f"{fa_mean:5.1f} ± {fa_std:4.1f}%          "
              f"{delta:+.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Repeated independent experiment (N runs × R rounds)")
    parser.add_argument("--n",      type=int, default=8, help="Number of independent runs (default: 8)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per run (default: 3)")
    args = parser.parse_args()

    tasks = load_tasks()
    total_tasks = len(tasks)
    all_runs: list[dict] = []

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    # Start fresh
    with open(RESULTS_PATH, "w") as f:
        json.dump([], f)

    print(f"\nStarting N={args.n} independent runs × {args.rounds} rounds × {total_tasks} tasks")
    print(f"Model: qwen2.5-coder:latest  |  temperature: 1.2\n")
    print("=" * 90)

    for run_id in range(1, args.n + 1):
        print(f"\n▶ Run {run_id}/{args.n}")
        try:
            run_data = run_single_independent_run(tasks, args.rounds, run_id, args.n)
            all_runs.append(run_data)
        except Exception as exc:
            print(f"  [Run {run_id}] FAILED — skipping. Error: {exc}")
            traceback.print_exc()
            continue

        # Persist after every run so partial results survive a crash
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, ensure_ascii=False, indent=2)

        # Running tally
        print_live_summary(all_runs, args.rounds)

    print("=" * 90)
    print(f"Done. {len(all_runs)}/{args.n} runs completed.")
    print(f"Results saved to: {RESULTS_PATH}")
    print("\nRun `python experiments/analyze_results.py` for full statistics.\n")


if __name__ == "__main__":
    main()
