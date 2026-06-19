"""
analyze_results.py — Statistical analysis of repeated_experiment_results.json.

Outputs:
  1. Per-round mean ± std table (Baseline vs FA)
  2. Per-task failure frequency and cross-task rescue rate
  3. results/summary_chart.png  (matplotlib if available, else ASCII fallback)

Usage:
  python experiments/analyze_results.py
  python experiments/analyze_results.py --no-chart
"""
import argparse
import json
import os
import statistics
import sys

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "repeated_experiment_results.json")
CHART_PATH   = os.path.join(os.path.dirname(__file__), "..", "results", "summary_chart.png")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Per-round statistics
# ──────────────────────────────────────────────────────────────────────────────

def per_round_stats(runs: list[dict]) -> dict:
    """Return {round: {baseline: {mean, std, rates[]}, failure_aware: {...}}}"""
    rounds_data: dict = {}
    for run in runs:
        for rd in run["rounds"]:
            rnd = rd["round"]
            if rnd not in rounds_data:
                rounds_data[rnd] = {
                    "baseline":      {"1st": [], "final": []},
                    "failure_aware": {"1st": [], "final": []},
                }
            rounds_data[rnd]["baseline"]["1st"].append(rd["baseline"]["first_attempt_pass_rate"])
            rounds_data[rnd]["baseline"]["final"].append(rd["baseline"]["final_pass_rate"])
            rounds_data[rnd]["failure_aware"]["1st"].append(rd["failure_aware"]["first_attempt_pass_rate"])
            rounds_data[rnd]["failure_aware"]["final"].append(rd["failure_aware"]["final_pass_rate"])
    return rounds_data


def _fmt(rates: list[float]) -> str:
    m = statistics.mean(rates) * 100
    s = (statistics.stdev(rates) * 100) if len(rates) > 1 else 0.0
    return f"{m:5.1f} ± {s:4.1f}%"


def print_round_table(rounds_data: dict, n_runs: int) -> None:
    print(f"\n{'='*80}")
    print(f"PER-ROUND STATISTICS  (N={n_runs} independent runs)")
    print(f"{'='*80}")
    header = f"{'Round':<7} {'Baseline 1st% (mean±σ)':<26} {'FA 1st% (mean±σ)':<26} {'Δ(FA−B)':<10} {'B Final%':<14} {'FA Final%'}"
    print(header)
    print("─" * len(header))
    for rnd in sorted(rounds_data.keys()):
        rd = rounds_data[rnd]
        b1  = rd["baseline"]["1st"]
        fa1 = rd["failure_aware"]["1st"]
        bf  = rd["baseline"]["final"]
        ff  = rd["failure_aware"]["final"]
        delta = (statistics.mean(fa1) - statistics.mean(b1)) * 100
        print(
            f"{rnd:<7} {_fmt(b1):<26} {_fmt(fa1):<26} "
            f"{delta:+.1f}%{'':5} {statistics.mean(bf)*100:5.1f}%{'':8} {statistics.mean(ff)*100:5.1f}%"
        )
    print("─" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Per-task failure analysis & cross-task rescue rate
# ──────────────────────────────────────────────────────────────────────────────

def per_task_analysis(runs: list[dict]) -> dict:
    """
    For each (run, round, task):
      baseline_fail1: baseline failed attempt 1
      fa_fail1:       FA failed attempt 1
      rescue:         baseline failed attempt 1 AND FA succeeded attempt 1
    """
    task_stats: dict = {}

    for run in runs:
        for rd in run["rounds"]:
            b_tasks  = {r["task_id"]: r for r in rd["baseline"]["task_results"]}
            fa_tasks = {r["task_id"]: r for r in rd["failure_aware"]["task_results"]}

            for tid, b_r in b_tasks.items():
                if tid not in task_stats:
                    task_stats[tid] = {
                        "total":        0,
                        "b_fail1":      0,
                        "fa_fail1":     0,
                        "rescue":       0,   # b failed, FA succeeded on attempt 1
                    }
                s = task_stats[tid]
                s["total"] += 1

                b_failed  = not b_r["first_attempt_passed"]
                fa_passed = fa_tasks.get(tid, {}).get("first_attempt_passed", True)

                if b_failed:
                    s["b_fail1"] += 1
                    if fa_passed:
                        s["rescue"] += 1
                if not fa_tasks.get(tid, {}).get("first_attempt_passed", True):
                    s["fa_fail1"] += 1

    return task_stats


def print_task_table(task_stats: dict) -> float:
    print(f"\n{'='*80}")
    print("PER-TASK FAILURE & RESCUE ANALYSIS")
    print(f"{'='*80}")
    header = f"{'Task':<26} {'B fail1':<10} {'FA fail1':<10} {'Rescues':<10} {'Rescue rate'}"
    print(header)
    print("─" * len(header))

    total_b_fail = 0
    total_rescue = 0

    for tid, s in sorted(task_stats.items(), key=lambda x: -x[1]["b_fail1"]):
        rr = s["rescue"] / s["b_fail1"] if s["b_fail1"] else 0.0
        b_rate  = s["b_fail1"]  / s["total"] * 100
        fa_rate = s["fa_fail1"] / s["total"] * 100
        print(
            f"{tid:<26} "
            f"{s['b_fail1']:>3}/{s['total']} ({b_rate:4.1f}%)  "
            f"{s['fa_fail1']:>3}/{s['total']} ({fa_rate:4.1f}%)  "
            f"{s['rescue']:>3}/{s['b_fail1'] or 1}  "
            f"({rr*100:.1f}%)"
        )
        total_b_fail += s["b_fail1"]
        total_rescue += s["rescue"]

    overall_rr = total_rescue / total_b_fail if total_b_fail else 0.0
    print("─" * len(header))
    print(f"{'OVERALL':<26} {'B fail1':>10}  {'FA fail1':>10}  {'Rescues':>10}  {overall_rr*100:.1f}%")
    print(f"\n  Total Baseline 1st-attempt failures : {total_b_fail}")
    print(f"  Cases where FA succeeded (rescues)  : {total_rescue}")
    print(f"  Overall cross-task rescue rate       : {overall_rr*100:.1f}%")
    return overall_rr


# ──────────────────────────────────────────────────────────────────────────────
# 3. Chart
# ──────────────────────────────────────────────────────────────────────────────

def ascii_chart(rounds_data: dict) -> None:
    print(f"\n{'='*60}")
    print("ASCII CHART — 1st-Attempt Pass Rate by Round")
    print(f"{'='*60}")
    for rnd in sorted(rounds_data.keys()):
        rd = rounds_data[rnd]
        b  = statistics.mean(rd["baseline"]["1st"]) * 100
        fa = statistics.mean(rd["failure_aware"]["1st"]) * 100
        b_bar  = "█" * int(b  / 5)
        fa_bar = "█" * int(fa / 5)
        print(f"  R{rnd} Baseline │ {b_bar:<20} {b:5.1f}%")
        print(f"  R{rnd} FA       │ {fa_bar:<20} {fa:5.1f}%")
        print()


def save_chart(rounds_data: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        rounds = sorted(rounds_data.keys())
        b_means  = [statistics.mean(rounds_data[r]["baseline"]["1st"])      * 100 for r in rounds]
        fa_means = [statistics.mean(rounds_data[r]["failure_aware"]["1st"]) * 100 for r in rounds]
        b_stds   = [(statistics.stdev(rounds_data[r]["baseline"]["1st"])      * 100 if len(rounds_data[r]["baseline"]["1st"]) > 1 else 0) for r in rounds]
        fa_stds  = [(statistics.stdev(rounds_data[r]["failure_aware"]["1st"]) * 100 if len(rounds_data[r]["failure_aware"]["1st"]) > 1 else 0) for r in rounds]

        x = np.arange(len(rounds))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars_b  = ax.bar(x - width/2, b_means,  width, label="Baseline",      color="#e74c3c", alpha=0.85, yerr=b_stds,  capsize=5)
        bars_fa = ax.bar(x + width/2, fa_means, width, label="Failure-Aware", color="#2ecc71", alpha=0.85, yerr=fa_stds, capsize=5)

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("1st-Attempt Pass Rate (%)", fontsize=12)
        ax.set_title("Baseline vs Failure-Aware Agent\n1st-Attempt Pass Rate (mean ± σ, N=8 runs)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Round {r}" for r in rounds])
        ax.set_ylim(0, 115)
        ax.legend(fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)

        for bar in bars_b:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
        for bar in bars_fa:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
        plt.savefig(CHART_PATH, dpi=150)
        plt.close()
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(RESULTS_PATH):
        print(f"Results file not found: {RESULTS_PATH}")
        print("Run `python experiments/run_experiment_repeated.py --n 8 --rounds 3` first.")
        sys.exit(1)

    runs = load_results()
    if not runs:
        print("No completed runs found in results file.")
        sys.exit(1)

    n = len(runs)
    print(f"\nLoaded {n} completed run(s) from {RESULTS_PATH}")

    rounds_data = per_round_stats(runs)
    print_round_table(rounds_data, n)

    task_stats = per_task_analysis(runs)
    rescue_rate = print_task_table(task_stats)

    if not args.no_chart:
        saved = save_chart(rounds_data)
        if saved:
            print(f"\n  Chart saved to: {CHART_PATH}")
        else:
            ascii_chart(rounds_data)
            print("  (matplotlib not available — ASCII chart shown above)")

    # Print markdown table for README copy-paste
    print(f"\n{'='*80}")
    print("MARKDOWN TABLE (for README)")
    print(f"{'='*80}")
    print()
    print("| Round | Baseline 1st% (mean±σ) | FA 1st% (mean±σ) | Δ(FA−B) | B Final% | FA Final% |")
    print("|-------|------------------------|------------------|---------|----------|-----------|")
    for rnd in sorted(rounds_data.keys()):
        rd = rounds_data[rnd]
        b1  = rd["baseline"]["1st"]
        fa1 = rd["failure_aware"]["1st"]
        bf  = rd["baseline"]["final"]
        ff  = rd["failure_aware"]["final"]
        bm,bs   = statistics.mean(b1)*100,  (statistics.stdev(b1)*100  if len(b1)>1  else 0)
        fm,fs   = statistics.mean(fa1)*100, (statistics.stdev(fa1)*100 if len(fa1)>1 else 0)
        delta = fm - bm
        print(f"| {rnd}     | {bm:.1f} ± {bs:.1f}%             | {fm:.1f} ± {fs:.1f}%          | {delta:+.1f}%   | "
              f"{statistics.mean(bf)*100:.1f}%     | {statistics.mean(ff)*100:.1f}%      |")

    print(f"\n  Overall cross-task rescue rate: {rescue_rate*100:.1f}%")
    print(f"  (of all Baseline 1st-attempt failures, FA succeeded on {rescue_rate*100:.1f}% of them)")


if __name__ == "__main__":
    main()
