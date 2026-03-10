"""Cross-system comparison: load two result sets, normalize, and generate outputs."""

import argparse
import json
from pathlib import Path

from common.normalize import normalize_results, perf_per_dollar, perf_per_watt, energy_per_task
from common.utils import load_json, save_json


# Mapping of benchmark file -> (metric_key, metric_label)
BENCHMARK_METRICS = {
    "llm": [("tokens_per_sec", "LLM Tokens/sec")],
    "vision": [("fps", "Vision FPS")],
    "gemm": [("tflops_avg", "GEMM TFLOPS")],
    "bandwidth": [("gb_per_sec", "Bandwidth GB/s")],
    "pipeline": [("inferences_per_sec", "Pipeline Inferences/sec")],
}


def load_system_results(results_dir: str) -> dict:
    """Load all benchmark JSON files from a results directory."""
    results = {}
    config = {}
    for json_file in Path(results_dir).glob("*.json"):
        data = load_json(str(json_file))
        results[json_file.stem] = data
        if not config and "config" in data:
            config = data["config"]
    return results, config


def compare_systems(dir_a: str, dir_b: str) -> dict:
    """Compare two system result directories and produce normalized metrics."""
    results_a, config_a = load_system_results(dir_a)
    results_b, config_b = load_system_results(dir_b)

    cost_a = config_a.get("system", {}).get("cost_usd", 1)
    cost_b = config_b.get("system", {}).get("cost_usd", 1)
    name_a = config_a.get("system", {}).get("name", "System A")
    name_b = config_b.get("system", {}).get("name", "System B")

    comparisons = []

    for bench_name, metrics in BENCHMARK_METRICS.items():
        if bench_name not in results_a or bench_name not in results_b:
            continue

        for metric_key, metric_label in metrics:
            comparison = normalize_results(
                results_a[bench_name],
                results_b[bench_name],
                metric_key,
                cost_a,
                cost_b,
            )
            comparison["metric"] = metric_label
            comparisons.append(comparison)

    return {
        "system_a": {"name": name_a, "cost_usd": cost_a, "results_dir": dir_a},
        "system_b": {"name": name_b, "cost_usd": cost_b, "results_dir": dir_b},
        "comparisons": comparisons,
    }


def print_summary(comparison: dict):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 80)

    name_a = comparison["system_a"]["name"]
    name_b = comparison["system_b"]["name"]
    cost_a = comparison["system_a"]["cost_usd"]
    cost_b = comparison["system_b"]["cost_usd"]

    print(f"\n  {name_a} (${cost_a:,}) vs {name_b} (${cost_b:,})\n")
    print(f"  {'Metric':<30s} {'':>12s} {'':>12s} {'Perf/$':>10s} {'Perf/$':>10s} {'Winner':>10s}")
    print(f"  {'':30s} {name_a:>12s} {name_b:>12s} {name_a:>10s} {name_b:>10s}")
    print("  " + "-" * 84)

    for c in comparison["comparisons"]:
        raw_a = c["system_a"]["raw"]
        raw_b = c["system_b"]["raw"]
        ppd_a = c["system_a"]["perf_per_dollar"]
        ppd_b = c["system_b"]["perf_per_dollar"]
        winner = name_a if raw_a > raw_b else name_b

        print(
            f"  {c['metric']:<30s} "
            f"{raw_a:>12.2f} {raw_b:>12.2f} "
            f"{ppd_a:>10.4f} {ppd_b:>10.4f} "
            f"{winner:>10s}"
        )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between two systems")
    parser.add_argument("--system-a", required=True, help="Results directory for system A")
    parser.add_argument("--system-b", required=True, help="Results directory for system B")
    parser.add_argument("--output", default="results/comparison.json", help="Output comparison JSON")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    comparison = compare_systems(args.system_a, args.system_b)
    save_json(comparison, args.output)
    print(f"Comparison saved to {args.output}")

    print_summary(comparison)

    if args.plot:
        from plots.generate_plots import main as generate_plots_main
        import sys
        sys.argv = [
            "generate_plots",
            "--results-a", args.system_a,
            "--results-b", args.system_b,
            "--comparison", args.output,
            "--output", args.plots_dir,
        ]
        generate_plots_main()


if __name__ == "__main__":
    main()
