"""Generate comparison plots from benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plots.style import apply_style, get_system_color


def load_results(results_dir: str) -> dict:
    """Load all JSON result files from a directory."""
    results = {}
    for json_file in Path(results_dir).glob("*.json"):
        if json_file.name == "config.yaml":
            continue
        with open(json_file) as f:
            results[json_file.stem] = json.load(f)
    return results


def plot_bar_comparison(
    data_a: list[dict],
    data_b: list[dict],
    name_a: str,
    name_b: str,
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    output_path: str,
    xlabel: str = "",
):
    """Create a side-by-side bar chart comparing two systems."""
    apply_style()

    # Extract values
    labels_a = [str(d.get(x_key, "")) for d in data_a]
    labels_b = [str(d.get(x_key, "")) for d in data_b]
    values_a = [d.get(y_key, 0) for d in data_a]
    values_b = [d.get(y_key, 0) for d in data_b]

    # Align labels
    all_labels = list(dict.fromkeys(labels_a + labels_b))
    vals_a = {str(d.get(x_key, "")): d.get(y_key, 0) for d in data_a}
    vals_b = {str(d.get(x_key, "")): d.get(y_key, 0) for d in data_b}

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, [vals_a.get(l, 0) for l in all_labels], width,
           label=name_a, color=get_system_color(name_a, 0))
    ax.bar(x + width / 2, [vals_b.get(l, 0) for l in all_labels], width,
           label=name_b, color=get_system_color(name_b, 1))

    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_llm_comparison(results_a: dict, results_b: dict, name_a: str, name_b: str, output_dir: str):
    """Generate LLM benchmark comparison plots."""
    if "llm" not in results_a or "llm" not in results_b:
        return

    data_a = results_a["llm"]["results"]
    data_b = results_b["llm"]["results"]

    # Tokens/sec by model (batch_size=1 for fair comparison)
    bs1_a = [r for r in data_a if r.get("batch_size") == 1]
    bs1_b = [r for r in data_b if r.get("batch_size") == 1]

    if bs1_a and bs1_b:
        plot_bar_comparison(
            bs1_a, bs1_b, name_a, name_b,
            x_key="model", y_key="tokens_per_sec",
            title="LLM Inference: Tokens/sec (batch_size=1)",
            ylabel="Tokens/sec",
            output_path=f"{output_dir}/llm_tokens_per_sec.png",
        )


def plot_vision_comparison(results_a: dict, results_b: dict, name_a: str, name_b: str, output_dir: str):
    """Generate vision benchmark comparison plots."""
    if "vision" not in results_a or "vision" not in results_b:
        return

    data_a = results_a["vision"]["results"]
    data_b = results_b["vision"]["results"]

    # FPS by model (batch_size=1)
    bs1_a = [r for r in data_a if r.get("batch_size") == 1]
    bs1_b = [r for r in data_b if r.get("batch_size") == 1]

    if bs1_a and bs1_b:
        plot_bar_comparison(
            bs1_a, bs1_b, name_a, name_b,
            x_key="model", y_key="fps",
            title="Vision Inference: Frames/sec (batch_size=1)",
            ylabel="Frames/sec",
            output_path=f"{output_dir}/vision_fps.png",
        )


def plot_gemm_comparison(results_a: dict, results_b: dict, name_a: str, name_b: str, output_dir: str):
    """Generate GEMM benchmark comparison plots."""
    if "gemm" not in results_a or "gemm" not in results_b:
        return

    data_a = results_a["gemm"]["results"]
    data_b = results_b["gemm"]["results"]

    # TFLOPS by matrix size for each dtype
    for dtype in ["float16", "float32"]:
        filtered_a = [r for r in data_a if r.get("dtype") == dtype and "label" not in r]
        filtered_b = [r for r in data_b if r.get("dtype") == dtype and "label" not in r]

        # Create label from matrix size
        for r in filtered_a:
            r["_label"] = f"{r['M']}x{r['N']}"
        for r in filtered_b:
            r["_label"] = f"{r['M']}x{r['N']}"

        if filtered_a and filtered_b:
            plot_bar_comparison(
                filtered_a, filtered_b, name_a, name_b,
                x_key="_label", y_key="tflops_avg",
                title=f"GEMM Compute: TFLOPS ({dtype})",
                ylabel="TFLOPS",
                output_path=f"{output_dir}/gemm_tflops_{dtype}.png",
                xlabel="Matrix Size",
            )


def plot_bandwidth_comparison(results_a: dict, results_b: dict, name_a: str, name_b: str, output_dir: str):
    """Generate bandwidth benchmark comparison plots."""
    if "bandwidth" not in results_a or "bandwidth" not in results_b:
        return

    data_a = results_a["bandwidth"]["results"]
    data_b = results_b["bandwidth"]["results"]

    # GB/s by direction (using largest size)
    for direction in ["D2D", "H2D", "D2H"]:
        dir_a = [r for r in data_a if r.get("direction") == direction]
        dir_b = [r for r in data_b if r.get("direction") == direction]

        for r in dir_a:
            r["_label"] = f"{r['size_mb']:.0f} MB"
        for r in dir_b:
            r["_label"] = f"{r['size_mb']:.0f} MB"

        if dir_a and dir_b:
            plot_bar_comparison(
                dir_a, dir_b, name_a, name_b,
                x_key="_label", y_key="gb_per_sec",
                title=f"Memory Bandwidth: {direction} (GB/s)",
                ylabel="GB/s",
                output_path=f"{output_dir}/bandwidth_{direction.lower()}.png",
                xlabel="Transfer Size",
            )


def plot_normalized_comparison(comparison: dict, output_dir: str):
    """Generate performance-per-dollar and performance-per-watt plots."""
    apply_style()

    if not comparison.get("comparisons"):
        return

    metrics = comparison["comparisons"]
    labels = [m["metric"] for m in metrics]

    # Perf per dollar
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (norm_key, title) in enumerate([
        ("perf_per_dollar", "Performance per Dollar"),
        ("perf_per_watt", "Performance per Watt"),
    ]):
        ax = axes[idx]
        name_a = metrics[0]["system_a"]["name"]
        name_b = metrics[0]["system_b"]["name"]

        vals_a = [m["system_a"].get(norm_key, 0) for m in metrics]
        vals_b = [m["system_b"].get(norm_key, 0) for m in metrics]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, vals_a, width, label=name_a, color=get_system_color(name_a, 0))
        ax.bar(x + width / 2, vals_b, width, label=name_b, color=get_system_color(name_b, 1))
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/normalized_comparison.png")
    plt.close()
    print(f"  Saved: {output_dir}/normalized_comparison.png")


def generate_summary_table(comparison: dict, output_dir: str):
    """Generate a summary table as a plot."""
    apply_style()

    if not comparison.get("comparisons"):
        return

    metrics = comparison["comparisons"]
    name_a = metrics[0]["system_a"]["name"]
    name_b = metrics[0]["system_b"]["name"]

    headers = ["Metric", name_a, name_b, "Winner"]
    rows = []
    for m in metrics:
        raw_a = m["system_a"]["raw"]
        raw_b = m["system_b"]["raw"]
        winner = name_a if raw_a > raw_b else name_b
        rows.append([
            m["metric"],
            f"{raw_a:.2f}",
            f"{raw_b:.2f}",
            winner,
        ])

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.5 + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Benchmark Summary", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_table.png")
    plt.close()
    print(f"  Saved: {output_dir}/summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--results-a", required=True, help="Results directory for system A")
    parser.add_argument("--results-b", required=True, help="Results directory for system B")
    parser.add_argument("--comparison", default=None, help="Path to comparison.json from compare.py")
    parser.add_argument("--output", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    results_a = load_results(args.results_a)
    results_b = load_results(args.results_b)

    # Determine system names
    name_a = "System A"
    name_b = "System B"
    for v in results_a.values():
        if "system" in v:
            name_a = v["system"]
            break
    for v in results_b.values():
        if "system" in v:
            name_b = v["system"]
            break

    print(f"Generating plots: {name_a} vs {name_b}")

    plot_llm_comparison(results_a, results_b, name_a, name_b, args.output)
    plot_vision_comparison(results_a, results_b, name_a, name_b, args.output)
    plot_gemm_comparison(results_a, results_b, name_a, name_b, args.output)
    plot_bandwidth_comparison(results_a, results_b, name_a, name_b, args.output)

    if args.comparison:
        with open(args.comparison) as f:
            comparison = json.load(f)
        plot_normalized_comparison(comparison, args.output)
        generate_summary_table(comparison, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
