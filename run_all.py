"""Cross-platform benchmark orchestration script (replaces run_all.sh on Windows)."""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


BENCHMARKS = [
    ("hpc_gemm.run", "gemm.json", "GEMM Compute"),
    ("bandwidth_test.run", "bandwidth.json", "Memory Bandwidth"),
    ("vision_benchmark.run", "vision.json", "Vision Model"),
    ("llm_benchmark.run", "llm.json", "LLM Inference"),
    ("pipeline_benchmark.run", "pipeline.json", "End-to-End Pipeline"),
]


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--benchmarks", nargs="+", help="Run only specific benchmarks (gemm, bandwidth, vision, llm, pipeline)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        print("Usage: python run_all.py [config.yaml]")
        print("  e.g. python run_all.py configs/jetson_thor.yaml")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    system_name = config["system"]["name"]
    results_dir = config["output"]["results_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / f"{system_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(f"  Benchmark Suite: {system_name}")
    print(f"  Config: {args.config}")
    print(f"  Results: {run_dir}")
    print(f"  Started: {datetime.now()}")
    print("=" * 50)

    # Save config for reproducibility
    shutil.copy2(config_path, run_dir / "config.yaml")

    # Save system info
    try:
        info_path = str(run_dir / "system_info.json").replace("\\", "/")
        subprocess.run(
            [sys.executable, "-c",
             "from common.device_info import get_device_info; from common.utils import save_json; "
             f"save_json(get_device_info(), '{info_path}')"],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        print("WARN: Could not save system info")

    # Filter benchmarks if requested
    benchmarks = BENCHMARKS
    if args.benchmarks:
        filter_set = set(args.benchmarks)
        benchmarks = [b for b in benchmarks if any(f in b[0] for f in filter_set)]

    # Run each benchmark
    total = len(benchmarks)
    for i, (module, output_file, label) in enumerate(benchmarks, 1):
        print(f"\n[{i}/{total}] {label} Benchmark")
        print("-" * 50)

        output_path = run_dir / output_file
        cmd = [sys.executable, "-m", module, "--config", str(config_path), "--output", str(output_path)]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"WARN: {label} benchmark failed")
        except FileNotFoundError:
            print(f"WARN: {label} benchmark skipped (module not found)")

    print()
    print("=" * 50)
    print(f"  All benchmarks complete!")
    print(f"  Results saved to: {run_dir}")
    print(f"  Finished: {datetime.now()}")
    print("=" * 50)
    print()
    print("To compare results between systems, run:")
    print(f"  python compare.py --system-a results/<system_a_dir> --system-b results/<system_b_dir> --plot")


if __name__ == "__main__":
    main()
