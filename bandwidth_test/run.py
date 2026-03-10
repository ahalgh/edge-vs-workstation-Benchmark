"""CLI entry point for memory bandwidth benchmark."""

from bandwidth_test.benchmark import BandwidthBenchmark
from common.config import get_benchmark_args, load_config
from common.utils import setup_logging
from power_monitor import PowerMonitor


def main():
    args = get_benchmark_args()
    config = load_config(args.config, args.remaining)
    logger = setup_logging()

    output = args.output or f"{config['output']['results_dir']}/bandwidth.json"

    logger.info("Starting memory bandwidth benchmark")
    pm = PowerMonitor(config)
    benchmark = BandwidthBenchmark(config, pm)
    result = benchmark.run()
    result.to_json(output)
    logger.info(f"Results saved to {output}")

    # Print summary
    for r in result.results:
        print(f"  {r['direction']} {r['size_mb']:>8.0f} MB: {r['gb_per_sec']:.2f} GB/s")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
