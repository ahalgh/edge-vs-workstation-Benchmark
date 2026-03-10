"""CLI entry point for vision model benchmark."""

from common.config import get_benchmark_args, load_config
from common.utils import setup_logging
from power_monitor import PowerMonitor
from vision_benchmark.benchmark import VisionBenchmark


def main():
    args = get_benchmark_args()
    config = load_config(args.config, args.remaining)
    logger = setup_logging()

    output = args.output or f"{config['output']['results_dir']}/vision.json"

    logger.info("Starting vision model benchmark")
    pm = PowerMonitor(config)
    benchmark = VisionBenchmark(config, pm)
    result = benchmark.run()
    result.to_json(output)
    logger.info(f"Results saved to {output}")

    for r in result.results:
        print(f"  {r['model']:>10s} bs={r['batch_size']:>2d}: {r['fps']:.1f} fps, {r['latency_per_frame_ms']:.2f} ms/frame")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
