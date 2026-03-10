"""CLI entry point for end-to-end pipeline benchmark."""

from common.config import get_benchmark_args, load_config
from common.utils import setup_logging
from pipeline_benchmark.benchmark import PipelineBenchmark
from power_monitor import PowerMonitor


def main():
    args = get_benchmark_args()
    config = load_config(args.config, args.remaining)
    logger = setup_logging()

    output = args.output or f"{config['output']['results_dir']}/pipeline.json"

    logger.info("Starting end-to-end pipeline benchmark")
    pm = PowerMonitor(config)
    benchmark = PipelineBenchmark(config, pm)
    result = benchmark.run()
    result.to_json(output)
    logger.info(f"Results saved to {output}")

    for r in result.results:
        print(f"  E2E: {r['e2e_latency_ms']:.1f} ms ({r['inferences_per_sec']:.2f} inf/s)")
        print(f"    CLIP: {r['clip_latency_ms']:.1f} ms, LLM: {r['llm_latency_ms']:.1f} ms")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
