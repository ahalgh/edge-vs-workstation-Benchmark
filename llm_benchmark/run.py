"""CLI entry point for LLM inference benchmark."""

from common.config import get_benchmark_args, load_config
from common.utils import setup_logging
from llm_benchmark.benchmark import LLMBenchmark
from power_monitor import PowerMonitor


def main():
    args = get_benchmark_args()
    config = load_config(args.config, args.remaining)
    logger = setup_logging()

    output = args.output or f"{config['output']['results_dir']}/llm.json"

    logger.info("Starting LLM inference benchmark")
    pm = PowerMonitor(config)
    benchmark = LLMBenchmark(config, pm)
    result = benchmark.run()
    result.to_json(output)
    logger.info(f"Results saved to {output}")

    for r in result.results:
        print(
            f"  {r['model']:>30s} bs={r['batch_size']:>2d}: "
            f"{r['tokens_per_sec']:.1f} tok/s, "
            f"{r['latency_per_token_ms']:.2f} ms/tok, "
            f"{r['gpu_memory_mb']:.0f} MB"
        )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
