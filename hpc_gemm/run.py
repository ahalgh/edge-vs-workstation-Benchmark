"""CLI entry point for GEMM compute benchmark."""

from common.config import get_benchmark_args, load_config
from common.utils import setup_logging
from hpc_gemm.benchmark import GEMMBenchmark
from power_monitor import PowerMonitor


def main():
    args = get_benchmark_args()
    config = load_config(args.config, args.remaining)
    logger = setup_logging()

    output = args.output or f"{config['output']['results_dir']}/gemm.json"

    logger.info("Starting GEMM compute benchmark")
    pm = PowerMonitor(config)
    benchmark = GEMMBenchmark(config, pm)
    result = benchmark.run()
    result.to_json(output)
    logger.info(f"Results saved to {output}")

    # Print summary
    for r in result.results:
        label = r.get("label", f"{r['M']}x{r['N']}x{r['K']}")
        print(f"  {r['dtype']:>8s} {label:>20s}: {r['tflops_avg']:.2f} TFLOPS (avg), {r['tflops_peak']:.2f} TFLOPS (peak)")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
