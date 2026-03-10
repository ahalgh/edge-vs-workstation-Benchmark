"""Abstract base benchmark with warmup/measure/teardown template."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import torch

from common.device_info import get_device_info
from common.result_schema import BenchmarkResult, PowerResult
from common.utils import get_gpu_memory_usage, set_reproducibility, setup_logging

logger = setup_logging()


class BaseBenchmark(ABC):
    """Template-method base class for all benchmarks.

    Subclasses implement setup(), run_single(), teardown().
    The run() method handles warmup, power-monitored measurement, and result packaging.
    """

    name: str = "base"

    def __init__(self, config: dict, power_monitor=None):
        self.config = config
        self.power_monitor = power_monitor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_reproducibility(config.get("benchmarks", {}).get("seed", 42))

    @abstractmethod
    def setup(self) -> None:
        """Initialize models, data, etc."""

    @abstractmethod
    def run_single(self, **kwargs) -> dict:
        """Run a single benchmark iteration. Returns a dict of metrics."""

    def teardown(self) -> None:
        """Clean up resources. Override if needed."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self) -> BenchmarkResult:
        """Execute the full benchmark: setup → warmup → measure → teardown."""
        warmup_iters = self.config.get("benchmarks", {}).get("warmup_iterations", 5)
        bench_iters = self.config.get("benchmarks", {}).get("benchmark_iterations", 50)

        logger.info(f"[{self.name}] Setting up...")
        self.setup()

        logger.info(f"[{self.name}] Warming up ({warmup_iters} iterations)...")
        for _ in range(warmup_iters):
            self.run_single()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"[{self.name}] Benchmarking ({bench_iters} iterations)...")
        power_result = PowerResult()
        results = []

        if self.power_monitor:
            with self.power_monitor as pm:
                for _ in range(bench_iters):
                    results.append(self.run_single())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                power_result = pm.get_results()
        else:
            for _ in range(bench_iters):
                results.append(self.run_single())
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        logger.info(f"[{self.name}] Tearing down...")
        self.teardown()

        return BenchmarkResult(
            system=self.config.get("system", {}).get("name", "unknown"),
            benchmark_name=self.name,
            results=results,
            power=power_result,
            device_info=get_device_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )
