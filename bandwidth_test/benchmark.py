"""Memory bandwidth benchmark: D2D, H2D, D2H transfer rates."""

from datetime import datetime, timezone

import torch

from common.base_benchmark import BaseBenchmark
from common.device_info import get_device_info
from common.result_schema import BenchmarkResult, PowerResult
from common.timer import CUDATimer
from common.utils import setup_logging


class BandwidthBenchmark(BaseBenchmark):
    name = "memory_bandwidth"

    def setup(self) -> None:
        pass

    def run_single(self, **kwargs) -> dict:
        return {}

    def _measure_d2d(self, size_bytes: int, iterations: int) -> dict:
        """Device-to-device bandwidth via tensor clone."""
        src = torch.empty(size_bytes // 4, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(3):
            _ = src.clone()
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            with CUDATimer() as timer:
                _ = src.clone()
            timings.append(timer.elapsed_ms)

        avg_ms = sum(timings) / len(timings)
        median_ms = sorted(timings)[len(timings) // 2]
        gb_per_sec = (size_bytes / 1e9) / (median_ms / 1000)

        del src
        return {
            "direction": "D2D",
            "size_mb": size_bytes / (1024 * 1024),
            "avg_ms": avg_ms,
            "median_ms": median_ms,
            "gb_per_sec": gb_per_sec,
        }

    def _measure_h2d(self, size_bytes: int, iterations: int) -> dict:
        """Host-to-device bandwidth with pinned memory."""
        src = torch.empty(size_bytes // 4, dtype=torch.float32, device="cpu").pin_memory()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(3):
            _ = src.cuda()
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            with CUDATimer() as timer:
                _ = src.cuda()
            timings.append(timer.elapsed_ms)

        avg_ms = sum(timings) / len(timings)
        median_ms = sorted(timings)[len(timings) // 2]
        gb_per_sec = (size_bytes / 1e9) / (median_ms / 1000)

        del src
        return {
            "direction": "H2D",
            "size_mb": size_bytes / (1024 * 1024),
            "avg_ms": avg_ms,
            "median_ms": median_ms,
            "gb_per_sec": gb_per_sec,
        }

    def _measure_d2h(self, size_bytes: int, iterations: int) -> dict:
        """Device-to-host bandwidth."""
        src = torch.empty(size_bytes // 4, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()

        # Warmup
        for _ in range(3):
            _ = src.cpu()
        torch.cuda.synchronize()

        timings = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            with CUDATimer() as timer:
                _ = src.cpu()
            timings.append(timer.elapsed_ms)

        avg_ms = sum(timings) / len(timings)
        median_ms = sorted(timings)[len(timings) // 2]
        gb_per_sec = (size_bytes / 1e9) / (median_ms / 1000)

        del src
        return {
            "direction": "D2H",
            "size_mb": size_bytes / (1024 * 1024),
            "avg_ms": avg_ms,
            "median_ms": median_ms,
            "gb_per_sec": gb_per_sec,
        }

    def run(self) -> BenchmarkResult:
        logger = setup_logging()
        sizes_mb = self.config.get("bandwidth", {}).get("sizes_mb", [1, 10, 100, 500, 1000])
        iterations = self.config.get("bandwidth", {}).get("iterations", 100)

        all_results = []
        for size_mb in sizes_mb:
            size_bytes = int(size_mb * 1024 * 1024)
            logger.info(f"[Bandwidth] Testing {size_mb} MB")

            for measure_fn in [self._measure_d2d, self._measure_h2d, self._measure_d2h]:
                try:
                    result = measure_fn(size_bytes, iterations)
                    all_results.append(result)
                    logger.info(f"  {result['direction']}: {result['gb_per_sec']:.2f} GB/s")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"  OOM at {size_mb} MB, skipping")
                    break

            torch.cuda.empty_cache()

        self.teardown()

        return BenchmarkResult(
            system=self.config.get("system", {}).get("name", "unknown"),
            benchmark_name=self.name,
            results=all_results,
            power=PowerResult(),
            device_info=get_device_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )
