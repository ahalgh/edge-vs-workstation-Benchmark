"""GEMM compute benchmark measuring FP16/FP32 TFLOPS via PyTorch (cuBLAS)."""

import torch

from common.base_benchmark import BaseBenchmark
from common.timer import CUDATimer


class GEMMBenchmark(BaseBenchmark):
    name = "gemm_compute"

    def setup(self) -> None:
        self._gemm_configs = []
        dtypes_str = self.config.get("gemm", {}).get("dtypes", ["float16", "float32"])
        sizes = self.config.get("gemm", {}).get("matrix_sizes", [1024, 2048, 4096])

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        for dtype_name in dtypes_str:
            dtype = dtype_map.get(dtype_name, torch.float32)
            for size in sizes:
                self._gemm_configs.append((size, dtype, dtype_name))

        self._config_idx = 0

    def run_single(self, **kwargs) -> dict:
        M, dtype, dtype_name = self._gemm_configs[self._config_idx % len(self._gemm_configs)]
        N, K = M, M

        A = torch.randn(M, K, dtype=dtype, device=self.device)
        B = torch.randn(K, N, dtype=dtype, device=self.device)

        torch.cuda.synchronize()
        with CUDATimer() as timer:
            torch.mm(A, B)

        # FLOPS = 2 * M * N * K for matrix multiply
        flops = 2 * M * N * K
        tflops = flops / (timer.elapsed_ms / 1000) / 1e12

        result = {
            "M": M,
            "N": N,
            "K": K,
            "dtype": dtype_name,
            "elapsed_ms": timer.elapsed_ms,
            "tflops": tflops,
        }

        self._config_idx += 1
        return result

    def run(self):
        """Override to iterate over all GEMM configs with proper warmup per config."""
        from datetime import datetime, timezone

        from common.device_info import get_device_info
        from common.result_schema import BenchmarkResult, PowerResult
        from common.utils import setup_logging

        logger = setup_logging()
        warmup_iters = self.config.get("benchmarks", {}).get("warmup_iterations", 5)
        bench_iters = self.config.get("benchmarks", {}).get("benchmark_iterations", 50)

        self.setup()
        all_results = []

        dtypes_str = self.config.get("gemm", {}).get("dtypes", ["float16", "float32"])
        sizes = self.config.get("gemm", {}).get("matrix_sizes", [1024, 2048, 4096])
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

        for dtype_name in dtypes_str:
            dtype = dtype_map.get(dtype_name, torch.float32)
            for M in sizes:
                logger.info(f"[GEMM] {dtype_name} {M}x{M}x{M}")

                # Warmup
                A = torch.randn(M, M, dtype=dtype, device=self.device)
                B = torch.randn(M, M, dtype=dtype, device=self.device)
                for _ in range(warmup_iters):
                    torch.mm(A, B)
                torch.cuda.synchronize()

                # Benchmark
                timings = []
                for _ in range(bench_iters):
                    torch.cuda.synchronize()
                    with CUDATimer() as timer:
                        torch.mm(A, B)
                    timings.append(timer.elapsed_ms)

                flops = 2 * M * M * M
                avg_ms = sum(timings) / len(timings)
                min_ms = min(timings)
                tflops_avg = flops / (avg_ms / 1000) / 1e12
                tflops_peak = flops / (min_ms / 1000) / 1e12

                all_results.append({
                    "M": M, "N": M, "K": M,
                    "dtype": dtype_name,
                    "avg_ms": avg_ms,
                    "min_ms": min_ms,
                    "tflops_avg": tflops_avg,
                    "tflops_peak": tflops_peak,
                    "iterations": bench_iters,
                })

                del A, B
                torch.cuda.empty_cache()

        # Also test transformer-shaped matrices (batch*seq_len x hidden x hidden)
        transformer_shapes = [
            (4096, 4096, 11008, "transformer_ffn"),
            (4096, 4096, 4096, "transformer_attn"),
        ]
        for M, K, N, label in transformer_shapes:
            for dtype_name in dtypes_str:
                dtype = dtype_map.get(dtype_name, torch.float32)
                try:
                    A = torch.randn(M, K, dtype=dtype, device=self.device)
                    B = torch.randn(K, N, dtype=dtype, device=self.device)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"[GEMM] OOM for {label} {dtype_name}, skipping")
                    continue

                for _ in range(warmup_iters):
                    torch.mm(A, B)
                torch.cuda.synchronize()

                timings = []
                for _ in range(bench_iters):
                    torch.cuda.synchronize()
                    with CUDATimer() as timer:
                        torch.mm(A, B)
                    timings.append(timer.elapsed_ms)

                flops = 2 * M * N * K
                avg_ms = sum(timings) / len(timings)
                min_ms = min(timings)

                all_results.append({
                    "M": M, "N": N, "K": K,
                    "label": label,
                    "dtype": dtype_name,
                    "avg_ms": avg_ms,
                    "min_ms": min_ms,
                    "tflops_avg": flops / (avg_ms / 1000) / 1e12,
                    "tflops_peak": flops / (min_ms / 1000) / 1e12,
                    "iterations": bench_iters,
                })

                del A, B
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
