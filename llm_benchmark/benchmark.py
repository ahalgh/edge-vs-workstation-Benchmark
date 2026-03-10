"""LLM inference benchmark measuring tokens/sec, latency, and memory usage."""

from datetime import datetime, timezone

import torch

from common.device_info import get_device_info
from common.result_schema import BenchmarkResult, PowerResult
from common.timer import CUDATimer
from common.utils import get_gpu_memory_usage, setup_logging
from llm_benchmark.backends.base import create_backend
from llm_benchmark.prompts import get_prompts


class LLMBenchmark:
    name = "llm_inference"

    def __init__(self, config: dict, power_monitor=None):
        self.config = config
        self.power_monitor = power_monitor

    def run(self) -> BenchmarkResult:
        logger = setup_logging()
        llm_config = self.config.get("llm", {})
        backend_name = llm_config.get("backend", "vllm")
        models = llm_config.get("models", [])
        batch_sizes = llm_config.get("batch_sizes", [1, 4, 8])
        max_tokens = llm_config.get("max_tokens", 128)
        warmup_iters = self.config.get("benchmarks", {}).get("warmup_iterations", 5)
        bench_iters = self.config.get("benchmarks", {}).get("benchmark_iterations", 50)

        backend = create_backend(backend_name)
        all_results = []

        for model_name in models:
            logger.info(f"[LLM] Loading {model_name} with {backend_name}")
            try:
                backend.load_model(model_name, self.config)
            except Exception as e:
                logger.error(f"[LLM] Failed to load {model_name}: {e}")
                continue

            for batch_size in batch_sizes:
                logger.info(f"[LLM] {model_name} batch_size={batch_size}")
                prompts = get_prompts(batch_size, prompt_type="medium")

                # Warmup
                for _ in range(warmup_iters):
                    backend.generate(prompts, max_tokens)

                # Benchmark
                timings = []
                token_counts = []
                for _ in range(bench_iters):
                    with CUDATimer() as timer:
                        result = backend.generate(prompts, max_tokens)
                    timings.append(timer.elapsed_ms)
                    token_counts.append(result.completion_tokens)

                avg_ms = sum(timings) / len(timings)
                avg_tokens = sum(token_counts) / len(token_counts)
                tokens_per_sec = avg_tokens / (avg_ms / 1000)
                latency_per_token_ms = avg_ms / avg_tokens if avg_tokens > 0 else 0
                memory = backend.get_memory_usage()

                all_results.append({
                    "model": model_name,
                    "batch_size": batch_size,
                    "tokens_per_sec": tokens_per_sec,
                    "latency_per_token_ms": latency_per_token_ms,
                    "avg_completion_tokens": avg_tokens,
                    "avg_ms": avg_ms,
                    "gpu_memory_mb": memory.get("allocated_mb", 0),
                    "iterations": bench_iters,
                })

            backend.cleanup()

        return BenchmarkResult(
            system=self.config.get("system", {}).get("name", "unknown"),
            benchmark_name=self.name,
            results=all_results,
            power=PowerResult(),
            device_info=get_device_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )
