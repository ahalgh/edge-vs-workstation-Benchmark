"""End-to-end AI pipeline benchmark."""

from datetime import datetime, timezone

import torch

from common.device_info import get_device_info
from common.result_schema import BenchmarkResult, PowerResult
from common.timer import CUDATimer
from common.utils import setup_logging
from vision_benchmark.data import generate_synthetic_images_float


class PipelineBenchmark:
    name = "pipeline_e2e"

    def __init__(self, config: dict, power_monitor=None):
        self.config = config
        self.power_monitor = power_monitor

    def run(self) -> BenchmarkResult:
        logger = setup_logging()
        warmup_iters = self.config.get("benchmarks", {}).get("warmup_iterations", 3)
        bench_iters = self.config.get("benchmarks", {}).get("benchmark_iterations", 20)
        llm_config = self.config.get("llm", {})

        # Load CLIP
        logger.info("[Pipeline] Loading CLIP model")
        from vision_benchmark.models.clip_model import CLIPModel
        clip_model = CLIPModel()

        # Load LLM
        logger.info("[Pipeline] Loading LLM")
        from llm_benchmark.backends.base import create_backend
        llm_backend = create_backend(llm_config.get("backend", "vllm"))
        model_name = llm_config.get("models", ["meta-llama/Llama-3.1-8B"])[0]
        llm_backend.load_model(model_name, self.config)

        # Build pipeline
        from pipeline_benchmark.pipeline import AIPipeline
        pipeline = AIPipeline(clip_model, llm_backend, self.config)

        # Generate test images (224x224 for CLIP)
        test_images = generate_synthetic_images_float(bench_iters + warmup_iters, (224, 224), "cuda")

        # Warmup
        logger.info(f"[Pipeline] Warming up ({warmup_iters} iterations)")
        for i in range(warmup_iters):
            pipeline.run(test_images[i : i + 1])
        torch.cuda.synchronize()

        # Benchmark
        logger.info(f"[Pipeline] Benchmarking ({bench_iters} iterations)")
        timings_total = []
        timings_clip = []
        timings_llm = []

        for i in range(bench_iters):
            image = test_images[warmup_iters + i : warmup_iters + i + 1]

            # Time CLIP step
            torch.cuda.synchronize()
            with CUDATimer() as clip_timer:
                image_emb = clip_model.encode_image(image)

            # Time LLM step (includes prompt construction)
            image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)
            sims = (image_emb_norm @ pipeline.text_embeddings.T).softmax(dim=-1)
            top_labels = [pipeline.labels[j] for j in sims.topk(3).indices[0].cpu().tolist()]

            prompt = (
                f"Image contains: {', '.join(top_labels)}. "
                f"Classify and explain in one sentence."
            )

            torch.cuda.synchronize()
            with CUDATimer() as llm_timer:
                llm_result = llm_backend.generate([prompt], max_tokens=64)

            total_ms = clip_timer.elapsed_ms + llm_timer.elapsed_ms
            timings_total.append(total_ms)
            timings_clip.append(clip_timer.elapsed_ms)
            timings_llm.append(llm_timer.elapsed_ms)

        avg_total = sum(timings_total) / len(timings_total)
        avg_clip = sum(timings_clip) / len(timings_clip)
        avg_llm = sum(timings_llm) / len(timings_llm)

        results = [{
            "model_llm": model_name,
            "model_clip": "ViT-B-32",
            "e2e_latency_ms": avg_total,
            "clip_latency_ms": avg_clip,
            "llm_latency_ms": avg_llm,
            "inferences_per_sec": 1000.0 / avg_total,
            "iterations": bench_iters,
        }]

        # Cleanup
        clip_model.cleanup()
        llm_backend.cleanup()

        return BenchmarkResult(
            system=self.config.get("system", {}).get("name", "unknown"),
            benchmark_name=self.name,
            results=results,
            power=PowerResult(),
            device_info=get_device_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )
