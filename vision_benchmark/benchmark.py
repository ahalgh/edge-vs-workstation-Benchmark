"""Vision model benchmark measuring frames/sec and latency."""

from datetime import datetime, timezone

import torch

from common.device_info import get_device_info
from common.result_schema import BenchmarkResult, PowerResult
from common.timer import CUDATimer
from common.utils import setup_logging
from vision_benchmark.data import generate_numpy_images, generate_synthetic_images_float


class VisionBenchmark:
    name = "vision_inference"

    def __init__(self, config: dict, power_monitor=None):
        self.config = config
        self.power_monitor = power_monitor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_name: str):
        if model_name.startswith("yolov8"):
            from vision_benchmark.models.yolov8 import YOLOv8Model
            return YOLOv8Model(variant=model_name), "numpy"
        elif model_name == "clip":
            from vision_benchmark.models.clip_model import CLIPModel
            return CLIPModel(), "tensor"
        elif model_name == "sam":
            from vision_benchmark.models.sam import SAMModel
            return SAMModel(), "numpy"
        else:
            raise ValueError(f"Unknown vision model: {model_name}")

    def run(self) -> BenchmarkResult:
        logger = setup_logging()
        models = self.config.get("vision", {}).get("models", ["yolov8n", "clip"])
        batch_sizes = self.config.get("vision", {}).get("batch_sizes", [1, 4, 8])
        resolution = tuple(self.config.get("vision", {}).get("input_resolution", [640, 640]))
        warmup_iters = self.config.get("benchmarks", {}).get("warmup_iterations", 5)
        bench_iters = self.config.get("benchmarks", {}).get("benchmark_iterations", 50)

        all_results = []

        for model_name in models:
            try:
                model, input_type = self._load_model(model_name)
            except (ImportError, Exception) as e:
                logger.warning(f"[Vision] Skipping {model_name}: {e}")
                continue

            for batch_size in batch_sizes:
                logger.info(f"[Vision] {model_name} batch_size={batch_size}")

                # Generate input data
                if input_type == "numpy":
                    images = generate_numpy_images(batch_size, resolution)
                else:
                    # CLIP uses 224x224 float tensors
                    clip_res = (224, 224)
                    images = generate_synthetic_images_float(batch_size, clip_res, str(self.device))

                # Warmup
                for _ in range(warmup_iters):
                    if input_type == "numpy":
                        model.inference(images)
                    else:
                        model.inference(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Benchmark
                timings = []
                for _ in range(bench_iters):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    with CUDATimer() as timer:
                        num_processed = model.inference(images)
                    timings.append(timer.elapsed_ms)

                avg_ms = sum(timings) / len(timings)
                total_frames = num_processed
                fps = total_frames / (avg_ms / 1000)
                latency_per_frame_ms = avg_ms / total_frames

                all_results.append({
                    "model": model_name,
                    "batch_size": batch_size,
                    "avg_ms": avg_ms,
                    "fps": fps,
                    "latency_per_frame_ms": latency_per_frame_ms,
                    "iterations": bench_iters,
                })

            try:
                model.cleanup()
            except Exception:
                pass
            torch.cuda.empty_cache()

        return BenchmarkResult(
            system=self.config.get("system", {}).get("name", "unknown"),
            benchmark_name=self.name,
            results=all_results,
            power=PowerResult(),
            device_info=get_device_info(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self.config,
        )
