"""TensorRT-LLM inference backend (stub)."""

from llm_benchmark.backends.base import GenerationResult, LLMBackend


class TensorRTBackend(LLMBackend):
    """Placeholder for TensorRT-LLM backend.

    TensorRT-LLM provides optimized inference on NVIDIA GPUs, especially
    useful on Jetson platforms with INT4/FP8 quantization support.

    To implement:
    1. Install TensorRT-LLM for your platform
    2. Build engine files for target models
    3. Implement the generate() method using tensorrt_llm.runtime
    """

    def load_model(self, model_name: str, config: dict) -> None:
        raise NotImplementedError(
            "TensorRT-LLM backend not yet implemented. "
            "Use --llm.backend vllm or set llm.backend: vllm in your config. "
            "See https://github.com/NVIDIA/TensorRT-LLM for setup instructions."
        )

    def generate(self, prompts: list[str], max_tokens: int) -> GenerationResult:
        raise NotImplementedError("TensorRT-LLM backend not yet implemented.")

    def get_memory_usage(self) -> dict:
        return {"allocated_mb": 0, "reserved_mb": 0}

    def cleanup(self) -> None:
        pass
