"""Abstract base class for LLM inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    outputs: list[str]
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


class LLMBackend(ABC):
    """Abstract interface for LLM inference engines."""

    @abstractmethod
    def load_model(self, model_name: str, config: dict) -> None:
        """Load a model with the given configuration."""

    @abstractmethod
    def generate(self, prompts: list[str], max_tokens: int) -> GenerationResult:
        """Generate completions for a batch of prompts."""

    @abstractmethod
    def get_memory_usage(self) -> dict:
        """Return current GPU memory usage in MB."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources."""


def create_backend(backend_name: str) -> LLMBackend:
    """Factory function to create an LLM backend by name."""
    if backend_name == "vllm":
        from llm_benchmark.backends.vllm_backend import VLLMBackend
        return VLLMBackend()
    elif backend_name == "tensorrt":
        from llm_benchmark.backends.tensorrt_backend import TensorRTBackend
        return TensorRTBackend()
    else:
        raise ValueError(f"Unknown LLM backend: {backend_name}. Choose 'vllm' or 'tensorrt'.")
