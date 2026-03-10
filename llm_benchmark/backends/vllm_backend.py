"""vLLM inference backend."""

import time

import torch

from llm_benchmark.backends.base import GenerationResult, LLMBackend


class VLLMBackend(LLMBackend):
    def __init__(self):
        self.llm = None
        self.sampling_params = None

    def load_model(self, model_name: str, config: dict) -> None:
        from vllm import LLM, SamplingParams

        llm_config = config.get("llm", {})
        quantization = llm_config.get("quantization")
        max_tokens = llm_config.get("max_tokens", 128)

        self.llm = LLM(
            model=model_name,
            quantization=quantization,
            dtype="auto",
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # deterministic for reproducibility
        )

    def generate(self, prompts: list[str], max_tokens: int) -> GenerationResult:
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if max_tokens != self.sampling_params.max_tokens:
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        start = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params)
        elapsed = time.perf_counter() - start

        total_prompt = 0
        total_completion = 0
        output_texts = []

        for output in outputs:
            total_prompt += len(output.prompt_token_ids)
            for completion in output.outputs:
                total_completion += len(completion.token_ids)
                output_texts.append(completion.text)

        return GenerationResult(
            outputs=output_texts,
            total_tokens=total_prompt + total_completion,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            elapsed_seconds=elapsed,
        )

    def get_memory_usage(self) -> dict:
        if not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        }

    def cleanup(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        torch.cuda.empty_cache()
