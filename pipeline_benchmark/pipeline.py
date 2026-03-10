"""End-to-end AI pipeline: image -> CLIP embedding -> LLM prompt -> classification."""

import torch

from llm_benchmark.backends.base import LLMBackend
from vision_benchmark.models.clip_model import CLIPModel


CLASSIFICATION_LABELS = [
    "cat", "dog", "car", "truck", "person", "bicycle", "bird", "boat",
    "building", "tree", "flower", "food", "airplane", "train", "horse",
    "mountain", "beach", "city", "forest", "sunset",
]


class AIPipeline:
    """Simulates a realistic AI workflow: image -> CLIP -> LLM -> classification."""

    def __init__(self, clip_model: CLIPModel, llm_backend: LLMBackend, config: dict):
        self.clip = clip_model
        self.llm = llm_backend
        self.config = config
        self.labels = CLASSIFICATION_LABELS

        # Pre-compute text embeddings for labels
        self.text_embeddings = self.clip.encode_text(self.labels)
        self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)

    def run(self, image: torch.Tensor) -> dict:
        """Run the full pipeline on a single image.

        Args:
            image: (1, 3, 224, 224) float32 tensor on CUDA.

        Returns:
            Dict with classification results and intermediate data.
        """
        # Step 1: CLIP image embedding
        image_embedding = self.clip.encode_image(image)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Step 2: Compute similarities to find top labels
        similarities = (image_embedding @ self.text_embeddings.T).softmax(dim=-1)
        top_indices = similarities.topk(3).indices[0]
        top_labels = [self.labels[i] for i in top_indices.cpu().tolist()]
        top_scores = similarities[0, top_indices].cpu().tolist()

        # Step 3: Build LLM prompt from CLIP context
        prompt = (
            f"An image analysis system detected the following objects with confidence scores: "
            f"{top_labels[0]} ({top_scores[0]:.2f}), "
            f"{top_labels[1]} ({top_scores[1]:.2f}), "
            f"{top_labels[2]} ({top_scores[2]:.2f}). "
            f"Based on these detections, classify this image into one primary category "
            f"and explain your reasoning in one sentence."
        )

        # Step 4: LLM classification
        max_tokens = self.config.get("llm", {}).get("max_tokens", 64)
        result = self.llm.generate([prompt], max_tokens=min(max_tokens, 64))

        return {
            "top_clip_labels": top_labels,
            "top_clip_scores": top_scores,
            "llm_output": result.outputs[0] if result.outputs else "",
            "classification": top_labels[0],
            "llm_tokens": result.completion_tokens,
        }
