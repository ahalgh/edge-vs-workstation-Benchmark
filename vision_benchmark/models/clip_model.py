"""CLIP model wrapper for benchmarking using open_clip."""

import torch


class CLIPModel:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.cuda().eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model_name = model_name

    def warmup(self, images: torch.Tensor) -> None:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            self.model.encode_image(images[:1])

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images. Input: (B, 3, 224, 224) float32 tensor."""
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return self.model.encode_image(images)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            return self.model.encode_text(tokens)

    def inference(self, images: torch.Tensor) -> int:
        """Run image encoding. Returns number of images processed."""
        self.encode_image(images)
        return images.shape[0]

    def cleanup(self) -> None:
        del self.model
        torch.cuda.empty_cache()
