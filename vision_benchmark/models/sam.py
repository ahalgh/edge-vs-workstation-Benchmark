"""Segment Anything Model wrapper (optional)."""

import numpy as np
import torch


class SAMModel:
    def __init__(self, variant: str = "vit_b"):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            raise ImportError(
                "segment_anything not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        checkpoint_map = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }
        checkpoint = checkpoint_map.get(variant, checkpoint_map["vit_b"])
        self.model = sam_model_registry[variant](checkpoint=checkpoint).cuda().eval()
        self.predictor = SamPredictor(self.model)
        self.variant = variant

    def warmup(self, images: list[np.ndarray]) -> None:
        self.predictor.set_image(images[0])
        self.predictor.predict(
            point_coords=np.array([[320, 320]]),
            point_labels=np.array([1]),
        )

    def inference(self, images: list[np.ndarray]) -> int:
        """Run SAM on a list of images. Returns number processed."""
        for img in images:
            self.predictor.set_image(img)
            self.predictor.predict(
                point_coords=np.array([[img.shape[1] // 2, img.shape[0] // 2]]),
                point_labels=np.array([1]),
            )
        return len(images)

    def cleanup(self) -> None:
        del self.model, self.predictor
        torch.cuda.empty_cache()
