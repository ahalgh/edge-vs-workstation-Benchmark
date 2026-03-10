"""YOLOv8 model wrapper for benchmarking."""

import numpy as np


class YOLOv8Model:
    def __init__(self, variant: str = "yolov8n"):
        from ultralytics import YOLO
        self.model = YOLO(f"{variant}.pt")
        self.variant = variant

    def warmup(self, images: list[np.ndarray]) -> None:
        self.model(images[0], verbose=False)

    def inference(self, images: list[np.ndarray]) -> int:
        """Run inference on a list of numpy images. Returns number of images processed."""
        self.model(images, verbose=False)
        return len(images)

    def cleanup(self) -> None:
        del self.model
