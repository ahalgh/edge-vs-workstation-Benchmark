"""Synthetic and sample image generation for vision benchmarks."""

import numpy as np
import torch


def generate_synthetic_images(
    batch_size: int,
    resolution: tuple[int, int] = (640, 640),
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a batch of random images as uint8 tensors (B, 3, H, W)."""
    h, w = resolution
    return torch.randint(0, 256, (batch_size, 3, h, w), dtype=torch.uint8, device=device)


def generate_synthetic_images_float(
    batch_size: int,
    resolution: tuple[int, int] = (640, 640),
    device: str = "cuda",
) -> torch.Tensor:
    """Generate a batch of random images as float32 tensors (B, 3, H, W), normalized [0, 1]."""
    h, w = resolution
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32, device=device)


def generate_numpy_images(
    batch_size: int,
    resolution: tuple[int, int] = (640, 640),
) -> list[np.ndarray]:
    """Generate a list of random numpy images (H, W, 3) uint8 for YOLO/SAM input."""
    h, w = resolution
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(batch_size)]
