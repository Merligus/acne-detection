import os
import math
from typing import List, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Density-map utilities
# ---------------------------------------------------------------------------


def parse_yolo_label_file(path: str) -> torch.Tensor:
    """
    Read a YOLO-format .txt label file. Each line: 'class cx cy w h'.
    Returns [N, 4] tensor of normalized (cx, cy, w, h). Empty/missing -> [0,4].
    """
    if not os.path.isfile(path):
        return torch.zeros((0, 4), dtype=torch.float32)
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            rows.append([cx, cy, w, h])
    if not rows:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def boxes_to_density_map(
    boxes: torch.Tensor,
    map_size: int = 56,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Convert YOLO-normalized boxes [N, 4] (cx, cy, w, h in [0, 1]) into a
    Gaussian density map [map_size, map_size] whose total sum equals the
    number of boxes (within the on-grid clipping error).

    Sigma is in density-map pixels. With map_size=56 and sigma=2, the FWHM
    is roughly 5 px, which is a reasonable footprint for a single acne lesion.
    """
    density = torch.zeros(map_size, map_size, dtype=torch.float32)
    if boxes is None or boxes.numel() == 0:
        return density

    cx = (boxes[:, 0] * map_size).clamp(0, map_size - 1)
    cy = (boxes[:, 1] * map_size).clamp(0, map_size - 1)

    ys, xs = torch.meshgrid(
        torch.arange(map_size, dtype=torch.float32),
        torch.arange(map_size, dtype=torch.float32),
        indexing="ij",
    )

    two_sigma_sq = 2.0 * sigma * sigma
    norm = 1.0 / (math.pi * two_sigma_sq)  # 2D Gaussian normalizer => sum~=1 each

    for i in range(boxes.shape[0]):
        d2 = (xs - cx[i]) ** 2 + (ys - cy[i]) ** 2
        density += norm * torch.exp(-d2 / two_sigma_sq)

    return density


# ---------------------------------------------------------------------------
# Reference dataset: image + YOLO label .txt + severity int
# ---------------------------------------------------------------------------


class MultiTaskAcneDataset(Dataset):
    """
    samples : list of (image_path, yolo_label_txt_path)
    image_processor : HF AutoImageProcessor configured for the chosen DINOv3.

    Note on coordinates: HuggingFace processors usually resize and may
    center-crop. ACNE04 images are roughly square so the drift introduced by
    a center crop is minor; if you need exact alignment, pre-resize images
    to a square shape on disk and disable cropping in the processor config.
    """

    def __init__(
        self,
        samples: List[Tuple[str, str]],
        image_processor,
        density_map_size: int = 56,
        sigma: float = 2.0,
    ):
        self.samples = samples
        self.image_processor = image_processor
        self.density_map_size = density_map_size
        self.sigma = sigma

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _detections_to_severity(n: int) -> int:
        # Mild (Grade 0): 0–5 inflammatory lesions.
        # Moderate (Grade 1): 6–20 inflammatory lesions.
        # Severe (Grade 2): More than 21 inflammatory lesions.
        if n <= 5:
            return 0
        if n <= 20:
            return 1
        return 2

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        boxes = parse_yolo_label_file(label_path)
        density = boxes_to_density_map(boxes, self.density_map_size, self.sigma)
        count = boxes.shape[0]
        severity = self._detections_to_severity(count)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(severity, dtype=torch.long),
            "density_target": density,
            "count": torch.tensor(float(count), dtype=torch.float32),
        }
