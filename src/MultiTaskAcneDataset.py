import os
import math
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
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
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Convert YOLO-normalized boxes [N, 4] (cx, cy, w, h in [0, 1]) into a
    Gaussian density map [map_size, map_size]. With weights=None (default),
    each Gaussian's per-box integral is ~1 so the map sums to N (within
    on-grid clipping error). With weights=[N], each per-box integral becomes
    weights[i] so the map sums to weights.sum() — used when YOLO confidences
    drive the teacher.

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
        amplitude = norm if weights is None else norm * float(weights[i].item())
        density += amplitude * torch.exp(-d2 / two_sigma_sq)

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
        density_source: str = "gt",
        yolo_model=None,
        yolo_conf: float = 0.25,
        segformer_model=None,
        segformer_device: str = "cuda",
        logger=None,
    ):
        self.samples = samples
        self.image_processor = image_processor
        self.density_map_size = density_map_size
        self.sigma = sigma
        self.density_source = density_source
        self.yolo_conf = yolo_conf

        self.yolo_cache = None
        self.segformer_cache = None
        if density_source == "yolo":
            assert yolo_model is not None, "yolo_model is required when density_source='yolo'"
            self.yolo_cache = self._build_yolo_cache(yolo_model, logger=logger)
        elif density_source == "segformer":
            assert segformer_model is not None, "segformer_model is required when density_source='segformer'"
            self.segformer_cache = self._build_segformer_cache(segformer_model, segformer_device, logger=logger)

    def _build_yolo_cache(self, yolo_model, logger=None):
        """
        Precompute YOLO predictions for every sample once. Each cache entry is
        (boxes_xywhn_cpu [N,4], confs_cpu [N]). Memory: ~1 KB per image for
        typical acne04 detection counts.
        """
        if logger is not None:
            logger.info(f"YOLO teacher precompute: {len(self.samples)} images at conf={self.yolo_conf}")
        cache = []
        for img_path, _ in self.samples:
            res = yolo_model.predict(source=img_path, conf=self.yolo_conf, verbose=False)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                boxes = res[0].boxes.xywhn.cpu().float()  # [N, 4]
                confs = res[0].boxes.conf.cpu().float()  # [N]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                confs = torch.zeros((0,), dtype=torch.float32)
            cache.append((boxes, confs))
        if logger is not None:
            total_detections = sum(c[1].shape[0] for c in cache)
            logger.info(f"YOLO teacher precompute done: {total_detections} total detections")
        return cache

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _build_segformer_cache(self, model, device, logger=None):
        """
        Precompute SegFormer-derived density targets, already downsampled to
        density_map_size and normalized so each non-empty map integrates to
        gt_count. Edge cases:
          - gt_count == 0           -> zeros map
          - gt_count > 0, mask sum ~= 0 -> fallback to gt-Gaussian (gt path)
          - otherwise               -> mask * (gt_count / mask.sum())
        """
        if logger is not None:
            logger.info(f"SegFormer teacher precompute: {len(self.samples)} images")
        cache = []
        mean = self._IMAGENET_MEAN.to(device)
        std = self._IMAGENET_STD.to(device)
        empty_fallback = 0
        zero_count = 0
        with torch.no_grad():
            for img_path, label_path in self.samples:
                # Load image as 512x512 RGB. acne04-dataset-512 is already 512;
                # PIL resize is a no-op for matching sizes, kept for safety.
                img = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                tensor = (tensor - mean) / std

                out = model(pixel_values=tensor)
                logits = F.interpolate(out.logits, size=(512, 512), mode="bilinear", align_corners=False)
                # SegFormer head is 2-channel (bg, acne); take channel 1.
                prob = torch.sigmoid(logits[:, 1:2])  # [1, 1, 512, 512]
                mask = F.adaptive_avg_pool2d(prob, (self.density_map_size, self.density_map_size))[0, 0].cpu()

                gt_boxes = parse_yolo_label_file(label_path)
                gt_count = gt_boxes.shape[0]
                if gt_count == 0:
                    density = torch.zeros_like(mask)
                    zero_count += 1
                elif mask.sum().item() <= 1e-6:
                    density = boxes_to_density_map(gt_boxes, self.density_map_size, self.sigma)
                    empty_fallback += 1
                else:
                    density = mask * (gt_count / mask.sum().item())
                cache.append(density)
        if logger is not None:
            logger.info(f"SegFormer teacher precompute done: {len(cache)} cached " f"({zero_count} gt-empty, {empty_fallback} empty-mask fallbacks)")
        return cache

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

        # Severity label always derives from GT boxes (we want to predict real severity).
        gt_boxes = parse_yolo_label_file(label_path)
        severity = self._detections_to_severity(gt_boxes.shape[0])

        if self.density_source == "yolo":
            yolo_boxes, yolo_confs = self.yolo_cache[idx]
            raw = boxes_to_density_map(
                yolo_boxes,
                self.density_map_size,
                self.sigma,
                weights=yolo_confs,
            )
            gt_count = gt_boxes.shape[0]
            rs = raw.sum().item()
            if gt_count == 0:
                density = torch.zeros_like(raw)
            elif rs <= 1e-6:
                # YOLO found nothing — fall back to gt-Gaussian for this sample.
                density = boxes_to_density_map(gt_boxes, self.density_map_size, self.sigma)
            else:
                density = raw * (gt_count / rs)
            count_value = float(gt_count)
        elif self.density_source == "segformer":
            density = self.segformer_cache[idx]
            # density is already normalized so it sums to gt_count (or zero when
            # gt_count == 0 / fallback to gt-Gaussian when mask was empty).
            count_value = float(gt_boxes.shape[0])
        else:
            density = boxes_to_density_map(gt_boxes, self.density_map_size, self.sigma)
            count_value = float(gt_boxes.shape[0])

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(severity, dtype=torch.long),
            "density_target": density,
            "count": torch.tensor(count_value, dtype=torch.float32),
        }
