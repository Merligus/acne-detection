"""
Multi-task DINOv3 classifier for ACNE04 / Hayashi grading.

Main task : severity classification (CrossEntropy on the CLS token).
Aux task  : per-pixel density map predicted from the patch tokens, supervised
            by Gaussians placed at the YOLO box centers. Sum of the density
            map approximates the lesion count, and is also pushed toward the
            true count with a Smooth-L1 term.

The shared backbone is forced to learn features that are simultaneously useful
for "what's the severity?" and "where/how many lesions are there?" -- the same
inductive bias Hayashi's rule encodes by hand.

Drop-in replacement for the original ImageClassifier; same train/evaluate/infer
surface, plus a `density_target` and `count` field on each batch.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
)

# ---------------------------------------------------------------------------
# Heads + multi-task model
# ---------------------------------------------------------------------------


class DensityHead(nn.Module):
    """
    Conv decoder: patch tokens [B, N, D] -> density map [B, target, target].
    The patch grid side is inferred from N at forward time, so this head
    works with any DINOv3 input resolution.
    """

    def __init__(self, hidden_size: int, target_size: int = 56):
        super().__init__()
        self.target_size = target_size

        self.proj = nn.Conv2d(hidden_size, 128, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = patch_tokens.shape
        side = int(round(math.sqrt(N)))
        assert side * side == N, f"non-square patch grid (N={N})"
        x = patch_tokens.transpose(1, 2).reshape(B, D, side, side)

        x = self.proj(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = self.out(x)
        x = F.relu(x)  # density >= 0

        if x.shape[-1] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return x.squeeze(1)  # [B, target, target]


class MultiTaskDinoV3(nn.Module):
    def __init__(
        self,
        backbone: AutoModel,
        num_classes: int,
        freeze_backbone: bool = True,
        density_map_size: int = 56,
    ):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden_size = getattr(backbone.config, "hidden_size", None)
        self.cls_head = nn.Linear(hidden_size, num_classes)
        self.density_head = DensityHead(hidden_size, target_size=density_map_size)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, 1+N, D]
        cls_token = last_hidden[:, 0]
        patch_tokens = last_hidden[:, 1:]

        logits = self.cls_head(cls_token)
        density = self.density_head(patch_tokens)
        return logits, density
