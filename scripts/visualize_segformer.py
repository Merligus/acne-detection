"""
Run a trained SegFormer-B1 acne-segmentation checkpoint on a handful of
images from a YOLO-format dataset split and write side-by-side overlays.

Used as the visual gate before launching the SegFormer-as-density-teacher
sweep — if the predicted masks look like noise on these images, the rest
of Phase 5 is moot.
"""

import os
import sys
import random
import argparse

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_segformer(base_model: str, weights_path: str, device: str):
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "background", 1: "acne"},
        label2id={"background": 0, "acne": 1},
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(device).eval()


def predict_mask(model, image: Image.Image, device: str, img_size: int = 512) -> np.ndarray:
    img = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        logits = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits[0, 1]).cpu().numpy()  # [H, W] in [0, 1]
    return prob


def overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a red heatmap of `mask` (values in [0, 1]) on a BGR image."""
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.zeros_like(image_bgr)
    heat[..., 2] = (mask * 255).astype(np.uint8)  # red channel
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heat, alpha, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segformer-weights", required=True, help="Path to SegFormer state_dict .pth file.")
    parser.add_argument("--segformer-base-model", default="nvidia/segformer-b1-finetuned-ade-512-512")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=".", help="Where to write check_seg_*.jpg files.")
    args = parser.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_segformer(args.segformer_base_model, args.segformer_weights, device)

    images_dir = os.path.join(args.dataset_name, args.split, "images")
    candidates = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(IMAGE_EXTS)]
    picked = random.sample(candidates, min(args.n, len(candidates)))

    os.makedirs(args.output_dir, exist_ok=True)
    for fname in picked:
        img_path = os.path.join(images_dir, fname)
        image = Image.open(img_path).convert("RGB")
        mask = predict_mask(model, image, device)

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (mask.shape[1], mask.shape[0]))
        out = overlay(bgr, mask)

        # also write the raw mask intensity as a side panel
        panel = np.concatenate([bgr, out], axis=1)
        out_path = os.path.join(args.output_dir, f"check_seg_{os.path.splitext(fname)[0]}.jpg")
        cv2.imwrite(out_path, panel)
        print(f"wrote {out_path}  mask_sum={mask.sum():.1f}  mask_max={mask.max():.3f}")


if __name__ == "__main__":
    sys.exit(main())
