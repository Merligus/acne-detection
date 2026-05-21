"""
Test-time augmentation (TTA) inference for a trained MultiTaskImageClassifier.

For each image in the chosen split, run the model 4 times (identity, hflip,
rotation +5°, rotation -5°), mean the softmax probs for classification, and
mean the integrated count for the count metric. Reports both no-TTA and TTA
metrics side-by-side so the gain is directly visible.

Why this is safe: counts are scalar sums over the density map — invariant to
spatial transforms. Cls softmax probs are averaged after each forward; the
fusion is a standard test-time ensemble.
"""

import os
import sys
import math
import argparse
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from MultiTaskClassifier import MultiTaskImageClassifier
from MultiTaskAcneDataset import parse_yolo_label_file, MultiTaskAcneDataset

load_dotenv()

logging.basicConfig(
    filename="tta_eval.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_samples(split_dir: str) -> List[Tuple[str, str]]:
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    samples = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        samples.append((os.path.join(images_dir, fname), os.path.join(labels_dir, stem + ".txt")))
    return samples


def tta_variants(image: Image.Image):
    """Yield 4 TTA-augmented copies: identity, hflip, rot+5, rot-5."""
    yield image
    yield TF.hflip(image)
    yield TF.rotate(image, 5.0, fill=0)
    yield TF.rotate(image, -5.0, fill=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--mt-ckpt", required=True)
    parser.add_argument(
        "--mt-model-name",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = MultiTaskImageClassifier(
        model_name=args.mt_model_name,
        token=os.environ.get("HF_TOKEN", ""),
        num_classes=3,
        classes=[0, 1, 2],
        ckpt_path=args.mt_ckpt,
        no_resume=False,
    )
    mt.model.eval()

    samples = build_samples(os.path.join(args.dataset_name, args.split))
    logging.info(f"Loaded {len(samples)} samples from {args.dataset_name}/{args.split}; ckpt={args.mt_ckpt}")

    n = 0
    notta_correct = tta_correct = 0
    notta_mae = tta_mae = 0.0
    notta_sq = tta_sq = 0.0

    autocast_enabled = torch.cuda.is_available()
    for img_path, label_path in samples:
        gt_boxes = parse_yolo_label_file(label_path)
        gt_count = int(gt_boxes.shape[0])
        gt_sev = MultiTaskAcneDataset._detections_to_severity(gt_count)
        image = Image.open(img_path).convert("RGB")

        # Collect predictions from each TTA variant.
        cls_probs_list = []
        count_list = []
        with torch.no_grad():
            for variant in tta_variants(image):
                inputs = mt.image_processor(images=variant, return_tensors="pt").to(device)
                with torch.amp.autocast("cuda", enabled=autocast_enabled):
                    logits, density = mt.model(inputs["pixel_values"])
                probs = torch.softmax(logits, dim=-1).squeeze(0).float().cpu().numpy()
                cls_probs_list.append(probs)
                count_list.append(float(density.flatten(1).sum(dim=1).item()))

        # No-TTA = identity-only (first variant).
        notta_probs = cls_probs_list[0]
        notta_count = count_list[0]
        notta_sev_pred = int(np.argmax(notta_probs))

        # TTA = mean over all 4.
        tta_probs = np.mean(cls_probs_list, axis=0)
        tta_count = float(np.mean(count_list))
        tta_sev_pred = int(np.argmax(tta_probs))

        n += 1
        notta_correct += int(notta_sev_pred == gt_sev)
        tta_correct += int(tta_sev_pred == gt_sev)
        notta_mae += abs(notta_count - gt_count)
        tta_mae += abs(tta_count - gt_count)
        notta_sq += (notta_count - gt_count) ** 2
        tta_sq += (tta_count - gt_count) ** 2

    notta_acc = notta_correct / max(n, 1)
    tta_acc = tta_correct / max(n, 1)
    notta_rmse = math.sqrt(notta_sq / max(n, 1))
    tta_rmse = math.sqrt(tta_sq / max(n, 1))
    notta_mae /= max(n, 1)
    tta_mae /= max(n, 1)

    msg = (
        f"--- TTA EVAL ({args.split}, n={n}) ---\n"
        f"               test_acc  test_mae  test_rmse\n"
        f"  no-TTA       {notta_acc*100:6.2f}%   {notta_mae:6.2f}   {notta_rmse:6.2f}\n"
        f"  TTA (x4)     {tta_acc*100:6.2f}%   {tta_mae:6.2f}   {tta_rmse:6.2f}\n"
        f"  delta        {(tta_acc-notta_acc)*100:+6.2f}%   {tta_mae-notta_mae:+6.2f}   {tta_rmse-notta_rmse:+6.2f}"
    )
    logging.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
