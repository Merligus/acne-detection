"""
Phase 8 / A2: IoU comparison of acne04 bbox annotations vs seg pixel masks.

For every (acne04 image, seg mask) pair found via the `_jpg.rf.<hash>` stem
mapping, build a `bbox_mask` (filled rectangles at the YOLO boxes) and a
`seg_mask` (letterboxed binary mask) on the same 512x512 canvas, then compute:

  iou = |bbox & seg| / |bbox | seg|
  seg_in_bbox = |bbox & seg| / |seg|              # mask pixels inside any bbox
  bbox_filled_by_seg = |bbox & seg| / |bbox|      # bbox pixels also in mask
  area_ratio = |seg| / |bbox|                     # mask-area vs bbox-area

Aggregates mean / median / std across the 1419 paired stems, per-split
breakdowns, and an IoU histogram. Diagnostic only — does not modify any
training assets.
"""

import os
import re
import csv
import logging
import argparse
from collections import Counter

import numpy as np
import cv2

ACNE04_ROOT = "/home/merligus/AcneDetectionAnalysis/acne04-dataset-512"
SEG_MASK_DIR = "/home/merligus/acne-segmentation/data/mask"
SUFFIX_RE = re.compile(r"_jpg\.rf\.[A-Za-z0-9]+\.jpg$")
SIZE = 512


def to_seg_stem(roboflow_name: str) -> str:
    return SUFFIX_RE.sub(".jpg", roboflow_name)


def letterbox_mask(mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """NEAREST-interp letterbox to a square canvas, zero pad. Matches the
    transform used in scripts/letterbox_dataset.py."""
    h, w = mask.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size, size), dtype=mask.dtype)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    out[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return out


def bbox_mask_from_yolo(label_path: str, size: int = SIZE) -> np.ndarray:
    """Return a binary 512x512 mask with 1s inside each YOLO box."""
    out = np.zeros((size, size), dtype=np.uint8)
    if not os.path.isfile(label_path):
        return out
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = float(cx) * size, float(cy) * size, float(w) * size, float(h) * size
            x1 = max(0, int(round(cx - w / 2)))
            y1 = max(0, int(round(cy - h / 2)))
            x2 = min(size - 1, int(round(cx + w / 2)))
            y2 = min(size - 1, int(round(cy + h / 2)))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), color=1, thickness=-1)
    return out


def iter_acne04_pairs():
    for split in ("train", "valid", "test"):
        images_dir = os.path.join(ACNE04_ROOT, split, "images")
        labels_dir = os.path.join(ACNE04_ROOT, split, "labels")
        if not os.path.isdir(images_dir):
            continue
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            label_path = os.path.join(labels_dir, stem + ".txt")
            seg_name = to_seg_stem(fname)
            mask_path = os.path.join(SEG_MASK_DIR, seg_name)
            if os.path.isfile(mask_path):
                yield split, fname, label_path, mask_path


def compute_metrics(label_path: str, mask_path: str):
    bbox_mask = bbox_mask_from_yolo(label_path)
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None
    seg_mask = (letterbox_mask(mask_raw) > 127).astype(np.uint8)

    bbox_sum = int(bbox_mask.sum())
    seg_sum = int(seg_mask.sum())
    intersection = int((bbox_mask & seg_mask).sum())
    union = int((bbox_mask | seg_mask).sum())

    has_boxes = bbox_sum > 0
    has_mask = seg_sum > 0
    iou = intersection / union if union else None
    seg_in_bbox = intersection / seg_sum if has_mask else None
    bbox_filled_by_seg = intersection / bbox_sum if has_boxes else None
    area_ratio = seg_sum / bbox_sum if has_boxes else None

    return {
        "bbox_sum": bbox_sum,
        "seg_sum": seg_sum,
        "intersection": intersection,
        "union": union,
        "iou": iou,
        "seg_in_bbox": seg_in_bbox,
        "bbox_filled_by_seg": bbox_filled_by_seg,
        "area_ratio": area_ratio,
        "has_boxes": has_boxes,
        "has_mask": has_mask,
    }


def stats(values):
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}


def fmt_stats(name, s):
    if s["n"] == 0:
        return f"  {name:<22} n=0 (no usable samples)"
    return f"  {name:<22} n={s['n']:4d}  mean={s['mean']:.4f}  median={s['median']:.4f}  std={s['std']:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", default="compare_acne04_vs_seg_iou.csv")
    parser.add_argument("--log-out", default="compare_acne04_vs_seg_iou.log")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_out,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    per_split = {"train": [], "valid": [], "test": []}
    rows = [("split", "image", "n_bbox_px", "n_seg_px", "iou", "seg_in_bbox", "bbox_filled_by_seg", "area_ratio")]
    empties = Counter()

    for split, fname, label_path, mask_path in iter_acne04_pairs():
        r = compute_metrics(label_path, mask_path)
        if r is None:
            continue
        if not r["has_boxes"] and not r["has_mask"]:
            empties["both_empty"] += 1
        elif not r["has_boxes"]:
            empties["no_boxes"] += 1
        elif not r["has_mask"]:
            empties["no_mask"] += 1
        per_split[split].append(r)
        rows.append((split, fname, r["bbox_sum"], r["seg_sum"], "" if r["iou"] is None else f"{r['iou']:.4f}", "" if r["seg_in_bbox"] is None else f"{r['seg_in_bbox']:.4f}", "" if r["bbox_filled_by_seg"] is None else f"{r['bbox_filled_by_seg']:.4f}", "" if r["area_ratio"] is None else f"{r['area_ratio']:.4f}"))

    with open(args.csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    all_results = per_split["train"] + per_split["valid"] + per_split["test"]

    def report_block(name, results):
        ious = [r["iou"] for r in results]
        sib = [r["seg_in_bbox"] for r in results]
        bfs = [r["bbox_filled_by_seg"] for r in results]
        ar = [r["area_ratio"] for r in results]
        lines = [
            f"== {name} == (n={len(results)})",
            fmt_stats("iou", stats(ious)),
            fmt_stats("seg_in_bbox", stats(sib)),
            fmt_stats("bbox_filled_by_seg", stats(bfs)),
            fmt_stats("area_ratio", stats(ar)),
        ]
        return "\n".join(lines)

    out_blocks = [
        report_block("OVERALL", all_results),
        report_block("train", per_split["train"]),
        report_block("valid", per_split["valid"]),
        report_block("test", per_split["test"]),
    ]

    # IoU histogram
    iou_vals = np.array([r["iou"] for r in all_results if r["iou"] is not None], dtype=np.float64)
    hist, edges = np.histogram(iou_vals, bins=np.linspace(0, 1, 11))
    hist_lines = ["IoU histogram (overall):"]
    for i, c in enumerate(hist):
        hist_lines.append(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}): {c:4d}")
    out_blocks.append("\n".join(hist_lines))

    out_blocks.append(f"Edge cases: {dict(empties)}")

    summary = "\n\n".join(out_blocks)
    logging.info("\n" + summary)
    print(summary)


if __name__ == "__main__":
    main()
