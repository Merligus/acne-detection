"""
Phase 8 / A: compare each acne04 bbox annotation against the matching seg mask.

For every image in acne04-dataset-512 train+valid+test that has a matching seg
mask (via the `_jpg.rf.<hash>` stem-stripping), report whether the box set and
the mask's connected components describe the same lesions:

  Sample is "equal" iff every mask component has at least one pixel inside a
  YOLO box, AND every YOLO box contains at least one mask pixel.

Outputs: console summary, compare_acne04_vs_seg.log, compare_acne04_vs_seg.csv.

If pass_rate >= 90% the user can skip the Phase B/C retrain; below 90% means
the acne04 boxes and the seg masks disagree on enough samples to justify
retraining YOLO on mask-derived labels.
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
    """Same letterbox transform as scripts/letterbox_dataset.py but NEAREST."""
    h, w = mask.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size, size), dtype=mask.dtype)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    out[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return out


def parse_yolo_labels(path: str):
    """Return [N, 4] array of (x1, y1, x2, y2) in 512-pixel coords."""
    if not os.path.isfile(path):
        return np.zeros((0, 4), dtype=np.int32)
    out = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = float(cx) * SIZE, float(cy) * SIZE, float(w) * SIZE, float(h) * SIZE
            x1 = max(0, int(round(cx - w / 2)))
            y1 = max(0, int(round(cy - h / 2)))
            x2 = min(SIZE - 1, int(round(cx + w / 2)))
            y2 = min(SIZE - 1, int(round(cy + h / 2)))
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
    return np.array(out, dtype=np.int32) if out else np.zeros((0, 4), dtype=np.int32)


def compare_pair(label_path: str, mask_path: str):
    """Return dict with the per-sample metrics for one (image, mask) pair."""
    boxes = parse_yolo_labels(label_path)
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        return None
    mask = letterbox_mask(mask_raw)
    binary = (mask > 127).astype(np.uint8)

    # Mask used for "is there any mask pixel here" tests
    n_labels, _comp_lbl_map, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)
    # stats[0] is background; rest are components
    n_components = max(0, n_labels - 1)

    n_boxes = boxes.shape[0]

    # Component -> matched if any pixel inside any box.
    # Box -> matched if any mask pixel inside it.
    matched_components = 0
    matched_boxes = 0

    if n_components > 0 and n_boxes > 0:
        # Per-box: slice the binary mask and check any-mask-pixel.
        for x1, y1, x2, y2 in boxes:
            sub = binary[y1 : y2 + 1, x1 : x2 + 1]
            if sub.size > 0 and sub.any():
                matched_boxes += 1

        # Per-component: check if component's bbox overlaps any YOLO box, and
        # whether at least one component pixel falls inside that overlap.
        for comp_id in range(1, n_labels):
            cx0, cy0, cw, ch, _area = stats[comp_id]
            cx1, cy1 = cx0 + cw, cy0 + ch  # exclusive
            # Component bbox: [cx0, cy0, cx1, cy1)
            for x1, y1, x2, y2 in boxes:
                ix0 = max(cx0, x1)
                iy0 = max(cy0, y1)
                ix1 = min(cx1, x2 + 1)
                iy1 = min(cy1, y2 + 1)
                if ix0 < ix1 and iy0 < iy1:
                    # Bounding rects overlap; check if any component pixel is there.
                    region_comp = _comp_lbl_map[iy0:iy1, ix0:ix1] == comp_id
                    if region_comp.any():
                        matched_components += 1
                        break

    equal = matched_components == n_components and matched_boxes == n_boxes
    return {
        "n_boxes": n_boxes,
        "n_components": n_components,
        "matched_boxes": matched_boxes,
        "matched_components": matched_components,
        "box_recall": (matched_boxes / n_boxes) if n_boxes else 1.0,
        "comp_recall": (matched_components / n_components) if n_components else 1.0,
        "equal": equal,
    }


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", default="compare_acne04_vs_seg.csv")
    parser.add_argument("--log-out", default="compare_acne04_vs_seg.log")
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_out,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    n_total, n_equal = 0, 0
    sum_box_recall, sum_comp_recall = 0.0, 0.0
    pair_hist = Counter()
    rows = [("split", "image", "n_boxes", "n_components", "matched_boxes", "matched_components", "box_recall", "comp_recall", "equal")]

    for split, fname, label_path, mask_path in iter_acne04_pairs():
        r = compare_pair(label_path, mask_path)
        if r is None:
            continue
        n_total += 1
        if r["equal"]:
            n_equal += 1
        sum_box_recall += r["box_recall"]
        sum_comp_recall += r["comp_recall"]
        pair_hist[(r["n_boxes"], r["n_components"])] += 1
        rows.append((split, fname, r["n_boxes"], r["n_components"], r["matched_boxes"], r["matched_components"], f"{r['box_recall']:.3f}", f"{r['comp_recall']:.3f}", int(r["equal"])))

    with open(args.csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    pass_rate = n_equal / max(n_total, 1) * 100
    avg_box_recall = sum_box_recall / max(n_total, 1)
    avg_comp_recall = sum_comp_recall / max(n_total, 1)

    summary = f"compared n={n_total}  equal={n_equal}  pass_rate={pass_rate:.2f}%  " f"avg_box_recall={avg_box_recall:.3f}  avg_comp_recall={avg_comp_recall:.3f}"
    logging.info(summary)
    print(summary)

    top_diffs = sorted(pair_hist.items(), key=lambda kv: -kv[1])[:10]
    msg = "Top (n_boxes, n_components) pairs:\n" + "\n".join(f"  ({nb}, {nc}): {ct}" for (nb, nc), ct in top_diffs)
    logging.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
