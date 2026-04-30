"""
Letterbox-resize a YOLO dataset to a fixed square size and rewrite YOLO
label files so the boxes still point at the correct pixels.

Each split (train/valid/test) is processed image-by-image:
  - the image is scaled by `size / max(W, H)` so the longer side hits `size`
  - the shorter side is padded with `fill` to make the image square
  - YOLO normalized (cx, cy, w, h) coordinates are remapped into the new
    letterboxed canvas (still in [0, 1])

Top-level non-split files (data.yaml, READMEs, ...) are copied as-is. YOLO
caches such as `labels.cache` are not copied; they will be regenerated.
"""

import os
import shutil
import argparse

from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")


def letterbox(image: Image.Image, size: int, fill=(0, 0, 0)):
    w, h = image.size
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    padded = Image.new("RGB", (size, size), fill)
    padded.paste(resized, (pad_left, pad_top))
    return padded, scale, pad_left, pad_top


def transform_label_line(line: str, size: int, scale: float, pad_left: int, pad_top: int, orig_w: int, orig_h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    new_cx = (cx * orig_w * scale + pad_left) / size
    new_cy = (cy * orig_h * scale + pad_top) / size
    new_w_box = (w * orig_w * scale) / size
    new_h_box = (h * orig_h * scale) / size
    return f"{cls} {new_cx:.6f} {new_cy:.6f} {new_w_box:.6f} {new_h_box:.6f}"


def process_split(src_split: str, dst_split: str, size: int):
    src_images = os.path.join(src_split, "images")
    src_labels = os.path.join(src_split, "labels")
    dst_images = os.path.join(dst_split, "images")
    dst_labels = os.path.join(dst_split, "labels")
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    n_imgs = 0
    n_labels = 0
    for fname in sorted(os.listdir(src_images)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue

        src_img = os.path.join(src_images, fname)
        with Image.open(src_img) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            padded, scale, pad_left, pad_top = letterbox(image, size=size)
        padded.save(os.path.join(dst_images, fname), quality=95)
        n_imgs += 1

        stem = os.path.splitext(fname)[0]
        src_lbl = os.path.join(src_labels, stem + ".txt")
        dst_lbl = os.path.join(dst_labels, stem + ".txt")
        if not os.path.isfile(src_lbl):
            continue

        with open(src_lbl, "r") as f:
            src_lines = f.readlines()
        new_lines = []
        for line in src_lines:
            transformed = transform_label_line(line, size, scale, pad_left, pad_top, orig_w, orig_h)
            if transformed is not None:
                new_lines.append(transformed)
        with open(dst_lbl, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
        n_labels += 1

    return n_imgs, n_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source dataset folder (containing train/valid/test).")
    parser.add_argument("--dst", required=True, help="Destination dataset folder. Must not exist.")
    parser.add_argument("--size", type=int, default=512, help="Square output size in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing destination.")
    args = parser.parse_args()

    if os.path.exists(args.dst) and not args.overwrite:
        raise SystemExit(f"Destination already exists: {args.dst} (use --overwrite to write into it)")

    os.makedirs(args.dst, exist_ok=True)

    for entry in sorted(os.listdir(args.src)):
        sp = os.path.join(args.src, entry)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(args.dst, entry))

    for split in SPLITS:
        sp = os.path.join(args.src, split)
        if not os.path.isdir(sp):
            continue
        dp = os.path.join(args.dst, split)
        n_imgs, n_labels = process_split(sp, dp, args.size)
        print(f"{split}: wrote {n_imgs} images, {n_labels} label files")

    print(f"Done. Output at {args.dst}")


if __name__ == "__main__":
    main()
