"""
Generate acne04-aligned, leak-free split lists for the SegFormer retrain.

For each acne04-dataset-512 split (train/valid/test), iterate the image
filenames, strip the Roboflow `_jpg.rf.<hash>` suffix to recover the original
"levle*_<N>.jpg" stem, and keep stems that have a corresponding mask under
~/acne-segmentation/data/mask/. Write the surviving stems to
acne-segmentation/data/acne04_<split>.txt.

The output text files match the format `with_mask_train.txt` expects in the
sibling segmentation repo: one `<stem>.jpg` per line.
"""

import os
import re

ACNE04_ROOT = "/home/merligus/AcneDetectionAnalysis/acne04-dataset-512"
MASKS_DIR = "/home/merligus/acne-segmentation/data/mask"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "data")

SUFFIX_RE = re.compile(r"_jpg\.rf\.[A-Za-z0-9]+\.jpg$")


def to_seg_stem(roboflow_name: str) -> str:
    """`levle0_44_jpg.rf.<hash>.jpg` -> `levle0_44.jpg`."""
    return SUFFIX_RE.sub(".jpg", roboflow_name)


def build_split(split: str) -> tuple:
    images_dir = os.path.join(ACNE04_ROOT, split, "images")
    seen, kept, dropped = set(), [], []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem_name = to_seg_stem(fname)
        if stem_name in seen:
            continue
        seen.add(stem_name)
        mask_path = os.path.join(MASKS_DIR, stem_name)
        if os.path.isfile(mask_path):
            kept.append(stem_name)
        else:
            dropped.append(stem_name)
    return kept, dropped


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ("train", "valid", "test"):
        kept, dropped = build_split(split)
        out_path = os.path.join(OUT_DIR, f"acne04_{split}.txt")
        with open(out_path, "w") as f:
            for stem in kept:
                f.write(f"{stem}\n")
        print(f"{split:5s}: kept={len(kept):4d}  dropped(no-mask)={len(dropped):3d}  -> {out_path}")


if __name__ == "__main__":
    main()
