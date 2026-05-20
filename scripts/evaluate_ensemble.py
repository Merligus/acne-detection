"""
Evaluate a (MultiTaskImageClassifier + YOLO) ensemble on a YOLO-format
dataset's test split.

Reports per-branch metrics so we can see which signal each model contributes:
    cls_acc        — classifier-only severity accuracy
    yolo_sev_acc   — YOLO-count -> severity bucket accuracy
    ensemble_acc   — fused severity accuracy (alpha * cls + (1-alpha) * yolo)
    cls_count_mae  — classifier density-sum count MAE (usually broken)
    yolo_count_mae — YOLO count MAE
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from EnsembleInfer import MultiTaskYOLOEnsemble

load_dotenv()

logging.basicConfig(
    filename="ensemble_eval.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_samples(split_dir):
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    samples = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        img_path = os.path.join(images_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        samples.append((img_path, label_path))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="Dataset folder, e.g. acne04-dataset-512.")
    parser.add_argument("--split", required=False, default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--mt-ckpt",
        required=True,
        help="Path to the multi-task checkpoint .pt file (the file saved by train_multitask.py).",
    )
    parser.add_argument(
        "--mt-model-name",
        required=False,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HF model name used when training the multi-task checkpoint.",
    )
    parser.add_argument(
        "--yolo-weights",
        required=True,
        help="Path to YOLO best.pt (e.g. runs/detect/Acne_Detection/yolo26s7/weights/best.pt).",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Severity-fusion weight on the classifier (0..1).")
    parser.add_argument("--sigma-count", type=float, default=0.6, help="Spread of count-derived severity probs.")
    args = parser.parse_args()

    samples = build_samples(os.path.join(args.dataset_name, args.split))
    logging.info(f"Loaded {len(samples)} samples from {args.dataset_name}/{args.split} " f"mt_ckpt={args.mt_ckpt} yolo={args.yolo_weights} " f"alpha={args.alpha} yolo_conf={args.yolo_conf}")

    ensemble = MultiTaskYOLOEnsemble(
        mt_ckpt_path=args.mt_ckpt,
        mt_model_name=args.mt_model_name,
        hf_token=os.environ.get("HF_TOKEN", ""),
        yolo_weights_path=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        alpha=args.alpha,
        sigma_count=args.sigma_count,
    )

    metrics = ensemble.evaluate(samples)
    msg = f"--- ENSEMBLE EVAL ({args.split}, n={metrics['n']}) ---\n" f"  cls_acc           {metrics['cls_acc']*100:.2f}%\n" f"  yolo_sev_acc      {metrics['yolo_sev_acc']*100:.2f}%\n" f"  ensemble_acc      {metrics['ensemble_acc']*100:.2f}%\n" f"  cls_count_mae     {metrics['cls_count_mae']:.2f}\n" f"  yolo_count_mae    {metrics['yolo_count_mae']:.2f}"
    logging.info(msg)
    print(msg)


if __name__ == "__main__":
    main()
