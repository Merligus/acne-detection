import os
import sys
import logging
import argparse

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# make src/ importable when running this script from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from MultiTaskClassifier import MultiTaskImageClassifier
from MultiTaskAcneDataset import MultiTaskAcneDataset

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# configure logging
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_samples(split_dir):
    """
    Scan {split_dir}/images and pair each image with its YOLO label .txt.
    """
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


if __name__ == "__main__":
    # set seed for reproducibility
    torch.manual_seed(7)

    # parse arguments
    parser = argparse.ArgumentParser(description="Train the multi-task DINOv3 classifier")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset folder name (e.g. acne04-dataset).",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        default=15,
        type=int,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        default=4,
        type=int,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--model-name",
        required=False,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        type=str,
        help="Model architecture to use. One of the dinov3 versions S/S+/B/H+/7B",
    )
    parser.add_argument(
        "--learning-rate",
        required=False,
        default=5e-4,
        type=float,
        help="Training learning rate parameter.",
    )
    parser.add_argument(
        "--weight-decay",
        required=False,
        default=1e-4,
        type=float,
        help="Training weight decay parameter.",
    )
    parser.add_argument(
        "--warmup-ratio",
        required=False,
        default=0.05,
        type=float,
        help="Training warmup ratio parameter.",
    )
    parser.add_argument(
        "--eval-every-steps",
        required=False,
        default=100,
        type=int,
        help="Training evaluation steps parameter.",
    )
    parser.add_argument(
        "--density-map-size",
        required=False,
        default=56,
        type=int,
        help="Side of the (square) density map produced by the aux head.",
    )
    parser.add_argument(
        "--density-sigma",
        required=False,
        default=2.0,
        type=float,
        help="Gaussian sigma (in density-map pixels) for each lesion.",
    )
    parser.add_argument(
        "--density-loss-weight",
        required=False,
        default=20.0,
        type=float,
        help="Multiplier on the density MSE term in the composite loss.",
    )
    parser.add_argument(
        "--count-loss-weight",
        required=False,
        default=0.05,
        type=float,
        help="Multiplier on the Smooth-L1 count term in the composite loss.",
    )
    parser.add_argument(
        "--ldl-sigma",
        required=False,
        default=1.0,
        type=float,
        help="Sigma of the discrete Gaussian target in LabelDistributionLoss.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=False,
        default="./saved_models",
        type=str,
        help="Directory to save (and possibly auto-resume from) model checkpoints.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip auto-resume from any existing checkpoint in --checkpoint-dir.",
    )
    parser.add_argument(
        "--class-weights",
        required=False,
        default="",
        type=str,
        help="Comma-separated per-class multipliers for LDL (e.g. '1.0,0.8,1.6'). Empty disables.",
    )
    args = parser.parse_args()

    DATA_DIR = args.dataset_name
    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = min(2, os.cpu_count() or 2)
    EPOCHS = args.epochs
    LR = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    WARMUP_RATIO = args.warmup_ratio
    EVAL_EVERY_STEPS = args.eval_every_steps
    DENSITY_MAP_SIZE = args.density_map_size
    DENSITY_SIGMA = args.density_sigma
    DENSITY_LOSS_WEIGHT = args.density_loss_weight
    COUNT_LOSS_WEIGHT = args.count_loss_weight
    LDL_SIGMA = args.ldl_sigma
    CHECKPOINT_DIR = args.checkpoint_dir
    NO_RESUME = args.no_resume
    CLASS_WEIGHTS = [float(x) for x in args.class_weights.split(",")] if args.class_weights else None

    train_samples = build_samples(os.path.join(DATA_DIR, "train"))
    val_samples = build_samples(os.path.join(DATA_DIR, "valid"))
    test_samples = build_samples(os.path.join(DATA_DIR, "test"))

    classes = [0, 1, 2]
    num_classes = len(classes)
    logging.info(f"Loaded {len(train_samples)} train / {len(val_samples)} val / " f"{len(test_samples)} test samples across classes {classes}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dino = MultiTaskImageClassifier(
        model_name=MODEL_NAME,
        token=os.environ["HF_TOKEN"],
        num_classes=num_classes,
        classes=classes,
        checkpoint_dir=CHECKPOINT_DIR,
        logging=logging,
        density_map_size=DENSITY_MAP_SIZE,
        density_loss_weight=DENSITY_LOSS_WEIGHT,
        count_loss_weight=COUNT_LOSS_WEIGHT,
        ldl_sigma=LDL_SIGMA,
        class_weights=CLASS_WEIGHTS,
        no_resume=NO_RESUME,
    )

    train_dataset = MultiTaskAcneDataset(
        samples=train_samples,
        image_processor=dino.image_processor,
        density_map_size=DENSITY_MAP_SIZE,
        sigma=DENSITY_SIGMA,
    )
    val_dataset = MultiTaskAcneDataset(
        samples=val_samples,
        image_processor=dino.image_processor,
        density_map_size=DENSITY_MAP_SIZE,
        sigma=DENSITY_SIGMA,
    )
    test_dataset = MultiTaskAcneDataset(
        samples=test_samples,
        image_processor=dino.image_processor,
        density_map_size=DENSITY_MAP_SIZE,
        sigma=DENSITY_SIGMA,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    dino.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LR,
        eval_every_steps=EVAL_EVERY_STEPS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        checkpoint_dir=CHECKPOINT_DIR,
        classes=classes,
    )
