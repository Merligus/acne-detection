import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F  # For SegFormer upsampling
import albumentations as albu
import segmentation_models_pytorch as smp

# Make the local `dataset/` package importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import AcneDataset
from torch.utils.data import DataLoader

# Import SMP metrics functions
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score
from tqdm.auto import tqdm
import argparse
import time
import matplotlib.pyplot as plt

# --- Conditional Import for SegFormer ---
try:
    from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. SegFormer architecture will not be available.")
    print("Install it with: pip install transformers")


# --- Configuration (Defaults) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ARCHITECTURE = "SegFormer"
DEFAULT_MODEL_PATH = ""  # required via --model; no default checkpoint
DEFAULT_INPUT_TXT = os.path.join(_SCRIPT_DIR, "data", "acne04_test.txt")
DEFAULT_INPUT_IMG_DIR = "/home/merligus/acne-segmentation/data/JPEGImages/"
DEFAULT_INPUT_MASK_DIR = "/home/merligus/acne-segmentation/data/mask/"
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results")
DEFAULT_IMG_SIZE = (512, 512)
DEFAULT_UNET_ENCODER = "mobilenet_v2"
DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b1-finetuned-ade-512-512"
DEFAULT_THRESHOLD = 0.5
DEFAULT_DARKEN_FACTOR = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Evaluate segmentation model (Unet or SegFormer) on images listed in a text file.")
parser.add_argument(
    "--architecture",
    type=str,
    default=DEFAULT_MODEL_ARCHITECTURE,
    choices=["Unet", "SegFormer"],
    help=f"Model architecture to use. Default: {DEFAULT_MODEL_ARCHITECTURE}",
)
parser.add_argument(
    "--model",
    type=str,
    default=DEFAULT_MODEL_PATH,
    help=f"Path to the trained model checkpoint (.pth file). Default: {DEFAULT_MODEL_PATH}",
)
parser.add_argument(
    "--unet_encoder",
    type=str,
    default=DEFAULT_UNET_ENCODER,
    help=f"Encoder for Unet architecture (ignored if architecture=SegFormer). Default: {DEFAULT_UNET_ENCODER}",
)
parser.add_argument(
    "--segformer_model",
    type=str,
    default=DEFAULT_SEGFORMER_MODEL,
    help="Pretrained SegFormer model name from Hugging Face Hub (ignored if architecture=Unet).",
)
parser.add_argument(
    "--input_txt",
    type=str,
    default=DEFAULT_INPUT_TXT,
    help=f"Path to the text file listing input image filenames. Default: {DEFAULT_INPUT_TXT}",
)
parser.add_argument(
    "--img_dir",
    type=str,
    default=DEFAULT_INPUT_IMG_DIR,
    help=f"Base directory containing the images listed in input_txt. Default: {DEFAULT_INPUT_IMG_DIR}",
)
parser.add_argument(
    "--mask_dir",
    type=str,
    default=DEFAULT_INPUT_MASK_DIR,
    help="Base directory containing the ground truth masks. If not provided, metrics won't be calculated.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=DEFAULT_OUTPUT_DIR,
    help=f"Directory to save the highlighted output images. Default: {DEFAULT_OUTPUT_DIR}",
)
parser.add_argument(
    "--img_size",
    type=int,
    nargs=2,
    default=list(DEFAULT_IMG_SIZE),
    help=f"Target image size (height width) model was trained on. Default: {DEFAULT_IMG_SIZE}",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    help=f"Probability threshold for binarizing the segmentation mask. Default: {DEFAULT_THRESHOLD}",
)
parser.add_argument(
    "--darken",
    type=float,
    default=DEFAULT_DARKEN_FACTOR,
    help=f"Factor to darken background pixels (0.0=black, 1.0=original). Default: {DEFAULT_DARKEN_FACTOR}",
)

args = parser.parse_args()

# Check if SegFormer was chosen but transformers is not available
if args.architecture == "SegFormer" and not TRANSFORMERS_AVAILABLE:
    print("Error: SegFormer architecture chosen, but 'transformers' library is not installed.")
    exit()

# Use parsed arguments
MODEL_ARCHITECTURE = args.architecture
MODEL_PATH = args.model
INPUT_TXT_FILE = args.input_txt
INPUT_IMAGE_DIR = args.img_dir
INPUT_MASK_DIR = args.mask_dir  # Get mask directory
OUTPUT_DIR = args.output_dir
IMG_SIZE = tuple(args.img_size)
THRESHOLD = args.threshold
DARKEN_FACTOR = args.darken

# Model specific args
UNET_ENCODER = args.unet_encoder
SEGFORMER_MODEL_NAME = args.segformer_model


# --- Preprocessing / Augmentation Function ---
def get_preprocessing_transform(img_size):
    """Resize longest side, pad for inference."""
    max_size = max(img_size)
    test_transform = [
        albu.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_LINEAR),
        albu.PadIfNeeded(
            min_height=img_size[0],
            min_width=img_size[1],
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # Normalization is done separately based on architecture
    ]
    return albu.Compose(test_transform)


# --- Load Model (Conditional) ---
print(f"--- Loading Model ---")
print(f"Architecture: {MODEL_ARCHITECTURE}")
model = None
preprocessing_fn = None  # Will store the correct normalization function

try:
    if MODEL_ARCHITECTURE == "Unet":
        print(f"Unet Encoder: {UNET_ENCODER}")
        model = smp.Unet(
            encoder_name=UNET_ENCODER,
            encoder_weights=None,  # Load weights from checkpoint
            in_channels=3,
            classes=1,
        )
        # Get SMP's normalization for Unet
        preprocessing_fn = smp.encoders.get_preprocessing_fn(UNET_ENCODER, "imagenet")

    elif MODEL_ARCHITECTURE == "SegFormer":
        print(f"SegFormer Model: {SEGFORMER_MODEL_NAME}")
        id2label = {0: "background", 1: "acne"}  # Define labels for binary task
        label2id = {v: k for k, v in id2label.items()}
        model = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_MODEL_NAME,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,  # Allows loading weights even if head size differs
        )
        # For SegFormer, we'll use standard ImageNet normalization manually
        # Alternatively, load AutoImageProcessor and use it, but manual keeps changes smaller
        normalize_transform = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # Define a wrapper function to match the expected preprocessing_fn signature
        def segformer_preprocess(image, **kwargs):
            return normalize_transform(image=image)["image"]

        preprocessing_fn = segformer_preprocess

    # --- Load Checkpoint Weights ---
    print(f"Loading weights from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove DataParallel prefix if needed

    # Use strict=False for SegFormer head mismatch, maybe True for Unet resuming
    load_strict = MODEL_ARCHITECTURE == "Unet"
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=load_strict)
    if missing_keys:
        print(f"  Warning: Missing keys during load: {missing_keys}")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys during load: {unexpected_keys}")

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"Error loading model or checkpoint: {e}")
    exit()

# --- Prepare Inference Transform ---
# This applies only geometric transforms (resize/pad)
transform_geom = get_preprocessing_transform(IMG_SIZE)

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

valid_dataset = AcneDataset(
    INPUT_IMAGE_DIR,
    INPUT_MASK_DIR,
    INPUT_TXT_FILE,
    architecture=MODEL_ARCHITECTURE,
    unet_encoder=UNET_ENCODER if MODEL_ARCHITECTURE == "Unet" else None,
    unet_encoder_weights="imagenet" if MODEL_ARCHITECTURE == "Unet" else None,
    target_img_size=IMG_SIZE,
    augmentation=transform_geom,
    inference_mode=INPUT_MASK_DIR is None,  # Add inference_mode flag
)
print(f"Dataset size: {len(valid_dataset)}")

if len(valid_dataset) == 0:
    print("Error: Dataset is empty. Please check file paths and content.")
    exit()


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of images and masks,
    ensuring correct types and handling potential None items from dataset errors.
    """
    # Filter out None items that might be returned by __getitem__ on error
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Check if we're in inference mode (single tensor) or evaluation mode (tuple)
    if isinstance(batch[0], tuple):
        # Evaluation mode - handle image and mask
        images = [item[0] for item in batch]
        masks = [item[1] for item in batch]
        try:
            images_batch = torch.stack([img.float() for img in images])
            masks_batch = torch.stack([mask.float() for mask in masks])
            return images_batch, masks_batch
        except RuntimeError as e:
            print(f"Error during torch.stack in custom_collate_fn: {e}")
            return None
    else:
        # Inference mode - handle only images
        try:
            images_batch = torch.stack([img.float() for img in batch])
            return images_batch
        except RuntimeError as e:
            print(f"Error during torch.stack in custom_collate_fn: {e}")
            return None


valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=max(1, os.cpu_count() // 2),
    pin_memory=True,
    drop_last=False,
    collate_fn=custom_collate_fn,
)

# --- Process Images and Accumulate Stats ---
processed_count = 0
error_count = 0
tp_list, fp_list, fn_list, tn_list = [], [], [], []
inference_times = []
per_image_results = []  # [(filename, iou_score)]
pbar_valid = tqdm(valid_loader, desc=f"Validation", leave=False)

with torch.no_grad():  # Disable gradient calculation for inference
    for batch_idx, batch_data in enumerate(pbar_valid):
        if batch_data is None:
            print(f"Warning: Skipping validation batch {batch_idx} due to data loading error.")
            continue

        if INPUT_MASK_DIR:
            images, masks_true = batch_data
            masks_true = masks_true.to(DEVICE).float()
        else:
            images = batch_data

        images = images.to(DEVICE).float()

        # --- Conditional Forward Pass & Upsampling ---
        # Start timing
        start_time = time.time()
        if MODEL_ARCHITECTURE == "Unet":
            masks_pred_logits = model(images)
            upsampled_logits = masks_pred_logits
        elif MODEL_ARCHITECTURE == "SegFormer":
            outputs = model(pixel_values=images)
            masks_pred_logits = outputs.logits
        # End timing and store
        inference_times.append(time.time() - start_time)

        if MODEL_ARCHITECTURE == "SegFormer":
            # Upsample SegFormer logits for metrics/loss
            upsampled_logits = F.interpolate(
                masks_pred_logits,
                size=IMG_SIZE,
                mode="bilinear",
                align_corners=False,
            )
            # Handle SegFormer's potential 2-channel output
            if upsampled_logits.shape[1] == 2:
                upsampled_logits = upsampled_logits[:, 1:2, :, :]
            elif upsampled_logits.shape[1] != 1:
                print(f"Warning: Skipping metrics val batch {batch_idx}, unexpected SegFormer channels: {upsampled_logits.shape[1]}")
                continue  # Skip metrics calculation for this batch

        # Calculate metrics using upsampled logits
        masks_pred_prob = torch.sigmoid(upsampled_logits.float())  # Ensure float for sigmoid
        # Threshold probabilities -> binary masks (0 or 1)
        masks_pred_binary = (masks_pred_prob > 0.5).long()

        # Only calculate metrics if mask directory is provided
        if INPUT_MASK_DIR:
            # Ensure masks_true is integer type for metrics
            masks_true_int = masks_true.long()

            # Calculate tp, fp, fn, tn for the batch
            tp, fp, fn, tn = get_stats(masks_pred_binary, masks_true_int, mode="binary")

            # Append batch stats to lists
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)

        filename = valid_dataset.image_paths[batch_idx]
        image_orig = cv2.imread(filename)
        image_orig_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_orig.shape[:2]

        # --- Unpad/Resize Mask to Original Image Size ---
        scale = max(IMG_SIZE) / max(original_h, original_w)
        resized_h = int(original_h * scale)
        resized_w = int(original_w * scale)
        pad_h = IMG_SIZE[0] - resized_h
        pad_w = IMG_SIZE[1] - resized_w
        top_pad = pad_h // 2
        left_pad = pad_w // 2
        crop_h_start = max(0, top_pad)
        crop_w_start = max(0, left_pad)
        crop_h_end = min(IMG_SIZE[0], crop_h_start + resized_h)
        crop_w_end = min(IMG_SIZE[1], crop_w_start + resized_w)

        pred_mask_resized = masks_pred_prob.squeeze().cpu().numpy()  # (H_img_size, W_img_size)

        # Ensure cropping indices are valid
        if crop_h_start >= crop_h_end or crop_w_start >= crop_w_end:
            print(f"Warning: Invalid crop dimensions for {filename}. Skipping metric calculation.")
            pred_mask_binary = np.zeros_like(masks_true_int.squeeze().cpu().numpy())  # Create dummy mask for saving image
        else:
            pred_mask_unpadded = pred_mask_resized[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            pred_mask_orig_size = cv2.resize(
                pred_mask_unpadded,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )
            # Binarize final prediction mask
            pred_mask_binary = (pred_mask_orig_size > THRESHOLD).astype(np.uint8)  # Prediction mask at original size {0, 1}

        # --- Create Highlighted Image ---
        darkened_image = (image_orig * DARKEN_FACTOR).astype(np.uint8)
        # Use the final binary prediction mask (at original size) for highlighting
        binary_mask_3ch = cv2.cvtColor(pred_mask_binary * 255, cv2.COLOR_GRAY2BGR) // 255
        highlighted_image = np.where(binary_mask_3ch == 1, image_orig, darkened_image)

        # --- Save Highlighted Image ---
        filename = filename.split("/")[-1].split(".")[0]  # Returns "levle3_104"
        output_filename = filename + "_highlighted.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, highlighted_image)

        # Store per-image IoU if mask directory is provided
        if INPUT_MASK_DIR:
            # Calculate IoU for this image
            tp_img = tp_list[-1]
            fp_img = fp_list[-1]
            fn_img = fn_list[-1]
            tn_img = tn_list[-1]
            iou_img = iou_score(tp_img, fp_img, fn_img, tn_img, reduction="micro").item()
            # Store only filename and score, not the image
            per_image_results.append((filename, iou_img))

        processed_count += 1

print(f"\n--- Inference Complete ---")
print(f"Processed and saved images: {processed_count}")
print(f"Skipped/Errors: {error_count}")
print(f"Average inference time: {(sum(inference_times) / len(inference_times)):.4f} seconds")
print(f"Highlighted outputs saved to: {OUTPUT_DIR}")

# --- Calculate and Print Overall Metrics ---
# Check if any stats were accumulated (i.e., processing didn't fail for all images)
if INPUT_MASK_DIR and tp_list:
    tp_total = torch.cat(tp_list).sum()
    fp_total = torch.cat(fp_list).sum()
    fn_total = torch.cat(fn_list).sum()
    tn_total = torch.cat(tn_list).sum()  # Although tn is not used in IoU/F1 for binary

    # Calculate overall IoU (Jaccard) and F1 (Dice) Score
    overall_iou = iou_score(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()
    overall_f1 = f1_score(tp_total, fp_total, fn_total, tn_total, reduction="micro").item()

    # Plot best and worst performing images
    if per_image_results:
        # Sort by IoU score
        sorted_results = sorted(per_image_results, key=lambda x: x[1], reverse=True)
        best_3 = sorted_results[:3]
        worst_3 = sorted_results[-3:]

        plt.figure(figsize=(15, 8))
        plt.suptitle("Best and Worst Predictions by IoU", fontsize=16)

        def plot_result(subplot_idx, fname, iou, best_or_worst):
            # Load and process image only when needed for plotting
            img_path = os.path.join(OUTPUT_DIR, f"{fname}_highlighted.png")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load {img_path}")
                return
            plt.subplot(2, 3, subplot_idx)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{best_or_worst} #{subplot_idx if subplot_idx <= 3 else subplot_idx-3}\nIoU: {iou:.4f}")
            plt.axis("off")

        # Plot in batches to manage memory
        for i, (fname, iou) in enumerate(best_3, 1):
            plot_result(i, fname, iou, "Best")
            plt.gcf().canvas.flush_events()  # Force update

        for i, (fname, iou) in enumerate(worst_3, 1):
            plot_result(i + 3, fname, iou, "Worst")
            plt.gcf().canvas.flush_events()  # Force update

        if MODEL_ARCHITECTURE == "Unet":
            MODEL_NAME_FOR_SAVE = f"unet_{UNET_ENCODER}"
        elif MODEL_ARCHITECTURE == "SegFormer":
            MODEL_NAME = args.segformer_model
            MODEL_NAME_FOR_SAVE = f"segformer_{MODEL_NAME.replace('/', '_')}"

        output_filename = f"best_worst_predictions_{MODEL_NAME_FOR_SAVE}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
        plt.close("all")  # Explicitly close all figures
        plt.clf()  # Clear current figure
        print(f"\nBest/worst predictions visualization saved as {output_filename}")

    print(f"\n--- Evaluation Metrics (Overall for {len(tp_list)} images) ---")  # Indicate how many images contributed to metrics
    print(f"IoU Score (Jaccard): {overall_iou:.4f}")
    print(f"F1 Score (Dice):     {overall_f1:.4f}")
    print(f"------------------------------------")
elif not INPUT_MASK_DIR:
    print("\nNo mask directory provided - metrics were not calculated.")
else:
    print("\nNo images were successfully processed for metric calculation.")
