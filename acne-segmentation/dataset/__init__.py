import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import segmentation_models_pytorch as smp


# --- Dataset Class ---
class AcneDataset(BaseDataset):
    """
    Reads images and masks based on filenames listed in a text file.
    Handles varying image sizes and ensures binary masks (0/1).
    Applies preprocessing based on the chosen model architecture.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        file_list_path,
        architecture,  # Pass the architecture choice
        unet_encoder=None,  # Needed for SMP preprocessing
        unet_encoder_weights=None,  # Needed for SMP preprocessing
        target_img_size=(256, 256),
        augmentation=None,
        inference_mode=False,
    ):
        self.image_paths = []
        self.mask_paths = []
        self.architecture = architecture
        self.target_img_size = target_img_size
        self.inference_mode = inference_mode
        skipped_files = 0

        print(f"Reading file list from: {file_list_path}")
        try:
            ext = ".jpg"
            with open(file_list_path, "r") as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    filename = parts[0]

                    img_path = os.path.join(images_dir, filename)

                    if self.inference_mode:
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)
                        else:
                            skipped_files += 1
                    else:
                        base_name = os.path.splitext(filename)[0]
                        mask_filename = base_name + ext
                        mask_path = os.path.join(masks_dir, mask_filename)

                        if os.path.exists(img_path) and os.path.exists(mask_path):
                            self.image_paths.append(img_path)
                            self.mask_paths.append(mask_path)
                        else:
                            skipped_files += 1
        except FileNotFoundError:
            print(f"Error: File not found at {file_list_path}")
            raise
        if not self.image_paths:
            raise RuntimeError(f"No valid image/mask pairs found from {file_list_path}.")
        if skipped_files > 0:
            print(f"Warning: Skipped {skipped_files} files (missing image or mask).")

        self.augmentation = augmentation

        # --- Conditional Preprocessing Setup ---
        if self.architecture == "Unet":
            if unet_encoder is None or unet_encoder_weights is None:
                raise ValueError("Unet encoder and weights must be provided for Unet architecture.")
            self.preprocessing_fn = smp.encoders.get_preprocessing_fn(unet_encoder, unet_encoder_weights)
        elif self.architecture == "SegFormer":
            # Use standard ImageNet normalization, applied manually after augmentation
            self.normalize_transform = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            # Fallback or error if needed
            self.normalize_transform = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            print(f"Warning: Unknown architecture '{self.architecture}' in Dataset. Using default ImageNet normalization.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        # Read image
        try:
            image = cv2.imread(self.image_paths[i])
            if image is None:
                raise IOError(f"Could not read image: {self.image_paths[i]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Only read mask if not in inference mode
            if not self.inference_mode:
                mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise IOError(f"Could not read mask: {self.mask_paths[i]}")

                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            # Apply augmentations
            if self.augmentation:
                if self.inference_mode:
                    sample = self.augmentation(image=image)
                    image = sample["image"]
                else:
                    sample = self.augmentation(image=image, mask=mask)
                    image, mask = sample["image"], sample["mask"]

            # --- Apply Conditional Preprocessing ---
            if self.architecture == "Unet":
                image = self.preprocessing_fn(image)  # Apply SMP normalization
                image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW tensor
            elif self.architecture == "SegFormer":
                image = self.normalize_transform(image=image)["image"]  # Apply manual normalization
                image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW tensor
            else:  # Default/Fallback
                image = self.normalize_transform(image=image)["image"]
                image = torch.from_numpy(image).permute(2, 0, 1).float()

            if self.inference_mode:
                return image

            # Mask to Tensor (common step)
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)  # 1, H, W
            mask = torch.from_numpy(mask).float()

        except Exception as e:
            # print(f"Error loading item {i}: Image: {self.image_paths[i]}, Mask: {self.mask_paths[i]}, Error: {e}")
            dummy_image = torch.zeros(
                (3, self.target_img_size[0], self.target_img_size[1]),
                dtype=torch.float32,
            )
            if self.inference_mode:
                return dummy_image

            dummy_mask = torch.zeros(
                (1, self.target_img_size[0], self.target_img_size[1]),
                dtype=torch.float32,
            )
            return dummy_image, dummy_mask

        return image, mask
