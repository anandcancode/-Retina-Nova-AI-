"""
Diabetic Retinopathy Detection - Image Preprocessing Module
Implements standard clinical retinal fundus image processing pipeline:
1. Black boundary removal (auto-cropping the retinal circle)
2. Ben Graham's local contrast enhancement (Gaussian weighted subtraction)
3. Circular masking to remove peripheral border artifacts
4. Image resizing & normalization for deep learning inference
"""

import os
import cv2
import numpy as np
from PIL import Image

# Standard image size for hybrid deep learning model
IMAGE_SIZE = (224, 224)


def crop_image_from_gray(img, tol=7):
    """
    Remove black borders around the retinal fundus image.
    Works by finding the bounding mask of pixels brighter than tolerance.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is entirely too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def ben_graham_enhancement(img, sigma_x=10):
    """
    Applies Ben Graham's method (1st place solution in Kaggle Diabetic Retinopathy Detection):
    Blends original image with a Gaussian blurred version to normalize lighting variations
    and emphasize microaneurysms, hemorrhages, exudates, and neovascularization.
    formula: 4 * img - 4 * GaussianBlur(img) + 128
    """
    enhanced = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma_x), -4, 128)
    return enhanced


def apply_circular_mask(img):
    """
    Creates a smooth circular mask matching the fundus aperture to clear out boundary artifacts.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(w, h) * 0.48)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def preprocess_fundus_image(image_path, target_size=IMAGE_SIZE, save_enhanced_path=None):
    """
    Full clinical preprocessing pipeline for input fundus image:
    1. Reads RGB image from disk
    2. Crops uninformative black background
    3. Resizes to target dimension (e.g., 224x224)
    4. Computes Ben Graham enhanced version for lesion visual inspection
    5. Normalizes pixel values to [0.0, 1.0] for model inference

    Returns:
        input_tensor: numpy array of shape (1, 224, 224, 3), float32 normalized [0, 1]
        enhanced_bgr: uint8 BGR image ready to save or display in the UI
        quality_info: dict containing basic retinal quality metrics (dimensions, mean brightness)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image in RGB format
    pil_img = Image.open(image_path).convert('RGB')
    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]

    # Step 1: Crop black border
    cropped = crop_image_from_gray(img_np)
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        cropped = img_np

    # Step 2: Resize to target dimension
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply Ben Graham's enhancement
    enhanced = ben_graham_enhancement(resized, sigma_x=10)

    # Step 4: Circular mask
    enhanced_masked = apply_circular_mask(enhanced)

    # Optionally save the enhanced visualization
    if save_enhanced_path:
        enhanced_bgr = cv2.cvtColor(enhanced_masked, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_enhanced_path, enhanced_bgr)

    # Step 5: Normalize for neural network inference [0.0, 1.0]
    normalized = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Retinal quality metadata
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    quality_info = {
        "original_width": orig_w,
        "original_height": orig_h,
        "mean_intensity": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "quality_flag": "Adequate" if (mean_val > 25 and std_val > 15) else "Low Contrast / Suboptimal"
    }

    return input_tensor, enhanced_masked, quality_info


def validate_image_file(filename, file_size_bytes=None, max_size_mb=15):
    """
    Validates file extension and size.
    Allowed extensions: jpg, jpeg, png
    """
    allowed_exts = {'.jpg', '.jpeg', '.png'}
    _, ext = os.path.splitext(filename.lower())
    if ext not in allowed_exts:
        return False, f"Unsupported file type '{ext}'. Please upload JPG, JPEG, or PNG."

    if file_size_bytes is not None:
        max_bytes = max_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            return False, f"File size exceeds {max_size_mb}MB limit. Please upload a smaller image."

    return True, "Valid"

