"""
Image preprocessing pipeline for UPI receipt images.

Pipeline stages:
  1. Decode uploaded bytes → NumPy array
  2. Resize to a safe maximum dimension (avoids OOM on large photos)
  3. Convert to grayscale
  4. Remove noise (fastNlMeansDenoising)
  5. CLAHE contrast enhancement (handles dark / faded receipts)
  6. Adaptive threshold → binary image (maximises OCR contrast)
  7. Optional deskew (straightens rotated screenshots)
"""

from __future__ import annotations

import math
import numpy as np
import cv2
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Accept raw image bytes and return a preprocessed NumPy image
    ready for OCR.

    Args:
        image_bytes: Raw byte content of the uploaded image file.

    Returns:
        Processed single-channel (grayscale) NumPy uint8 array.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    img = _decode_image(image_bytes)
    img = _resize_if_needed(img)
    img = _to_grayscale(img)
    img = _denoise(img)
    img = _enhance_contrast(img)
    img = _binarize(img)
    img = _deskew(img)
    logger.debug("Image preprocessing complete", extra={"shape": img.shape})
    return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a BGR NumPy array."""
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Could not decode image. Ensure the file is a valid JPG/PNG/BMP/TIFF/WEBP."
        )
    logger.debug("Image decoded", extra={"shape": img.shape})
    return img


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    """
    Downscale images whose longest edge exceeds IMG_MAX_DIMENSION.
    Upscaling is intentionally skipped — small images are padded at OCR stage.
    """
    max_dim = settings.IMG_MAX_DIMENSION
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_dim:
        return img

    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized image", extra={"new_shape": img.shape})
    return img


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(img.shape) == 2:
        return img  # Already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means denoising.
    h=10 is a balanced value — increase for noisier photos.
    """
    return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Dramatically improves readability of dark or low-contrast receipts.
    """
    clip_limit = settings.IMG_CLAHE_CLIP_LIMIT
    tile_size = settings.IMG_CLAHE_TILE_GRID
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    return clahe.apply(img)


def _binarize(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding produces a clean black-and-white image.
    Handles uneven lighting (e.g. phone camera photos with glare).
    """
    return cv2.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,    # neighbourhood size (must be odd)
        C=10,            # constant subtracted from mean
    )


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Correct slight rotation using projection-profile skew estimation.
    Skips correction if the detected angle is less than 0.5°.

    Only handles ± 45° skew — larger rotations are intentionally ignored
    because PaddleOCR's angle classifier handles those.
    """
    try:
        angle = _compute_skew_angle(img)
        if abs(angle) < 0.5:
            return img

        logger.debug("Deskewing image", extra={"angle_deg": angle})
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception as exc:                          # pragma: no cover
        logger.warning("Deskew failed, skipping", extra={"error": str(exc)})
        return img


def _compute_skew_angle(img: np.ndarray) -> float:
    """
    Estimate skew angle via the Projection Profile Method.
    Returns angle in degrees (positive = counter-clockwise).
    """
    # Invert so text pixels are white on black
    inv = cv2.bitwise_not(img)

    # Scan angles in [-20, 20] and find one with maximum projection variance
    best_angle = 0.0
    best_score = -1.0
    h, w = inv.shape

    for angle in np.arange(-20, 20.5, 0.5):
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            inv, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        projection = rotated.sum(axis=1).astype(np.float32)
        score = float(projection.var())
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle
