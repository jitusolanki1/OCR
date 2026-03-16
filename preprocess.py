"""
preprocess.py — OpenCV image preprocessing pipeline for UPI receipt OCR.

Pipeline (applied in order):
  1. Decode raw bytes → NumPy BGR array
  2. Resize:  cap longest edge at MAX_DIM to save RAM on Render free tier
  3. Dark-image detection & inversion  (WhatsApp Pay, PhonePe dark theme)
  4. Grayscale conversion
  5. Gaussian denoise
  6. CLAHE contrast enhancement  (handles dark / faded receipts)
  7. Adaptive threshold → clean binary image
  8. Sharpening kernel            (crisp text edges → better char recognition)
  9. Deskew                       (straightens rotated screenshots ±20°)
"""

from __future__ import annotations

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
# OPTIMIZED FOR SPEED: Lower image size → 10x faster EasyOCR on CPU
MAX_DIM              = 1280   # px  (reduced from 1920 for speed)
CLAHE_CLIP           = 2.0
CLAHE_TILE           = 8
DENOISE_H            = 2       # minimal denoise for speed
ADAPTIVE_BLOCK       = 21     # simplified from 31 for speed
ADAPTIVE_C           = 5      # reduced from 10
SHARPEN_STRENGTH     = 1.0    # reduced from 1.5
DESKEW_ANGLE_RANGE   = 5      # skip deskew in fast mode anyway
DESKEW_ANGLE_STEP    = 2.0
DARK_PIXEL_THRESHOLD = 127    # mean brightness below this → dark/night image
FAST_MODE_ENABLED    = True   # skip expensive ops like deskew by default


# ── Public entry point ───────────────────────────────────────────────────────

def preprocess(image_bytes: bytes, fast_mode: bool = True) -> np.ndarray:
    """
    Accept raw upload bytes and return a preprocessed uint8 NumPy array
    ready for EasyOCR.

    Args:
        image_bytes: raw image bytes
        fast_mode: if True, skip expensive ops (deskew, ++) for speed

    Raises:
        ValueError: if the bytes cannot be decoded as an image.
    """
    img = _decode(image_bytes)
    img = _resize(img)
    img = _invert_if_dark(img)   # ← handles dark-theme screenshots
    img = _to_gray(img)
    img = _denoise(img)
    # CLAHE is expensive — skip in fast mode
    if not fast_mode:
        img = _clahe(img)
    img = _binarize(img)
    img = _sharpen(img)
    # Deskew is very expensive (scans ±10-20 angles) — skip in fast mode
    if not fast_mode:
        img = _deskew(img)
    return img
    img = _denoise(img)
    img = _clahe(img)
    img = _binarize(img)
    img = _sharpen(img)
    img = _deskew(img)
    logger.debug("Preprocessing done  shape=%s", img.shape)
    return img


# ── Stages ───────────────────────────────────────────────────────────────────

def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Cannot decode image. Upload a valid JPG / PNG / WEBP / BMP file."
        )
    return img


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_DIM:
        return img
    scale  = MAX_DIM / longest
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA)
    logger.debug("Resized to %s", img.shape)
    return img


def _invert_if_dark(img: np.ndarray) -> np.ndarray:
    """
    Detect dark-background screenshots (WhatsApp Pay, PhonePe dark mode,
    bank app night themes) and invert them so text becomes dark-on-white.
    This dramatically improves OCR accuracy on dark UIs.

    Strategy: convert to grayscale, compute mean brightness.
    If mean < DARK_PIXEL_THRESHOLD the image is predominantly dark.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean_brightness = float(np.mean(gray))
    if mean_brightness < DARK_PIXEL_THRESHOLD:
        logger.debug("Dark image detected (mean=%.1f) — inverting", mean_brightness)
        img = cv2.bitwise_not(img)
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        img, h=DENOISE_H,
        templateWindowSize=7, searchWindowSize=21
    )


def _clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(CLAHE_TILE, CLAHE_TILE)
    )
    return clahe.apply(img)


def _binarize(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK,
        ADAPTIVE_C,
    )


def _sharpen(img: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharp   = cv2.addWeighted(img, 1 + SHARPEN_STRENGTH,
                              blurred, -SHARPEN_STRENGTH, 0)
    return sharp


def _deskew(img: np.ndarray) -> np.ndarray:
    try:
        angle = _best_angle(img)
        if abs(angle) < 0.5:
            return img
        logger.debug("Deskewing %.1f°", angle)
        h, w  = img.shape[:2]
        M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception as exc:
        logger.warning("Deskew skipped: %s", exc)
        return img


def _best_angle(img: np.ndarray) -> float:
    inv  = cv2.bitwise_not(img)
    h, w = inv.shape
    best_angle, best_var = 0.0, -1.0
    for angle in np.arange(-DESKEW_ANGLE_RANGE,
                            DESKEW_ANGLE_RANGE + DESKEW_ANGLE_STEP,
                            DESKEW_ANGLE_STEP):
        M       = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(inv, M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        var = float(rotated.sum(axis=1).astype(np.float32).var())
        if var > best_var:
            best_var, best_angle = var, angle
    return best_angle
