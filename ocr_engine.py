"""
ocr_engine.py — EasyOCR singleton + inference helpers.

Design:
- The Reader is initialised ONCE at module-import time (lazy on first call).
  Re-creating it on every request would reload ~300 MB of models.
- EasyOCR is NOT async; we run it in a ThreadPoolExecutor so FastAPI's
  event loop is never blocked.
- Low-memory flags are set for Render free-tier (512 MB RAM).
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import easyocr
import numpy as np

logger = logging.getLogger(__name__)

# ── Singleton ─────────────────────────────────────────────────────────────────
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        logger.info("Loading EasyOCR model (first request) …")
        _reader = easyocr.Reader(
            ["en"],          # English only — smaller footprint than multi-lang
            gpu=False,       # Render free tier has no GPU
            model_storage_directory=".easyocr_models",   # local cache dir
            download_enabled=True,
            # Quantise to reduce RAM: EasyOCR uses this internally when set
            quantize=True,
        )
        logger.info("EasyOCR ready")
    return _reader


def warmup_model():
    """
    Pre-load EasyOCR model at application startup.
    Prevents 5-15 second delay on first request.
    """
    logger.info("Pre-warming EasyOCR model…")
    reader = _get_reader()
    # Dummy inference on 1x1 pixel to fully load weights into RAM/VRAM
    try:
        reader.readtext(np.zeros((1, 1, 3), dtype=np.uint8), detail=1, batch_size=1, workers=0)
        logger.info("EasyOCR model warmed up successfully")
    except Exception as e:
        logger.warning("Model warmup inference failed (non-critical): %s", e)


# ── Public API ─────────────────────────────────────────────────────────────────

# Type alias: list of (text, confidence) pairs sorted top-to-bottom
OCRLines = List[Tuple[str, float]]


def extract_text(image: np.ndarray) -> OCRLines:
    """
    Run EasyOCR on a preprocessed image.

    Args:
        image: Preprocessed uint8 NumPy array (grayscale or BGR).

    Returns:
        List of (text, confidence) sorted top-to-bottom by y-coordinate.

    Raises:
        RuntimeError: on EasyOCR inference failure.
    """
    try:
        reader = _get_reader()
    except Exception as exc:
        logger.exception("Failed to load EasyOCR model - check memory/disk")
        raise RuntimeError(f"Model load failed: {exc}") from exc
    
    try:
        # readtext returns: [ (bbox, text, confidence), ... ]
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,     # Keep individual word/line boxes
            batch_size=1,        # Minimise RAM: process one image strip at a time
            workers=0,           # No subprocesses — avoids fork issues on Render
        )
    except Exception as exc:
        logger.exception("EasyOCR inference failed")
        raise RuntimeError(f"OCR engine error: {exc}") from exc

    if not results:
        logger.warning("EasyOCR returned no results")
        return []

    # Sort by top-left y then x so text reads in natural document order
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    lines: OCRLines = []
    for bbox, text, conf in results:
        text = text.strip()
        if text:
            lines.append((text, float(conf)))

    logger.debug("OCR extracted %d lines", len(lines))
    return lines


def lines_to_text(lines: OCRLines) -> str:
    """Flatten OCR lines to a single newline-separated string."""
    return "\n".join(t for t, _ in lines)
