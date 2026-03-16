"""
OCR service wrapping PaddleOCR.

Design decisions:
- The PaddleOCR engine is initialised ONCE at module import time (singleton).
  This avoids reloading 100 MB+ models on every request.
- The engine is NOT thread-safe; FastAPI runs on asyncio so concurrent
  requests are handled via async endpoints and a thread executor.
- The service accepts a preprocessed NumPy array and returns a plain list
  of (text, confidence) tuples sorted top-to-bottom, left-to-right.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from paddleocr import PaddleOCR

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton engine initialisation
# ---------------------------------------------------------------------------

# PaddleOCR logs a lot of initialisation spam — level is suppressed in logger.py
_ocr_engine: PaddleOCR | None = None


def _get_engine() -> PaddleOCR:
    """
    Lazily initialise and cache the PaddleOCR engine.
    Thread-safety is acceptable here because FastAPI is single-threaded
    for startup events and the global is set only once.
    """
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initialising PaddleOCR engine …")
        _ocr_engine = PaddleOCR(
            use_angle_cls=settings.OCR_USE_ANGLE_CLS,
            lang=settings.OCR_LANG,
            use_gpu=settings.OCR_USE_GPU,
            # Suppress model download progress bars in production logs
            show_log=False,
            # Use the most accurate detection model
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
        )
        logger.info("PaddleOCR engine ready")
    return _ocr_engine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

OCRResult = List[Tuple[str, float]]   # [(text, confidence), ...]


def run_ocr(image: np.ndarray) -> OCRResult:
    """
    Run PaddleOCR on a preprocessed image and return extracted text lines.

    Args:
        image: Preprocessed grayscale (or colour) NumPy uint8 array.

    Returns:
        List of (text, confidence) tuples, ordered top-to-bottom.

    Raises:
        RuntimeError: If the OCR engine fails for any reason.
    """
    engine = _get_engine()

    try:
        # PaddleOCR accepts BGR/grayscale NumPy arrays directly.
        # result shape: [[ [[box], (text, conf)], ... ]]
        raw_result = engine.ocr(image, cls=settings.OCR_USE_ANGLE_CLS)
    except Exception as exc:
        logger.exception("PaddleOCR inference failed")
        raise RuntimeError(f"OCR engine error: {exc}") from exc

    if not raw_result or raw_result[0] is None:
        logger.warning("OCR returned empty result")
        return []

    lines: OCRResult = []
    for page in raw_result:
        if page is None:
            continue
        for detection in page:
            # detection = [bounding_box_points, (text, confidence)]
            try:
                text: str = detection[1][0].strip()
                confidence: float = float(detection[1][1])
                if text:
                    lines.append((text, confidence))
            except (IndexError, TypeError, ValueError):
                continue  # Malformed detection — skip

    logger.debug("OCR completed", extra={"lines_extracted": len(lines)})
    return lines


def lines_to_text(lines: OCRResult) -> str:
    """Concatenate OCR line tuples into a single newline-separated string."""
    return "\n".join(text for text, _ in lines)
