"""
FastAPI route definitions.

Endpoint:
  POST /scan-slip   — upload a payment receipt image, receive structured JSON
"""

from __future__ import annotations

import asyncio
from functools import partial

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.response import OCRResponse, ErrorResponse, TransactionData
from app.services.image_processor import preprocess_image
from app.services.ocr_service import run_ocr, lines_to_text
from app.services.extractor import extract_all
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Allowed MIME types → extension mapping
# ---------------------------------------------------------------------------
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp",
}

_MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# POST /scan-slip
# ---------------------------------------------------------------------------

@router.post(
    "/scan-slip",
    response_model=OCRResponse,
    responses={
        200: {"model": OCRResponse, "description": "Transaction data extracted"},
        400: {"model": ErrorResponse, "description": "Invalid file"},
        422: {"model": ErrorResponse, "description": "Unprocessable image"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Scan a UPI / bank payment receipt",
    description=(
        "Upload a JPG/PNG/WEBP screenshot or photo of a payment receipt from "
        "Google Pay, PhonePe, Paytm, BHIM, or any bank app. "
        "Returns structured transaction data extracted via OCR."
    ),
    tags=["OCR"],
)
async def scan_slip(
    file: UploadFile = File(
        ...,
        description="Payment receipt image (JPG / PNG / WEBP / BMP / TIFF, max 10 MB)",
    ),
) -> OCRResponse:
    # ---- 1. Validate MIME type -------------------------------------------
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ---- 2. Read bytes & enforce size limit ------------------------------
    image_bytes = await file.read()
    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large ({len(image_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    logger.info(
        "Received image for OCR",
        extra={
            "filename": file.filename,
            "content_type": content_type,
            "size_bytes": len(image_bytes),
        },
    )

    # ---- 3. Preprocess (CPU-bound — run in thread pool) ------------------
    loop = asyncio.get_running_loop()
    try:
        preprocessed = await loop.run_in_executor(
            None, partial(preprocess_image, image_bytes)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Image preprocessing failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image preprocessing failed: {exc}",
        ) from exc

    # ---- 4. OCR (CPU-bound — run in thread pool) -------------------------
    try:
        ocr_lines = await loop.run_in_executor(
            None, partial(run_ocr, preprocessed)
        )
    except RuntimeError as exc:
        logger.exception("OCR engine failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    raw_text = lines_to_text(ocr_lines)
    logger.debug("Raw OCR text", extra={"text_preview": raw_text[:200]})

    # ---- 5. Extraction ---------------------------------------------------
    extracted = extract_all(raw_text)

    # ---- 6. Build response -----------------------------------------------
    transaction = TransactionData(**extracted)

    logger.info(
        "OCR scan complete",
        extra={
            "amount": transaction.amount,
            "utr": transaction.utr,
            "payment_app": transaction.payment_app,
        },
    )

    return OCRResponse(
        success=True,
        message="Transaction data extracted successfully",
        data=transaction,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="API is healthy",
)
async def health_check() -> dict:
    return {"status": "ok", "version": settings.APP_VERSION}
