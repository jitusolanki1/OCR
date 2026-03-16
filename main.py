"""
main.py — FastAPI application entry point.

Run locally:
    uvicorn main:app --reload

Run in production (single worker — EasyOCR model is not fork-safe):
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback
from functools import partial

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from preprocess import preprocess
from ocr_engine import extract_text, lines_to_text
from extractor import extract_all
from utils import setup_logging, read_and_validate_image

import logging

# ── Bootstrap logging before anything else ────────────────────────────────────
LOG_LEVEL  = os.getenv("LOG_LEVEL",  "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")   # "json" for production
setup_logging(level=LOG_LEVEL, fmt=LOG_FORMAT)
logger = logging.getLogger(__name__)

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ── Pydantic models ───────────────────────────────────────────────────────────

class TransactionData(BaseModel):
    """Inner data object — all detected fields from the slip."""
    transactionId:      Optional[str]   = None   # UTR / UPI Ref / Txn ID
    amount:             Optional[float] = None   # e.g. 409.0
    date:               Optional[str]   = None   # ISO datetime  2026-02-27T18:34:00 or date-only
    merchant:           Optional[str]   = None   # receiver name
    sender:             Optional[str]   = None
    bank:               Optional[str]   = None
    paymentApp:         Optional[str]   = None   # Google Pay / PhonePe / Paytm …
    paymentMethod:      Optional[str]   = None   # UPI / NEFT / RTGS …
    transactionType:    Optional[str]   = None   # "debit" or "credit"  (real detection)
    transactionStatus:  Optional[str]   = None   # Success / Failed / Pending
    receiverUpi:        Optional[str]   = None
    senderUpi:          Optional[str]   = None
    confidencePercent:  int             = 0      # 0-100 based on fields found
    rawText:            Optional[str]   = None
    success:            bool            = True
    errorMessage:       Optional[str]   = None


class TransactionResponse(BaseModel):
    """Top-level API response envelope."""
    success:   bool             = True
    message:   str              = "OCR extraction completed successfully."
    data:      Optional[TransactionData] = None
    errors:    Optional[str]    = None
    timestamp: Optional[str]   = None


class ErrorResponse(BaseModel):
    success: bool   = False
    message: str
    detail:  Optional[str] = None


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="UPI OCR API",
    version="1.0.0",
    description=(
        "Scan UPI / bank payment receipt images (Google Pay, PhonePe, Paytm, "
        "BHIM, bank apps) and receive structured transaction data via OCR."
    ),
    docs_url  ="/docs"        if DEBUG else None,
    redoc_url ="/redoc"       if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def _timing(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(ms)
    logger.info("%s %s  %d  %s ms",
                request.method, request.url.path,
                response.status_code, ms)
    return response

# ── Global exception handlers ──────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            message="Request validation failed",
            detail=str(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def _generic_error(request: Request, exc: Exception):
    logger.error("Unhandled exception\n%s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            message="Internal server error",
            detail=str(exc) if DEBUG else None,
        ).model_dump(),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":          "UPI OCR API",
        "version":       "1.0.0",
        "scan_endpoint": "/scan-slip",
        "health":        "/health",
    }


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post(
    "/scan-slip",
    response_model=TransactionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Scan a UPI / bank payment receipt",
    description=(
        "Upload a JPG / PNG / WEBP screenshot or photo of a payment receipt. "
        "Supports Google Pay, PhonePe, Paytm, BHIM, and any bank app. "
        "Returns structured transaction data."
    ),
    tags=["OCR"],
)
async def scan_slip(
    file: UploadFile = File(
        ...,
        description="Payment receipt image — JPG / PNG / WEBP / BMP / TIFF (max 10 MB)",
    ),
) -> TransactionResponse:
    # 1. Validate + read bytes
    image_bytes = await read_and_validate_image(file)

    loop = asyncio.get_running_loop()

    # 2. Preprocess (CPU-bound → thread pool keeps event loop free)
    try:
        processed = await loop.run_in_executor(
            None, partial(preprocess, image_bytes)
        )
    except ValueError as exc:
        raise  # re-raises as 400 via validation error path
    except Exception as exc:
        logger.exception("Preprocessing failed")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                message="Image preprocessing failed",
                detail=str(exc),
            ).model_dump(),
        )

    # 3. OCR (CPU-bound → thread pool)
    try:
        ocr_lines = await loop.run_in_executor(
            None, partial(extract_text, processed)
        )
    except RuntimeError as exc:
        logger.exception("OCR failed")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                message="OCR engine failed",
                detail=str(exc),
            ).model_dump(),
        )

    # 4. Extract structured fields
    raw_text  = lines_to_text(ocr_lines)
    extracted = extract_all(raw_text)

    # 5. Build response
    from datetime import datetime as _dt, timezone as _tz

    # Combine date + time into ISO 8601 datetime string (or date-only if no time)
    date_str = extracted.get("date")   # "2026-02-27"
    time_str = extracted.get("time")   # "18:34" or None
    if date_str and time_str:
        iso_dt = f"{date_str}T{time_str}:00"
    else:
        iso_dt = date_str  # None or date-only string

    data = TransactionData(
        transactionId     = extracted.get("utr"),
        amount            = extracted.get("amount"),
        date              = iso_dt,
        merchant          = extracted.get("receiver"),
        sender            = extracted.get("sender"),
        bank              = extracted.get("bank"),
        paymentApp        = extracted.get("payment_app"),
        paymentMethod     = extracted.get("payment_method"),
        transactionType   = extracted.get("transaction_type"),   # real detection
        transactionStatus = extracted.get("status"),
        receiverUpi       = extracted.get("receiver_upi"),
        senderUpi         = extracted.get("sender_upi"),
        confidencePercent = extracted.get("confidence", 0),      # real score 0-100
        rawText           = extracted.get("raw_text"),
        success           = True,
        errorMessage      = None,
    )

    logger.info(
        "Scan complete  amount=%s  utr=%s  app=%s  type=%s  confidence=%s",
        data.amount, data.transactionId,
        data.paymentApp, data.transactionType, data.confidencePercent,
    )

    return TransactionResponse(
        success   = True,
        message   = "OCR extraction completed successfully.",
        data      = data,
        errors    = None,
        timestamp = _dt.now(_tz.utc).isoformat(),
    )


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    logger.info("UPI OCR API starting — DEBUG=%s", DEBUG)
    # DISABLED: Do NOT pre-warm EasyOCR model at startup on free tier (512MB RAM).
    # The model (300+ MB) will be loaded on first request instead (lazy loading).
    # from ocr_engine import warmup_model
    # warmup_model()


@app.on_event("shutdown")
async def _shutdown():
    logger.info("UPI OCR API shutting down")
