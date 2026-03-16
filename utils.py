"""
utils.py — Shared utilities: logging setup, validation, timing helpers.
"""

from __future__ import annotations

import logging
import sys
import json
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, Any

from fastapi import UploadFile, HTTPException, status

# ── Logging ───────────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Machine-readable JSON log lines — easy to aggregate on Render / Railway."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "fn":      record.funcName,
            "line":    record.lineno,
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Call once at startup.  fmt='json' | 'text'"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level.upper())
    h.setFormatter(_JSONFormatter() if fmt == "json" else _TextFormatter())
    root.addHandler(h)

    # Silence noisy third-party loggers
    for lib in ("easyocr", "PIL", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── Upload validation ─────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp",  "image/tiff", "image/webp",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB


async def read_and_validate_image(file: UploadFile) -> bytes:
    """
    Read the uploaded file, enforce MIME type and size limits.

    Returns raw bytes on success.
    Raises HTTPException (400) on validation failure.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    data = await file.read()

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File too large "
                f"({len(data) / 1_048_576:.1f} MB). Max 10 MB."
            ),
        )

    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return data


# ── Timing decorator ──────────────────────────────────────────────────────────

def timed(fn: Callable) -> Callable:
    """Log wall-clock time of any sync function."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0     = time.perf_counter()
        result = fn(*args, **kwargs)
        ms     = round((time.perf_counter() - t0) * 1000, 1)
        logging.getLogger(fn.__module__).debug(
            "%s took %s ms", fn.__qualname__, ms
        )
        return result
    return wrapper
