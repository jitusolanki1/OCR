"""
Structured logging setup using Python's standard logging module.
Outputs JSON-formatted logs for easy parsing in production (Render / Railway).
"""

import logging
import sys
import json
from datetime import datetime, timezone
from app.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured, machine-readable logs."""

    LEVEL_LABELS = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": self.LEVEL_LABELS.get(record.levelno, "UNKNOWN"),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Attach exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Attach any extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging() -> None:
    """Configure root logger based on settings."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    # Avoid duplicate handlers when module is re-imported
    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(settings.LOG_LEVEL.upper())

    if settings.LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy_lib in ("paddle", "ppocr", "uvicorn.access"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call after setup_logging)."""
    return logging.getLogger(name)
