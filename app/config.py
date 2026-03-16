"""
Application configuration settings.
Centralizes all environment variables and constants.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Metadata
    APP_NAME: str = "UPI OCR API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-ready OCR API for detecting UPI/bank payment transaction slips"
    )
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # File upload limits
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

    # OCR engine settings
    OCR_LANG: str = "en"                 # Language for PaddleOCR
    OCR_USE_ANGLE_CLS: bool = True       # Auto-rotate skewed receipts
    OCR_USE_GPU: bool = False            # Set True if CUDA is available

    # Image preprocessing
    IMG_MAX_DIMENSION: int = 2048        # Resize cap to avoid OOM
    IMG_CLAHE_CLIP_LIMIT: float = 2.0
    IMG_CLAHE_TILE_GRID: int = 8

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"             # "json" | "text"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()


settings = get_settings()
