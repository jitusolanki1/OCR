# ============================================================
# Dockerfile — UPI OCR API (EasyOCR + FastAPI)
# Multi-stage build:  builder installs deps, runtime is lean
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker-requirements.txt .

# Install PyTorch CPU wheel first — prevents pip from pulling the 2 GB CUDA build
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r docker-requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL description="UPI OCR API — EasyOCR + FastAPI"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1001 appuser
USER appuser
WORKDIR /home/appuser/app

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

# Copy flat source files only
COPY --chown=appuser:appuser main.py ocr_engine.py preprocess.py extractor.py utils.py ./

# Pre-warm EasyOCR model cache at build time
RUN python -c "\
import easyocr; \
easyocr.Reader(['en'], gpu=False, \
               model_storage_directory='.easyocr_models', \
               download_enabled=True, quantize=True)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; \
urllib.request.urlopen('http://localhost:8000/health')"

# --workers 1: EasyOCR Reader is not fork-safe
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning"]
