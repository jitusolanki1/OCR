# Minimal Dockerfile - Python 3.12 only, no buildpack interference
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU first (prevents CUDA wheel bloat)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ocr_engine.py preprocess.py extractor.py utils.py ./

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
