# UPI OCR API

A **production-ready REST API** that extracts structured payment data from UPI receipt images using **EasyOCR** and **FastAPI**.

---

## Features

| Feature                | Detail                                                                     |
| ---------------------- | -------------------------------------------------------------------------- |
| OCR Engine             | PaddleOCR (PP-OCRv4, multi-lingual)                                        |
| Image Preprocessing    | OpenCV: resize → grayscale → denoise → CLAHE → adaptive threshold → deskew |
| Information Extraction | Regex with 50+ UPI-specific patterns                                       |
| Framework              | FastAPI + Pydantic v2                                                      |
| Deployment             | Docker (multi-stage), Render, Railway                                      |
| Logging                | JSON-structured logs (stdout)                                              |
| Testing                | pytest + httpx async client                                                |

---

## Supported Payment Apps

- Google Pay / GPay
- PhonePe
- Paytm
- BHIM
- HDFC, ICICI, SBI, Axis, Kotak — and any bank app

---

## Extracted Fields

```json
{
  "amount": 2500.0,
  "currency": "INR",
  "receiver": "Rahul Patel",
  "sender": "Amit Shah",
  "receiver_upi_id": "rahul@okicici",
  "sender_upi_id": "amit@oksbi",
  "utr": "320309141501",
  "order_id": null,
  "date": "2026-03-09",
  "time": "14:35",
  "bank": "HDFC Bank",
  "payment_app": "Google Pay",
  "payment_method": "UPI",
  "status": "Payment Successful",
  "raw_text": "Google Pay\nPayment Successful\n..."
}
```

---

## Project Structure

```
pythonOCR/
├── app/
│   ├── main.py                  # FastAPI app factory + middleware
│   ├── config.py                # Settings (pydantic-settings + .env)
│   ├── api/
│   │   └── routes.py            # POST /scan-slip, GET /health
│   ├── models/
│   │   └── response.py          # Pydantic response / error models
│   ├── services/
│   │   ├── image_processor.py   # OpenCV preprocessing pipeline
│   │   ├── ocr_service.py       # PaddleOCR singleton + inference
│   │   └── extractor.py         # Regex extraction (50+ patterns)
│   └── utils/
│       └── logger.py            # JSON + text structured logging
├── tests/
│   ├── test_extractor.py        # Unit tests for regex extractors
│   └── test_api.py              # Integration tests (async httpx)
├── Dockerfile                   # Multi-stage Docker build
├── .dockerignore
├── .env.example
├── .gitignore
├── pytest.ini
└── requirements.txt
```

---

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd pythonOCR
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed (defaults work for local dev)
```

### 3. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> First run downloads ~100 MB PaddleOCR models from PPOCR servers. Subsequent runs use the local cache (`~/.paddleocr/`).

### 4. Open API docs

```
http://localhost:8000/docs
```

> `docs_url` is only enabled when `DEBUG=true` in `.env`

---

## API Reference

### `POST /api/v1/scan-slip`

Upload a payment receipt image and receive structured transaction data.

**Request**

```
Content-Type: multipart/form-data
Body:
  file: <image file>   (JPG / PNG / WEBP / BMP / TIFF, max 10 MB)
```

**cURL example**

```bash
curl -X POST http://localhost:8000/api/v1/scan-slip \
  -F "file=@/path/to/receipt.jpg"
```

**Python example**

```python
import httpx

with open("receipt.png", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/scan-slip",
        files={"file": ("receipt.png", f, "image/png")},
    )

print(response.json())
```

**JavaScript (fetch) example**

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("http://localhost:8000/api/v1/scan-slip", {
  method: "POST",
  body: formData,
});

const data = await response.json();
console.log(data);
```

**Success Response (200)**

```json
{
  "success": true,
  "message": "Transaction data extracted successfully",
  "data": {
    "amount": 2500.0,
    "currency": "INR",
    "receiver": "Rahul Patel",
    "sender": "Amit Shah",
    "receiver_upi_id": "rahul@okicici",
    "sender_upi_id": "amit@oksbi",
    "utr": "320309141501",
    "order_id": null,
    "date": "2026-03-09",
    "time": "14:35",
    "bank": "HDFC Bank",
    "payment_app": "Google Pay",
    "payment_method": "UPI",
    "status": "Payment Successful",
    "raw_text": "Google Pay\nPayment Successful\n..."
  }
}
```

**Error Response (400)**

```json
{
  "success": false,
  "message": "Failed to process image",
  "detail": "Unsupported file type 'application/pdf'."
}
```

---

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok", "version": "1.0.0"}
```

---

## Docker

### Build

```bash
docker build -t upi-ocr-api .
```

### Run

```bash
docker run -p 8000:8000 \
  -e DEBUG=false \
  -e LOG_FORMAT=json \
  upi-ocr-api
```

> The Dockerfile pre-warms PaddleOCR models at build time so the container starts instantly without internet access at runtime.

---

## Run Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_extractor.py::TestExtractAmount::test_rupee_symbol PASSED
tests/test_extractor.py::TestExtractDate::test_iso_format    PASSED
tests/test_extractor.py::TestExtractAll::test_full_gpay_slip PASSED
tests/test_api.py::test_health_check                         PASSED
tests/test_api.py::test_scan_slip_returns_200                PASSED
...
```

---

## Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set **Environment** → Docker
4. Add environment variables from `.env.example`
5. Deploy ✓

> Render free tier has 512 MB RAM — the PaddleOCR CPU model uses ~400 MB peak. Use the **Starter** plan (512 MB+) or upgrade to **Standard** for reliable performance.

## Deploy to Railway

```bash
railway login
railway init
railway up
```

Set env vars via the Railway dashboard or `railway variables set KEY=VALUE`.

---

## Configuration Reference

| Variable             | Default | Description                              |
| -------------------- | ------- | ---------------------------------------- |
| `DEBUG`              | `false` | Enable `/docs`, `/redoc`, verbose errors |
| `PORT`               | `8000`  | Uvicorn listening port                   |
| `MAX_UPLOAD_SIZE_MB` | `10`    | Maximum image file size                  |
| `OCR_LANG`           | `en`    | PaddleOCR language                       |
| `OCR_USE_ANGLE_CLS`  | `true`  | Auto-rotate skewed images                |
| `OCR_USE_GPU`        | `false` | Enable CUDA (requires paddlepaddle-gpu)  |
| `IMG_MAX_DIMENSION`  | `2048`  | Resize cap (px) for large photos         |
| `LOG_LEVEL`          | `INFO`  | Logging verbosity                        |
| `LOG_FORMAT`         | `json`  | `json` or `text`                         |

---

## Image Preprocessing Pipeline

```
Upload bytes
    │
    ▼
Decode (OpenCV)
    │
    ▼
Resize (max 2048px, INTER_AREA)
    │
    ▼
Grayscale
    │
    ▼
Denoise (fastNlMeansDenoising h=10)
    │
    ▼
CLAHE contrast enhancement
    │
    ▼
Adaptive threshold → binary image
    │
    ▼
Deskew (projection-profile, ±20°)
    │
    ▼
PaddleOCR inference
```

---

## Accuracy Notes

- **Best accuracy**: clear screenshots from Google Pay / PhonePe / Paytm apps
- **Good accuracy**: photos of phone screens in normal light
- **Reduced accuracy**: low-resolution images, extreme glare, heavily compressed JPEGs
- **UTR extraction**: 12-digit NPCI UTR numbers extracted reliably; non-standard IDs use relaxed alphanumeric patterns
- To improve further: fine-tune PaddleOCR on a custom UPI dataset or add a post-processing LLM pass

---

## Project Structure

```
pythonOCR/
├── main.py          FastAPI app, routes, middleware, error handlers
├── ocr_engine.py    EasyOCR singleton + inference helper
├── preprocess.py    OpenCV preprocessing pipeline
├── extractor.py     Regex extraction (amount, date, UTR, names …)
├── utils.py         Logging setup, upload validation, timing
├── Dockerfile       Multi-stage Docker build
└── requirements.txt Pinned Python dependencies
```

## Run Locally

```powershell
# Install PyTorch CPU first (avoids 2 GB CUDA download)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt

# Start with interactive docs
$env:DEBUG="true"; uvicorn main:app --reload
```

Open: http://localhost:8000/docs

## Docker

```bash
docker build -t upi-ocr-api .
docker run -p 8000:8000 upi-ocr-api
```
#   O C R  
 #   O C R  
 