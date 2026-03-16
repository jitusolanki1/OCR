"""
API integration tests using httpx AsyncClient.
Run with: pytest tests/ -v --asyncio-mode=auto
"""

from __future__ import annotations

import io
import pytest
import numpy as np
import cv2
from httpx import AsyncClient, ASGITransport

from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image_bytes(text: str = "") -> bytes:
    """
    Create a minimal valid PNG image with optional text burned in.
    Used to simulate a payment slip without needing a real receipt.
    """
    # White 400x200 image
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255

    if text:
        lines = text.strip().split("\n")
        y = 20
        for line in lines:
            cv2.putText(
                img, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
            y += 22

    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_scan_slip_returns_200():
    image_bytes = _make_dummy_image_bytes(
        "Google Pay\nPayment Successful\nPaid to Rahul Patel\n"
        "₹2500\n09/03/2026 14:35\nUTR: 123456789012\nFrom: Amit Shah"
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/scan-slip",
            files={"file": ("slip.png", io.BytesIO(image_bytes), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "data" in body
    # Amount should be extracted
    assert body["data"]["amount"] == 2500.0


@pytest.mark.asyncio
async def test_scan_slip_invalid_file_type():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/scan-slip",
            files={"file": ("document.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_scan_slip_corrupt_image():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/scan-slip",
            files={"file": ("bad.png", io.BytesIO(b"not-an-image"), "image/png")},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert "scan_endpoint" in response.json()
