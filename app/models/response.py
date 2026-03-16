"""
Pydantic response and error models.

All fields are Optional so partial extraction still returns useful JSON
instead of 422 validation errors.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class TransactionData(BaseModel):
    """Structured representation of an extracted payment slip."""

    # ---- Core financial fields ----
    amount: Optional[float] = Field(
        None,
        description="Transaction amount in INR (e.g. 2500.00)",
        examples=[2500.0],
    )
    currency: str = Field(
        "INR",
        description="Currency code — always INR for UPI transactions",
    )

    # ---- Parties ----
    receiver: Optional[str] = Field(
        None,
        description="Name of the recipient / beneficiary",
        examples=["Rahul Patel"],
    )
    sender: Optional[str] = Field(
        None,
        description="Name of the sender / payer",
        examples=["Amit Shah"],
    )
    receiver_upi_id: Optional[str] = Field(
        None,
        description="UPI VPA of the receiver (e.g. rahul@okicici)",
        examples=["rahul@okicici"],
    )
    sender_upi_id: Optional[str] = Field(
        None,
        description="UPI VPA of the sender",
        examples=["amit@oksbi"],
    )

    # ---- Transaction identifiers ----
    utr: Optional[str] = Field(
        None,
        description="UTR / UPI Reference Number / Transaction ID",
        examples=["123456789012"],
    )
    order_id: Optional[str] = Field(
        None,
        description="Merchant order ID if present",
    )

    # ---- Temporal ----
    date: Optional[str] = Field(
        None,
        description="Transaction date in YYYY-MM-DD format",
        examples=["2026-03-09"],
    )
    time: Optional[str] = Field(
        None,
        description="Transaction time in HH:MM (24-hr) format",
        examples=["14:35"],
    )

    # ---- Institution ----
    bank: Optional[str] = Field(
        None,
        description="Bank name associated with the transaction",
        examples=["HDFC Bank"],
    )
    payment_app: Optional[str] = Field(
        None,
        description="Payment app detected (Google Pay / PhonePe / Paytm / BHIM / etc.)",
        examples=["Google Pay"],
    )
    payment_method: Optional[str] = Field(
        None,
        description="Payment method: UPI / NEFT / IMPS / RTGS",
        examples=["UPI"],
    )

    # ---- Status ----
    status: Optional[str] = Field(
        None,
        description="Transaction status string extracted from slip",
        examples=["Success"],
    )

    # ---- Raw OCR output ----
    raw_text: Optional[str] = Field(
        None,
        description="Full concatenated OCR text from the image",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 2500.0,
                "currency": "INR",
                "receiver": "Rahul Patel",
                "sender": "Amit Shah",
                "receiver_upi_id": "rahul@okicici",
                "sender_upi_id": "amit@oksbi",
                "utr": "123456789012",
                "order_id": None,
                "date": "2026-03-09",
                "time": "14:35",
                "bank": "HDFC Bank",
                "payment_app": "Google Pay",
                "payment_method": "UPI",
                "status": "Success",
                "raw_text": "Payment Successful\nRahul Patel\n₹2,500\n...",
            }
        }


class OCRResponse(BaseModel):
    """Top-level API response envelope."""

    success: bool = True
    message: str = "Transaction data extracted successfully"
    data: TransactionData

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Transaction data extracted successfully",
                "data": TransactionData.Config.json_schema_extra["example"],
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    success: bool = False
    message: str
    detail: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Failed to process image",
                "detail": "Unsupported file format",
            }
        }
