"""
Regex-based extraction of structured transaction fields from raw OCR text.

Each extractor function accepts the full raw OCR text string and returns
a Value or None. All functions are pure and stateless.

Pattern design philosophy:
- Patterns are ordered by specificity (most specific first).
- Patterns are case-insensitive.
- Non-capturing groups are used where the group value is unneeded.
- UPI-specific vocabulary is baked in (UTR, VPA, Paid to, etc.).
"""

from __future__ import annotations

import re
from typing import Optional
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Amount
# ===========================================================================

# Matches: ₹2,500   Rs. 2500.00   INR 2500   ₹ 2,500.50
_AMOUNT_PATTERNS = [
    # ₹ 2,500.00  or  ₹2500
    r"[₹]\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # Rs. 2500  Rs 2,500.50
    r"[Rr]s\.?\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # INR 2500
    r"INR\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # Paid  2500   Amount: 2500
    r"(?:paid|amount|total)[:\s]+([0-9,]+(?:\.[0-9]{1,2})?)",
]


def extract_amount(text: str) -> Optional[float]:
    for pattern in _AMOUNT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "")
            try:
                return float(raw)
            except ValueError:
                continue
    return None


# ===========================================================================
# Date
# ===========================================================================

# Handles DD Mon YYYY, DD/MM/YYYY, YYYY-MM-DD, D MMM YYYY, etc.
_DATE_PATTERNS = [
    # 2026-03-09  (ISO)
    (r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", "%Y-%m-%d"),
    # 09/03/2026  09-03-2026
    (r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", "%d-%m-%Y"),
    # 09 Mar 2026  9 March 2026
    (r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", "%d %b %Y"),
    # Mar 09, 2026
    (r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", "%b %d %Y"),
]

_MONTH_ABBR = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
    # Full month names
    "january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
    "june": "Jun", "july": "Jul", "august": "Aug", "september": "Sep",
    "october": "Oct", "november": "Nov", "december": "Dec",
}


def extract_date(text: str) -> Optional[str]:
    for pattern, fmt in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = list(match.groups())

            # Normalise month abbreviation if present
            for i, g in enumerate(groups):
                if g.lower() in _MONTH_ABBR:
                    groups[i] = _MONTH_ABBR[g.lower()]

            date_str = " ".join(groups)
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


# ===========================================================================
# Time
# ===========================================================================

_TIME_PATTERNS = [
    # 14:35:22  14:35  with optional AM/PM
    r"\b(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\b",
]


def extract_time(text: str) -> Optional[str]:
    for pattern in _TIME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1).strip()
            # Normalise to HH:MM (24-hr)
            for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
                try:
                    dt = datetime.strptime(raw.upper(), fmt)
                    return dt.strftime("%H:%M")
                except ValueError:
                    continue
            # Fallback: return as-is (already HH:MM or HH:MM:SS)
            return raw[:5]
    return None


# ===========================================================================
# UTR / Transaction ID
# ===========================================================================

_UTR_PATTERNS = [
    # UPI Ref No / UTR  followed by 12-22 digit number
    r"(?:UTR|UPI\s*Ref(?:erence)?\s*(?:No\.?|Number)?|Transaction\s*(?:ID|No\.?|Ref))[:\s#]*([A-Z0-9]{8,22})",
    # Standalone 12-digit numeric UTR (NPCI standard)
    r"\b([0-9]{12})\b",
    # AlphaNumeric txn IDs seen on Paytm / PhonePe
    r"\b([A-Z]{1,4}[0-9]{8,18})\b",
]


def extract_utr(text: str) -> Optional[str]:
    for pattern in _UTR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ===========================================================================
# UPI VPA
# ===========================================================================

_VPA_PATTERN = r"([a-zA-Z0-9._\-+]+@[a-zA-Z0-9]+)"


def extract_vpa(text: str) -> list[str]:
    """Return all VPAs found in text (there may be sender + receiver)."""
    return re.findall(_VPA_PATTERN, text)


# ===========================================================================
# Receiver name
# ===========================================================================

_RECEIVER_PATTERNS = [
    r"(?:Paid\s+to|To|Sent\s+to|Transferred\s+to|Received\s+by)[:\s]+([A-Za-z][A-Za-z\s.'-]{2,40})",
    r"(?:Payee|Beneficiary|Recipient)[:\s]+([A-Za-z][A-Za-z\s.'-]{2,40})",
]


def extract_receiver(text: str) -> Optional[str]:
    for pattern in _RECEIVER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))
    return None


# ===========================================================================
# Sender name
# ===========================================================================

_SENDER_PATTERNS = [
    r"(?:From|Paid\s+by|Sender|Deducted\s+from|Sent\s+by)[:\s]+([A-Za-z][A-Za-z\s.'-]{2,40})",
    r"(?:Payer)[:\s]+([A-Za-z][A-Za-z\s.'-]{2,40})",
]


def extract_sender(text: str) -> Optional[str]:
    for pattern in _SENDER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_name(match.group(1))
    return None


# ===========================================================================
# Bank name
# ===========================================================================

_BANK_KEYWORDS: dict[str, str] = {
    "hdfc": "HDFC Bank",
    "icici": "ICICI Bank",
    "sbi": "SBI",
    "axis": "Axis Bank",
    "kotak": "Kotak Mahindra Bank",
    "pnb": "Punjab National Bank",
    "punjab national": "Punjab National Bank",
    "bank of baroda": "Bank of Baroda",
    "bob": "Bank of Baroda",
    "yes bank": "Yes Bank",
    "indusind": "IndusInd Bank",
    "canara": "Canara Bank",
    "union bank": "Union Bank of India",
    "idfc": "IDFC First Bank",
    "federal": "Federal Bank",
    "rbl": "RBL Bank",
    "bandhan": "Bandhan Bank",
    "paytm payments": "Paytm Payments Bank",
    "airtel payments": "Airtel Payments Bank",
    "fino": "Fino Payments Bank",
}


def extract_bank(text: str) -> Optional[str]:
    lower = text.lower()
    for keyword, canonical in _BANK_KEYWORDS.items():
        if keyword in lower:
            return canonical
    return None


# ===========================================================================
# Payment app detection
# ===========================================================================

_APP_KEYWORDS: dict[str, str] = {
    "google pay": "Google Pay",
    "gpay": "Google Pay",
    "phonepe": "PhonePe",
    "phone pe": "PhonePe",
    "paytm": "Paytm",
    "bhim": "BHIM",
    "amazon pay": "Amazon Pay",
    "whatsapp pay": "WhatsApp Pay",
    "mobikwik": "MobiKwik",
    "freecharge": "FreeCharge",
    "airtel money": "Airtel Money",
    "jio pay": "JioPay",
    "navi": "Navi UPI",
    "slice": "Slice",
    "cred": "CRED",
}


def extract_payment_app(text: str) -> Optional[str]:
    lower = text.lower()
    for keyword, canonical in _APP_KEYWORDS.items():
        if keyword in lower:
            return canonical
    return None


# ===========================================================================
# Payment method
# ===========================================================================

_PAYMENT_METHOD_PATTERN = r"\b(UPI|NEFT|RTGS|IMPS|NACH|DD|Cheque)\b"


def extract_payment_method(text: str) -> Optional[str]:
    match = re.search(_PAYMENT_METHOD_PATTERN, text, re.IGNORECASE)
    return match.group(1).upper() if match else "UPI"


# ===========================================================================
# Transaction status
# ===========================================================================

_STATUS_PATTERNS = [
    r"\b(Payment\s+Successful|Transfer\s+Successful|Transaction\s+Successful|Success(?:ful)?)\b",
    r"\b(Payment\s+Failed|Transfer\s+Failed|Transaction\s+Failed|Failed)\b",
    r"\b(Pending|Processing|In\s+Progress)\b",
]


def extract_status(text: str) -> Optional[str]:
    for pattern in _STATUS_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# ===========================================================================
# Order ID
# ===========================================================================

_ORDER_ID_PATTERNS = [
    r"(?:Order\s*(?:ID|No\.?|Number))[:\s#]+([A-Z0-9_-]{6,30})",
    r"(?:Merchant\s*Ref(?:erence)?)[:\s#]+([A-Z0-9_-]{6,30})",
]


def extract_order_id(text: str) -> Optional[str]:
    for pattern in _ORDER_ID_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ===========================================================================
# Master extraction function
# ===========================================================================

def extract_all(raw_text: str) -> dict:
    """
    Run all extractors on raw_text and return a dict aligned with
    TransactionData field names.
    """
    vpas = extract_vpa(raw_text)
    receiver_vpa = vpas[0] if len(vpas) > 0 else None
    sender_vpa = vpas[1] if len(vpas) > 1 else None

    extracted = {
        "amount": extract_amount(raw_text),
        "receiver": extract_receiver(raw_text),
        "sender": extract_sender(raw_text),
        "receiver_upi_id": receiver_vpa,
        "sender_upi_id": sender_vpa,
        "utr": extract_utr(raw_text),
        "order_id": extract_order_id(raw_text),
        "date": extract_date(raw_text),
        "time": extract_time(raw_text),
        "bank": extract_bank(raw_text),
        "payment_app": extract_payment_app(raw_text),
        "payment_method": extract_payment_method(raw_text),
        "status": extract_status(raw_text),
        "raw_text": raw_text,
    }

    logger.debug(
        "Extraction complete",
        extra={k: v for k, v in extracted.items() if k != "raw_text"},
    )
    return extracted


# ===========================================================================
# Helpers
# ===========================================================================

def _clean_name(name: str) -> str:
    """Strip trailing punctuation and extra whitespace from a name."""
    name = re.sub(r"[^\w\s.'-]", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()
