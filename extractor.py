"""
extractor.py — Advanced regex-based structured field extraction from raw OCR text.

Supports: Google Pay, PhonePe, Paytm, BHIM, WhatsApp Pay, all bank apps.
Handles: light & dark themes, multi-line labels, OCR symbol corruption.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# AMOUNT
# ═══════════════════════════════════════════════════════════════════════════

_AMOUNT_PATTERNS = [
    # ₹50.00  ₹2,500  ₹ 700
    r"[₹\u20b9]\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # Rs. 1200  Rs 999
    r"[Rr]s\.?\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # INR 500
    r"INR\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    # Amount: 500  Paid: 700  Total 1200  Credited 300  Debited 500
    r"(?:paid|amount|total|credited|debited|sent)[:\s]+([0-9,]+(?:\.[0-9]{1,2})?)",
    # Paytm: number right after "Successfully" line
    r"(?:successfully|success)\s*[\r\n]+\s*([0-9,]{2,10})\b",
    # Paytm: number right before "Rupees" line
    r"([0-9,]{2,10})[\r\n]+[Rr]upees\b",
    # Standalone 3–7 digit line (last resort)
    r"(?:^|\n)([0-9,]{3,7})(?:\.[0-9]{1,2})?(?:\n|$)",
]

# Indian number words → integer (for Paytm "Rupees Seven Hundred Only")
_W2N_ONES: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_W2N_SCALE: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "lakh": 1_00_000, "lac": 1_00_000, "lakhs": 1_00_000,
    "crore": 1_00_00_000, "crores": 1_00_00_000,
}


def _words_to_int(phrase: str) -> Optional[int]:
    tokens = re.sub(r"[^a-zA-Z\s]", "", phrase).lower().split()
    tokens = [t for t in tokens if t not in ("and", "only", "rupees", "paise")]
    if not tokens:
        return None
    total, current = 0, 0
    for token in tokens:
        if token in _W2N_ONES:
            current += _W2N_ONES[token]
        elif token in _W2N_SCALE:
            scale = _W2N_SCALE[token]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
        else:
            return None
    return total + current or None


def _extract_amount_from_words(text: str) -> Optional[float]:
    """Paytm prints 'Rupees Seven Hundred Only' — most reliable source."""
    m = re.search(r"[Rr]upees?\s+(.+?)\s+[Oo]nly\b", text, re.IGNORECASE)
    if not m:
        return None
    val = _words_to_int(m.group(1).strip())
    return float(val) if val else None


# ₹ OCR alias digits — symbol frequently read as one of these
_RUPEE_ALIASES = {"7", "2"}


def _rupee_in_text(text: str) -> bool:
    """True when EasyOCR recognised the ₹ symbol correctly."""
    return bool(re.search(r"[₹\u20b9]", text))


def _apply_rupee_strip(val_str: str, text: str) -> float:
    """
    If ₹ was NOT recognised in the OCR output, try stripping a leading
    ₹-alias digit (2 or 7) from val_str before converting to float.

    Safe because:
      ₹250  OCR-failed → '2250' → strip '2' → 250 ✓
      ₹409  OCR-failed → '2409' → strip '2' → 409 ✓
      ₹8000 OCR-failed → '78,000' → (Case-A handles comma form)
      ₹2500 OCR-correct → '₹2500' caught by symbol pattern first,
                          never reaches here (rupee_in_text=True → no strip)
    """
    clean = val_str.replace(",", "")
    if not _rupee_in_text(text) and clean and clean[0] in _RUPEE_ALIASES:
        stripped = clean[1:]
        if re.match(r"^[1-9]\d*(?:\.\d{1,2})?$", stripped):
            logger.debug(
                "_apply_rupee_strip: '%s' → '%s'", clean, stripped
            )
            return float(stripped)
    return float(clean)


def _amount_from_debit_section(text: str) -> Optional[float]:
    """
    Google Pay / PhonePe / ICICI show the amount in the 'Debited from' row.
    Layout (two variants):
      Debited from              Debited from
      2250                      ********6281        ₹250
      ********6281
    We grab the FIRST number on the line right after the label.
    """
    m = re.search(
        r"(?:Debited\s+from|Deducted\s+from|Amount\s+paid|Amount\s+debited)"
        r"[^\r\n]*[\r\n]+[^\r\n]*?([0-9,]+(?:\.[0-9]{1,2})?)",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return _apply_rupee_strip(m.group(1), text)
    except (ValueError, IndexError):
        return None


def _amount_from_header_anchor(text: str) -> Optional[float]:
    """
    WhatsApp Pay / Google Pay show the amount on the line right after the
    'Payment details' heading.  The ₹ symbol is frequently OCR-d as a single
    stray digit (usually '7' or '2').

    Two corruption patterns handled:
      A) With comma:    ₹8,000 → '78,000'  (comma at index 2, first digit alias)
      B) Without comma: ₹50    → '250.00'  (≤3 integer digits, first digit alias)

    Safety rule for case B: only strip if the ₹ symbol does NOT appear correctly
    in the raw text (i.e. if ₹ was OCR-d correctly we trust the symbol pattern
    in _AMOUNT_PATTERNS instead, and we skip stripping here to avoid corrupting
    genuine ₹200, ₹250 amounts).
    """
    m = re.search(
        r"Payment\s+details?\s*[\r\n]+\s*([0-9][^\r\n]{0,20})",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    line = m.group(1).strip()
    nm = re.search(r"(\d[\d,]*(?:\.\d{1,2})?)", line)
    if not nm:
        return None
    val_str = nm.group(1)
    comma_idx = val_str.find(",")

    # ₹ OCR aliases that appear as a single leading digit
    _RUPEE_ALIASES = {"7", "2"}
    rupee_correctly_ocrd = bool(re.search(r"[₹\u20b9]", text))

    # Delegate to unified helper
    try:
        return _apply_rupee_strip(val_str, text)
    except (ValueError, IndexError):
        return None


def extract_amount(text: str) -> Optional[float]:
    # 1. ₹/Rs/INR symbol patterns — highest confidence (OCR got ₹ right)
    for pat in _AMOUNT_PATTERNS[:3]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if val > 0:
                    return val
            except ValueError:
                pass

    # 2. Words-based (Paytm "Rupees Seven Hundred Only")
    words_val = _extract_amount_from_words(text)
    if words_val is not None:
        return words_val

    # 3. Labeled debit row (Google Pay / PhonePe / ICICI)
    debit_val = _amount_from_debit_section(text)
    if debit_val is not None:
        return debit_val

    # 4. Header-anchored (WhatsApp Pay: amount right after 'Payment details')
    header_val = _amount_from_header_anchor(text)
    if header_val is not None:
        return header_val

    # 5. Remaining context patterns — apply strip to each candidate
    for pat in _AMOUNT_PATTERNS[3:]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return _apply_rupee_strip(m.group(1), text)
            except (ValueError, IndexError):
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DATE
# ═══════════════════════════════════════════════════════════════════════════

_MONTH_MAP = {
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
    "january": "Jan",  "february": "Feb", "march": "Mar",
    "april": "Apr",    "june": "Jun",     "july": "Jul",
    "august": "Aug",   "september": "Sep","october": "Oct",
    "november": "Nov", "december": "Dec",
}

_DATE_PATTERNS = [
    # ISO: 2026-03-09
    (r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})", "%Y-%m-%d"),
    # 09/03/2026  09-03-2026  09.03.2026
    (r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})", "%d-%m-%Y"),
    # 26-Feb-2026  26/Feb/2026  26 Feb 2026  (with separator)
    (r"(\d{1,2})[/\-\.\s]+([A-Za-z]{3,9})[/\-\.\s,]+(\d{4})", "%d %b %Y"),
    # 13Jan2023  13Jan 2023 (no separator — Google Pay header compact format)
    (r"(?<![A-Za-z])(\d{1,2})([A-Za-z]{3,9})(\d{4})(?!\d)", "%d %b %Y"),
    # Feb 26, 2026  February 26 2026
    (r"([A-Za-z]{3,9})[/\-\.\s]+(\d{1,2})[,\s]+(\d{4})", "%b %d %Y"),
]


def extract_date(text: str) -> Optional[str]:
    for pat, fmt in _DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            groups = list(m.groups())
            # Normalise month name
            groups = [_MONTH_MAP.get(g.lower(), g) for g in groups]
            try:
                dt = datetime.strptime(" ".join(groups), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
# TIME
# ═══════════════════════════════════════════════════════════════════════════

# Matches: 14:35  5:23 pm  09.59 PM  14:35:22
_TIME_PAT = r"\b(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[AaPp][Mm])?)\b"

# Date-then-time pattern: "09-Mar-2026, 3:54 pm"  or  "23 Mar 2024, 09.59 PM"
_DATE_TIME_PAT = (
    r"\d{1,2}[/\-.](?:\d{1,2}|[A-Za-z]{3,9})[/\-.,\s]+\d{2,4}"
    r"[,\s]+"
    r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[AaPp][Mm])?)"
)


def _parse_time_str(raw: str) -> Optional[str]:
    raw = raw.strip().replace(".", ":")   # normalise dot-separator → colon
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(raw.upper(), fmt).strftime("%H:%M")
        except ValueError:
            continue
    return raw[:5]


def extract_time(text: str) -> Optional[str]:
    # Prefer time that appears right after a date (transaction time, not status-bar time)
    m = re.search(_DATE_TIME_PAT, text, re.IGNORECASE)
    if m:
        result = _parse_time_str(m.group(1))
        if result:
            return result
    # Fallback: first time token anywhere in text
    m = re.search(_TIME_PAT, text)
    if not m:
        return None
    return _parse_time_str(m.group(1))


# ═══════════════════════════════════════════════════════════════════════════
# UTR / TRANSACTION ID
# ═══════════════════════════════════════════════════════════════════════════

_UTR_PATTERNS = [
    # ── Priority 1: explicit UTR / UPI Ref label (always preferred over Txn ID) ──
    r"(?:UTR|UPI\s*Ref(?:erence)?(?:\s*No\.?)?)\s*[:\s#]*([A-Z0-9]{8,22})",
    r"(?:UTR|UPI\s*Ref(?:erence)?(?:\s*No\.?))\s*[\r\n]+\s*([A-Z0-9]{8,22})",
    # ── Priority 2: Transaction ID / Txn No (same line) ─────────────────────────
    r"(?:Transaction\s*(?:ID|No\.?|Ref)|Txn\s*(?:ID|No\.?)|Reference\s*(?:No\.?|ID))"
    r"[:\s#]*([A-Z0-9]{8,22})",
    # ── Priority 3: Transaction ID on next line ──────────────────────────────────
    r"(?:Transaction\s*(?:ID|No\.?|Ref)|Txn\s*(?:ID|No\.?)|Reference\s*(?:No\.?))"
    r"\s*[\r\n]+\s*([A-Z0-9]{8,22})",
    # ── Priority 4: OCR-corrupt label 'Transaction (D' / '1D' etc. ──────────────
    r"Transaction\s*[\(\[|1Iil][Dd\)\]][:\s]*([0-9]{8,22})",
    r"Transaction\s*[\(\[|1Iil][Dd\)\]]\s*[\r\n]+\s*([0-9]{8,22})",
    # ── Priority 5: standalone NPCI 12-digit UTR ─────────────────────────────────
    r"\b([0-9]{12})\b",
    # ── Priority 6: alphanumeric IDs (Paytm / PhonePe) ──────────────────────────
    r"\b([A-Z]{1,4}[0-9]{8,18})\b",
]


def extract_utr(text: str) -> Optional[str]:
    for pat in _UTR_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


# ═══════════════════════════════════════════════════════════════════════════
# UPI VPA
# ═══════════════════════════════════════════════════════════════════════════

_VPA_RE = r"([a-zA-Z0-9._+\-]+@[a-zA-Z0-9]+)"


def extract_vpas(text: str) -> list[str]:
    return re.findall(_VPA_RE, text)


# ═══════════════════════════════════════════════════════════════════════════
# RECEIVER NAME
# ═══════════════════════════════════════════════════════════════════════════

# Patterns: label followed by name on SAME line OR NEXT line
_RECEIVER_INLINE = [
    r"(?:Paid\s+to|Sent\s+to|To|Transferred\s+to|Received\s+by|Payee|Beneficiary|Recipient)"
    r"[:\s]+([A-Z][A-Z0-9 .''-]{2,50})",
]
_RECEIVER_NEXTLINE = [
    r"(?:Paid\s+to|Sent\s+to|To|Transferred\s+to|Payee|Beneficiary|Recipient)"
    r"\s*[\r\n]+\s*([A-Z][A-Z0-9 .''-]{2,50})",
]


def extract_receiver(text: str) -> Optional[str]:
    # Try inline first (case-insensitive, name may be Title Case or UPPERCASE)
    for pat in _RECEIVER_INLINE:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return _clean_name(m.group(1))
    # Try next-line layout (WhatsApp Pay, PhonePe style)
    for pat in _RECEIVER_NEXTLINE:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return _clean_name(m.group(1))
    # Fallback: "Payment should now be in <NAME>'s bank account"
    m = re.search(
        r"(?:payment\s+should\s+now\s+be\s+in)\s+([A-Z][A-Z0-9 .''-]{2,50})'?s?\s+bank",
        text, re.IGNORECASE,
    )
    if m:
        return _clean_name(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SENDER NAME
# ═══════════════════════════════════════════════════════════════════════════

_SENDER_INLINE = [
    r"(?:From|Paid\s+by|Sender|Sent\s+by|Deducted\s+from|Payer)"
    r"[:\s]+([A-Za-z][A-Za-z0-9 .''-]{2,50})",
]
_SENDER_NEXTLINE = [
    r"(?:From|Paid\s+by|Sender|Sent\s+by|Payer)"
    r"\s*[\r\n]+\s*([A-Za-z][A-Za-z0-9 .''-]{2,50})",
]


def extract_sender(text: str) -> Optional[str]:
    for pat in _SENDER_INLINE:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return _clean_name(m.group(1))
    for pat in _SENDER_NEXTLINE:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return _clean_name(m.group(1))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# BANK NAME
# ═══════════════════════════════════════════════════════════════════════════

# Ordered longest-match first to avoid "sbi" matching inside "kotak sbi"
_BANKS: list[tuple[str, str]] = [
    ("paytm payments bank",  "Paytm Payments Bank"),
    ("airtel payments bank", "Airtel Payments Bank"),
    ("fino payments bank",   "Fino Payments Bank"),
    ("punjab national",      "Punjab National Bank"),
    ("bank of baroda",       "Bank of Baroda"),
    ("union bank",           "Union Bank of India"),
    ("state bank of india",  "State Bank Of India"),
    ("hdfc",                 "HDFC Bank"),
    ("icici",                "ICICI Bank"),
    ("axis",                 "Axis Bank"),
    ("kotak",                "Kotak Mahindra Bank"),
    ("indusind",             "IndusInd Bank"),
    ("yes bank",             "Yes Bank"),
    ("canara",               "Canara Bank"),
    ("idfc",                 "IDFC First Bank"),
    ("federal",              "Federal Bank"),
    ("rbl",                  "RBL Bank"),
    ("bandhan",              "Bandhan Bank"),
    ("pnb",                  "Punjab National Bank"),
    ("sbi",                  "State Bank Of India"),   # short form last
]


def extract_bank(text: str) -> Optional[str]:
    lower = text.lower()
    for kw, name in _BANKS:
        if kw in lower:
            return name
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PAYMENT APP
# ═══════════════════════════════════════════════════════════════════════════

_APPS: list[tuple[str, str]] = [
    ("whatsapp pay",       "WhatsApp Pay"),
    ("payments on whatsapp","WhatsApp Pay"),  # WhatsApp Pay footer text
    ("google pay",         "Google Pay"),
    ("gpay",               "Google Pay"),
    ("phonepe",            "PhonePe"),
    ("phone pe",           "PhonePe"),
    ("paytm",              "Paytm"),
    ("bhim",               "BHIM"),
    ("amazon pay",         "Amazon Pay"),
    ("mobikwik",           "MobiKwik"),
    ("freecharge",         "FreeCharge"),
    ("cred",               "CRED"),
    ("navi",               "Navi UPI"),
    ("slice",              "Slice"),
    ("jio pay",            "JioPay"),
    ("airtel money",       "Airtel Money"),
]


def extract_payment_app(text: str) -> Optional[str]:
    lower = text.lower()
    for kw, name in _APPS:
        if kw in lower:
            return name
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PAYMENT METHOD
# ═══════════════════════════════════════════════════════════════════════════

def extract_payment_method(text: str) -> str:
    m = re.search(r"\b(UPI|NEFT|RTGS|IMPS|NACH)\b", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UPI"


# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION STATUS
# ═══════════════════════════════════════════════════════════════════════════

_STATUS_PATTERNS = [
    (r"\b(Payment\s+Successful|Transfer\s+Successful|Transaction\s+Successful"
     r"|Money\s+Sent\s+Successfully|Sent\s+Successfully|Success(?:ful)?)\b",
     "Success"),
    (r"\b(Completed?)\b",  "Success"),      # WhatsApp Pay / bank apps
    (r"\b(Payment\s+Failed|Transfer\s+Failed|Transaction\s+Failed|Failed)\b",
     "Failed"),
    (r"\b(Pending|Processing|In\s+Progress)\b", "Pending"),
    (r"\b(Refunded|Reversed)\b", "Refunded"),
]


def extract_status(text: str) -> Optional[str]:
    for pat, normalised in _STATUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return normalised
    return None


# ═══════════════════════════════════════════════════════════════════════════
# TRANSACTION TYPE  (debit / credit)
# ═══════════════════════════════════════════════════════════════════════════

_DEBIT_PATTERNS = [
    r"\b(paid\s+to|sent\s+to|debited|deducted|transferred\s+to|payment\s+to|money\s+sent|send\s+again|paid|sent)\b",
]
_CREDIT_PATTERNS = [
    r"\b(received|credited|money\s+received|amount\s+credited|refund|cashback|added\s+to|deposited)\b",
]


def extract_transaction_type(text: str) -> str:
    """Return 'credit' or 'debit' based on keywords in OCR text."""
    lower = text.lower()
    for pat in _CREDIT_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return "credit"
    for pat in _DEBIT_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return "debit"
    return "debit"   # default: most slips are outgoing payments


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE
# ═══════════════════════════════════════════════════════════════════════════

def compute_confidence(data: dict) -> int:
    """
    Returns 0-100 based on how many key transaction fields were extracted.
    Weights:
      amount   30 pts  — most critical
      utr      25 pts  — unique identifier
      date     20 pts  — when
      receiver 15 pts  — who
      bank      5 pts
      time      5 pts
    """
    score = 0
    if data.get("amount"):    score += 30
    if data.get("utr"):       score += 25
    if data.get("date"):      score += 20
    if data.get("receiver"): score += 15
    if data.get("bank"):      score += 5
    if data.get("time"):      score += 5
    return min(score, 100)


def extract_all(raw_text: str) -> dict:
    """
    Run every extractor and return a dict whose keys match the API response model.
    """
    vpas = extract_vpas(raw_text)

    result = {
        "amount":           extract_amount(raw_text),
        "receiver":         extract_receiver(raw_text),
        "sender":           extract_sender(raw_text),
        "utr":              extract_utr(raw_text),
        "date":             extract_date(raw_text),
        "time":             extract_time(raw_text),
        "bank":             extract_bank(raw_text),
        "payment_app":      extract_payment_app(raw_text),
        "payment_method":   extract_payment_method(raw_text),
        "status":           extract_status(raw_text),
        "transaction_type": extract_transaction_type(raw_text),
        "receiver_upi":     vpas[0] if len(vpas) > 0 else None,
        "sender_upi":       vpas[1] if len(vpas) > 1 else None,
        "raw_text":         raw_text,
    }
    result["confidence"] = compute_confidence(result)

    logger.debug(
        "Extracted: amount=%s utr=%s type=%s confidence=%s",
        result["amount"], result["utr"],
        result["transaction_type"], result["confidence"],
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _clean_name(name: str) -> str:
    """Strip trailing noise, extra spaces, stray punctuation."""
    # Remove anything after common noise words
    name = re.split(r"\b(?:UPI|via|bank|account|•|\|)\b", name, flags=re.IGNORECASE)[0]
    name = re.sub(r"[^\w\s.''-]", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip(". -")

