"""
Unit tests for the regex extractor module.
Run with: pytest tests/ -v
"""

import pytest
from app.services.extractor import (
    extract_amount,
    extract_date,
    extract_time,
    extract_utr,
    extract_receiver,
    extract_sender,
    extract_bank,
    extract_payment_app,
    extract_payment_method,
    extract_status,
    extract_all,
)


# ===========================================================================
# Amount tests
# ===========================================================================

class TestExtractAmount:
    def test_rupee_symbol(self):
        assert extract_amount("₹2,500") == 2500.0

    def test_rupee_symbol_decimal(self):
        assert extract_amount("₹ 2,500.50") == 2500.50

    def test_rs_abbreviation(self):
        assert extract_amount("Rs. 1200") == 1200.0

    def test_inr_prefix(self):
        assert extract_amount("INR 999") == 999.0

    def test_paid_keyword(self):
        assert extract_amount("Paid: 3500") == 3500.0

    def test_amount_keyword(self):
        assert extract_amount("Amount 10,000.00") == 10000.0

    def test_no_amount(self):
        assert extract_amount("No monetary value here") is None


# ===========================================================================
# Date tests
# ===========================================================================

class TestExtractDate:
    def test_iso_format(self):
        assert extract_date("2026-03-09") == "2026-03-09"

    def test_dd_mm_yyyy(self):
        assert extract_date("09/03/2026") == "2026-03-09"

    def test_dd_mon_yyyy(self):
        assert extract_date("09 Mar 2026") == "2026-03-09"

    def test_full_month(self):
        assert extract_date("9 March 2026") == "2026-03-09"

    def test_no_date(self):
        assert extract_date("No date present") is None


# ===========================================================================
# Time tests
# ===========================================================================

class TestExtractTime:
    def test_24hr(self):
        assert extract_time("14:35") == "14:35"

    def test_24hr_with_seconds(self):
        assert extract_time("14:35:22") == "14:35"

    def test_12hr_pm(self):
        result = extract_time("2:35 PM")
        assert result == "14:35"

    def test_no_time(self):
        assert extract_time("No time here") is None


# ===========================================================================
# UTR tests
# ===========================================================================

class TestExtractUTR:
    def test_utr_label(self):
        assert extract_utr("UTR: 123456789012") == "123456789012"

    def test_upi_ref(self):
        assert extract_utr("UPI Ref No: 320309141501") == "320309141501"

    def test_transaction_id(self):
        assert extract_utr("Transaction ID: TXN2026030912345") == "TXN2026030912345"

    def test_standalone_12_digit(self):
        assert extract_utr("123456789012") == "123456789012"

    def test_no_utr(self):
        assert extract_utr("No reference") is None


# ===========================================================================
# Receiver / Sender tests
# ===========================================================================

class TestExtractNames:
    def test_receiver_paid_to(self):
        assert extract_receiver("Paid to Rahul Patel") == "Rahul Patel"

    def test_receiver_to(self):
        assert extract_receiver("To: Anita Sharma") == "Anita Sharma"

    def test_sender_from(self):
        assert extract_sender("From: Amit Shah") == "Amit Shah"

    def test_sender_paid_by(self):
        assert extract_sender("Paid by Vikram Singh") == "Vikram Singh"


# ===========================================================================
# Bank / App / Method / Status tests
# ===========================================================================

class TestBankAppMethodStatus:
    def test_bank_hdfc(self):
        assert extract_bank("Transferred via HDFC Bank") == "HDFC Bank"

    def test_bank_sbi(self):
        assert extract_bank("SBI account deducted") == "SBI"

    def test_app_gpay(self):
        assert extract_payment_app("Sent via Google Pay") == "Google Pay"

    def test_app_phonepe(self):
        assert extract_payment_app("PhonePe Transfer") == "PhonePe"

    def test_app_paytm(self):
        assert extract_payment_app("Paytm Wallet") == "Paytm"

    def test_method_upi(self):
        assert extract_payment_method("UPI Transfer") == "UPI"

    def test_method_neft(self):
        assert extract_payment_method("NEFT Transfer confirmed") == "NEFT"

    def test_status_success(self):
        assert extract_status("Payment Successful") is not None

    def test_status_failed(self):
        assert extract_status("Payment Failed") is not None


# ===========================================================================
# Integration test — extract_all
# ===========================================================================

SAMPLE_GPAY_TEXT = """
Google Pay
Payment Successful
Paid to Rahul Patel
rahul@okhdfc
₹2,500
09 Mar 2026 | 14:35:22
UPI Ref No: 320309141501
From: Amit Shah
amit@oksbi
HDFC Bank
"""


class TestExtractAll:
    def test_full_gpay_slip(self):
        result = extract_all(SAMPLE_GPAY_TEXT)
        assert result["amount"] == 2500.0
        assert result["receiver"] == "Rahul Patel"
        assert result["sender"] == "Amit Shah"
        assert result["utr"] == "320309141501"
        assert result["date"] == "2026-03-09"
        assert result["time"] == "14:35"
        assert result["bank"] == "HDFC Bank"
        assert result["payment_app"] == "Google Pay"
        assert result["status"] is not None
        assert result["raw_text"] == SAMPLE_GPAY_TEXT
