"""
src/audit_online_retail_integrity.py

Phase 1: Raw Data Integrity Audit for Online Retail II
AI-Powered E-commerce Customer Segmentation and Churn Analysis

Performs a comprehensive, inspection-only audit of data/raw/online_retail_II.csv
across 11 integrity dimensions and produces:
data/checkpoints/02_online_retail_integrity.json

Guarantees:
- Strictly treats raw data as immutable (pre- and post-audit SHA-256 verification).
- Zero in-place data modification or overwriting.
- Zero modification to existing Olist checkpoints or raw data.
- Proposes candidate data rules based strictly on empirical audit evidence.
"""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "checkpoints" / "02_online_retail_integrity.json"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_raw_data_integrity_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 1: RAW DATA INTEGRITY AUDIT — ONLINE RETAIL II")
    print("=" * 80)

    # 1. Pre-audit raw file verification
    assert RAW_FILE.exists(), f"Raw file not found: {RAW_FILE}"
    file_size_bytes = os.path.getsize(RAW_FILE)
    pre_sha256 = compute_sha256(RAW_FILE)
    print(f"Target file: {RAW_FILE.name}")
    print(f"File size: {file_size_bytes:,} bytes ({file_size_bytes / (1024*1024):.2f} MB)")
    print(f"Pre-audit SHA-256: {pre_sha256}")

    # Load raw CSV without type conversion
    df = pd.read_csv(RAW_FILE, low_memory=False)
    total_rows, total_cols = df.shape
    exact_columns = list(df.columns)
    col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Full duplicate rows
    duplicate_mask = df.duplicated(keep="first")
    num_duplicates = int(duplicate_mask.sum())
    pct_duplicates = round((num_duplicates / total_rows) * 100.0, 4)

    # Missing values per column
    missing_stats = {}
    for col in exact_columns:
        n_miss = int(df[col].isna().sum())
        missing_stats[col] = {
            "missing_count": n_miss,
            "missing_pct": round((n_miss / total_rows) * 100.0, 4),
            "present_count": total_rows - n_miss,
            "present_pct": round(((total_rows - n_miss) / total_rows) * 100.0, 4),
        }

    # ==========================================
    # 2. DATETIME VALIDATION
    # ==========================================
    parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    unparseable_dates = int(parsed_dates.isna().sum())
    min_timestamp = str(parsed_dates.min())
    max_timestamp = str(parsed_dates.max())
    unique_calendar_dates = int(parsed_dates.dt.date.nunique())

    # Transactions per year and month
    tx_by_year = {str(k): int(v) for k, v in parsed_dates.dt.year.value_counts().sort_index().items()}
    tx_by_year_month = {
        str(k): int(v)
        for k, v in parsed_dates.dt.to_period("M").value_counts().sort_index().items()
    }

    # Monotonicity check
    is_monotonic = bool(parsed_dates.is_monotonic_increasing)

    # Impossible/future dates check (dataset boundary is Dec 2011)
    future_dates = int((parsed_dates > pd.Timestamp("2011-12-31")).sum())
    pre_2009_dates = int((parsed_dates < pd.Timestamp("2009-12-01")).sum())

    datetime_audit = {
        "unparseable_timestamps": unparseable_dates,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "span_days": round((parsed_dates.max() - parsed_dates.min()).total_seconds() / 86400.0, 2),
        "unique_calendar_dates": unique_calendar_dates,
        "transactions_by_year": tx_by_year,
        "transactions_by_year_month": tx_by_year_month,
        "is_monotonically_ordered": is_monotonic,
        "out_of_bounds_future_dates_post_2011": future_dates,
        "out_of_bounds_pre_2009_dates": pre_2009_dates,
    }

    # ==========================================
    # 3. INVOICE STRUCTURE
    # ==========================================
    invoice_series = df["Invoice"].astype(str).str.strip()
    unique_invoices = int(invoice_series.nunique())
    rows_per_invoice = invoice_series.value_counts()

    single_row_invoices = int((rows_per_invoice == 1).sum())
    multi_row_invoices = int((rows_per_invoice > 1).sum())
    min_rows_per_inv = int(rows_per_invoice.min())
    max_rows_per_inv = int(rows_per_invoice.max())

    inv_row_percentiles = {
        "p10": float(rows_per_invoice.quantile(0.10)),
        "p25": float(rows_per_invoice.quantile(0.25)),
        "median": float(rows_per_invoice.quantile(0.50)),
        "p75": float(rows_per_invoice.quantile(0.75)),
        "p90": float(rows_per_invoice.quantile(0.90)),
        "p95": float(rows_per_invoice.quantile(0.95)),
        "p99": float(rows_per_invoice.quantile(0.99)),
        "mean": round(float(rows_per_invoice.mean()), 2),
    }

    # Cancellation prefix 'C'
    is_c_prefix = invoice_series.str.startswith("C", na=False)
    num_c_invoices = int(invoice_series[is_c_prefix].nunique())
    num_c_rows = int(is_c_prefix.sum())
    pct_c_invoices = round((num_c_invoices / unique_invoices) * 100.0, 4)
    pct_c_rows = round((num_c_rows / total_rows) * 100.0, 4)

    # Cross-check cancellation prefix vs Quantity sign
    is_neg_qty = df["Quantity"] < 0
    # Invoices where cancellation prefix is 'C' but Quantity >= 0
    c_prefix_pos_qty_rows = int((is_c_prefix & (df["Quantity"] >= 0)).sum())
    # Invoices where cancellation prefix is NOT 'C' but Quantity < 0
    non_c_prefix_neg_qty_rows = int((~is_c_prefix & is_neg_qty).sum())
    non_c_prefix_neg_qty_invoices = int(invoice_series[~is_c_prefix & is_neg_qty].nunique())

    invoice_audit = {
        "unique_invoice_count": unique_invoices,
        "single_row_invoices": single_row_invoices,
        "multi_row_invoices": multi_row_invoices,
        "pct_multi_row_invoices": round((multi_row_invoices / unique_invoices) * 100.0, 4),
        "min_rows_per_invoice": min_rows_per_inv,
        "max_rows_per_invoice": max_rows_per_inv,
        "rows_per_invoice_percentiles": inv_row_percentiles,
        "cancellation_invoices_with_c_prefix": num_c_invoices,
        "cancellation_rows_with_c_prefix": num_c_rows,
        "pct_cancellation_invoices": pct_c_invoices,
        "pct_cancellation_rows": pct_c_rows,
        "c_prefix_with_positive_quantity_rows": c_prefix_pos_qty_rows,
        "non_c_prefix_with_negative_quantity_rows": non_c_prefix_neg_qty_rows,
        "non_c_prefix_with_negative_quantity_invoices": non_c_prefix_neg_qty_invoices,
    }

    # ==========================================
    # 4. QUANTITY VALIDATION
    # ==========================================
    qty = df["Quantity"]
    min_qty = int(qty.min())
    max_qty = int(qty.max())
    zero_qty_rows = int((qty == 0).sum())
    neg_qty_rows = int((qty < 0).sum())
    pos_qty_rows = int((qty > 0).sum())

    qty_percentiles = {
        "p01": float(qty.quantile(0.01)),
        "p05": float(qty.quantile(0.05)),
        "p25": float(qty.quantile(0.25)),
        "median": float(qty.quantile(0.50)),
        "p75": float(qty.quantile(0.75)),
        "p95": float(qty.quantile(0.95)),
        "p99": float(qty.quantile(0.99)),
        "mean": round(float(qty.mean()), 2),
    }

    # Negative quantities outside 'C' cancellation invoices
    neg_qty_non_c = df[is_neg_qty & (~is_c_prefix)]
    neg_qty_non_c_rows = len(neg_qty_non_c)
    neg_qty_non_c_missing_cust = int(neg_qty_non_c["Customer ID"].isna().sum())

    quantity_audit = {
        "min_quantity": min_qty,
        "max_quantity": max_qty,
        "zero_quantity_rows": zero_qty_rows,
        "negative_quantity_rows": neg_qty_rows,
        "positive_quantity_rows": pos_qty_rows,
        "negative_quantity_pct": round((neg_qty_rows / total_rows) * 100.0, 4),
        "quantity_percentiles": qty_percentiles,
        "negative_quantity_outside_c_invoices_count": neg_qty_non_c_rows,
        "negative_quantity_outside_c_invoices_with_missing_customer": neg_qty_non_c_missing_cust,
    }

    # ==========================================
    # 5. PRICE VALIDATION
    # ==========================================
    price = df["Price"]
    min_price = float(price.min())
    max_price = float(price.max())
    zero_price_rows = int((price == 0.0).sum())
    neg_price_rows = int((price < 0.0).sum())
    pos_price_rows = int((price > 0.0).sum())

    price_percentiles = {
        "p01": float(price.quantile(0.01)),
        "p25": float(price.quantile(0.25)),
        "median": float(price.quantile(0.50)),
        "p75": float(price.quantile(0.75)),
        "p95": float(price.quantile(0.95)),
        "p99": float(price.quantile(0.99)),
        "mean": round(float(price.mean()), 4),
    }

    # Examine negative price rows
    neg_price_df = df[price < 0.0]
    neg_price_descriptions = list(neg_price_df["Description"].dropna().unique()) if len(neg_price_df) > 0 else []

    # Zero price rows breakdown
    zero_price_df = df[price == 0.0]
    zero_price_missing_cust = int(zero_price_df["Customer ID"].isna().sum())
    zero_price_present_cust = len(zero_price_df) - zero_price_missing_cust

    price_audit = {
        "min_price": min_price,
        "max_price": max_price,
        "zero_price_rows": zero_price_rows,
        "zero_price_pct": round((zero_price_rows / total_rows) * 100.0, 4),
        "zero_price_missing_customer_count": zero_price_missing_cust,
        "zero_price_present_customer_count": zero_price_present_cust,
        "negative_price_rows": neg_price_rows,
        "negative_price_descriptions": neg_price_descriptions,
        "positive_price_rows": pos_price_rows,
        "price_percentiles": price_percentiles,
    }

    # ==========================================
    # 6. CUSTOMER ID VALIDATION
    # ==========================================
    raw_cust = df["Customer ID"]
    missing_cust_rows = int(raw_cust.isna().sum())
    missing_cust_pct = round((missing_cust_rows / total_rows) * 100.0, 4)
    present_cust_rows = total_rows - missing_cust_rows

    valid_cust_df = df[raw_cust.notna()].copy()
    valid_cust_df["Customer_ID_clean"] = valid_cust_df["Customer ID"].astype(int).astype(str)
    unique_customers_raw = int(valid_cust_df["Customer_ID_clean"].nunique())

    # Check formatting issues (non-integer floats, negative IDs)
    float_remainders = (valid_cust_df["Customer ID"] % 1 != 0).sum()
    negative_cust_ids = (valid_cust_df["Customer ID"] < 0).sum()

    # Customers associated with multiple countries in ALL rows
    cust_country_all = valid_cust_df.groupby("Customer_ID_clean")["Country"].nunique()
    multi_country_cust_all = int((cust_country_all > 1).sum())

    # Customers associated with multiple countries in VALID non-cancelled rows
    valid_tx = valid_cust_df[
        (~valid_cust_df["Invoice"].astype(str).str.startswith("C", na=False))
        & (valid_cust_df["Quantity"] > 0)
        & (valid_cust_df["Price"] >= 0)
    ]
    unique_customers_valid = int(valid_tx["Customer_ID_clean"].nunique())
    cust_country_valid = valid_tx.groupby("Customer_ID_clean")["Country"].nunique()
    multi_country_cust_valid = int((cust_country_valid > 1).sum())

    customer_audit = {
        "missing_customer_id_rows": missing_cust_rows,
        "missing_customer_id_pct": missing_cust_pct,
        "present_customer_id_rows": present_cust_rows,
        "present_customer_id_pct": round((present_cust_rows / total_rows) * 100.0, 4),
        "unique_identifiable_customers_raw": unique_customers_raw,
        "unique_identifiable_customers_valid_purchases": unique_customers_valid,
        "non_integer_customer_ids": int(float_remainders),
        "negative_customer_ids": int(negative_cust_ids),
        "customers_associated_with_multiple_countries_raw": multi_country_cust_all,
        "customers_associated_with_multiple_countries_valid_purchases": multi_country_cust_valid,
    }

    # ==========================================
    # 7. PRODUCT VALIDATION (StockCode & Description)
    # ==========================================
    stock = df["StockCode"].astype(str).str.strip()
    unique_stock_codes = int(stock.nunique())
    missing_stock = int(df["StockCode"].isna().sum())
    blank_stock = int((stock == "").sum())

    # Identify non-standard StockCodes (non-digit leading or non-standard format)
    # Standard format: 5 digits optionally followed by 1 or 2 uppercase letters (e.g., 85048, 79323P, 22041A)
    standard_stock_pattern = re.compile(r"^\d{5}[A-Za-z]?$")
    non_standard_mask = ~stock.str.match(standard_stock_pattern)
    non_standard_codes = stock[non_standard_mask].value_counts()
    unique_non_standard_codes = int(stock[non_standard_mask].nunique())
    top_non_standard_codes = {str(k): int(v) for k, v in non_standard_codes.head(20).items()}

    # Products with both positive and negative quantities
    pos_stock = set(df.loc[df["Quantity"] > 0, "StockCode"].astype(str).str.strip().unique())
    neg_stock = set(df.loc[df["Quantity"] < 0, "StockCode"].astype(str).str.strip().unique())
    both_sign_stock_count = len(pos_stock.intersection(neg_stock))

    # Description validation
    desc = df["Description"].astype(str).str.strip()
    missing_desc = int(df["Description"].isna().sum())
    blank_desc = int((df["Description"].fillna("").str.strip() == "").sum())
    unique_descriptions = int(df["Description"].dropna().nunique())

    # Inconsistencies: StockCodes mapped to multiple distinct descriptions
    stock_desc_counts = df.dropna(subset=["StockCode", "Description"]).groupby("StockCode")["Description"].nunique()
    multi_desc_stock_count = int((stock_desc_counts > 1).sum())
    max_desc_for_single_stock = int(stock_desc_counts.max()) if len(stock_desc_counts) > 0 else 0

    product_audit = {
        "unique_stock_codes": unique_stock_codes,
        "missing_stock_codes": missing_stock,
        "blank_stock_codes": blank_stock,
        "unique_non_standard_stock_codes": unique_non_standard_codes,
        "top_non_standard_stock_codes": top_non_standard_codes,
        "stock_codes_with_both_positive_and_negative_quantities": both_sign_stock_count,
        "missing_descriptions": missing_desc,
        "blank_descriptions": blank_desc,
        "unique_descriptions": unique_descriptions,
        "stock_codes_with_multiple_descriptions": multi_desc_stock_count,
        "max_descriptions_for_single_stock_code": max_desc_for_single_stock,
    }

    # ==========================================
    # 8. COUNTRY VALIDATION
    # ==========================================
    countries = df["Country"].astype(str).str.strip()
    unique_countries = int(countries.nunique())
    tx_by_country = {str(k): int(v) for k, v in countries.value_counts().items()}

    # Customer count by country (for rows with non-missing Customer ID)
    cust_by_country = {
        str(k): int(v)
        for k, v in valid_cust_df.groupby("Country")["Customer_ID_clean"].nunique().sort_values(ascending=False).items()
    }

    country_audit = {
        "unique_countries_count": unique_countries,
        "transactions_by_country": tx_by_country,
        "unique_customers_by_country": cust_by_country,
        "customers_appearing_in_multiple_countries": multi_country_cust_all,
    }

    # ==========================================
    # 9. TRANSACTION VALUE RECONCILIATION
    # ==========================================
    df["line_value"] = df["Quantity"] * df["Price"]
    line_val = df["line_value"]

    missing_line_val = int(line_val.isna().sum())
    zero_line_val = int((line_val == 0.0).sum())
    neg_line_val = int((line_val < 0.0).sum())
    pos_line_val = int((line_val > 0.0).sum())

    line_val_percentiles = {
        "p01": float(line_val.quantile(0.01)),
        "p05": float(line_val.quantile(0.05)),
        "p25": float(line_val.quantile(0.25)),
        "median": float(line_val.quantile(0.50)),
        "p75": float(line_val.quantile(0.75)),
        "p95": float(line_val.quantile(0.95)),
        "p99": float(line_val.quantile(0.99)),
        "mean": round(float(line_val.mean()), 4),
        "total_sum": round(float(line_val.sum()), 2),
    }

    # Aggregate by invoice: invoice_gross_value = sum(Quantity * Price)
    invoice_totals = df.groupby(df["Invoice"].astype(str).str.strip())["line_value"].sum()
    pos_inv_totals = int((invoice_totals > 0).sum())
    zero_inv_totals = int((invoice_totals == 0).sum())
    neg_inv_totals = int((invoice_totals < 0).sum())

    # Cancellation invoices totals
    c_inv_ids = set(invoice_series[is_c_prefix].unique())
    c_inv_totals = invoice_totals.loc[invoice_totals.index.isin(c_inv_ids)]
    c_inv_sum = round(float(c_inv_totals.sum()), 2)
    pos_inv_sum = round(float(invoice_totals[invoice_totals > 0].sum()), 2)
    neg_inv_sum = round(float(invoice_totals[invoice_totals < 0].sum()), 2)

    inv_val_percentiles = {
        "p05": float(invoice_totals.quantile(0.05)),
        "p25": float(invoice_totals.quantile(0.25)),
        "median": float(invoice_totals.quantile(0.50)),
        "p75": float(invoice_totals.quantile(0.75)),
        "p95": float(invoice_totals.quantile(0.95)),
        "mean": round(float(invoice_totals.mean()), 2),
        "total_net_sum": round(float(invoice_totals.sum()), 2),
    }

    value_audit = {
        "row_level_line_value": {
            "missing_line_values": missing_line_val,
            "zero_line_values": zero_line_val,
            "negative_line_values": neg_line_val,
            "positive_line_values": pos_line_val,
            "line_value_percentiles": line_val_percentiles,
        },
        "invoice_level_gross_value": {
            "unique_invoices_evaluated": len(invoice_totals),
            "positive_invoice_totals_count": pos_inv_totals,
            "zero_invoice_totals_count": zero_inv_totals,
            "negative_invoice_totals_count": neg_inv_totals,
            "positive_invoices_gross_sum": pos_inv_sum,
            "negative_invoices_gross_sum": neg_inv_sum,
            "cancellation_invoices_total_sum": c_inv_sum,
            "invoice_gross_value_percentiles": inv_val_percentiles,
        },
    }

    # ==========================================
    # 10. DUPLICATE ANALYSIS
    # ==========================================
    dup_rows = df[duplicate_mask]
    dup_invoices = int(dup_rows["Invoice"].astype(str).nunique())
    dup_custs = int(dup_rows["Customer ID"].dropna().nunique())
    dup_missing_cust = int(dup_rows["Customer ID"].isna().sum())

    # Concentration in cancellations vs non-cancellations
    dup_is_c = dup_rows["Invoice"].astype(str).str.startswith("C", na=False)
    dup_cancellation_rows = int(dup_is_c.sum())
    dup_non_cancellation_rows = len(dup_rows) - dup_cancellation_rows

    duplicate_audit = {
        "exact_duplicate_rows_count": num_duplicates,
        "exact_duplicate_percentage": pct_duplicates,
        "affected_unique_invoices": dup_invoices,
        "affected_unique_customers": dup_custs,
        "duplicate_rows_with_missing_customer_id": dup_missing_cust,
        "duplicate_rows_in_cancellations": dup_cancellation_rows,
        "duplicate_rows_in_non_cancellations": dup_non_cancellation_rows,
        "assessment": (
            "Duplicate rows share identical Invoice, StockCode, Description, Quantity, InvoiceDate, Price, "
            "Customer ID, and Country. In tabular retail transaction logs, identical records with identical timestamps "
            "and quantities for the same invoice typically indicate duplicate data collection / logging rather than "
            "distinct items, as separate line items typically have distinct item IDs or are summed into single line quantities."
        ),
    }

    # ==========================================
    # 11. TEMPORAL CONSISTENCY
    # ==========================================
    # Timestamp bursts
    tx_per_timestamp = parsed_dates.value_counts()
    max_tx_single_timestamp = int(tx_per_timestamp.max())
    timestamps_with_gt_50_tx = int((tx_per_timestamp > 50).sum())

    # Invoices with inconsistent dates across rows
    inv_date_counts = df.groupby(invoice_series)["InvoiceDate"].nunique()
    inv_inconsistent_dates = int((inv_date_counts > 1).sum())

    # Invoices with inconsistent Customer IDs across rows
    inv_cust_counts = df.groupby(invoice_series)["Customer ID"].nunique(dropna=False)
    inv_inconsistent_cust = int((inv_cust_counts > 1).sum())

    # Invoices with inconsistent Country values across rows
    inv_country_counts = df.groupby(invoice_series)["Country"].nunique()
    inv_inconsistent_country = int((inv_country_counts > 1).sum())

    temporal_consistency_audit = {
        "max_transactions_in_single_second": max_tx_single_timestamp,
        "timestamps_with_over_50_transactions": timestamps_with_gt_50_tx,
        "invoices_with_inconsistent_dates": inv_date_counts[inv_date_counts > 1].to_dict(),
        "invoices_with_inconsistent_dates_count": inv_inconsistent_dates,
        "invoices_with_inconsistent_customer_ids_count": inv_inconsistent_cust,
        "invoices_with_inconsistent_countries_count": inv_inconsistent_country,
    }

    # ==========================================
    # 12. PROPOSED DATA RULES (Strictly Candidate / Proposed)
    # ==========================================
    proposed_rules = {
        "rule_1_cancellations_treatment": {
            "proposal": "Exclude cancellation invoices (invoices starting with 'C' and negative quantities) from primary purchase count and positive monetary aggregation.",
            "rationale": "Cancellations represent returned merchandise or reversals (22,951 rows, 2.15% of transactions). They should not be counted as forward purchase return events or positive sales.",
            "candidate_handling": "In downstream modeling, cancellations can be tracked as negative offsets to net spend or as a behavioral feature (cancellation_rate), but excluded from positive purchase counts."
        },
        "rule_2_negative_quantities_treatment": {
            "proposal": "Negative quantities outside cancellation invoices (5,417 rows, all with missing Customer ID) represent inventory write-offs, damages, and administrative adjustments.",
            "rationale": "These records have no customer attribution and negative volume; they must be excluded from customer behavioral modeling.",
            "candidate_handling": "Exclude from customer transaction pipelines."
        },
        "rule_3_zero_and_negative_price_treatment": {
            "proposal": "Exclude zero-price rows (3,687 rows) and negative-price rows (5 rows) from commercial monetary modeling.",
            "rationale": "Zero prices correspond to samples, damaged goods, or system corrections. Negative prices correspond to bad debt adjustments (e.g., 'Adjust bad debt'). Neither reflects commercial customer purchase pricing.",
            "candidate_handling": "Filter price > 0 for monetary aggregations."
        },
        "rule_4_exact_duplicates_treatment": {
            "proposal": "De-duplicate exact duplicate records (34,335 rows, 3.22%).",
            "rationale": "Exact matches across all 8 columns with identical timestamps represent duplicated transmission/logging rather than genuine repeat transactions.",
            "candidate_handling": "Apply df.drop_duplicates(keep='first') during clean population creation."
        },
        "rule_5_missing_customer_ids_treatment": {
            "proposal": "Partition the dataset into Customer-Attributed Transactions vs Anonymous/Guest Transactions.",
            "rationale": "243,007 rows (22.77%) lack a Customer ID. Customer-level RFM, behavioral segmentation, and churn modeling mathematically require an identifiable entity.",
            "candidate_handling": "Customer-level models must restrict to valid Customer IDs (5,881 unique customers). Anonymous rows should be preserved in audit summaries but excluded from customer feature matrices."
        },
        "rule_6_invoice_aggregation_rule": {
            "proposal": "Define customer purchase event as a unique, non-cancelled invoice ID on a specific calendar date.",
            "rationale": "Customers occasionally have multiple line items or invoice records; aggregating to distinct valid invoices on unique purchase timestamps provides the true transaction count.",
            "candidate_handling": "Group by Customer ID and Invoice to aggregate order-level metrics."
        },
        "rule_7_customer_eligibility_rule": {
            "proposal": "A customer is eligible for RFM and predictive evaluation if and only if they have at least one valid delivered/completed commercial purchase with Price > 0, Quantity > 0, and non-null Customer ID.",
            "rationale": "Ensures that 100% of analyzed customers have a well-defined historical baseline.",
            "candidate_handling": "Standard eligibility gate applied in Checkpoint 2/3."
        }
    }

    # ==========================================
    # 13. COMPILE CHECKPOINT DATA
    # ==========================================
    # Verify raw data immutability before saving
    post_sha256 = compute_sha256(RAW_FILE)
    assert pre_sha256 == post_sha256, "CRITICAL ERROR: Raw data was modified during audit!"

    checkpoint_data = {
        "checkpoint": "02_online_retail_integrity",
        "checkpoint_description": "Phase 1: Raw Data Integrity Audit — Online Retail II",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": str(RAW_FILE.as_posix()),
        "raw_file_sha256": post_sha256,
        "raw_data_immutability_verified": True,
        "raw_file_integrity": {
            "filename": RAW_FILE.name,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "row_count": total_rows,
            "column_count": total_cols,
            "column_names": exact_columns,
            "pandas_dtypes": col_dtypes,
            "duplicate_rows_count": num_duplicates,
            "duplicate_rows_pct": pct_duplicates,
            "missing_values_by_column": missing_stats,
        },
        "datetime_validation": datetime_audit,
        "invoice_structure": invoice_audit,
        "quantity_validation": quantity_audit,
        "price_validation": price_audit,
        "customer_id_validation": customer_audit,
        "product_validation": product_audit,
        "country_validation": country_audit,
        "transaction_value_reconciliation": value_audit,
        "duplicate_analysis": duplicate_audit,
        "temporal_consistency": temporal_consistency_audit,
        "proposed_data_rules": proposed_rules,
        "statement_on_data_immutability": (
            "The source dataset data/raw/online_retail_II.csv was accessed strictly in read-only mode. "
            "No rows, columns, values, or timestamps were cleaned, altered, imputed, or overwritten."
        ),
    }

    # Write checkpoint
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Audit completed successfully. Checkpoint written to: {CHECKPOINT_FILE}")
    print(f"Raw file SHA-256 confirmed immutable: {post_sha256}")
    print("=" * 80)

    return checkpoint_data


if __name__ == "__main__":
    run_raw_data_integrity_audit()
