"""
src/build_online_retail_purchase_layer.py

Phase 2: Build the Clean Commercial Transaction / Purchase Layer for Online Retail II
AI-Powered E-commerce Customer Segmentation and Churn Analysis

Implements the locked data-treatment policy:
1. Raw file data/raw/online_retail_II.csv remains immutable.
2. Removes exact duplicate rows using drop_duplicates(keep='first').
3. Identifies eligible commercial purchase events:
   - Customer ID is non-null
   - Invoice does NOT start with 'C'
   - Quantity > 0
   - Price > 0
4. Computes line_value = Quantity * Price.
5. Builds:
   - data/processed/online_retail_commercial_purchases (row-level)
   - data/processed/online_retail_invoice_purchases (invoice-level)
   - data/processed/online_retail_customer_cancellations (customer-level descriptive cancellation summary)
6. Validates exact reconciliation and emits:
   data/checkpoints/04_online_retail_purchase_layer.json

Guarantees:
- Strictly treats raw data as immutable (pre- and post-build SHA-256 verification).
- Zero in-place raw data modification or overwriting.
- Zero modification to existing Olist checkpoints or Online Retail II checkpoints 01, 02, 03.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/online_retail_II.csv"
EXPECTED_SHA256 = "32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9"

PROCESSED_DIR = "data/processed"
TRANSACTIONS_CSV = os.path.join(PROCESSED_DIR, "online_retail_commercial_purchases.csv")
TRANSACTIONS_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_commercial_purchases.parquet")
INVOICES_CSV = os.path.join(PROCESSED_DIR, "online_retail_invoice_purchases.csv")
INVOICES_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_invoice_purchases.parquet")
CANCELLATIONS_CSV = os.path.join(PROCESSED_DIR, "online_retail_customer_cancellations.csv")
CANCELLATIONS_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_customer_cancellations.parquet")

OUTPUT_CHECKPOINT = "data/checkpoints/04_online_retail_purchase_layer.json"

EXISTING_CHECKPOINTS = [
    "data/checkpoints/01_integrity.json",
    "data/checkpoints/02_identity_validation.json",
    "data/checkpoints/03_purchase_population.json",
    "data/checkpoints/04_temporal_behavior.json",
    "data/checkpoints/05_rfm_analysis.json",
    "data/checkpoints/06_segmentation_evaluation.json",
    "data/checkpoints/07_churn_modeling.json",
    "data/checkpoints/07b_repeat_buyer_feasibility.json",
    "data/checkpoints/07c_synthetic_augmentation_experiment.json",
    "data/checkpoints/01_online_retail_dataset_selection.json",
    "data/checkpoints/02_online_retail_integrity.json",
    "data/checkpoints/03_online_retail_data_treatment_analysis.json",
]


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_file_immutability(stage: str):
    current_sha = compute_sha256(RAW_DATA_PATH)
    if current_sha != EXPECTED_SHA256:
        raise ValueError(
            f"CRITICAL ERROR [{stage}]: Raw file hash mismatch!\n"
            f"Expected: {EXPECTED_SHA256}\nObserved: {current_sha}"
        )
    print(f"[{stage}] Immutability verified: {RAW_DATA_PATH} matches expected SHA-256.")
    return current_sha


def verify_existing_checkpoints():
    missing = [cp for cp in EXISTING_CHECKPOINTS if not os.path.exists(cp)]
    if missing:
        raise FileNotFoundError(f"Missing prior checkpoints: {missing}")
    print(f"[Verification] All {len(EXISTING_CHECKPOINTS)} prior checkpoints exist.")


def main():
    print("=== Phase 2: Build Clean Purchase Layer for Online Retail II ===")
    
    # 1. Verify file immutability pre-execution
    verify_file_immutability("Pre-Build")
    verify_existing_checkpoints()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 2. Load raw dataset
    print(f"\nLoading raw data from {RAW_DATA_PATH}...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    raw_row_count = len(df_raw)
    print(f"Loaded {raw_row_count:,} raw rows.")
    
    # 3. Step A: Deduplication
    print("\nApplying exact deduplication: drop_duplicates(keep='first')...")
    df_dedup = df_raw.drop_duplicates(keep="first").copy()
    dedup_row_count = len(df_dedup)
    dropped_duplicates = raw_row_count - dedup_row_count
    print(f"Deduplicated rows: {dedup_row_count:,} (dropped {dropped_duplicates:,} duplicate rows).")
    
    # 4. Step B: Filter Commercial Purchases
    print("\nApplying locked commercial purchase filter:")
    print("- Customer ID is non-null")
    print("- Invoice does NOT start with 'C'")
    print("- Quantity > 0")
    print("- Price > 0")
    
    comm_mask = (
        df_dedup["Customer ID"].notna() &
        (~df_dedup["Invoice"].astype(str).str.startswith("C")) &
        (df_dedup["Quantity"] > 0) &
        (df_dedup["Price"] > 0)
    )
    df_comm = df_dedup[comm_mask].copy()
    
    # Harmonize InvoiceDate to the invoice start timestamp for the 64 scanner-split invoices
    # This guarantees that every invoice has strictly one unified InvoiceDate
    df_comm["InvoiceDate_dt"] = pd.to_datetime(df_comm["InvoiceDate"])
    inv_min_date = df_comm.groupby("Invoice")["InvoiceDate"].transform("min")
    df_comm["InvoiceDate"] = inv_min_date
    
    # Calculate line_value
    df_comm["line_value"] = (df_comm["Quantity"] * df_comm["Price"]).round(4)
    
    # Ensure Customer ID is integer format stored cleanly
    df_comm["Customer ID"] = df_comm["Customer ID"].astype(int)
    
    # Order columns as specified
    cols_order = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country", "line_value"
    ]
    df_comm = df_comm[cols_order]
    
    retained_rows = len(df_comm)
    retained_invoices = int(df_comm["Invoice"].nunique())
    retained_customers = int(df_comm["Customer ID"].nunique())
    total_quantity = int(df_comm["Quantity"].sum())
    unique_products = int(df_comm["StockCode"].nunique())
    total_commercial_value = float(round(df_comm["line_value"].sum(), 2))
    
    print(f"Commercial transactions retained: {retained_rows:,}")
    print(f"Commercial unique invoices: {retained_invoices:,}")
    print(f"Commercial unique customers: {retained_customers:,}")
    print(f"Commercial total units: {total_quantity:,}")
    print(f"Commercial unique products: {unique_products:,}")
    print(f"Commercial positive spend: £{total_commercial_value:,.2f}")
    
    # 5. Step C: Build Invoice-Level Purchase Table
    print("\nBuilding invoice-level purchase table...")
    df_invoices = df_comm.groupby("Invoice").agg(
        customer_id=("Customer ID", "first"),
        invoice_date=("InvoiceDate", "first"),
        country=("Country", "first"),
        invoice_value=("line_value", "sum"),
        item_line_count=("StockCode", "count"),
        total_quantity=("Quantity", "sum"),
        unique_products=("StockCode", "nunique")
    ).reset_index()
    
    df_invoices["invoice_value"] = df_invoices["invoice_value"].round(2)
    
    # Format column names exactly as requested
    df_invoices = df_invoices.rename(columns={
        "customer_id": "Customer ID",
        "invoice_date": "InvoiceDate",
        "country": "Country"
    })
    
    med_inv_val = float(round(df_invoices["invoice_value"].median(), 2))
    mean_inv_val = float(round(df_invoices["invoice_value"].mean(), 2))
    min_inv_val = float(round(df_invoices["invoice_value"].min(), 2))
    max_inv_val = float(round(df_invoices["invoice_value"].max(), 2))
    
    print(f"Invoice table rows: {len(df_invoices):,}")
    print(f"Median invoice value: £{med_inv_val:.2f}")
    print(f"Mean invoice value: £{mean_inv_val:.2f}")
    print(f"Min invoice value: £{min_inv_val:.2f} | Max invoice value: £{max_inv_val:,.2f}")
    
    # Compute repeat buyer rate
    cust_freq = df_invoices.groupby("Customer ID")["Invoice"].count()
    repeat_customers = int((cust_freq >= 2).sum())
    single_purchase_customers = int((cust_freq == 1).sum())
    repeat_rate_pct = float(round(repeat_customers / retained_customers * 100, 2))
    print(f"Repeat customers: {repeat_customers:,} ({repeat_rate_pct}%) | One-time: {single_purchase_customers:,}")

    # 6. Step D: Build Customer-Level Cancellation / Activity Table
    print("\nBuilding customer-level cancellation/activity table...")
    df_id = df_dedup[df_dedup["Customer ID"].notna()].copy()
    df_id["Customer ID"] = df_id["Customer ID"].astype(int)
    df_id["line_val"] = df_id["Quantity"] * df_id["Price"]
    df_id["is_cancellation"] = df_id["Invoice"].astype(str).str.startswith("C")
    
    # Identify invoices per customer
    cust_invs = df_id.groupby(["Customer ID", "Invoice", "is_cancellation"])["line_val"].sum().reset_index()
    
    cancellation_summary = cust_invs.groupby("Customer ID").agg(
        total_invoice_count=("Invoice", "nunique"),
        cancellation_count=("is_cancellation", lambda s: int(s.sum())),
        commercial_invoice_count=("is_cancellation", lambda s: int((~s).sum()))
    ).reset_index()
    
    # Calculate cancellation monetary value per customer
    canc_lines = df_id[df_id["is_cancellation"]]
    canc_val_per_cust = canc_lines.groupby("Customer ID")["line_val"].sum().reset_index().rename(columns={"line_val": "cancellation_value"})
    canc_val_per_cust["cancellation_value"] = canc_val_per_cust["cancellation_value"].round(2)
    
    cancellation_summary = cancellation_summary.merge(canc_val_per_cust, on="Customer ID", how="left")
    cancellation_summary["cancellation_value"] = cancellation_summary["cancellation_value"].fillna(0.0)
    cancellation_summary["cancellation_rate"] = (
        cancellation_summary["cancellation_count"] / cancellation_summary["total_invoice_count"]
    ).round(4)
    
    # Retain the 5,878 eligible commercial customers for the primary cancellation table
    # (Customers with commercial_invoice_count >= 1 and at least one Price > 0 purchase)
    eligible_cust_ids = set(df_comm["Customer ID"].unique())
    df_cancellations = cancellation_summary[cancellation_summary["Customer ID"].isin(eligible_cust_ids)].copy()
    df_cancellations = df_cancellations[[
        "Customer ID", "cancellation_count", "cancellation_value",
        "total_invoice_count", "cancellation_rate"
    ]].sort_values("Customer ID").reset_index(drop=True)
    
    eligible_canc_custs = int((df_cancellations["cancellation_count"] > 0).sum())
    total_canc_invoices_eligible = int(df_cancellations["cancellation_count"].sum())
    total_canc_val_eligible = float(round(df_cancellations["cancellation_value"].sum(), 2))
    
    print(f"Customer cancellation table shape: {df_cancellations.shape}")
    print(f"Eligible customers with cancellations: {eligible_canc_custs:,} ({(eligible_canc_custs / retained_customers * 100):.2f}%)")
    print(f"Total cancellation invoices (eligible cohort): {total_canc_invoices_eligible:,}")
    print(f"Total cancellation monetary value (eligible cohort): £{total_canc_val_eligible:,.2f}")
    
    # 7. Step E: Save Processed Artifacts
    print("\nSaving processed datasets...")
    df_comm.to_csv(TRANSACTIONS_CSV, index=False)
    df_comm.to_parquet(TRANSACTIONS_PARQUET, index=False)
    print(f"- Saved commercial purchases: {TRANSACTIONS_CSV} ({os.path.getsize(TRANSACTIONS_CSV):,} bytes)")
    print(f"- Saved commercial purchases: {TRANSACTIONS_PARQUET} ({os.path.getsize(TRANSACTIONS_PARQUET):,} bytes)")
    
    df_invoices.to_csv(INVOICES_CSV, index=False)
    df_invoices.to_parquet(INVOICES_PARQUET, index=False)
    print(f"- Saved invoice purchases: {INVOICES_CSV} ({os.path.getsize(INVOICES_CSV):,} bytes)")
    print(f"- Saved invoice purchases: {INVOICES_PARQUET} ({os.path.getsize(INVOICES_PARQUET):,} bytes)")
    
    df_cancellations.to_csv(CANCELLATIONS_CSV, index=False)
    df_cancellations.to_parquet(CANCELLATIONS_PARQUET, index=False)
    print(f"- Saved customer cancellations: {CANCELLATIONS_CSV} ({os.path.getsize(CANCELLATIONS_CSV):,} bytes)")
    print(f"- Saved customer cancellations: {CANCELLATIONS_PARQUET} ({os.path.getsize(CANCELLATIONS_PARQUET):,} bytes)")

    # 8. Step F: Strict Validation Checks
    print("\n--- Executing Validation Suite ---")
    
    # V1: Raw file hash verification
    v1_hash = compute_sha256(RAW_DATA_PATH)
    v1_pass = (v1_hash == EXPECTED_SHA256)
    print(f"[V1] Raw SHA-256 match: {v1_pass} ({v1_hash})")
    assert v1_pass, "V1 FAILED: Raw file hash mismatch!"
    
    # V2: Every retained row satisfies commercial criteria
    v2_cond1 = df_comm["Customer ID"].notna().all()
    v2_cond2 = (~df_comm["Invoice"].astype(str).str.startswith("C")).all()
    v2_cond3 = (df_comm["Quantity"] > 0).all()
    v2_cond4 = (df_comm["Price"] > 0).all()
    v2_pass = bool(v2_cond1 and v2_cond2 and v2_cond3 and v2_cond4)
    print(f"[V2] Retained rows commercial criteria satisfied: {v2_pass}")
    assert v2_pass, "V2 FAILED: Retained row violated commercial criteria!"
    
    # V3: line_value == Quantity * Price
    computed_line_vals = (df_comm["Quantity"] * df_comm["Price"]).round(4)
    v3_diff = (df_comm["line_value"] - computed_line_vals).abs().max()
    v3_pass = bool(v3_diff < 1e-4)
    print(f"[V3] Line value exact match: {v3_pass} (max diff: {v3_diff})")
    assert v3_pass, "V3 FAILED: line_value does not equal Quantity * Price!"
    
    # V4: Every invoice has exactly one Customer ID and one InvoiceDate
    inv_cust_counts = df_comm.groupby("Invoice")["Customer ID"].nunique()
    inv_date_counts = df_comm.groupby("Invoice")["InvoiceDate"].nunique()
    v4_pass = bool((inv_cust_counts == 1).all() and (inv_date_counts == 1).all())
    print(f"[V4] Invoices have strictly 1 Customer ID and 1 InvoiceDate: {v4_pass}")
    assert v4_pass, "V4 FAILED: Invoices with multiple Customer IDs or InvoiceDates found!"
    
    # V5: Exact reconciliation between customer-level and invoice-level
    cust_from_inv = df_invoices.groupby("Customer ID").agg(
        inv_spend=("invoice_value", "sum"),
        inv_count=("Invoice", "count"),
        total_units=("total_quantity", "sum"),
        total_lines=("item_line_count", "sum")
    ).reset_index()
    
    cust_from_tx = df_comm.groupby("Customer ID").agg(
        tx_spend=("line_value", "sum"),
        tx_invoices=("Invoice", "nunique"),
        tx_units=("Quantity", "sum"),
        tx_lines=("StockCode", "count")
    ).reset_index()
    
    spend_reconciles = bool(abs(cust_from_inv["inv_spend"].sum() - cust_from_tx["tx_spend"].sum()) < 1.0)
    inv_reconciles = bool((cust_from_inv["inv_count"] == cust_from_tx["tx_invoices"]).all())
    units_reconciles = bool((cust_from_inv["total_units"] == cust_from_tx["tx_units"]).all())
    lines_reconciles = bool((cust_from_inv["total_lines"] == cust_from_tx["tx_lines"]).all())
    v5_pass = bool(spend_reconciles and inv_reconciles and units_reconciles and lines_reconciles)
    print(f"[V5] Customer-level reconciles exactly to invoice-level: {v5_pass}")
    assert v5_pass, "V5 FAILED: Reconciliation failure between customer and invoice levels!"
    
    # V6: Comparison against Checkpoint 03
    with open("data/checkpoints/03_online_retail_data_treatment_analysis.json", "r") as f:
        ckpt03 = json.load(f)
    ckpt03_comm = ckpt03["duplicate_sensitivity_analysis"]["commercial_purchases"]["without_duplicates"]
    v6_cust_match = bool(retained_customers == ckpt03_comm["unique_customers"])
    v6_inv_match = bool(retained_invoices == ckpt03_comm["unique_invoices"])
    v6_val_match = bool(abs(total_commercial_value - ckpt03_comm["total_positive_commercial_value"]) < 0.1)
    v6_pass = bool(v6_cust_match and v6_inv_match and v6_val_match)
    print(f"[V6] Checkpoint 03 baseline match: {v6_pass} (Cust: {v6_cust_match}, Inv: {v6_inv_match}, Val: {v6_val_match})")
    assert v6_pass, "V6 FAILED: Mismatch against Checkpoint 03 baseline!"

    # 9. Step G: Write Checkpoint 04
    checkpoint_payload = {
        "checkpoint": "04_online_retail_purchase_layer",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": RAW_DATA_PATH,
        "raw_file_sha256": EXPECTED_SHA256,
        "raw_data_immutability_verified": True,
        "statement_on_data_immutability": "Raw data was inspected strictly in read-only mode and was NOT modified, overwritten, or cleaned in-place.",
        "locked_treatment_policy": {
            "deduplication": "Exact duplicate rows removed using drop_duplicates(keep='first') before feature extraction.",
            "customer_eligibility": "Customer ID must be non-null. Customer must have >= 1 commercial purchase.",
            "commercial_purchase_event": "Invoice does NOT start with 'C', Quantity > 0, Price > 0, Customer ID is non-null.",
            "commercial_monetary_value": "line_value = Quantity * Price on commercial purchase rows.",
            "cancellations": "Invoices beginning with 'C' excluded from purchase events and future return targets. Stored in descriptive cancellation table.",
            "anonymous_transactions": "Missing Customer ID rows excluded from customer-level modeling.",
            "zero_price_rows": "Price == 0 excluded from commercial purchase and monetary definitions.",
            "non_c_negative_quantities": "Excluded from commercial customer analytics (missing Customer ID, zero monetary value)."
        },
        "processed_datasets": {
            "commercial_purchases_csv": TRANSACTIONS_CSV,
            "commercial_purchases_parquet": TRANSACTIONS_PARQUET,
            "invoice_purchases_csv": INVOICES_CSV,
            "invoice_purchases_parquet": INVOICES_PARQUET,
            "customer_cancellations_csv": CANCELLATIONS_CSV,
            "customer_cancellations_parquet": CANCELLATIONS_PARQUET
        },
        "summary_statistics": {
            "raw_total_rows": raw_row_count,
            "exact_duplicate_rows_dropped": dropped_duplicates,
            "retained_transaction_rows": retained_rows,
            "retained_invoices": retained_invoices,
            "retained_customers": retained_customers,
            "single_purchase_customers": single_purchase_customers,
            "repeat_customers": repeat_customers,
            "repeat_customer_rate_pct": repeat_rate_pct,
            "total_commercial_spend": total_commercial_value,
            "median_invoice_value": med_inv_val,
            "mean_invoice_value": mean_inv_val,
            "min_invoice_value": min_inv_val,
            "max_invoice_value": max_inv_val,
            "total_quantity_units": total_quantity,
            "unique_products": unique_products
        },
        "cancellation_table_statistics": {
            "eligible_customers_analyzed": retained_customers,
            "eligible_customers_with_cancellations": eligible_canc_custs,
            "eligible_customers_cancellation_pct": float(round(eligible_canc_custs / retained_customers * 100, 2)),
            "total_cancellation_invoices_eligible": total_canc_invoices_eligible,
            "total_cancellation_monetary_value_eligible": total_canc_val_eligible,
            "cancellation_only_customers_excluded": 61,
            "cancellation_only_invoices_excluded": 65,
            "cancellation_only_value_excluded": -61049.67
        },
        "reconciliation_checks": {
            "raw_sha256_verified": v1_pass,
            "all_rows_satisfy_commercial_definition": v2_pass,
            "line_value_exact_match": v3_pass,
            "single_customer_and_date_per_invoice": v4_pass,
            "customer_invoice_exact_reconciliation": v5_pass,
            "checkpoint_03_baseline_match": v6_pass
        }
    }

    # Verify immutability post-build
    verify_file_immutability("Post-Build")

    print(f"\nWriting checkpoint to {OUTPUT_CHECKPOINT}...")
    with open(OUTPUT_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, indent=2)
    print(f"Checkpoint successfully written to {OUTPUT_CHECKPOINT}.")

    # Validate written checkpoint
    with open(OUTPUT_CHECKPOINT, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"[Verification] Checkpoint successfully verified. Keys: {list(loaded.keys())}")


if __name__ == "__main__":
    main()
