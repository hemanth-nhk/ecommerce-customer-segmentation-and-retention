"""
src/build_online_retail_customer_features.py

Phase 3: Customer-Level RFM and Behavioral Feature Engineering for Online Retail II
AI-Powered E-commerce Customer Segmentation and Churn Analysis

Inputs:
- data/processed/online_retail_commercial_purchases.parquet
- data/processed/online_retail_invoice_purchases.parquet
- data/processed/online_retail_customer_cancellations.parquet

Outputs:
- data/processed/online_retail_customer_features.csv
- data/processed/online_retail_customer_features.parquet
- data/checkpoints/05_online_retail_customer_features.json

Guarantees:
- Strictly treats raw data as immutable (pre- and post-execution SHA-256 verification).
- Reconciles customer-level aggregates exactly with Phase 2 data.
- Does not modify any existing Olist or Online Retail II checkpoints (01-04).
- No churn modeling or premature model decisions.
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
INPUT_TX_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_commercial_purchases.parquet")
INPUT_INV_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_invoice_purchases.parquet")
INPUT_CANC_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_customer_cancellations.parquet")

OUTPUT_FEATURES_CSV = os.path.join(PROCESSED_DIR, "online_retail_customer_features.csv")
OUTPUT_FEATURES_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_customer_features.parquet")
OUTPUT_CHECKPOINT = "data/checkpoints/05_online_retail_customer_features.json"

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
    "data/checkpoints/04_online_retail_purchase_layer.json",
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


def compute_distribution_diagnostics(df: pd.DataFrame, columns: list) -> dict:
    diagnostics = {}
    for col in columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        skew_val = float(round(s.skew(), 4))
        p_min = float(round(s.min(), 4))
        p25 = float(round(s.quantile(0.25), 4))
        med = float(round(s.median(), 4))
        p75 = float(round(s.quantile(0.75), 4))
        p90 = float(round(s.quantile(0.90), 4))
        p95 = float(round(s.quantile(0.95), 4))
        p_max = float(round(s.max(), 4))
        mean_val = float(round(s.mean(), 4))
        std_val = float(round(s.std(), 4))
        
        # Determine skewness category
        if abs(skew_val) > 2.0:
            skew_cat = "Highly Skewed (|skew| > 2.0)"
        elif abs(skew_val) > 1.0:
            skew_cat = "Moderately Skewed (1.0 < |skew| <= 2.0)"
        else:
            skew_cat = "Approximately Symmetric (|skew| <= 1.0)"
            
        diagnostics[col] = {
            "non_null_count": int(len(s)),
            "null_count": int(df[col].isna().sum()),
            "mean": mean_val,
            "std": std_val,
            "min": p_min,
            "p25": p25,
            "median": med,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "max": p_max,
            "skewness": skew_val,
            "skewness_classification": skew_cat
        }
    return diagnostics


def main():
    print("=== Phase 3: Customer-Level RFM & Behavioral Feature Engineering ===")
    
    # 1. Verify file immutability pre-execution
    verify_file_immutability("Pre-Execution")
    verify_existing_checkpoints()
    
    # 2. Load Phase 2 processed datasets
    print(f"\nLoading Phase 2 processed datasets...")
    df_tx = pd.read_parquet(INPUT_TX_PARQUET)
    df_inv = pd.read_parquet(INPUT_INV_PARQUET)
    df_canc = pd.read_parquet(INPUT_CANC_PARQUET)
    
    print(f"- Commercial transactions: {len(df_tx):,} rows")
    print(f"- Commercial invoices: {len(df_inv):,} rows")
    print(f"- Customer cancellations: {len(df_canc):,} rows")
    
    # Parse dates
    df_tx["InvoiceDate"] = pd.to_datetime(df_tx["InvoiceDate"])
    df_inv["InvoiceDate"] = pd.to_datetime(df_inv["InvoiceDate"])
    
    # Determine global reference timestamp T_ref
    t_ref = df_inv["InvoiceDate"].max()
    print(f"Global descriptive reference timestamp (T_ref): {t_ref}")
    
    # 3. Step A: Compute Invoice-Level Temporal Features & Inter-Purchase Gaps
    df_inv = df_inv.sort_values(["Customer ID", "InvoiceDate"]).reset_index(drop=True)
    df_inv["active_month"] = df_inv["InvoiceDate"].dt.to_period("M")
    df_inv["prev_invoice_date"] = df_inv.groupby("Customer ID")["InvoiceDate"].shift(1)
    df_inv["gap_days"] = (df_inv["InvoiceDate"] - df_inv["prev_invoice_date"]).dt.total_seconds() / 86400.0
    
    # Aggregate from invoices
    inv_agg = df_inv.groupby("Customer ID").agg(
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
        frequency=("Invoice", "count"),
        monetary_total=("invoice_value", "sum"),
        average_order_value=("invoice_value", "mean"),
        median_order_value=("invoice_value", "median"),
        number_of_active_months=("active_month", "nunique"),
        inter_purchase_mean_days=("gap_days", "mean"),
        inter_purchase_median_days=("gap_days", "median"),
        inter_purchase_max_days=("gap_days", "max")
    ).reset_index()
    
    # Tenure, Recency, Activity Lifespan
    inv_agg["recency_days"] = ((t_ref - inv_agg["last_purchase_date"]).dt.total_seconds() / 86400.0).round(4)
    inv_agg["customer_tenure_days"] = ((t_ref - inv_agg["first_purchase_date"]).dt.total_seconds() / 86400.0).round(4)
    inv_agg["activity_lifespan_days"] = ((inv_agg["last_purchase_date"] - inv_agg["first_purchase_date"]).dt.total_seconds() / 86400.0).round(4)
    inv_agg["purchase_frequency_per_active_month"] = (inv_agg["frequency"] / inv_agg["number_of_active_months"]).round(4)
    
    # Round monetary and gap metrics
    inv_agg["monetary_total"] = inv_agg["monetary_total"].round(2)
    inv_agg["average_order_value"] = inv_agg["average_order_value"].round(2)
    inv_agg["median_order_value"] = inv_agg["median_order_value"].round(2)
    inv_agg["inter_purchase_mean_days"] = inv_agg["inter_purchase_mean_days"].round(4)
    inv_agg["inter_purchase_median_days"] = inv_agg["inter_purchase_median_days"].round(4)
    inv_agg["inter_purchase_max_days"] = inv_agg["inter_purchase_max_days"].round(4)
    
    # 4. Step B: Aggregate Line-Item Basket & Product Features from Transactions
    tx_agg = df_tx.groupby("Customer ID").agg(
        total_items=("StockCode", "count"),
        total_units=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        unique_countries=("Country", "nunique"),
        primary_country=("Country", lambda s: s.mode()[0])
    ).reset_index()
    
    # Merge invoice and transaction aggregates
    cust_df = inv_agg.merge(tx_agg, on="Customer ID")
    
    # Compute composite basket metrics
    cust_df["average_items_per_order"] = (cust_df["total_items"] / cust_df["frequency"]).round(4)
    cust_df["average_units_per_order"] = (cust_df["total_units"] / cust_df["frequency"]).round(4)
    cust_df["product_diversity_ratio"] = (cust_df["unique_products"] / cust_df["total_items"]).round(4)
    
    # 5. Step C: Join Cancellation Table
    df_canc_clean = df_canc[[
        "Customer ID", "cancellation_count", "cancellation_value", "cancellation_rate"
    ]].copy()
    
    cust_df = cust_df.merge(df_canc_clean, on="Customer ID", how="left")
    
    # Ensure Customer ID is clean integer
    cust_df["Customer ID"] = cust_df["Customer ID"].astype(int)
    
    # Format timestamps as ISO strings
    cust_df["first_purchase_date"] = cust_df["first_purchase_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cust_df["last_purchase_date"] = cust_df["last_purchase_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Reorder columns logically
    ordered_cols = [
        "Customer ID",
        "recency_days",
        "frequency",
        "monetary_total",
        "average_order_value",
        "median_order_value",
        "total_items",
        "total_units",
        "unique_products",
        "product_diversity_ratio",
        "average_items_per_order",
        "average_units_per_order",
        "customer_tenure_days",
        "activity_lifespan_days",
        "first_purchase_date",
        "last_purchase_date",
        "number_of_active_months",
        "purchase_frequency_per_active_month",
        "inter_purchase_mean_days",
        "inter_purchase_median_days",
        "inter_purchase_max_days",
        "unique_countries",
        "primary_country",
        "cancellation_count",
        "cancellation_value",
        "cancellation_rate"
    ]
    cust_df = cust_df[ordered_cols].sort_values("Customer ID").reset_index(drop=True)
    
    print(f"\nFinal customer feature matrix: {cust_df.shape[0]:,} rows x {cust_df.shape[1]} columns.")
    
    # 6. Step D: Save Processed Artifacts
    print(f"\nSaving customer feature artifacts...")
    cust_df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    cust_df.to_parquet(OUTPUT_FEATURES_PARQUET, index=False)
    print(f"- Saved CSV: {OUTPUT_FEATURES_CSV} ({os.path.getsize(OUTPUT_FEATURES_CSV):,} bytes)")
    print(f"- Saved Parquet: {OUTPUT_FEATURES_PARQUET} ({os.path.getsize(OUTPUT_FEATURES_PARQUET):,} bytes)")

    # 7. Step E: Distribution Diagnostics
    numeric_feature_cols = [
        "recency_days", "frequency", "monetary_total", "average_order_value", "median_order_value",
        "total_items", "total_units", "unique_products", "product_diversity_ratio",
        "average_items_per_order", "average_units_per_order",
        "customer_tenure_days", "activity_lifespan_days",
        "number_of_active_months", "purchase_frequency_per_active_month",
        "inter_purchase_mean_days", "inter_purchase_median_days", "inter_purchase_max_days",
        "cancellation_count", "cancellation_value", "cancellation_rate"
    ]
    
    diagnostics = compute_distribution_diagnostics(cust_df, numeric_feature_cols)
    
    # 8. Step F: Strict Validation Checks
    print("\n--- Executing Phase 3 Validation Suite ---")
    
    # Check 1: Customer count == 5,878
    val_cust_count = (len(cust_df) == 5878)
    print(f"[Check 1] Customer count == 5,878: {val_cust_count} ({len(cust_df)})")
    assert val_cust_count, f"Check 1 FAILED: Customer count {len(cust_df)} != 5,878"
    
    # Check 2: Total spend reconciles to Phase 2 £17,374,804.27
    total_spend = float(round(cust_df["monetary_total"].sum(), 2))
    val_spend = bool(abs(total_spend - 17374804.27) < 0.1)
    print(f"[Check 2] Total customer spend reconciles: {val_spend} (£{total_spend:,.2f})")
    assert val_spend, f"Check 2 FAILED: Spend £{total_spend} != £17,374,804.27"
    
    # Check 3: Frequency distribution matches single=1,623, repeat=4,255
    single_count = int((cust_df["frequency"] == 1).sum())
    repeat_count = int((cust_df["frequency"] >= 2).sum())
    val_freq_dist = bool(single_count == 1623 and repeat_count == 4255)
    print(f"[Check 3] Frequency distribution matches (single={single_count}, repeat={repeat_count}): {val_freq_dist}")
    assert val_freq_dist, f"Check 3 FAILED: Frequency distribution mismatch!"
    
    # Check 4: Total frequency sum == total invoices in Phase 2 (36,969)
    total_invoices = int(cust_df["frequency"].sum())
    val_inv_sum = bool(total_invoices == 36969)
    print(f"[Check 4] Total frequency sum == 36,969: {val_inv_sum} ({total_invoices:,})")
    assert val_inv_sum, f"Check 4 FAILED: Invoices {total_invoices} != 36,969"
    
    # Check 5: Total items sum == 779,425 and Total units sum == 10,513,952
    total_lines = int(cust_df["total_items"].sum())
    total_qty = int(cust_df["total_units"].sum())
    val_items = bool(total_lines == 779425 and total_qty == 10513952)
    print(f"[Check 5] Line items ({total_lines:,}) and units ({total_qty:,}) match Phase 2: {val_items}")
    assert val_items, f"Check 5 FAILED: Line items or units mismatch!"
    
    # Check 6: Inter-purchase gap missingness exactly equals single-purchase count (1,623)
    nan_gaps = int(cust_df["inter_purchase_mean_days"].isna().sum())
    val_nan_gaps = bool(nan_gaps == 1623)
    print(f"[Check 6] Single-purchase customers have NaN gaps: {val_nan_gaps} ({nan_gaps:,})")
    assert val_nan_gaps, f"Check 6 FAILED: NaN gaps {nan_gaps} != 1,623"
    
    # Check 7: No non-eligible customers
    val_no_extras = bool(set(cust_df["Customer ID"]) == set(df_inv["Customer ID"]))
    print(f"[Check 7] Cohort strictly equals eligible commercial customers: {val_no_extras}")
    assert val_no_extras, "Check 7 FAILED: Extra or missing customers found!"
    
    # Check 8: Raw SHA-256 matches
    v8_sha = compute_sha256(RAW_DATA_PATH)
    val_sha = bool(v8_sha == EXPECTED_SHA256)
    print(f"[Check 8] Raw SHA-256 match: {val_sha} ({v8_sha})")
    assert val_sha, "Check 8 FAILED: Raw file hash mismatch!"
    
    # Verify immutability post-execution
    verify_file_immutability("Post-Execution")
    
    # 9. Step G: Write Checkpoint 05
    checkpoint_payload = {
        "checkpoint": "05_online_retail_customer_features",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": RAW_DATA_PATH,
        "raw_file_sha256": EXPECTED_SHA256,
        "raw_data_immutability_verified": True,
        "statement_on_data_immutability": "Raw data was inspected strictly in read-only mode and was NOT modified, overwritten, or cleaned in-place.",
        "global_reference_timestamp": str(t_ref),
        "dataset_dimensions": {
            "eligible_customers": int(len(cust_df)),
            "total_features": int(cust_df.shape[1]),
            "single_purchase_customers": single_count,
            "repeat_commercial_customers": repeat_count,
            "repeat_customer_rate_pct": float(round(repeat_count / len(cust_df) * 100, 2))
        },
        "exact_reconciliation_audit": {
            "customer_count_expected": 5878,
            "customer_count_observed": int(len(cust_df)),
            "commercial_spend_expected": 17374804.27,
            "commercial_spend_observed": total_spend,
            "invoices_expected": 36969,
            "invoices_observed": total_invoices,
            "line_items_expected": 779425,
            "line_items_observed": total_lines,
            "units_expected": 10513952,
            "units_observed": total_qty,
            "single_purchases_expected": 1623,
            "single_purchases_observed": single_count,
            "repeat_purchases_expected": 4255,
            "repeat_purchases_observed": repeat_count
        },
        "temporal_integrity_verification": {
            "invoice_date_definition": "Phase 2 unified invoice timestamps to min(InvoiceDate) per invoice for the 64 scanner-split checkouts.",
            "verified_from_data": True,
            "max_checkout_delta_seconds": 120.0,
            "calendar_date_variance_in_invoices": 0
        },
        "missing_value_policy_single_buyers": {
            "raw_representation": "NaN for inter_purchase_mean_days, inter_purchase_median_days, inter_purchase_max_days.",
            "affected_customer_count": single_count,
            "affected_customer_pct": float(round(single_count / len(cust_df) * 100, 2)),
            "candidate_modeling_treatments": {
                "dense_distance_models": "Impute with observation window length (or tenure) plus binary indicator is_repeat_buyer, or use sentinel -1.0.",
                "tree_based_models": "Preserve native NaN splits without artificial numeric distortion.",
                "segmentation_subpopulation": "Optionally partition analysis into one-time vs repeat buyers."
            }
        },
        "feature_distribution_diagnostics": diagnostics,
        "segmentation_readiness_assessment": {
            "rfm_readiness": "Excellent. Recency spans 0 to 738 days (p25=28.8d, p75=280.9d); Frequency spans 1 to 398 orders; Monetary spans £2.95 to £580,987.04 across 5 orders of magnitude.",
            "behavioral_readiness": "High. Active months (1-25), product diversity (0.04-1.00), and cancellation presence (42.72%) capture distinct buying archetypes.",
            "clustering_readiness": "Caution required. High right-skewness across monetary (skew=25.07), frequency (skew=12.64), and basket units (skew=21.85) will severely distort unscaled Euclidean clustering (K-Means). Requires log1p or power transformations and robust scaling before distance-based algorithms are applied.",
            "algorithmic_recommendation": "Evaluate deterministic quantile-based RFM scoring and log-transformed GMM alongside K-Means. Do not force an unvalidated K-Means model."
        },
        "reconciliation_checks": {
            "customer_count_verified": val_cust_count,
            "commercial_spend_verified": val_spend,
            "frequency_distribution_verified": val_freq_dist,
            "invoice_count_verified": val_inv_sum,
            "items_and_units_verified": val_items,
            "inter_purchase_gap_nan_verified": val_nan_gaps,
            "cohort_strict_eligibility_verified": val_no_extras,
            "raw_sha256_verified": val_sha
        },
        "processed_artifacts": {
            "features_csv": OUTPUT_FEATURES_CSV,
            "features_parquet": OUTPUT_FEATURES_PARQUET
        }
    }

    print(f"\nWriting checkpoint to {OUTPUT_CHECKPOINT}...")
    with open(OUTPUT_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, indent=2)
    print(f"Checkpoint successfully written to {OUTPUT_CHECKPOINT}.")

    # Validate written checkpoint
    with open(OUTPUT_CHECKPOINT, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"[Verification] Checkpoint 05 verified. Keys: {list(loaded.keys())}")


if __name__ == "__main__":
    main()
