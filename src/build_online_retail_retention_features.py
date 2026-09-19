"""
src/build_online_retail_retention_features.py

Phase 6: Leakage-Safe Point-in-Time Feature Matrix for 90-Day Future-Return Target
AI-Powered E-commerce Customer Segmentation and Churn Analysis

Inputs:
- data/processed/online_retail_invoice_purchases.parquet
- data/processed/online_retail_commercial_purchases.parquet
- data/raw/online_retail_II.csv (deduplicated for pre-cutoff cancellations)

Outputs:
- data/processed/online_retail_retention_features_90d.csv
- data/processed/online_retail_retention_features_90d.parquet
- data/checkpoints/08_online_retail_retention_features.json

Methodology:
1. Filters cohort to customers with >= 1 commercial purchase on or before T_obs (2011-09-10 12:50:00).
2. Computes binary return target in (T_obs, T_max] (2011-09-10 12:50:00, 2011-12-09 12:50:00].
3. Computes point-in-time RFM, temporal behavior, basket metrics, and cancellations strictly using transactions <= T_obs.
4. Computes point-in-time RFM scores and segment strictly on pre-cutoff quantiles.
5. Performs automated leakage audit (verifying zero post-cutoff data enters features).
6. Computes distribution diagnostics, skewness, and correlations.
7. Evaluates an earlier 90-day cutoff cohort for rolling-origin temporal robustness design.

Guarantees:
- Strictly treats raw data as immutable (pre- and post-execution SHA-256 verification).
- Does not modify any existing checkpoints (01-07).
- Zero ML model training, hyperparameter optimization, threshold tuning, or SMOTE augmentation.
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
INPUT_INVOICES_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_invoice_purchases.parquet")
INPUT_TRANSACTIONS_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_commercial_purchases.parquet")

OUTPUT_CSV = os.path.join(PROCESSED_DIR, "online_retail_retention_features_90d.csv")
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "online_retail_retention_features_90d.parquet")
OUTPUT_CHECKPOINT = "data/checkpoints/08_online_retail_retention_features.json"

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
    "data/checkpoints/05_online_retail_customer_features.json",
    "data/checkpoints/06_online_retail_segmentation_evaluation.json",
    "data/checkpoints/07_online_retail_retention_target.json",
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


def assign_pre_cutoff_rfm_segment(row) -> str:
    r, f = int(row["R_score_pre"]), int(row["F_score_pre"])
    if r in [4, 5]:
        if f in [4, 5]:
            return "Champions"
        elif f in [2, 3]:
            return "Potential Loyalists"
        elif f == 1:
            return "Recent Customers"
    elif r == 3:
        if f in [3, 4, 5]:
            return "Loyal Customers"
        elif f in [1, 2]:
            return "About to Sleep"
    elif r in [1, 2]:
        if f in [4, 5]:
            return "At Risk"
        elif f in [2, 3]:
            return "Hibernating"
        elif f == 1:
            return "About to Sleep" if r == 2 else "Lost"
    return "Other"


def main():
    print("=== Phase 6: Build Leakage-Safe Point-in-Time Feature Matrix ===")
    
    # 1. Verify file immutability pre-execution
    verify_file_immutability("Pre-Execution")
    verify_existing_checkpoints()
    
    # 2. STEP 1: Temporal Cohort Partitioning
    print("\nLoading Phase 2 invoice and transaction layers...")
    inv_df = pd.read_parquet(INPUT_INVOICES_PARQUET)
    tx_df = pd.read_parquet(INPUT_TRANSACTIONS_PARQUET)
    raw_df = pd.read_csv(RAW_DATA_PATH).drop_duplicates(keep="first")
    
    inv_df["InvoiceDate"] = pd.to_datetime(inv_df["InvoiceDate"])
    tx_df["InvoiceDate"] = pd.to_datetime(tx_df["InvoiceDate"])
    raw_df["InvoiceDate"] = pd.to_datetime(raw_df["InvoiceDate"])
    
    t_max = inv_df["InvoiceDate"].max()
    t_obs = t_max - pd.Timedelta(days=90)
    
    print(f"Target Horizon: W = 90 days")
    print(f"Observation Cutoff (T_obs): {t_obs}")
    print(f"Dataset End (T_max): {t_max}")
    print(f"Target Window: ({t_obs}, {t_max}]")
    
    # Partition invoice layer
    pre_inv = inv_df[inv_df["InvoiceDate"] <= t_obs].copy()
    post_inv = inv_df[(inv_df["InvoiceDate"] > t_obs) & (inv_df["InvoiceDate"] <= t_max)].copy()
    pre_tx = tx_df[tx_df["InvoiceDate"] <= t_obs].copy()
    
    eligible_cust_ids = sorted(list(pre_inv["Customer ID"].unique()))
    n_eligible = len(eligible_cust_ids)
    print(f"\n[STEP 1] Eligible cohort size (purchases on/before T_obs): {n_eligible:,}")
    assert n_eligible == 5281, f"Expected 5,281 eligible customers, found {n_eligible}"
    
    # Construct binary return target
    post_cust_ids = set(post_inv["Customer ID"].unique())
    returners = set(eligible_cust_ids).intersection(post_cust_ids)
    n_pos = len(returners)
    n_neg = n_eligible - n_pos
    pos_rate = float(round(n_pos / n_eligible * 100, 2))
    
    print(f"Target distribution: Positives={n_pos:,} ({pos_rate}%), Negatives={n_neg:,}")
    assert n_pos == 2292, f"Expected 2,292 positives, found {n_pos}"
    assert n_neg == 2989, f"Expected 2,989 negatives, found {n_neg}"
    
    # 3. STEP 2 & 3: Point-in-Time RFM & Temporal Behavior Features
    print("\nComputing point-in-time RFM and temporal behavior features...")
    pre_inv = pre_inv.sort_values(["Customer ID", "InvoiceDate"]).reset_index(drop=True)
    pre_inv["active_month"] = pre_inv["InvoiceDate"].dt.to_period("M")
    pre_inv["prev_date"] = pre_inv.groupby("Customer ID")["InvoiceDate"].shift(1)
    pre_inv["gap_days"] = (pre_inv["InvoiceDate"] - pre_inv["prev_date"]).dt.total_seconds() / 86400.0
    
    cust_rfm = pre_inv.groupby("Customer ID").agg(
        first_purchase_date_pre=("InvoiceDate", "min"),
        last_purchase_date_pre=("InvoiceDate", "max"),
        frequency_pre=("Invoice", "count"),
        monetary_total_pre=("invoice_value", "sum"),
        average_order_value_pre=("invoice_value", "mean"),
        median_order_value_pre=("invoice_value", "median"),
        number_of_active_months_pre=("active_month", "nunique"),
        inter_purchase_mean_days_pre=("gap_days", "mean"),
        inter_purchase_median_days_pre=("gap_days", "median"),
        inter_purchase_max_days_pre=("gap_days", "max")
    ).reset_index()
    
    cust_rfm["recency_days_pre"] = ((t_obs - cust_rfm["last_purchase_date_pre"]).dt.total_seconds() / 86400.0).round(4)
    cust_rfm["customer_tenure_days_pre"] = ((t_obs - cust_rfm["first_purchase_date_pre"]).dt.total_seconds() / 86400.0).round(4)
    cust_rfm["activity_lifespan_days_pre"] = ((cust_rfm["last_purchase_date_pre"] - cust_rfm["first_purchase_date_pre"]).dt.total_seconds() / 86400.0).round(4)
    cust_rfm["purchase_frequency_per_active_month_pre"] = (cust_rfm["frequency_pre"] / cust_rfm["number_of_active_months_pre"]).round(4)
    
    cust_rfm["monetary_total_pre"] = cust_rfm["monetary_total_pre"].round(2)
    cust_rfm["average_order_value_pre"] = cust_rfm["average_order_value_pre"].round(2)
    cust_rfm["median_order_value_pre"] = cust_rfm["median_order_value_pre"].round(2)
    cust_rfm["inter_purchase_mean_days_pre"] = cust_rfm["inter_purchase_mean_days_pre"].round(4)
    cust_rfm["inter_purchase_median_days_pre"] = cust_rfm["inter_purchase_median_days_pre"].round(4)
    cust_rfm["inter_purchase_max_days_pre"] = cust_rfm["inter_purchase_max_days_pre"].round(4)

    # 4. STEP 4: Basket & Product Behavior Features (Pre-Cutoff)
    print("Computing point-in-time basket and product diversity features...")
    cust_tx = pre_tx.groupby("Customer ID").agg(
        total_items_pre=("StockCode", "count"),
        total_units_pre=("Quantity", "sum"),
        unique_products_pre=("StockCode", "nunique"),
        unique_countries_pre=("Country", "nunique"),
        primary_country_pre=("Country", lambda s: s.mode()[0])
    ).reset_index()
    
    features = cust_rfm.merge(cust_tx, on="Customer ID")
    features["average_items_per_order_pre"] = (features["total_items_pre"] / features["frequency_pre"]).round(4)
    features["average_units_per_order_pre"] = (features["total_units_pre"] / features["frequency_pre"]).round(4)
    features["product_diversity_ratio_pre"] = (features["unique_products_pre"] / features["total_items_pre"]).round(4)

    # 5. STEP 5: Cancellation Features (Pre-Cutoff Only)
    print("Computing point-in-time cancellation features (pre-T_obs only)...")
    raw_id = raw_df[raw_df["Customer ID"].notna()].copy()
    raw_id["Customer ID"] = raw_id["Customer ID"].astype(int)
    raw_id["is_canc"] = raw_id["Invoice"].astype(str).str.startswith("C")
    
    pre_canc = raw_id[
        (raw_id["InvoiceDate"] <= t_obs) &
        (raw_id["is_canc"]) &
        (raw_id["Customer ID"].isin(set(eligible_cust_ids)))
    ].copy()
    pre_canc["line_val"] = pre_canc["Quantity"] * pre_canc["Price"]
    
    canc_stats = pre_canc.groupby("Customer ID").agg(
        cancellation_count_pre=("Invoice", "nunique"),
        cancellation_value_pre=("line_val", "sum")
    ).reset_index()
    
    features = features.merge(canc_stats, on="Customer ID", how="left")
    features["cancellation_count_pre"] = features["cancellation_count_pre"].fillna(0).astype(int)
    features["cancellation_value_pre"] = features["cancellation_value_pre"].fillna(0.0).round(2)
    features["cancellation_rate_pre"] = (
        features["cancellation_count_pre"] / (features["cancellation_count_pre"] + features["frequency_pre"])
    ).round(4)

    # 6. STEP 6: Point-in-Time RFM Segmentation
    print("Recomputing point-in-time RFM segment strictly from pre-cutoff quantiles...")
    features["R_score_pre"], r_edges = pd.qcut(features["recency_days_pre"], q=5, labels=[5, 4, 3, 2, 1], retbins=True)
    features["R_score_pre"] = features["R_score_pre"].astype(int)
    
    f_bins = [0, 1, 2, 4, 8, float("inf")]
    features["F_score_pre"] = pd.cut(features["frequency_pre"], bins=f_bins, labels=[1, 2, 3, 4, 5]).astype(int)
    
    features["M_score_pre"], m_edges = pd.qcut(features["monetary_total_pre"], q=5, labels=[1, 2, 3, 4, 5], retbins=True)
    features["M_score_pre"] = features["M_score_pre"].astype(int)
    
    features["pre_cutoff_rfm_segment"] = features.apply(assign_pre_cutoff_rfm_segment, axis=1)
    
    pre_rfm_boundaries = {
        "recency_edges": [float(round(e, 4)) for e in r_edges],
        "frequency_bins": [0, 1, 2, 4, 8, "inf"],
        "monetary_edges": [float(round(e, 4)) for e in m_edges],
        "segment_distribution": features["pre_cutoff_rfm_segment"].value_counts().to_dict()
    }
    print(f"Pre-cutoff RFM segment distribution:\n{pre_rfm_boundaries['segment_distribution']}")

    # 7. STEP 7: Append Binary Target and Format Matrix
    features["return_within_90d"] = features["Customer ID"].apply(lambda cid: 1 if cid in returners else 0).astype(int)
    
    # Format dates as ISO strings
    features["first_purchase_date_pre"] = features["first_purchase_date_pre"].dt.strftime("%Y-%m-%d %H:%M:%S")
    features["last_purchase_date_pre"] = features["last_purchase_date_pre"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    ordered_cols = [
        "Customer ID",
        "return_within_90d",
        "recency_days_pre",
        "frequency_pre",
        "monetary_total_pre",
        "average_order_value_pre",
        "median_order_value_pre",
        "customer_tenure_days_pre",
        "activity_lifespan_days_pre",
        "number_of_active_months_pre",
        "purchase_frequency_per_active_month_pre",
        "inter_purchase_mean_days_pre",
        "inter_purchase_median_days_pre",
        "inter_purchase_max_days_pre",
        "total_items_pre",
        "total_units_pre",
        "unique_products_pre",
        "average_items_per_order_pre",
        "average_units_per_order_pre",
        "product_diversity_ratio_pre",
        "unique_countries_pre",
        "primary_country_pre",
        "cancellation_count_pre",
        "cancellation_value_pre",
        "cancellation_rate_pre",
        "R_score_pre",
        "F_score_pre",
        "M_score_pre",
        "pre_cutoff_rfm_segment",
        "first_purchase_date_pre",
        "last_purchase_date_pre"
    ]
    features = features[ordered_cols].sort_values("Customer ID").reset_index(drop=True)
    
    print(f"\nFinal feature matrix: {features.shape[0]:,} rows x {features.shape[1]} columns.")
    
    # Save artifacts
    print("Saving processed predictive datasets...")
    features.to_csv(OUTPUT_CSV, index=False)
    features.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"- Saved CSV: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV):,} bytes)")
    print(f"- Saved Parquet: {OUTPUT_PARQUET} ({os.path.getsize(OUTPUT_PARQUET):,} bytes)")

    # 8. STEP 8 & 9: Automated Leakage Audit & Target Integrity Checks
    print("\n--- Executing Automated Leakage Audit ---")
    
    max_feature_tx_ts = pre_tx["InvoiceDate"].max()
    max_feature_inv_ts = pre_inv["InvoiceDate"].max()
    max_feature_canc_ts = pre_canc["InvoiceDate"].max() if len(pre_canc) > 0 else t_obs
    min_target_ts = post_inv["InvoiceDate"].min()
    max_target_ts = post_inv["InvoiceDate"].max()
    
    leakage_check_1 = bool(max_feature_tx_ts <= t_obs)
    leakage_check_2 = bool(max_feature_inv_ts <= t_obs)
    leakage_check_3 = bool(max_feature_canc_ts <= t_obs)
    leakage_check_4 = bool(min_target_ts > t_obs)
    leakage_check_5 = bool(max_target_ts <= t_max)
    
    print(f"[Leakage 1] Max feature transaction timestamp <= T_obs: {leakage_check_1} ({max_feature_tx_ts})")
    print(f"[Leakage 2] Max feature invoice timestamp <= T_obs: {leakage_check_2} ({max_feature_inv_ts})")
    print(f"[Leakage 3] Max feature cancellation timestamp <= T_obs: {leakage_check_3} ({max_feature_canc_ts})")
    print(f"[Leakage 4] Min target invoice timestamp > T_obs: {leakage_check_4} ({min_target_ts})")
    print(f"[Leakage 5] Max target invoice timestamp <= T_max: {leakage_check_5} ({max_target_ts})")
    
    assert leakage_check_1 and leakage_check_2 and leakage_check_3 and leakage_check_4 and leakage_check_5, "Leakage audit FAILED!"
    
    # Target integrity
    target_integrity_1 = bool(len(features) == 5281)
    target_integrity_2 = bool(features["Customer ID"].duplicated().sum() == 0)
    target_integrity_3 = bool(set(features["return_within_90d"].unique()) == {0, 1})
    target_integrity_4 = bool(features["return_within_90d"].isna().sum() == 0)
    target_integrity_5 = bool((features["return_within_90d"] == 1).sum() == 2292)
    target_integrity_6 = bool((features["return_within_90d"] == 0).sum() == 2989)
    
    assert target_integrity_1 and target_integrity_2 and target_integrity_3 and target_integrity_4 and target_integrity_5 and target_integrity_6, "Target integrity FAILED!"

    # 9. STEP 10: Feature Diagnostics & Correlations
    print("\n--- Feature Diagnostics & Distributions ---")
    numeric_feature_cols = [
        "recency_days_pre", "frequency_pre", "monetary_total_pre", "average_order_value_pre", "median_order_value_pre",
        "customer_tenure_days_pre", "activity_lifespan_days_pre", "number_of_active_months_pre",
        "purchase_frequency_per_active_month_pre", "inter_purchase_mean_days_pre", "inter_purchase_median_days_pre",
        "inter_purchase_max_days_pre", "total_items_pre", "total_units_pre", "unique_products_pre",
        "average_items_per_order_pre", "average_units_per_order_pre", "product_diversity_ratio_pre",
        "cancellation_count_pre", "cancellation_value_pre", "cancellation_rate_pre"
    ]
    
    diagnostics = {}
    highly_skewed_features = []
    zero_inflated_features = []
    log_transform_candidates = []
    
    for col in numeric_feature_cols:
        s = features[col].dropna()
        n_missing = int(features[col].isna().sum())
        pct_missing = float(round(n_missing / len(features) * 100, 2))
        skew_val = float(round(s.skew(), 4))
        p_min = float(round(s.min(), 4))
        med = float(round(s.median(), 4))
        p95 = float(round(s.quantile(0.95), 4))
        p_max = float(round(s.max(), 4))
        zero_pct = float(round((features[col] == 0).sum() / len(features) * 100, 2))
        
        if abs(skew_val) > 2.0:
            highly_skewed_features.append(col)
        if zero_pct > 30.0:
            zero_inflated_features.append(col)
        if skew_val > 2.0 and p_min >= 0:
            log_transform_candidates.append(col)
            
        diagnostics[col] = {
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "min": p_min,
            "median": med,
            "p95": p95,
            "max": p_max,
            "skewness": skew_val,
            "zero_pct": zero_pct
        }
        
    print(f"Highly skewed features (|skew| > 2.0): {len(highly_skewed_features)}")
    print(f"Zero-inflated features (>30% zeros): {zero_inflated_features}")
    print(f"Candidate log-transform features: {log_transform_candidates}")
    
    # Correlations among key RFM features
    rfm_corr_cols = [
        "recency_days_pre", "frequency_pre", "monetary_total_pre",
        "average_order_value_pre", "customer_tenure_days_pre",
        "total_items_pre", "total_units_pre", "return_within_90d"
    ]
    rfm_corr_matrix = features[rfm_corr_cols].corr().round(4).to_dict()
    print("\nCorrelations with Target (return_within_90d):")
    for col in rfm_corr_cols[:-1]:
        print(f"  {col:<25}: r = {rfm_corr_matrix['return_within_90d'][col]:>7.4f}")

    # 10. STEP 11: Temporal Robustness Design (Earlier 90-Day Cohort)
    print("\n--- STEP 11: Temporal Robustness Design ---")
    t_obs_early = t_obs - pd.Timedelta(days=90)  # 2011-06-12 12:50:00
    pre_early = inv_df[inv_df["InvoiceDate"] <= t_obs_early]
    post_early = inv_df[(inv_df["InvoiceDate"] > t_obs_early) & (inv_df["InvoiceDate"] <= t_obs)]
    
    elig_early = set(pre_early["Customer ID"].unique())
    post_custs_early = set(post_early["Customer ID"].unique())
    ret_early = elig_early.intersection(post_custs_early)
    
    temporal_robustness_spec = {
        "design_concept": "Rolling-Origin Out-of-Time Temporal Cross-Validation",
        "primary_cohort": {
            "name": "Cohort 1 (Primary Model Cohort)",
            "cutoff_t_obs": str(t_obs),
            "target_window": f"({str(t_obs)}, {str(t_max)}]",
            "eligible_customers": n_eligible,
            "positive_returners": n_pos,
            "non_returners": n_neg,
            "positive_rate_pct": pos_rate
        },
        "earlier_benchmark_cohort": {
            "name": "Cohort 0 (Earlier Out-of-Time Validation Cohort)",
            "cutoff_t_obs": str(t_obs_early),
            "target_window": f"({str(t_obs_early)}, {str(t_obs)}]",
            "pre_cutoff_history_days": float(round((t_obs_early - inv_df['InvoiceDate'].min()).total_seconds() / 86400.0, 2)),
            "eligible_customers": len(elig_early),
            "positive_returners": len(ret_early),
            "non_returners": len(elig_early) - len(ret_early),
            "positive_rate_pct": float(round(len(ret_early) / len(elig_early) * 100, 2)),
            "status": "Defined for future out-of-time evaluation; NOT used as training data yet."
        }
    }
    print(f"Earlier Cohort: Cutoff={t_obs_early} | Eligible={len(elig_early):,} | Returners={len(ret_early):,} ({len(ret_early)/len(elig_early)*100:.2f}%)")

    # 11. STEP 12: Checkpoint Emission & Immutability Verification
    verify_existing_checkpoints()
    verify_file_immutability("Post-Execution")
    
    checkpoint_payload = {
        "checkpoint": "08_online_retail_retention_features",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": RAW_DATA_PATH,
        "raw_file_sha256": EXPECTED_SHA256,
        "raw_data_immutability_verified": True,
        "statement_on_data_immutability": "Raw data was inspected strictly in read-only mode and was NOT modified, overwritten, or cleaned in-place.",
        "temporal_parameters": {
            "selected_window_days": 90,
            "t_obs": str(t_obs),
            "t_max": str(t_max),
            "pre_cutoff_span_days": float(round((t_obs - inv_df['InvoiceDate'].min()).total_seconds() / 86400.0, 2))
        },
        "cohort_and_target_summary": {
            "eligible_customers": n_eligible,
            "positive_returners": n_pos,
            "non_returners": n_neg,
            "positive_prevalence_pct": pos_rate,
            "expected_20pct_test_positives": float(round(n_pos * 0.20, 1)),
            "expected_80pct_train_positives": float(round(n_pos * 0.80, 1))
        },
        "pre_cutoff_rfm_segmentation": pre_rfm_boundaries,
        "leakage_audit_results": {
            "max_feature_transaction_timestamp": str(max_feature_tx_ts),
            "max_feature_invoice_timestamp": str(max_feature_inv_ts),
            "max_feature_cancellation_timestamp": str(max_feature_canc_ts),
            "min_target_timestamp": str(min_target_ts),
            "max_target_timestamp": str(max_target_ts),
            "temporal_isolation_verified": True,
            "zero_post_cutoff_leakage": True
        },
        "feature_diagnostics": diagnostics,
        "feature_engineering_recommendations": {
            "highly_skewed_features": highly_skewed_features,
            "zero_inflated_features": zero_inflated_features,
            "log_transform_candidates": log_transform_candidates,
            "missing_treatment_single_buyers": "inter_purchase_mean/median/max_days_pre are preserved as NaN for 1,576 single-purchase buyers (frequency_pre=1)."
        },
        "correlation_with_target": {col: rfm_corr_matrix["return_within_90d"][col] for col in rfm_corr_cols[:-1]},
        "temporal_robustness_specification": temporal_robustness_spec,
        "processed_artifacts": {
            "features_csv": OUTPUT_CSV,
            "features_parquet": OUTPUT_PARQUET
        },
        "validation_checks": {
            "eligible_customer_count_verified": target_integrity_1,
            "unique_customer_ids_verified": target_integrity_2,
            "binary_target_verified": target_integrity_3,
            "zero_missing_targets_verified": target_integrity_4,
            "positive_cases_verified": target_integrity_5,
            "negative_cases_verified": target_integrity_6,
            "leakage_tests_passed": bool(leakage_check_1 and leakage_check_2 and leakage_check_3 and leakage_check_4 and leakage_check_5),
            "raw_sha256_verified": bool(compute_sha256(RAW_DATA_PATH) == EXPECTED_SHA256)
        }
    }

    print(f"\nWriting checkpoint to {OUTPUT_CHECKPOINT}...")
    with open(OUTPUT_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, indent=2)
    print(f"Checkpoint successfully written to {OUTPUT_CHECKPOINT}.")

    # Validate written checkpoint
    with open(OUTPUT_CHECKPOINT, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"[Verification] Checkpoint 08 verified. Keys: {list(loaded.keys())}")


if __name__ == "__main__":
    main()
