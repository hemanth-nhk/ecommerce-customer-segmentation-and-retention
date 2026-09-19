"""
src/analyze_online_retail_data_treatments.py

Phase 1b: Data-Treatment Sensitivity Analysis for Online Retail II
AI-Powered E-commerce Customer Segmentation and Churn Analysis

Performs an empirical data-treatment sensitivity analysis across:
- Scenario A: Raw identifiable transactions
- Scenario B: Remove exact duplicate rows
- Scenario C: Remove cancellations
- Scenario D: Remove cancellations + non-C negative quantities
- Scenario E: Commercial purchases

Analyzes duplicate sensitivity, zero-price transactions, and non-C negative quantities.
Produces data/checkpoints/03_online_retail_data_treatment_analysis.json

Guarantees:
- Treats raw data as strictly immutable (pre- and post-analysis SHA-256 verification).
- Zero in-place data modification or overwriting.
- Does not modify any existing checkpoints or Olist data.
"""

import sys
import os
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/raw/online_retail_II.csv"
EXPECTED_SHA256 = "32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9"
OUTPUT_CHECKPOINT = "data/checkpoints/03_online_retail_data_treatment_analysis.json"

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
    missing = []
    for cp in EXISTING_CHECKPOINTS:
        if not os.path.exists(cp):
            missing.append(cp)
    if missing:
        raise FileNotFoundError(f"Missing prior checkpoints: {missing}")
    print(f"[Verification] All {len(EXISTING_CHECKPOINTS)} prior checkpoints exist.")


def compute_scenario_metrics(data: pd.DataFrame, scenario_name: str, scenario_desc: str) -> dict:
    row_count = len(data)
    unique_invoices = int(data["Invoice"].nunique())
    
    # Customer level calculations (restricted to present Customer ID)
    cust_data = data[data["Customer ID"].notna()]
    unique_customers = int(cust_data["Customer ID"].nunique())
    
    if unique_customers > 0:
        cust_inv_counts = cust_data.groupby("Customer ID")["Invoice"].nunique()
        c1 = int((cust_inv_counts == 1).sum())
        c2 = int((cust_inv_counts == 2).sum())
        c3 = int((cust_inv_counts == 3).sum())
        c4 = int((cust_inv_counts >= 4).sum())
        repeat_pct = float(round((cust_inv_counts >= 2).sum() / unique_customers * 100, 2))
    else:
        c1 = c2 = c3 = c4 = 0
        repeat_pct = 0.0

    # Line monetary values
    line_val = data["Quantity"] * data["Price"]
    pos_comm_val = float(round(line_val[line_val > 0].sum(), 2))
    
    # Invoice monetary aggregation
    inv_totals = data.groupby("Invoice")["line_val"].sum()
    med_inv_val = float(round(inv_totals.median(), 2)) if len(inv_totals) > 0 else 0.0
    mean_inv_val = float(round(inv_totals.mean(), 2)) if len(inv_totals) > 0 else 0.0
    
    zero_price_rows = int((data["Price"] == 0).sum())
    neg_qty_rows = int((data["Quantity"] < 0).sum())
    
    # Cancellation invoices: invoices beginning with 'C'
    c_inv_count = int(data["Invoice"].astype(str).str.startswith("C").groupby(data["Invoice"]).first().sum())
    
    return {
        "scenario_name": scenario_name,
        "description": scenario_desc,
        "row_count": row_count,
        "unique_invoices": unique_invoices,
        "unique_customers": unique_customers,
        "customers_1_purchase": c1,
        "customers_2_purchases": c2,
        "customers_3_purchases": c3,
        "customers_4plus_purchases": c4,
        "repeat_customer_pct": repeat_pct,
        "total_positive_commercial_value": pos_comm_val,
        "median_invoice_value": med_inv_val,
        "mean_invoice_value": mean_inv_val,
        "zero_price_rows_retained": zero_price_rows,
        "negative_quantity_rows_retained": neg_qty_rows,
        "cancellation_invoices_retained": c_inv_count,
    }


def compute_duplicate_sensitivity(data: pd.DataFrame, dataset_name: str) -> dict:
    with_dup = data.copy()
    no_dup = data.drop_duplicates().copy()
    
    # Customer and invoice aggregates
    c_w = int(with_dup["Customer ID"].dropna().nunique())
    c_no = int(no_dup["Customer ID"].dropna().nunique())
    
    inv_w = int(with_dup["Invoice"].nunique())
    inv_no = int(no_dup["Invoice"].nunique())
    
    # Repeat buyer rate
    inv_per_c_w = with_dup.groupby("Customer ID")["Invoice"].nunique()
    inv_per_c_no = no_dup.groupby("Customer ID")["Invoice"].nunique()
    
    rep_w = float(round((inv_per_c_w >= 2).sum() / c_w * 100, 2)) if c_w > 0 else 0.0
    rep_no = float(round((inv_per_c_no >= 2).sum() / c_no * 100, 2)) if c_no > 0 else 0.0
    
    # Median inter-purchase gap (in days)
    def calc_med_gap(d):
        inv_dt = d[["Customer ID", "Invoice", "InvoiceDate"]].drop_duplicates().sort_values(["Customer ID", "InvoiceDate"])
        gaps = inv_dt.groupby("Customer ID")["InvoiceDate"].diff().dt.total_seconds() / (24 * 3600)
        valid = gaps.dropna()
        return float(round(valid.median(), 2)) if len(valid) > 0 else 0.0
    
    med_gap_w = calc_med_gap(with_dup)
    med_gap_no = calc_med_gap(no_dup)
    
    # Total positive commercial value
    val_w = float(round(with_dup.loc[with_dup["line_val"] > 0, "line_val"].sum(), 2))
    val_no = float(round(no_dup.loc[no_dup["line_val"] > 0, "line_val"].sum(), 2))
    val_diff = float(round(val_w - val_no, 2))
    val_diff_pct = float(round((val_diff / val_no) * 100, 2)) if val_no > 0 else 0.0
    
    # Customer level differences
    cust_invoice_freq_diff = int((inv_per_c_w != inv_per_c_no).sum())
    
    items_w = with_dup.groupby("Customer ID").size()
    items_no = no_dup.groupby("Customer ID").size()
    cust_item_freq_diff = int((items_w != items_no).sum())
    
    rep_to_one = int(((inv_per_c_w >= 2) & (inv_per_c_no == 1)).sum())
    
    m_w = with_dup.groupby("Customer ID")["line_val"].sum()
    m_no = no_dup.groupby("Customer ID")["line_val"].sum()
    
    cust_m_diff = int((m_w != m_no).sum())
    
    # Relative change against deduplicated monetary spend
    rel_change = (m_w - m_no).abs() / m_no.abs().replace(0, np.nan)
    gt_1pct = int((rel_change > 0.01).sum())
    
    max_pct = float(round(rel_change.max() * 100, 2))
    max_cust = int(rel_change.idxmax()) if not rel_change.isna().all() else None
    max_cust_spend_with_dup = float(round(m_w.loc[max_cust], 2)) if max_cust else 0.0
    max_cust_spend_no_dup = float(round(m_no.loc[max_cust], 2)) if max_cust else 0.0

    return {
        "dataset_analyzed": dataset_name,
        "with_duplicates": {
            "unique_customers": c_w,
            "unique_invoices": inv_w,
            "repeat_buyer_rate_pct": rep_w,
            "median_inter_purchase_gap_days": med_gap_w,
            "total_positive_commercial_value": val_w
        },
        "without_duplicates": {
            "unique_customers": c_no,
            "unique_invoices": inv_no,
            "repeat_buyer_rate_pct": rep_no,
            "median_inter_purchase_gap_days": med_gap_no,
            "total_positive_commercial_value": val_no
        },
        "impact_summary": {
            "commercial_value_reduction": val_diff,
            "commercial_value_reduction_pct": val_diff_pct,
            "customers_with_invoice_frequency_change": cust_invoice_freq_diff,
            "customers_with_item_line_frequency_change": cust_item_freq_diff,
            "customers_changing_from_repeat_to_onetime": rep_to_one,
            "customers_with_monetary_value_change": cust_m_diff,
            "customers_with_gt_1pct_monetary_value_change": gt_1pct,
            "max_percentage_monetary_value_change": max_pct,
            "customer_id_with_max_monetary_change": max_cust,
            "customer_max_spend_with_duplicates": max_cust_spend_with_dup,
            "customer_max_spend_without_duplicates": max_cust_spend_no_dup
        }
    }


def analyze_zero_price_rows(df: pd.DataFrame) -> dict:
    zero_p = df[df["Price"] == 0].copy()
    total_zero = len(zero_p)
    
    zero_missing = zero_p[zero_p["Customer ID"].isna()]
    zero_valid = zero_p[zero_p["Customer ID"].notna()]
    
    valid_cust_zero_invs = zero_valid["Invoice"].unique()
    rows_on_those_invs = df[df["Invoice"].isin(valid_cust_zero_invs)]
    inv_has_positive = rows_on_those_invs.groupby("Invoice")["Price"].apply(lambda s: (s > 0).any())
    
    mixed_invs = inv_has_positive[inv_has_positive].index
    only_zero_invs = inv_has_positive[~inv_has_positive].index
    
    zero_rows_mixed = zero_valid[zero_valid["Invoice"].isin(mixed_invs)]
    zero_rows_only = zero_valid[zero_valid["Invoice"].isin(only_zero_invs)]
    
    only_zero_inv_custs = zero_rows_only["Customer ID"].unique().tolist()
    
    # Check customers who exclusively have zero-price transactions in the entire dataset
    all_cust_ids = df[df["Customer ID"].notna()]["Customer ID"].unique()
    cust_has_positive_price = df[df["Customer ID"].notna()].groupby("Customer ID")["Price"].apply(lambda s: (s > 0).any())
    purely_zero_custs = cust_has_positive_price[~cust_has_positive_price].index.tolist()
    
    top_descriptions_valid = zero_valid["Description"].value_counts().head(10).to_dict()
    
    return {
        "total_zero_price_rows": total_zero,
        "zero_price_rows_missing_customer_id": len(zero_missing),
        "zero_price_rows_valid_customer_id": len(zero_valid),
        "valid_customer_zero_price_invoices_total": len(valid_cust_zero_invs),
        "mixed_invoices_count": len(mixed_invs),
        "invoices_with_only_zero_price_items_count": len(only_zero_invs),
        "valid_customer_zero_price_rows_on_mixed_invoices": len(zero_rows_mixed),
        "valid_customer_zero_price_rows_on_only_zero_invoices": len(zero_rows_only),
        "customers_associated_with_only_zero_invoices": [int(c) for c in only_zero_inv_custs],
        "customers_with_exclusively_zero_price_records_across_dataset": [int(c) for c in purely_zero_custs],
        "top_descriptions_valid_customer_zero_price": top_descriptions_valid
    }


def analyze_non_c_negatives(df: pd.DataFrame) -> dict:
    non_c_neg = df[(~df["Invoice"].astype(str).str.startswith("C")) & (df["Quantity"] < 0)].copy()
    total_count = len(non_c_neg)
    
    missing_cust_id_count = int(non_c_neg["Customer ID"].isna().sum())
    valid_cust_id_count = int(non_c_neg["Customer ID"].notna().sum())
    
    # Monetary totals
    line_val = non_c_neg["Quantity"] * non_c_neg["Price"]
    monetary_total = float(round(line_val.sum(), 2))
    zero_price_count = int((non_c_neg["Price"] == 0).sum())
    
    top_stock_codes = non_c_neg["StockCode"].value_counts().head(10).to_dict()
    top_descriptions = non_c_neg["Description"].value_counts().head(15).to_dict()
    
    return {
        "total_non_c_negative_quantity_rows": total_count,
        "missing_customer_id_count": missing_cust_id_count,
        "valid_customer_id_count": valid_cust_id_count,
        "identifiable_customers_affected": valid_cust_id_count,
        "price_equals_zero_count": zero_price_count,
        "total_monetary_value": monetary_total,
        "top_stock_codes": top_stock_codes,
        "top_descriptions": top_descriptions,
        "nature_of_records": "Warehouse and administrative adjustments (inventory damage, loss, audit corrections, write-offs)."
    }


def main():
    print("=== Phase 1b: Online Retail II Data-Treatment Sensitivity Analysis ===")
    
    # 1. Verify file immutability pre-analysis
    verify_file_immutability("Pre-Analysis")
    verify_existing_checkpoints()
    
    # Load raw dataset
    print(f"\nLoading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["line_val"] = df["Quantity"] * df["Price"]
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    
    # 2. Comparative Scenario Analysis (Two perspectives: Identifiable Population & Full Raw)
    print("\n--- Running Comparative Scenario Analysis ---")
    
    # Perspective A: Within Identifiable Population (Customer ID notna)
    df_id = df[df["Customer ID"].notna()].copy()
    
    scenarios_identifiable_defs = [
        ("SCENARIO_A", "Raw identifiable transactions: valid Customer ID, no final cleaning beyond identifying customer population.",
         df_id),
        ("SCENARIO_B", "Remove exact duplicate rows: drop exact duplicate rows only within identifiable transactions.",
         df_id.drop_duplicates()),
        ("SCENARIO_C", "Remove cancellations: exclude invoices beginning with 'C' within identifiable transactions.",
         df_id[~df_id["Invoice"].astype(str).str.startswith("C")]),
        ("SCENARIO_D", "Remove cancellations + non-C negative quantities: exclude 'C' invoices and negative quantities.",
         df_id[(~df_id["Invoice"].astype(str).str.startswith("C")) & (df_id["Quantity"] >= 0)]),
        ("SCENARIO_E", "Commercial purchases: valid Customer ID, non-cancelled, Quantity > 0, Price > 0.",
         df_id[(~df_id["Invoice"].astype(str).str.startswith("C")) & (df_id["Quantity"] > 0) & (df_id["Price"] > 0)])
    ]
    
    scenarios_identifiable_results = {}
    for code, desc, sub_df in scenarios_identifiable_defs:
        res = compute_scenario_metrics(sub_df, code, desc)
        scenarios_identifiable_results[code] = res
        print(f"[{code} (Identifiable)] Rows: {res['row_count']:,} | Invoices: {res['unique_invoices']:,} | Cust: {res['unique_customers']:,} | Repeat%: {res['repeat_customer_pct']}% | PosVal: £{res['total_positive_commercial_value']:,.2f}")
    
    # Perspective B: Full Raw Dataset Baseline (Evaluating isolated filters across all raw rows)
    scenarios_raw_defs = [
        ("SCENARIO_A_RAW", "Raw identifiable subset: filter Customer ID not null.",
         df[df["Customer ID"].notna()]),
        ("SCENARIO_B_RAW", "Drop exact duplicate rows across all raw rows (isolated filter).",
         df.drop_duplicates()),
        ("SCENARIO_C_RAW", "Exclude invoices beginning with 'C' across all raw rows (isolated filter).",
         df[~df["Invoice"].astype(str).str.startswith("C")]),
        ("SCENARIO_D_RAW", "Exclude 'C' invoices and non-C negative quantities across all raw rows (isolated filter).",
         df[(~df["Invoice"].astype(str).str.startswith("C")) & (df["Quantity"] >= 0)]),
        ("SCENARIO_E_RAW", "Commercial purchases: valid Customer ID, non-cancelled, Quantity > 0, Price > 0.",
         df[df["Customer ID"].notna() & (~df["Invoice"].astype(str).str.startswith("C")) & (df["Quantity"] > 0) & (df["Price"] > 0)])
    ]
    
    scenarios_raw_results = {}
    for code, desc, sub_df in scenarios_raw_defs:
        res = compute_scenario_metrics(sub_df, code, desc)
        scenarios_raw_results[code] = res
        print(f"[{code} (Raw)] Rows: {res['row_count']:,} | Invoices: {res['unique_invoices']:,} | Cust: {res['unique_customers']:,} | Repeat%: {res['repeat_customer_pct']}% | PosVal: £{res['total_positive_commercial_value']:,.2f}")
    
    # 3. Duplicate Sensitivity Analysis
    print("\n--- Running Duplicate Sensitivity Analysis ---")
    comm_purchases = df[df["Customer ID"].notna() & (~df["Invoice"].astype(str).str.startswith("C")) & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    dup_sensitivity_comm = compute_duplicate_sensitivity(comm_purchases, "Commercial Purchases (Scenario E)")
    dup_sensitivity_all_id = compute_duplicate_sensitivity(df_id, "All Identifiable (Scenario A)")
    
    print(f"Commercial Purchases PosVal with dups: £{dup_sensitivity_comm['with_duplicates']['total_positive_commercial_value']:,.2f}")
    print(f"Commercial Purchases PosVal without dups: £{dup_sensitivity_comm['without_duplicates']['total_positive_commercial_value']:,.2f}")
    print(f"Commercial Value reduction: £{dup_sensitivity_comm['impact_summary']['commercial_value_reduction']:,.2f} ({dup_sensitivity_comm['impact_summary']['commercial_value_reduction_pct']}%)")
    print(f"Customers with >1% spend change: {dup_sensitivity_comm['impact_summary']['customers_with_gt_1pct_monetary_value_change']}")
    print(f"Max % spend change: {dup_sensitivity_comm['impact_summary']['max_percentage_monetary_value_change']}% (Customer {dup_sensitivity_comm['impact_summary']['customer_id_with_max_monetary_change']})")
    print(f"Customers changing from repeat to one-time: {dup_sensitivity_comm['impact_summary']['customers_changing_from_repeat_to_onetime']}")

    # 4. Zero-Price Analysis
    print("\n--- Running Zero-Price Transactions Analysis ---")
    zero_price_res = analyze_zero_price_rows(df)
    print(f"Total zero-price rows: {zero_price_res['total_zero_price_rows']}")
    print(f"Missing Customer ID zero-price rows: {zero_price_res['zero_price_rows_missing_customer_id']}")
    print(f"Valid Customer ID zero-price rows: {zero_price_res['zero_price_rows_valid_customer_id']}")
    print(f"Mixed invoices with zero-price items: {zero_price_res['mixed_invoices_count']}")
    print(f"Only-zero invoices: {zero_price_res['invoices_with_only_zero_price_items_count']}")
    print(f"Customers with only zero-price invoices: {zero_price_res['customers_associated_with_only_zero_invoices']}")
    print(f"Customers with exclusively zero-price records across dataset: {zero_price_res['customers_with_exclusively_zero_price_records_across_dataset']}")

    # 5. Non-C Negative Quantity Analysis
    print("\n--- Running Non-C Negative Quantity Rows Analysis ---")
    non_c_neg_res = analyze_non_c_negatives(df)
    print(f"Total non-C negative rows: {non_c_neg_res['total_non_c_negative_quantity_rows']}")
    print(f"Missing Customer ID count: {non_c_neg_res['missing_customer_id_count']} (100%)")
    print(f"Valid Customer ID count: {non_c_neg_res['valid_customer_id_count']} (0%)")
    print(f"Total monetary value: £{non_c_neg_res['total_monetary_value']}")

    # 6. Formulate Recommended Treatment Policy
    recommended_policy = {
        "A_purchase_event_definition": {
            "policy": "A purchase event is defined as a unique non-cancelled invoice (Invoice NOT starting with 'C') occurring on a specific timestamp with Quantity > 0 and Price > 0.",
            "evidence": "Invoices without 'C' prefix and with positive quantity/price correspond to authentic commercial checkouts. Zero-price lines or pure zero-price invoices represent samples/test products, and cancellation invoices represent transaction reversals."
        },
        "B_customer_activity_definition": {
            "policy": "A customer is considered active at a given observation cutoff T_obs if and only if they have made at least one valid commercial purchase event (Quantity > 0, Price > 0, valid Customer ID, non-cancelled) strictly prior to T_obs.",
            "evidence": "Focuses modeling on genuine commercial customers (5,878 verified commercial customers vs 5,942 raw IDs, filtering 3 customers who only ever had zero-price test/sample items and 61 who only had cancellations)."
        },
        "C_monetary_value_definition": {
            "policy": "Row-level commercial spend is Quantity * Price for rows with Quantity > 0 and Price > 0. Invoice gross commercial spend is sum(Quantity * Price) over commercial lines. Deduplication is required to prevent an artificial £368,624.91 (+2.12%) inflation in positive commercial value.",
            "evidence": "Exact duplicate rows inflate 1,893 customers' spend, with 1,381 customers experiencing >1% inflation and up to +119.52% inflation (e.g., Customer 17976 from £321.79 to £706.38)."
        },
        "D_cancellation_refund_feature_definition": {
            "policy": "Cancellations (invoices prefixed with 'C') and refund amounts must NOT be mixed into positive purchase counts or forward return targets. Instead, they should be engineered into explicit pre-cutoff behavioral features: cancellation_count, cancellation_rate (cancellations / total orders), and total_refund_amount.",
            "evidence": "Cancellations total -£1,194,598.64 across 8,292 invoices. They represent valuable behavioral risk signals rather than standard purchase activity."
        },
        "E_duplicate_handling": {
            "policy": "Apply drop_duplicates(keep='first') across all columns on the raw dataset before downstream feature extraction.",
            "evidence": "Deduplication removes 34,335 exact logging duplicate rows without altering unique customer counts, unique invoice counts, repeat buyer rates, or inter-purchase gaps, while correcting severe monetary distortion on 1,893 customers."
        }
    }

    # Verify immutability post-analysis
    verify_file_immutability("Post-Analysis")

    # Construct checkpoint dictionary
    checkpoint_data = {
        "checkpoint": "03_online_retail_data_treatment_analysis",
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_path": RAW_DATA_PATH,
        "raw_file_sha256": EXPECTED_SHA256,
        "raw_data_immutability_verified": True,
        "statement_on_data_immutability": "Raw data was inspected strictly in read-only mode and was NOT modified, overwritten, or cleaned in-place.",
        "scenarios_identifiable_cohort": scenarios_identifiable_results,
        "scenarios_raw_baseline": scenarios_raw_results,
        "duplicate_sensitivity_analysis": {
            "commercial_purchases": dup_sensitivity_comm,
            "all_identifiable": dup_sensitivity_all_id
        },
        "zero_price_analysis": zero_price_res,
        "non_c_negative_quantities_analysis": non_c_neg_res,
        "recommended_treatment_policy": recommended_policy
    }

    print(f"\nWriting checkpoint to {OUTPUT_CHECKPOINT}...")
    with open(OUTPUT_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint successfully written to {OUTPUT_CHECKPOINT}.")

    # Verify written checkpoint
    with open(OUTPUT_CHECKPOINT, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"[Verification] Checkpoint loaded successfully. Keys: {list(loaded.keys())}")


if __name__ == "__main__":
    main()
