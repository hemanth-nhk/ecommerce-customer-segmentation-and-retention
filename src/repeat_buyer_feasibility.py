"""
Repeat-Buyer Subpopulation Feasibility Assessment (Phase 7 V2 Feasibility)
AI-Powered E-commerce Customer Segmentation and Churn Analysis
Brazilian E-Commerce Public Dataset by Olist

Evaluates whether restricting predictive modeling strictly to customers with
>= 2 delivered purchases before T_obs satisfies predefined minimum-case gates:
- Gate 1: Total positive returners >= 200
- Gate 2: Projected/test positive returners >= 50 (at 20% stratified test split)

Strictly evaluates 120d and 180d windows identically without model training.
Preserves Phase 7 V1 artifacts intact.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
PROHIBITED_CHECKPOINT_KEYS = {
    "customer_id",
    "customer_unique_id",
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
}
HEX_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )

    return {"orders": orders, "customers": customers}


def audit_window_repeat_buyer_feasibility(
    deliv_orders: pd.DataFrame,
    T_obs: pd.Timestamp,
    window_days: int,
) -> Dict[str, Any]:
    """
    Evaluates repeat-buyer feasibility for a single window:
    1. Pre-cutoff history (<= T_obs): identifies customers with >= 2 delivered orders.
    2. Future window (T_obs, T_obs + window_days]: determines return (1) vs non-return (0).
    """
    T_end = T_obs + pd.Timedelta(days=window_days)

    # --- STAGE 1: Pre-cutoff repeat qualification (<= T_obs) ---
    pre_orders = deliv_orders[deliv_orders["order_purchase_timestamp"] <= T_obs].copy()

    # Temporal assertion: no pre-cutoff order exceeds T_obs
    assert (
        pre_orders["order_purchase_timestamp"].max() <= T_obs
    ), f"Leakage violation: pre-cutoff order timestamp > T_obs for {window_days}d"

    # Group by customer_unique_id and count delivered purchases before T_obs
    pre_counts = pre_orders.groupby("customer_unique_id")["order_id"].nunique()

    # Pre-cutoff repeat buyers: frequency_pre >= 2
    qualifying_repeat_customers = pre_counts[pre_counts >= 2].index
    n_eligible_repeat = len(qualifying_repeat_customers)

    # --- STAGE 2: Future target evaluation (T_obs, T_end] ---
    future_orders = deliv_orders[
        (deliv_orders["order_purchase_timestamp"] > T_obs)
        & (deliv_orders["order_purchase_timestamp"] <= T_end)
    ].copy()

    # Temporal assertions
    if len(future_orders) > 0:
        assert (
            future_orders["order_purchase_timestamp"].min() > T_obs
        ), f"Leakage violation: target order timestamp <= T_obs for {window_days}d"
        assert (
            future_orders["order_purchase_timestamp"].max() <= T_end
        ), f"Censoring violation: target order timestamp > T_end for {window_days}d"

    # Filter future orders to qualifying repeat customers
    future_repeat_orders = future_orders[
        future_orders["customer_unique_id"].isin(qualifying_repeat_customers)
    ]
    returning_repeat_customers = set(
        future_repeat_orders["customer_unique_id"].unique()
    )

    n_positives = len(returning_repeat_customers)
    n_negatives = n_eligible_repeat - n_positives
    return_rate_pct = (
        round((n_positives / n_eligible_repeat) * 100.0, 4)
        if n_eligible_repeat > 0
        else 0.0
    )

    # Construct binary vector for stratified test split evaluation
    y_vec = np.zeros(n_eligible_repeat, dtype=int)
    y_vec[:n_positives] = 1

    # Hypothetical 80/20 stratified split
    if n_positives >= 2 and n_negatives >= 2:
        _, _, y_train, y_test = train_test_split(
            np.zeros(n_eligible_repeat),
            y_vec,
            test_size=0.20,
            stratify=y_vec,
            random_state=RANDOM_STATE,
        )
        n_test_pos = int(np.sum(y_test))
        n_train_pos = int(np.sum(y_train))
        n_test_total = len(y_test)
        n_train_total = len(y_train)
    else:
        n_test_pos = int(round(n_positives * 0.20))
        n_train_pos = n_positives - n_test_pos
        n_test_total = int(round(n_eligible_repeat * 0.20))
        n_train_total = n_eligible_repeat - n_test_total

    # Preregistered gate evaluations
    gate_1_total_positives_pass = n_positives >= 200
    gate_2_test_positives_pass = n_test_pos >= 50
    overall_gate_pass = (
        gate_1_total_positives_pass and gate_2_test_positives_pass
    )

    return {
        "window_days": window_days,
        "cutoff_timestamp_T_obs": T_obs.isoformat(),
        "future_window_end_timestamp": T_end.isoformat(),
        "eligible_repeat_buyers": n_eligible_repeat,
        "future_returners": n_positives,
        "non_returners": n_negatives,
        "return_rate_pct": return_rate_pct,
        "positive_to_negative_ratio": (
            round(n_positives / n_negatives, 6) if n_negatives > 0 else 0.0
        ),
        "hypothetical_20pct_test_split": {
            "test_total_customers": n_test_total,
            "test_positive_returners": n_test_pos,
            "test_prevalence_pct": (
                round((n_test_pos / n_test_total) * 100.0, 4)
                if n_test_total > 0
                else 0.0
            ),
            "train_total_customers": n_train_total,
            "train_positive_returners": n_train_pos,
            "train_prevalence_pct": (
                round((n_train_pos / n_train_total) * 100.0, 4)
                if n_train_total > 0
                else 0.0
            ),
        },
        "preregistered_gates": {
            "gate_1_min_total_positives": {
                "threshold": 200,
                "observed": n_positives,
                "passed": gate_1_total_positives_pass,
            },
            "gate_2_min_test_positives": {
                "threshold": 50,
                "observed": n_test_pos,
                "passed": gate_2_test_positives_pass,
            },
            "overall_feasibility_gate": "PASS" if overall_gate_pass else "FAIL",
            "feasibility_assessment": (
                "FEASIBLE FOR CANDIDATE MODELING"
                if overall_gate_pass
                else "INFEASIBLE: INSUFFICIENT POSITIVE CASE VOLUME (UNDERPOWERED)"
            ),
        },
    }


def validate_checkpoint_privacy(data: Any, path: str = "") -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).lower() in PROHIBITED_CHECKPOINT_KEYS:
                raise ValueError(
                    f"Privacy violation: Prohibited key '{k}' found at path '{path}'!"
                )
            validate_checkpoint_privacy(v, f"{path}.{k}" if path else str(k))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            validate_checkpoint_privacy(item, f"{path}[{idx}]")
    elif isinstance(data, str):
        if HEX_ID_PATTERN.match(data):
            raise ValueError(
                f"Privacy violation: Raw 32-char hex ID '{data}' found at path '{path}'!"
            )


def run_feasibility_audit() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"
    checkpoints_dir = project_root / "data" / "checkpoints"

    # Pre-audit immutability
    all_raw_files = sorted(list(data_dir.glob("*.csv")))
    pre_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}

    tables = load_raw_data(data_dir)
    orders = tables["orders"]
    customers = tables["customers"]

    # Merge orders with customers on customer_id to obtain customer_unique_id
    deliv_orders = orders[orders["order_status"] == "delivered"].copy()
    deliv_orders = deliv_orders.dropna(subset=["order_purchase_timestamp"])
    deliv_orders = deliv_orders.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="inner",
    )

    T_max = deliv_orders["order_purchase_timestamp"].max()

    # 120d window
    T_obs_120 = T_max - pd.Timedelta(days=120)
    res_120 = audit_window_repeat_buyer_feasibility(
        deliv_orders, T_obs_120, 120
    )

    # 180d window
    T_obs_180 = T_max - pd.Timedelta(days=180)
    res_180 = audit_window_repeat_buyer_feasibility(
        deliv_orders, T_obs_180, 180
    )

    # Post-audit immutability
    post_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}
    immutability_verified = pre_hashes == post_hashes
    if not immutability_verified:
        raise RuntimeError("Raw data immutability verification failed!")

    checkpoint_data = {
        "checkpoint": "07b_repeat_buyer_feasibility",
        "checkpoint_description": (
            "Repeat-Buyer Subpopulation Feasibility Assessment (Pre-Cutoff Qualification)"
        ),
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_immutability_verified": immutability_verified,
        "dataset_reference_T_max": T_max.isoformat(),
        "preregistered_feasibility_criteria": {
            "gate_1_min_total_positives": 200,
            "gate_2_min_test_positives": 50,
            "qualification_rule": (
                "frequency_pre >= 2 delivered orders with timestamp <= T_obs"
            ),
            "target_rule": (
                "return_within_window = 1 if >= 1 delivered order in (T_obs, T_obs + W], else 0"
            ),
        },
        "candidate_windows": {
            "120d": res_120,
            "180d": res_180,
        },
        "comparative_feasibility_summary": {
            "120d_gate_status": res_120["preregistered_gates"]["overall_feasibility_gate"],
            "180d_gate_status": res_180["preregistered_gates"]["overall_feasibility_gate"],
        },
    }

    validate_checkpoint_privacy(checkpoint_data)

    out_path = checkpoints_dir / "07b_repeat_buyer_feasibility.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_data


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING REPEAT-BUYER SUBPOPULATION FEASIBILITY AUDIT")
    print("=" * 70)
    data = run_feasibility_audit()
    cw = data["candidate_windows"]
    for w_name in ["120d", "180d"]:
        info = cw[w_name]
        g = info["preregistered_gates"]
        print(f"\nWindow: {w_name}")
        print(f"  Eligible Repeat Buyers (F_pre >= 2): {info['eligible_repeat_buyers']:,}")
        print(f"  Future Returners: {info['future_returners']:,}")
        print(f"  Non-Returners: {info['non_returners']:,}")
        print(f"  Return Rate: {info['return_rate_pct']:.4f}%")
        print(f"  Projected Test Positives (20% split): {info['hypothetical_20pct_test_split']['test_positive_returners']}")
        print(f"  Gate 1 (Total Pos >= 200): {'PASS' if g['gate_1_min_total_positives']['passed'] else 'FAIL'} ({g['gate_1_min_total_positives']['observed']})")
        print(f"  Gate 2 (Test Pos >= 50): {'PASS' if g['gate_2_min_test_positives']['passed'] else 'FAIL'} ({g['gate_2_min_test_positives']['observed']})")
        print(f"  Overall Gate: {g['overall_feasibility_gate']} ({g['feasibility_assessment']})")
    print("\n" + "=" * 70)
    print("Feasibility audit completed successfully. Written to data/checkpoints/07b_repeat_buyer_feasibility.json")
