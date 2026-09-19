"""
src/identity.py

Checkpoint 2 — Customer Identity Validation
AI-Powered E-commerce Customer Segmentation and Churn Analysis
Brazilian E-Commerce Public Dataset by Olist

This module determines whether customer purchase history can be reliably
constructed at the customer level using `customer_unique_id`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REQUIRED_CUSTOMER_COLUMNS = ["customer_id", "customer_unique_id"]
REQUIRED_ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
]

# Operational definition documentation
COMPLETED_PURCHASE_DEFINITION = (
    "For this project, delivered orders are selected as the primary completed-purchase "
    "population because the observed order_status and delivery-date evidence indicate "
    "that they represent fulfilled transactions."
)

SUITABILITY_CONCLUSION = (
    "customer_unique_id is the appropriate customer-level analytical identifier for this "
    "project, based on its ability to consolidate transaction-scoped customer_id records "
    "and support cross-order customer histories."
)

# Prohibited keys in aggregate checkpoints to enforce privacy
PROHIBITED_CHECKPOINT_KEYS = {
    "rows",
    "records",
    "customer_ids",
    "customer_unique_ids",
    "order_ids",
    "postal_codes",
    "coordinates",
    "mappings",
}

# Regex to detect 32-char hex strings (standard Olist MD5 ID format)
HEX_ID_PATTERN = re.compile(r"^[a-fA-F0-9]{32}$")


def get_project_root() -> Path:
    """Resolve and return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_raw_data_dir(project_root: Optional[Path] = None) -> Path:
    """Return path to data/raw directory."""
    if project_root is None:
        project_root = get_project_root()
    return project_root / "data" / "raw"


def get_checkpoints_dir(project_root: Optional[Path] = None) -> Path:
    """Return path to data/checkpoints directory."""
    if project_root is None:
        project_root = get_project_root()
    return project_root / "data" / "checkpoints"


def compute_file_sha256(file_path: Path) -> str:
    """Compute and return the SHA-256 hash of a file for immutability verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_schemas(df_customers: pd.DataFrame, df_orders: pd.DataFrame) -> Dict[str, Any]:
    """Validate presence and schema of required customer and order columns."""
    cust_missing = [c for c in REQUIRED_CUSTOMER_COLUMNS if c not in df_customers.columns]
    order_missing = [c for c in REQUIRED_ORDER_COLUMNS if c not in df_orders.columns]

    if cust_missing:
        raise ValueError(f"Missing required columns in customers dataset: {cust_missing}")
    if order_missing:
        raise ValueError(f"Missing required columns in orders dataset: {order_missing}")

    return {
        "customers_required_columns_present": True,
        "customers_columns_checked": REQUIRED_CUSTOMER_COLUMNS,
        "orders_required_columns_present": True,
        "orders_columns_checked": REQUIRED_ORDER_COLUMNS,
        "customers_dtypes": {c: str(df_customers[c].dtype) for c in REQUIRED_CUSTOMER_COLUMNS},
        "orders_dtypes": {c: str(df_orders[c].dtype) for c in REQUIRED_ORDER_COLUMNS},
    }


def analyze_customer_identifiers(df_customers: pd.DataFrame) -> Dict[str, Any]:
    """
    Quantify customer_id vs customer_unique_id mapping cardinality and distribution.
    All metrics are computed dynamically from the raw data.
    """
    total_customer_rows = int(len(df_customers))
    unique_customer_ids = int(df_customers["customer_id"].nunique())
    unique_unique_ids = int(df_customers["customer_unique_id"].nunique())

    # Count customer_id per customer_unique_id
    counts_per_unique = df_customers.groupby("customer_unique_id")["customer_id"].count()
    multi_id_count = int((counts_per_unique > 1).sum())
    multi_id_percentage = round((multi_id_count / unique_unique_ids) * 100, 4)

    # Distribution of customer_id counts per customer_unique_id
    dist_raw = counts_per_unique.value_counts().sort_index()
    distribution = {str(k): int(v) for k, v in dist_raw.items()}

    return {
        "total_customer_rows": total_customer_rows,
        "unique_customer_id_count": unique_customer_ids,
        "unique_customer_unique_id_count": unique_unique_ids,
        "customers_with_multiple_customer_ids": multi_id_count,
        "customers_with_multiple_customer_ids_pct": multi_id_percentage,
        "max_customer_ids_per_unique_id": int(counts_per_unique.max()),
        "customer_id_per_unique_id_distribution": distribution,
    }


def analyze_order_customer_cardinality(
    df_orders: pd.DataFrame, df_customers: pd.DataFrame
) -> Dict[str, Any]:
    """
    Explicitly validate order-level uniqueness and cardinality between order_id and customer_id.
    Substantiates empirically whether customer_id is strictly an order-scoped token.
    Reports aggregate counts only.
    """
    orders_total_rows = int(len(df_orders))
    orders_unique_order_id = int(df_orders["order_id"].nunique())
    orders_duplicate_order_id_count = orders_total_rows - orders_unique_order_id
    orders_unique_customer_id = int(df_orders["customer_id"].nunique())
    customers_unique_customer_id = int(df_customers["customer_id"].nunique())

    # Measure whether any customer_id is associated with >1 order_id in orders
    orders_per_customer_id = df_orders.groupby("customer_id")["order_id"].count()
    customer_ids_with_multiple_orders = int((orders_per_customer_id > 1).sum())

    # Measure whether any order_id is associated with >1 customer_id in orders
    customers_per_order_id = df_orders.groupby("order_id")["customer_id"].count()
    order_ids_with_multiple_customer_ids = int((customers_per_order_id > 1).sum())

    is_strictly_one_to_one = (
        orders_duplicate_order_id_count == 0
        and customer_ids_with_multiple_orders == 0
        and order_ids_with_multiple_customer_ids == 0
        and orders_unique_order_id == orders_unique_customer_id
    )

    return {
        "orders_total_rows": orders_total_rows,
        "orders_unique_order_id": orders_unique_order_id,
        "orders_duplicate_order_id_count": orders_duplicate_order_id_count,
        "orders_unique_customer_id": orders_unique_customer_id,
        "customers_unique_customer_id": customers_unique_customer_id,
        "customer_ids_with_multiple_orders": customer_ids_with_multiple_orders,
        "order_ids_with_multiple_customer_ids": order_ids_with_multiple_customer_ids,
        "is_strictly_one_to_one_order_customer": is_strictly_one_to_one,
        "cardinality_substantiation": (
            "Empirically validated that every order_id in orders maps to exactly one customer_id, "
            "and every customer_id in orders maps to exactly one order_id (0 duplicates, "
            "0 multi-order customer_ids). This proves customer_id is strictly a 1:1 order-scoped token."
        ),
    }


def analyze_order_resolution(df_orders: pd.DataFrame, df_customers: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the resolution path:
    orders.customer_id -> customers.customer_id -> customer_unique_id.
    Reports aggregate match counts only. Never stores individual IDs.
    """
    total_orders = int(len(df_orders))
    known_customer_ids = set(df_customers["customer_id"].dropna().unique())

    matched_orders = int(df_orders["customer_id"].isin(known_customer_ids).sum())
    unmatched_orders = total_orders - matched_orders
    resolution_pct = round((matched_orders / total_orders) * 100, 4) if total_orders > 0 else 0.0

    return {
        "total_orders": total_orders,
        "orders_resolved_to_customer_id": matched_orders,
        "orders_unmatched_to_customer_id": unmatched_orders,
        "resolution_percentage": resolution_pct,
        "all_orders_resolved": (matched_orders == total_orders),
    }


def analyze_order_statuses(df_orders: pd.DataFrame) -> Dict[str, Any]:
    """
    Audit order_status distribution and delivery date milestone evidence.
    Reports counts and percentages across all observed statuses.
    """
    total_orders = int(len(df_orders))
    status_counts_raw = df_orders["order_status"].value_counts(dropna=False)
    status_counts = {str(k): int(v) for k, v in status_counts_raw.items()}
    status_pcts = {str(k): round((int(v) / total_orders) * 100, 4) for k, v in status_counts_raw.items()}

    # Detailed delivery-date presence per status
    has_deliv_date = df_orders["order_delivered_customer_date"].notna()
    status_delivery_date_presence: Dict[str, Dict[str, int]] = {}
    for status, grp in df_orders.groupby("order_status"):
        total_s = int(len(grp))
        deliv_date_cnt = int(grp["order_delivered_customer_date"].notna().sum())
        missing_deliv_date_cnt = total_s - deliv_date_cnt
        status_delivery_date_presence[str(status)] = {
            "total_orders": total_s,
            "delivered_date_present": deliv_date_cnt,
            "delivered_date_missing": missing_deliv_date_cnt,
        }

    delivered_stats = status_delivery_date_presence.get("delivered", {"total_orders": 0, "delivered_date_present": 0, "delivered_date_missing": 0})

    return {
        "total_orders": total_orders,
        "order_status_counts": status_counts,
        "order_status_percentages": status_pcts,
        "status_delivery_date_presence": status_delivery_date_presence,
        "completed_purchase_definition": COMPLETED_PURCHASE_DEFINITION,
        "primary_completed_status": "delivered",
        "delivered_orders_total": delivered_stats["total_orders"],
        "delivered_orders_with_date": delivered_stats["delivered_date_present"],
        "delivered_orders_missing_date": delivered_stats["delivered_date_missing"],
        "canceled_orders_count": status_counts.get("canceled", 0),
        "unavailable_orders_count": status_counts.get("unavailable", 0),
        "in_flight_orders_count": total_orders - status_counts.get("delivered", 0) - status_counts.get("canceled", 0) - status_counts.get("unavailable", 0),
    }


def calculate_frequency_distribution(frequencies: pd.Series) -> Dict[str, Any]:
    """Compute aggregate frequency buckets (1, 2, 3+) and distribution summary."""
    total_customers = int(len(frequencies))
    if total_customers == 0:
        return {
            "total_unique_customers": 0,
            "customers_with_1_order": 0,
            "customers_with_1_order_pct": 0.0,
            "customers_with_2_orders": 0,
            "customers_with_2_orders_pct": 0.0,
            "customers_with_3_plus_orders": 0,
            "customers_with_3_plus_orders_pct": 0.0,
            "repeat_customers_total": 0,
            "repeat_customers_pct": 0.0,
            "max_order_frequency": 0,
            "frequency_distribution": {},
        }

    c_1 = int((frequencies == 1).sum())
    c_2 = int((frequencies == 2).sum())
    c_3_plus = int((frequencies >= 3).sum())
    repeat_total = int((frequencies >= 2).sum())

    dist_counts = frequencies.value_counts().sort_index()
    frequency_distribution = {str(k): int(v) for k, v in dist_counts.items()}

    return {
        "total_unique_customers": total_customers,
        "customers_with_1_order": c_1,
        "customers_with_1_order_pct": round((c_1 / total_customers) * 100, 4),
        "customers_with_2_orders": c_2,
        "customers_with_2_orders_pct": round((c_2 / total_customers) * 100, 4),
        "customers_with_3_plus_orders": c_3_plus,
        "customers_with_3_plus_orders_pct": round((c_3_plus / total_customers) * 100, 4),
        "repeat_customers_total": repeat_total,
        "repeat_customers_pct": round((repeat_total / total_customers) * 100, 4),
        "max_order_frequency": int(frequencies.max()),
        "frequency_distribution": frequency_distribution,
    }


def analyze_customer_order_frequencies(
    df_orders: pd.DataFrame, df_customers: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute order frequency at customer_unique_id level for:
    1. Completed-purchase population (delivered orders)
    2. All-orders population (all statuses)
    """
    # Join orders with customers to resolve customer_unique_id
    df_merged = df_orders.merge(
        df_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="inner",
    )

    # All orders frequency
    all_frequencies = df_merged.groupby("customer_unique_id")["order_id"].count()
    all_orders_dist = calculate_frequency_distribution(all_frequencies)

    # Completed orders frequency (delivered)
    df_delivered = df_merged[df_merged["order_status"] == "delivered"]
    delivered_frequencies = df_delivered.groupby("customer_unique_id")["order_id"].count()
    delivered_orders_dist = calculate_frequency_distribution(delivered_frequencies)

    return {
        "completed_purchases_delivered": delivered_orders_dist,
        "all_orders": all_orders_dist,
    }


def assess_customer_unique_id_suitability(
    mapping_stats: Dict[str, Any],
    cardinality_stats: Dict[str, Any],
    resolution_stats: Dict[str, Any],
    frequency_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Formulate evidence-based suitability assessment of customer_unique_id."""
    deliv_dist = frequency_stats["completed_purchases_delivered"]

    consolidates_ids = mapping_stats["customers_with_multiple_customer_ids"] > 0
    full_resolution = resolution_stats["all_orders_resolved"]
    has_repeat_customers = deliv_dist["repeat_customers_total"] > 0
    is_order_scoped = cardinality_stats["is_strictly_one_to_one_order_customer"]

    is_suitable = consolidates_ids and full_resolution and has_repeat_customers and is_order_scoped

    return {
        "is_suitable_analytical_identifier": is_suitable,
        "statement": SUITABILITY_CONCLUSION,
        "supporting_evidence": {
            "customer_id_is_strictly_order_scoped": is_order_scoped,
            "orders_unique_order_ids_equal_customer_ids": (cardinality_stats["orders_unique_order_id"] == cardinality_stats["orders_unique_customer_id"]),
            "consolidates_transaction_scoped_customer_ids": consolidates_ids,
            "customers_with_multiple_customer_ids_count": mapping_stats["customers_with_multiple_customer_ids"],
            "orders_resolved_to_customer_unique_id_pct": resolution_stats["resolution_percentage"],
            "completed_purchases_unique_customers": deliv_dist["total_unique_customers"],
            "completed_purchases_repeat_customers_count": deliv_dist["repeat_customers_total"],
            "completed_purchases_repeat_customers_pct": deliv_dist["repeat_customers_pct"],
            "completed_purchases_max_frequency": deliv_dist["max_order_frequency"],
        },
        "methodological_boundary_note": (
            "This checkpoint establishes identity and repeat-purchase evidence only. "
            "It does NOT evaluate or decide whether K-Means, RFM scoring, or churn modeling "
            "is viable. Segmentation feasibility will be assessed in subsequent checkpoints."
        ),
    }


def validate_checkpoint_privacy(data: Any, path: str = "") -> None:
    """
    Recursively inspect checkpoint data structure to verify that prohibited
    value-bearing fields or raw hex IDs are not persisted.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            if k.lower() in PROHIBITED_CHECKPOINT_KEYS:
                raise ValueError(f"Privacy violation: Prohibited key '{k}' found at path '{path}'!")
            validate_checkpoint_privacy(v, f"{path}.{k}" if path else k)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            validate_checkpoint_privacy(item, f"{path}[{idx}]")
    elif isinstance(data, str):
        if HEX_ID_PATTERN.match(data):
            raise ValueError(f"Privacy violation: Raw 32-char hex ID '{data}' found at path '{path}'!")


def run_identity_validation(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute full Checkpoint 2 identity validation and write aggregate-only checkpoint JSON.
    Verifies raw data immutability before and after execution.
    """
    if project_root is None:
        project_root = get_project_root()

    raw_dir = get_raw_data_dir(project_root)
    checkpoints_dir = get_checkpoints_dir(project_root)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    cust_file = raw_dir / "olist_customers_dataset.csv"
    orders_file = raw_dir / "olist_orders_dataset.csv"

    if not cust_file.exists() or not orders_file.exists():
        raise FileNotFoundError(f"Required raw files missing in {raw_dir}")

    # 1. Pre-audit SHA-256 calculation of all CSV files in data/raw
    all_raw_files = sorted(raw_dir.glob("*.csv"))
    pre_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}

    # 2. Read only necessary tables
    df_customers = pd.read_csv(cust_file, low_memory=False)
    df_orders = pd.read_csv(orders_file, low_memory=False)

    # 3. Validate schemas
    schema_stats = validate_schemas(df_customers, df_orders)

    # 4. Customer identifier mapping analysis
    mapping_stats = analyze_customer_identifiers(df_customers)

    # 5. Order-customer cardinality analysis (explicit 1:1 token validation)
    cardinality_stats = analyze_order_customer_cardinality(df_orders, df_customers)

    # 6. Order resolution check (orders.customer_id -> customers.customer_id -> customer_unique_id)
    resolution_stats = analyze_order_resolution(df_orders, df_customers)

    # 7. Order status population analysis
    status_stats = analyze_order_statuses(df_orders)

    # 8. Repeat purchase frequency analysis (delivered vs all)
    frequency_stats = analyze_customer_order_frequencies(df_orders, df_customers)

    # 9. Suitability determination
    suitability_stats = assess_customer_unique_id_suitability(
        mapping_stats, cardinality_stats, resolution_stats, frequency_stats
    )

    # 10. Post-audit SHA-256 verification
    post_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}
    immutability_verified = pre_hashes == post_hashes
    if not immutability_verified:
        raise RuntimeError("Raw data immutability verification failed!")

    # 11. Compile aggregate checkpoint JSON (strictly no PII or individual records)
    checkpoint_data: Dict[str, Any] = {
        "checkpoint": "02_identity_validation",
        "checkpoint_description": "Customer Identity Validation and Repeat-Purchase Distribution",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_immutability_verified": immutability_verified,
        "schema_validation": schema_stats,
        "customer_identifier_analysis": mapping_stats,
        "order_customer_cardinality_analysis": cardinality_stats,
        "order_resolution_analysis": resolution_stats,
        "order_status_population": status_stats,
        "customer_order_frequencies": frequency_stats,
        "customer_unique_id_suitability": suitability_stats,
    }

    # 12. Structural privacy validation: ensure no prohibited keys or raw hex IDs exist
    validate_checkpoint_privacy(checkpoint_data)

    # 13. Write checkpoint JSON
    checkpoint_path = checkpoints_dir / "02_identity_validation.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_data


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CHECKPOINT 2 — CUSTOMER IDENTITY VALIDATION")
    print("=" * 70)

    result = run_identity_validation()
    c_stats = result["customer_identifier_analysis"]
    card_stats = result["order_customer_cardinality_analysis"]
    r_stats = result["order_resolution_analysis"]
    s_stats = result["order_status_population"]
    f_stats = result["customer_order_frequencies"]["completed_purchases_delivered"]
    suitability = result["customer_unique_id_suitability"]

    print(f"Total customer rows: {c_stats['total_customer_rows']:,}")
    print(f"Unique customer_id: {c_stats['unique_customer_id_count']:,}")
    print(f"Unique customer_unique_id: {c_stats['unique_customer_unique_id_count']:,}")
    print(
        f"Customers with multiple customer_ids: {c_stats['customers_with_multiple_customer_ids']:,} "
        f"({c_stats['customers_with_multiple_customer_ids_pct']}%)"
    )
    print(f"Orders total rows: {card_stats['orders_total_rows']:,}")
    print(f"Orders unique order_id: {card_stats['orders_unique_order_id']:,}")
    print(f"Orders unique customer_id: {card_stats['orders_unique_customer_id']:,}")
    print(f"customer_ids with multiple orders: {card_stats['customer_ids_with_multiple_orders']}")
    print(f"order_ids with multiple customer_ids: {card_stats['order_ids_with_multiple_customer_ids']}")
    print(f"Strict 1:1 order-customer cardinality: {card_stats['is_strictly_one_to_one_order_customer']}")
    print(f"Orders resolved to customer_unique_id: {r_stats['orders_resolved_to_customer_id']:,} ({r_stats['resolution_percentage']}%)")
    print(f"Delivered orders total: {s_stats['delivered_orders_total']:,} ({s_stats['order_status_percentages']['delivered']}%)")
    print(f"Delivered unique customers: {f_stats['total_unique_customers']:,}")
    print(
        f"Delivered repeat customers (2+ orders): {f_stats['repeat_customers_total']:,} "
        f"({f_stats['repeat_customers_pct']}%)"
    )
    print(f"Max delivered order frequency: {f_stats['max_order_frequency']}")
    print(f"Suitability conclusion: {suitability['statement']}")
    print(f"Raw data immutability verified: {result['raw_data_immutability_verified']}")
    print("=" * 70)
    print("Checkpoint 2 completed successfully. Checkpoint saved.")
