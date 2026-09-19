"""
src/audit.py

Checkpoint 1 — Dataset Integrity Audit
AI-Powered E-commerce Customer Segmentation and Churn Analysis
Brazilian E-Commerce Public Dataset by Olist

This module performs structural validation and integrity checks across the
9 raw Olist CSV datasets, generating an aggregate-only checkpoint artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Expected 9 Olist CSV files
EXPECTED_RAW_FILES: List[str] = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

# Patterns or column names indicating potential date/time fields
DATETIME_COLUMN_PATTERNS: Set[str] = {
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
}

# Structural column presence checks across related tables
# Validates only expected column presence and schema compatibility.
# Does NOT assert primary/foreign key semantics or perform customer identity validation.
EXPECTED_COLUMN_PRESENCE: List[Dict[str, str]] = [
    {
        "relationship_label": "orders <-> customers via customer_id",
        "table_a": "olist_orders_dataset",
        "column_a": "customer_id",
        "table_b": "olist_customers_dataset",
        "column_b": "customer_id",
    },
    {
        "relationship_label": "order_items <-> orders via order_id",
        "table_a": "olist_order_items_dataset",
        "column_a": "order_id",
        "table_b": "olist_orders_dataset",
        "column_b": "order_id",
    },
    {
        "relationship_label": "order_items <-> products via product_id",
        "table_a": "olist_order_items_dataset",
        "column_a": "product_id",
        "table_b": "olist_products_dataset",
        "column_b": "product_id",
    },
    {
        "relationship_label": "order_items <-> sellers via seller_id",
        "table_a": "olist_order_items_dataset",
        "column_a": "seller_id",
        "table_b": "olist_sellers_dataset",
        "column_b": "seller_id",
    },
    {
        "relationship_label": "order_payments <-> orders via order_id",
        "table_a": "olist_order_payments_dataset",
        "column_a": "order_id",
        "table_b": "olist_orders_dataset",
        "column_b": "order_id",
    },
    {
        "relationship_label": "order_reviews <-> orders via order_id",
        "table_a": "olist_order_reviews_dataset",
        "column_a": "order_id",
        "table_b": "olist_orders_dataset",
        "column_b": "order_id",
    },
    {
        "relationship_label": "products <-> translation via product_category_name",
        "table_a": "olist_products_dataset",
        "column_a": "product_category_name",
        "table_b": "product_category_name_translation",
        "column_b": "product_category_name",
    },
    {
        "relationship_label": "customers <-> geolocation via zip_code_prefix",
        "table_a": "olist_customers_dataset",
        "column_a": "customer_zip_code_prefix",
        "table_b": "olist_geolocation_dataset",
        "column_b": "geolocation_zip_code_prefix",
    },
    {
        "relationship_label": "sellers <-> geolocation via zip_code_prefix",
        "table_a": "olist_sellers_dataset",
        "column_a": "seller_zip_code_prefix",
        "table_b": "olist_geolocation_dataset",
        "column_b": "geolocation_zip_code_prefix",
    },
]


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


def verify_raw_files(raw_dir: Path) -> List[Path]:
    """
    Verify that all 9 expected CSV files exist in data/raw/.
    Fails clearly with FileNotFoundError if any are missing.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    existing_files = {p.name: p for p in raw_dir.glob("*.csv")}
    missing_files = [f for f in EXPECTED_RAW_FILES if f not in existing_files]

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} expected raw file(s) in {raw_dir}: {missing_files}"
        )

    return [existing_files[f] for f in EXPECTED_RAW_FILES]


def identify_candidate_identifiers(columns: List[str]) -> List[str]:
    """
    Identify candidate identifier/key columns strictly based on schema naming patterns.
    Does NOT assert primary-key, foreign-key, or customer-identity semantics at this checkpoint.
    """
    candidates: List[str] = []
    for col in columns:
        col_lower = col.lower()
        if (
            col_lower.endswith("_id")
            or col_lower == "id"
            or "_id_" in col_lower
            or col_lower.endswith("_prefix")
            or col_lower == "product_category_name"
        ):
            candidates.append(col)
    return candidates


def audit_datetime_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Safely identify, parse, and report statistics for date/time columns.
    Reports min valid date, max valid date, and count of invalid/unparseable dates.
    """
    datetime_stats: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        col_lower = col.lower()
        is_candidate = (
            col in DATETIME_COLUMN_PATTERNS
            or "timestamp" in col_lower
            or "date" in col_lower
            or col_lower.endswith("_at")
        )
        if not is_candidate:
            continue

        series = df[col]
        # Parse dates safely with coercion
        parsed = pd.to_datetime(series, errors="coerce")

        non_null_raw_count = int(series.notna().sum())
        valid_parsed_count = int(parsed.notna().sum())
        unparseable_count = non_null_raw_count - valid_parsed_count

        min_date_val = parsed.min()
        max_date_val = parsed.max()

        datetime_stats[col] = {
            "total_non_null_raw": non_null_raw_count,
            "valid_parsed_count": valid_parsed_count,
            "unparseable_count": unparseable_count,
            "min_valid_date": (
                min_date_val.isoformat() if pd.notna(min_date_val) else None
            ),
            "max_valid_date": (
                max_date_val.isoformat() if pd.notna(max_date_val) else None
            ),
        }

    return datetime_stats


def audit_single_table(csv_path: Path) -> Dict[str, Any]:
    """
    Audit an individual raw CSV table without modifying it.
    Returns an aggregate metadata dictionary.
    """
    table_name = csv_path.stem
    filename = csv_path.name

    # Load raw CSV read-only without data transformation
    df = pd.read_csv(csv_path, low_memory=False)

    row_count = int(len(df))
    column_count = int(len(df.columns))
    columns = list(df.columns)

    # Inferred pandas dtypes
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing value counts and percentages
    missing_counts: Dict[str, int] = {}
    missing_percentages: Dict[str, float] = {}
    for col in columns:
        m_count = int(df[col].isna().sum())
        missing_counts[col] = m_count
        missing_percentages[col] = (
            round((m_count / row_count) * 100, 4) if row_count > 0 else 0.0
        )

    # Duplicate row count
    duplicate_rows = int(df.duplicated().sum())

    # Candidate identifier columns based strictly on schema naming
    candidate_identifiers = identify_candidate_identifiers(columns)

    # Audit datetime columns
    datetime_stats = audit_datetime_columns(df)

    return {
        "filename": filename,
        "table_name": table_name,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_rows,
        "candidate_identifier_columns": candidate_identifiers,
        "datetime_columns": datetime_stats,
    }


def audit_structural_column_presence(
    tables_audit: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validate only expected column presence across related tables at a structural level.
    Does NOT assert primary/foreign key semantics or perform customer identity validation.
    """
    results: List[Dict[str, Any]] = []

    for check in EXPECTED_COLUMN_PRESENCE:
        tbl_a = check["table_a"]
        col_a = check["column_a"]
        tbl_b = check["table_b"]
        col_b = check["column_b"]

        tbl_a_present = tbl_a in tables_audit
        tbl_b_present = tbl_b in tables_audit

        col_a_present = tbl_a_present and (col_a in tables_audit[tbl_a]["columns"])
        col_b_present = tbl_b_present and (col_b in tables_audit[tbl_b]["columns"])

        structurally_valid = col_a_present and col_b_present

        results.append(
            {
                "relationship_label": check["relationship_label"],
                "table_a": tbl_a,
                "column_a": col_a,
                "column_a_present": col_a_present,
                "table_b": tbl_b,
                "column_b": col_b,
                "column_b_present": col_b_present,
                "both_columns_present": structurally_valid,
            }
        )

    return results


def run_audit(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute the full dataset integrity audit and save the aggregate-only checkpoint.
    Verifies raw data immutability using pre- and post-audit file hashes.
    """
    if project_root is None:
        project_root = get_project_root()

    raw_dir = get_raw_data_dir(project_root)
    checkpoints_dir = get_checkpoints_dir(project_root)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and verify raw files
    csv_paths = verify_raw_files(raw_dir)

    # 2. Record pre-audit file SHA-256 hashes to verify read-only immutability
    pre_hashes = {p.name: compute_file_sha256(p) for p in csv_paths}

    # 3. Audit each table
    tables_audit: Dict[str, Dict[str, Any]] = {}
    total_rows = 0
    tables_with_duplicates: Dict[str, int] = {}
    tables_with_missing_values: Dict[str, int] = {}

    for path in csv_paths:
        table_meta = audit_single_table(path)
        t_name = table_meta["table_name"]
        tables_audit[t_name] = table_meta

        total_rows += table_meta["row_count"]
        if table_meta["duplicate_rows"] > 0:
            tables_with_duplicates[t_name] = table_meta["duplicate_rows"]

        total_missing = sum(table_meta["missing_counts"].values())
        if total_missing > 0:
            tables_with_missing_values[t_name] = total_missing

    # 4. Check structural column presence
    structural_presence_checks = audit_structural_column_presence(tables_audit)

    # 5. Verify post-audit file hashes to guarantee zero modifications to data/raw/
    post_hashes = {p.name: compute_file_sha256(p) for p in csv_paths}
    immutability_verified = pre_hashes == post_hashes
    if not immutability_verified:
        mismatches = [
            f for f in pre_hashes if pre_hashes[f] != post_hashes.get(f)
        ]
        raise RuntimeError(
            f"Raw file immutability check failed for: {mismatches}!"
        )

    # 6. Compile aggregate checkpoint payload (strictly no row-level data or identifying values)
    checkpoint_data: Dict[str, Any] = {
        "checkpoint": "01_integrity",
        "checkpoint_description": "Dataset Integrity and Structural Audit",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_tables_audited": len(tables_audit),
            "expected_tables_count": len(EXPECTED_RAW_FILES),
            "total_row_count": total_rows,
            "tables_with_duplicates": tables_with_duplicates,
            "tables_with_missing_values": tables_with_missing_values,
            "all_structural_columns_present": all(
                r["both_columns_present"] for r in structural_presence_checks
            ),
            "raw_data_immutability_verified": immutability_verified,
        },
        "tables": tables_audit,
        "structural_column_presence_checks": structural_presence_checks,
    }

    # 7. Write aggregate checkpoint JSON
    checkpoint_file = checkpoints_dir / "01_integrity.json"
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_data


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CHECKPOINT 1 — DATASET INTEGRITY AUDIT")
    print("=" * 70)

    audit_result = run_audit()
    summary = audit_result["summary"]

    print(f"Total tables audited: {summary['total_tables_audited']}")
    print(f"Total rows audited: {summary['total_row_count']:,}")
    print(f"Tables with duplicate rows: {summary['tables_with_duplicates']}")
    print(f"Tables with missing values: {summary['tables_with_missing_values']}")
    print(
        f"All structural column presence checks valid: {summary['all_structural_columns_present']}"
    )
    print(
        f"Raw data immutability verified: {summary['raw_data_immutability_verified']}"
    )
    print("=" * 70)
    print("Audit completed successfully. Checkpoint saved.")
