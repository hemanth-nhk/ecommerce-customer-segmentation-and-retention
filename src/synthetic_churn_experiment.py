"""
src/synthetic_churn_experiment.py

SYNTHETIC TRAINING AUGMENTATION — EXPERIMENTAL
Phase 7 Churn Modeling Controlled Experiment

Evaluates whether synthetic minority-class augmentation (SMOTENC) applied strictly
to training data improves out-of-sample predictive performance of Balanced Logistic
Regression on the untouched real Olist test set.

Methodological Guardrails:
1. Raw Olist data remains immutable (SHA-256 verified).
2. Existing V1 results (07_churn_modeling.json) remain 100% untouched and preserved.
3. Target definition, 120d/180d cutoffs, and 80/20 stratified split are identical to V1.
4. X_test and y_test are 100% real, untouched, and never exposed to synthetic data.
5. In CV threshold selection, validation folds contain ONLY real observations;
   SMOTENC is applied strictly inside each CV training fold.
6. Tested models: Balanced Logistic Regression at 2x and 5x augmentation ratios.
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.churn import (
    RANDOM_STATE,
    TEST_SIZE,
    build_temporal_modeling_dataset,
    compute_file_sha256,
    evaluate_model_performance,
    load_raw_olist_tables,
)

EXPERIMENT_LABEL = "SYNTHETIC TRAINING AUGMENTATION — EXPERIMENTAL"


def find_optimal_threshold_cv_with_smotenc(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_indices: List[int],
    augmentation_multiplier: int,
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, Dict[str, Any]]:
    """
    Leakage-safe 5-fold CV threshold selection with SMOTENC applied inside each training fold:
    - Validation fold contains strictly 100% real observations.
    - SMOTENC fits strictly on the training fold; no validation sample is seen or interpolated.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    val_probs = np.zeros(len(y_train), dtype=float)
    fold_audit_records = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr_fold = X_train.iloc[tr_idx].copy()
        y_tr_fold = y_train[tr_idx].copy()

        X_val_fold = X_train.iloc[val_idx].copy()
        y_val_fold = y_train[val_idx].copy()

        n_pos_tr = int(np.sum(y_tr_fold == 1))
        n_neg_tr = int(np.sum(y_tr_fold == 0))
        target_pos_tr = n_pos_tr * augmentation_multiplier

        # Apply SMOTENC strictly to this fold's training portion
        smote = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy={0: n_neg_tr, 1: target_pos_tr},
            random_state=random_state + fold_idx,
        )
        X_tr_fold_aug, y_tr_fold_aug = smote.fit_resample(X_tr_fold, y_tr_fold)

        # Audit fold integrity
        fold_audit_records.append({
            "fold": fold_idx,
            "val_samples_real_only": len(X_val_fold),
            "val_positive_samples_real": int(np.sum(y_val_fold)),
            "train_samples_pre_aug": len(X_tr_fold),
            "train_positive_pre_aug": n_pos_tr,
            "train_samples_post_aug": len(X_tr_fold_aug),
            "train_positive_post_aug": int(np.sum(y_tr_fold_aug == 1)),
        })

        # Fit cloned pipeline on augmented fold training set
        fold_pipe = clone(base_pipeline)
        fold_pipe.fit(X_tr_fold_aug, y_tr_fold_aug)

        # Predict probabilities on strictly real validation fold
        val_probs[val_idx] = fold_pipe.predict_proba(X_val_fold)[:, 1]

    # Evaluate out-of-fold F1 scores on real y_train
    precisions, recalls, thresholds = precision_recall_curve(y_train, val_probs)
    f1_scores = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        denom = precisions[i] + recalls[i]
        f1_scores[i] = 2 * (precisions[i] * recalls[i]) / denom if denom > 0 else 0.0

    best_threshold = 0.50
    if len(thresholds) > 0 and np.max(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])

    return best_threshold, {"fold_records": fold_audit_records}


def run_single_synthetic_augmentation_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    window_days: int,
    augmentation_multiplier: int,  # 2 or 5
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    """
    Executes a single controlled augmentation experiment:
    1. Leakage-safe CV threshold tuning with SMOTENC inside CV folds.
    2. Fit final model on augmented X_train (real + synthetic training samples).
    3. Predict and evaluate strictly on untouched real X_test and y_test.
    """
    categorical_indices = [
        X_train.columns.get_loc(col) for col in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    base_lr_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 1. Leakage-safe CV threshold selection
    optimal_threshold, cv_audit = find_optimal_threshold_cv_with_smotenc(
        base_pipeline=base_lr_pipe,
        X_train=X_train,
        y_train=y_train,
        categorical_indices=categorical_indices,
        augmentation_multiplier=augmentation_multiplier,
        cv=5,
        random_state=RANDOM_STATE,
    )

    # 2. Final training set augmentation (strictly from X_train)
    orig_train_n = len(X_train)
    orig_pos_n = int(np.sum(y_train == 1))
    orig_neg_n = int(np.sum(y_train == 0))
    target_pos_n = orig_pos_n * augmentation_multiplier

    final_smote = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy={0: orig_neg_n, 1: target_pos_n},
        random_state=RANDOM_STATE,
    )
    X_train_aug, y_train_aug = final_smote.fit_resample(X_train, y_train)

    # Verify augmentation integrity
    assert len(X_train_aug) == orig_neg_n + target_pos_n
    assert int(np.sum(y_train_aug == 1)) == target_pos_n
    assert int(np.sum(y_train_aug == 0)) == orig_neg_n

    # 3. Fit final pipeline on augmented training set
    final_lr_pipe = clone(base_lr_pipe)
    final_lr_pipe.fit(X_train_aug, y_train_aug)

    # 4. Predict and evaluate strictly on untouched real test set
    test_proba = final_lr_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= optimal_threshold).astype(int)

    test_metrics = evaluate_model_performance(
        y_true=y_test,
        y_pred_binary=test_pred,
        y_pred_proba=test_proba,
        threshold_used=optimal_threshold,
    )

    return {
        "experiment_label": EXPERIMENT_LABEL,
        "window_days": window_days,
        "model_name": "Balanced Logistic Regression (SMOTENC Augmented)",
        "augmentation_method": "SMOTENC",
        "augmentation_multiplier": augmentation_multiplier,
        "augmentation_ratio_description": f"{augmentation_multiplier}x minority class",
        "original_train_size": orig_train_n,
        "augmented_train_size": len(X_train_aug),
        "synthetic_samples_added": len(X_train_aug) - orig_train_n,
        "original_train_positives": orig_pos_n,
        "augmented_train_positives": int(np.sum(y_train_aug == 1)),
        "original_train_prevalence_pct": round((orig_pos_n / orig_train_n) * 100.0, 4),
        "augmented_train_prevalence_pct": round((int(np.sum(y_train_aug == 1)) / len(X_train_aug)) * 100.0, 4),
        "test_size_real_untouched": len(X_test),
        "test_positives_real_untouched": int(np.sum(y_test)),
        "test_prevalence_pct": round((int(np.sum(y_test)) / len(y_test)) * 100.0, 4),
        "selected_cv_threshold": round(float(optimal_threshold), 4),
        "test_metrics": test_metrics,
        "cv_audit": cv_audit,
    }


def run_full_synthetic_augmentation_experiment() -> Dict[str, Any]:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    checkpoints_dir = PROJECT_ROOT / "data" / "checkpoints"

    # 1. Pre-audit SHA-256 verification of all 9 raw files
    all_raw_files = sorted(raw_dir.glob("*.csv"))
    assert len(all_raw_files) == 9, f"Expected 9 raw files, found {len(all_raw_files)}"
    pre_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}

    # Load V1 baseline checkpoint to ensure exact comparison against frozen V1 results
    v1_cp_path = checkpoints_dir / "07_churn_modeling.json"
    with open(v1_cp_path, "r", encoding="utf-8") as f:
        v1_cp = json.load(f)

    tables = load_raw_olist_tables(raw_dir)

    numeric_features = [
        "recency_days",
        "log1p_recency",
        "frequency",
        "monetary_total",
        "log1p_monetary",
        "mean_order_value",
        "tenure_days",
        "inter_purchase_mean_days",
        "inter_purchase_max_days",
        "total_items_purchased",
        "distinct_product_categories",
        "freight_value_ratio",
        "payment_installments_mean",
        "mean_review_score",
        "review_score_missing",
    ]
    categorical_features = [
        "payment_type_dominant",
        "behavioral_segment",
    ]

    results_by_window: Dict[str, Any] = {}

    for w_days, cutoff_str in [(120, "2018-05-01 15:00:37"), (180, "2018-03-02 15:00:37")]:
        cutoff_ts = pd.Timestamp(cutoff_str)
        cust_features, y_target, meta = build_temporal_modeling_dataset(
            tables, cutoff_ts, future_window_days=w_days
        )

        X = cust_features[numeric_features + categorical_features].copy()
        y = y_target.values

        # Exact real V1 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        # Retrieve V1 baseline Logistic Regression metrics from checkpoint
        v1_w_key = f"candidate_window_{w_days}d"
        v1_lr_metrics = v1_cp[v1_w_key]["modeling_experiment"]["models_evaluated"]["ml_1_balanced_logistic_regression"]["test_metrics"]

        # Run 2x augmentation experiment
        exp_2x = run_single_synthetic_augmentation_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            window_days=w_days,
            augmentation_multiplier=2,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Run 5x augmentation experiment
        exp_5x = run_single_synthetic_augmentation_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            window_days=w_days,
            augmentation_multiplier=5,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Integrity assertions
        assert len(X_test) == len(y_test)
        assert exp_2x["test_size_real_untouched"] == exp_5x["test_size_real_untouched"]
        assert exp_2x["test_positives_real_untouched"] == exp_5x["test_positives_real_untouched"]

        results_by_window[f"{w_days}d"] = {
            "v1_baseline_logistic_regression": {
                "window_days": w_days,
                "model_name": "Balanced Logistic Regression (V1 Baseline - No SMOTE)",
                "test_metrics": v1_lr_metrics,
            },
            "smotenc_2x": exp_2x,
            "smotenc_5x": exp_5x,
        }

    # Post-audit immutability
    post_hashes = {p.name: compute_file_sha256(p) for p in all_raw_files}
    assert pre_hashes == post_hashes, "Raw data immutability verification failed!"

    checkpoint_data = {
        "checkpoint": "07c_synthetic_augmentation_experiment",
        "experiment_label": EXPERIMENT_LABEL,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_immutability_verified": True,
        "leakage_safeguards": {
            "test_set_strictly_real": True,
            "synthetic_samples_in_test": 0,
            "cv_validation_folds_strictly_real": True,
            "test_labels_accessed_during_augmentation_or_tuning": False,
        },
        "experiments": results_by_window,
    }

    # Save to dedicated isolated checkpoint
    out_path = checkpoints_dir / "07c_synthetic_augmentation_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    return checkpoint_data


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING SYNTHETIC MINORITY-CLASS AUGMENTATION CONTROLLED EXPERIMENT")
    print("=" * 75)
    res = run_full_synthetic_augmentation_experiment()
    print("\nExperiment completed successfully.")
    print("Results written to: data/checkpoints/07c_synthetic_augmentation_experiment.json")
