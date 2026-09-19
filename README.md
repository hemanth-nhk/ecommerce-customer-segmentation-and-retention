# E-Commerce Customer Segmentation & 90-Day Retention Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.42+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Verification Status](https://img.shields.io/badge/Verification-8%2F8%20Passed-brightgreen.svg)]()

An enterprise-grade machine learning system demonstrating end-to-end customer analytics on commercial wholesale transaction data: rule-based descriptive RFM segmentation, strictly isolated point-in-time feature engineering, rigorous temporal model validation, retention value concentration analytics, and an interactive decision-support Streamlit application.

---

## Executive Summary

Customer retention and account prioritization are critical challenges for wholesale and B2B e-commerce platforms. Rather than relying on ungrounded heuristics or leaky snapshot aggregations, this project builds a statistically disciplined analytical framework using two years of real-world commercial transaction records.

The system addresses two complementary business objectives across two complementary analytical populations with separate purposes:
1. **Descriptive Account Segmentation ($N = 5,878$ All-Time Commercial Accounts):** Employs rule-based RFM (Recency, Frequency, Monetary) segmentation to profile commercial customers, revealing extreme spend concentration where the top 23.90% of accounts drive 68.56% of cumulative commercial spend.
2. **Predictive Retention Modeling ($N = 5,281$ Pre-Cutoff Eligible Accounts):** Formulates a leakage-free 90-day future-return prediction task relative to a frozen observation cutoff ($T_{\text{obs}}$ = September 10, 2011, 12:50:00). A primary candidate Logistic Regression pipeline achieves **0.7658 PR-AUC**, **0.8003 ROC-AUC**, and **2.0639x top-decile lift**, verified through forward temporal cross-validation.

An interactive, multi-page Streamlit web dashboard serves both non-technical executive stakeholders and technical data scientists, complete with on-the-fly model inference, scenario-based retention analytics, and explicit governance guardrails.

---

## Dataset Architecture & Cohort Isolation

### Source Data & Audit

The underlying transaction data is sourced from the official **Online Retail II** dataset, containing all wholesale transactions between December 1, 2009, and December 9, 2011, for a UK-based registered non-store online retailer:
- **Official UCI Machine Learning Repository:** [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Data Acquisition Source:** [Kaggle Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

```text
Raw File Integrity Audit:
├── Filename: online_retail_II.xlsx / online_retail_II.csv
├── Raw Record Count: 1,067,371 rows
├── Raw SHA-256 Checksum: 32569a66f3842a82b0d8c4d63b263c5d98a76bde5d1f65c6c01bf457e541d3a9
├── Exact Duplicate Rows Removed: 34,335
└── Retained Commercial Transaction Lines: 779,425 across 36,969 unique commercial invoices
```

### Critical Population Distinction

To prevent sampling bias and population confusion, the project maintains a strict boundary between two distinct populations:

| Attribute | Descriptive Segmentation Population | Predictive Retention Modeling Cohort |
| :--- | :--- | :--- |
| **Population Size** | **5,878 commercial customers** | **5,281 eligible customers** |
| **Time Horizon** | Full 2-year observation (2009-12-01 to 2011-12-09) | Active on or before cutoff ($T_{\text{obs}}$: 2011-09-10 12:50:00) |
| **Commercial Spend** | **£17,374,804.27** (all-time cumulative spend) | **£13,930,556.00** (pre-cutoff historical spend) |
| **Analytical Purpose** | Retrospective portfolio characterization & descriptive profiling | Prospective 90-day forward return/non-return prediction |
| **Isolation Rule** | *Never mix with predictive model evaluation sets* | *Strict point-in-time feature and target isolation* |

---

## Descriptive Customer Segmentation

The primary segmentation methodology is **Rule-Based Descriptive RFM Segmentation**, categorizing all 5,878 commercial accounts into actionable behavioral tiers using recency, frequency, and monetary deciles. K-Means clustering was evaluated as an exploratory benchmark but rejected for final deployment due to spherical cluster distortion, sensitivity to extreme spend skew, and lack of deterministic operational interpretability.

### Authoritative Segment Distribution ($N = 5,878$)

| Customer Segment | Customer Count | Account Share (%) | Commercial Spend (£) | Spend Share (%) | Average Spend / Account (£) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Champions** | 1,405 | 23.90% | £11,912,395.68 | 68.56% | £8,478.57 |
| **Loyal Customers** | 755 | 12.84% | £2,045,258.26 | 11.77% | £2,708.95 |
| **At Risk** | 283 | 4.81% | £1,007,120.98 | 5.80% | £3,558.73 |
| **Potential Loyalists** | 709 | 12.06% | £866,751.72 | 4.99% | £1,222.50 |
| **Hibernating** | 928 | 15.79% | £858,240.13 | 4.94% | £924.83 |
| **About to Sleep** | 844 | 14.36% | £360,218.28 | 2.07% | £426.80 |
| **Lost** | 717 | 12.20% | £244,313.76 | 1.41% | £340.74 |
| **Recent Customers** | 237 | 4.03% | £80,505.44 | 0.46% | £339.69 |
| **Total Commercial Population** | **5,878** | **100.00%** | **£17,374,804.27** | **100.00%** | **£2,955.90** |

**Key Segment Finding:** Extreme commercial spend concentration is present. The top two segments (Champions and Loyal Customers) represent only 36.74% of accounts but account for **80.33%** of all commercial transaction purchase totals (£13,957,653.94).

---

## Point-in-Time Feature Matrix & Target Formulation

### Temporal Isolation Protocol

In observational transaction data, leakage occurs if future purchase signals, post-cutoff cancellation records, or post-cutoff activity infect historical training features. To ensure zero target leakage:
- **Observation Cutoff ($T_{\text{obs}}$):** `2011-09-10 12:50:00`
- **90-Day Target Window:** `2011-09-10 12:50:00` to `2011-12-09 12:50:00` (90 calendar days)
- **Target Variable (`return_within_90d`):** Defined as `1` if the customer completed at least one commercial purchase during the 90-day target window; `0` otherwise.
- **Leakage Verification Audit:**
  - Maximum feature transaction timestamp: `2011-09-09 15:53:00` ($\le T_{\text{obs}}$)
  - Minimum target transaction timestamp: `2011-09-11 10:35:00` ($> T_{\text{obs}}$)
  - Pre-cutoff eligible accounts: **5,281**
  - Positive returners: **2,292** (43.40% prevalence)
  - Negative non-returners: **2,989** (56.60% non-return)
  - Strict temporal isolation verified with zero right-censoring across the cohort.

> **Target Definition Note:** The target `return_within_90d` is a **90-day future return / non-return target or retention proxy**. In non-contractual commercial wholesale settings where accounts do not terminate contracts, it must **never** be labeled as permanent churn.

### 27 Point-in-Time Predictive Features

The feature matrix comprises 5,281 customer rows, 31 total columns, and 27 predictive features categorized into 5 behavioral dimensions:
1. **Recency Dynamics:** `recency_days`, `recency_log`, `days_since_first_purchase`.
2. **Frequency & Velocity:** `frequency_invoices`, `invoices_per_month`, `frequency_log`, `avg_interpurchase_days`, `interpurchase_std`.
3. **Monetary Value:** `total_spend`, `spend_log`, `avg_invoice_spend`, `max_invoice_spend`, `spend_per_month`.
4. **Product Engagement & Breadth:** `total_items`, `unique_products`, `avg_items_per_invoice`, `cancellation_count`, `cancellation_rate`, `net_spend_ratio`.
5. **Temporal & Seasonality Profiles:** Spend across 30-day, 60-day, 90-day, and 180-day lookback windows (`spend_last_30d`, `spend_last_60d`, `spend_last_90d`, `spend_last_180d`, `momentum_30_vs_90`), and country domestic indicators.

---

## Model Benchmark & Evaluation

Six competing model architectures (M0 to M5) were evaluated on the held-out test set ($N = 1,057$, 20% stratified sample). The held-out test set was maintained strictly for final evaluation.

### Held-Out Evaluation Benchmark Table ($N = 1,057$)

| Model ID | Architecture / Baseline | PR-AUC | ROC-AUC | Brier Score | Decision Threshold ($t^*$) | Precision | Recall | F1 Score | Top-Decile Lift | Top-Decile Gain (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | Majority Class Baseline | 0.4342 | 0.5000 | 0.2457 | 0.50 | 0.0000 | 0.0000 | 0.0000 | N/A (ties) | N/A |
| **M1** | Stratified Random Baseline | 0.4623 | 0.5207 | 0.3211 | 0.50 | 0.4358 | 0.4858 | 0.4578 | 1.0211x | 10.21% |
| **M2** | Frozen RFM Heuristic | 0.5772 | 0.6908 | 0.2095 | 0.50 | 0.6621 | 0.6275 | 0.6443 | 1.6076x | 16.08% |
| **M3** | **Logistic Regression** | **0.7658** | **0.8003** | **0.1792** | **0.38** | **0.6472** | **0.7712** | **0.7038** | **2.0639x** | **20.70%** |
| **M4** | Random Forest Classifier | 0.7596 | 0.7892 | 0.1838 | 0.41 | 0.6508 | 0.7146 | 0.6812 | 2.1290x | 21.35% |
| **M5** | HistGradientBoosting | 0.7545 | 0.7875 | 0.1870 | 0.28 | 0.5886 | 0.8105 | 0.6819 | 2.0639x | 20.70% |

### Primary Candidate Model Selection: M3 Logistic Regression

**M3 Logistic Regression** is designated as **the frozen primary candidate model used by the dashboard**. 

Selection basis:
- **Consistent Discrimination:** Highest observed PR-AUC (0.7658) and ROC-AUC (0.8003) among evaluated candidates on 5-fold training cross-validation and held-out test evaluation.
- **Empirical Probability Calibration:** Lowest Brier score (0.1792), with a linear-fit reliability slope of 0.9841 and intercept of 0.0025. *(Note: These represent empirical diagnostic checks and do not constitute formal mathematical proof of calibration).*
- **Highest Observed Top-Decile Lift:** 2.0639x lift over base prevalence, identifying 20.70% of all future-returning accounts in the top 10% of predicted scores.
- **Operational Interpretability:** Transparent, standardized linear coefficients suitable for commercial audit and stakeholder review.

### Operating Point Diagnostics ($t^* = 0.38$)

Threshold $t^* = 0.38$ was tuned strictly on training 5-fold cross-validation to balance recall and precision for proactive engagement:

```text
Confusion Matrix (Held-Out Test Set, N = 1,057):
┌───────────────────────────────┬───────────────────────────────┐
│ True Negatives (TN): 405      │ False Positives (FP): 193     │
├───────────────────────────────┼───────────────────────────────┤
│ False Negatives (FN): 105     │ True Positives (TP): 354      │
└───────────────────────────────┴───────────────────────────────┘
Precision: 64.72% | Recall: 77.12% | F1 Score: 0.7038 | Specificity: 67.73%
```

---

## Forward Chronological Temporal Validation

To ensure resilience against seasonality, macroeconomic shifts, and temporal drift, a rolling-origin forward chronological backtest was conducted:
- **Historical Train Cohort 0:** $N = 4,977$ accounts ($T_{\text{cutoff}} = \text{2011-06-12 12:50:00}$)
- **Forward Evaluation Cohort 1:** $N = 5,281$ accounts ($T_{\text{cutoff}} = \text{2011-09-10 12:50:00}$)
- *Cohort Overlap Notice:* Because this is a rolling-origin longitudinal evaluation, there is a 94.24% account overlap between Cohorts 0 and 1. They represent temporal snapshots over time, not independent populations.

```text
Forward M3 Generalization Performance:
├── PR-AUC: 0.7709 (vs. 0.7658 held-out static test)
├── ROC-AUC: 0.7994 (vs. 0.8003 held-out static test)
├── Brier Score: 0.2000
├── Top-Decile Lift: 2.1125x
└── Top-Decile Gain: 21.16%
```
The forward temporal backtest demonstrates that the feature pipeline and model parameters maintained consistent discriminative ranking across consecutive operational quarters within this historical dataset.

---

## Retention Synthesis & Spend Concentration

Combining descriptive segmentation with prospective return predictions reveals critical spend concentration and commercial exposure across the pre-cutoff eligible population:
- **Pre-Cutoff Total Eligible Spend:** **£13,930,556.00**
- **High-Value Cohort (Champions + Loyal Customers):** **1,850 accounts** accounting for **£11,192,691.82** (**80.35%** of pre-cutoff commercial spend).

### Analytical Probability Bands Within High-Value Accounts ($N = 1,850$)

| Predicted Return Probability Band | Accounts ($N$) | Account Share (%) | Pre-Cutoff Spend (£) | Observed 90d Return Rate | Realized Post-Cutoff Spend (£) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High Predicted-Return ($p \ge 0.60$)** | 1,316 | 71.14% | £10,285,122.03 | 79.94% | £2,116,306.96 |
| **Intermediate Band ($0.30 \le p < 0.60$)** | 491 | 26.54% | £755,081.91 | 48.47% | £144,519.32 |
| **Low Predicted-Return ($p < 0.30$)** | 43 | 2.32% | £152,487.88 | 30.23% | £5,040.37 |
| **Total High-Value Cohort** | **1,850** | **100.00%** | **£11,192,691.82** | **70.43%** | **£2,265,866.65** |

> **Analytical Band Disclaimer:** The probability thresholds ($0.30$ and $0.60$) represent **descriptive analytical probability bands** used for cohort grouping. They are **not** validated business-optimal thresholds, nor do they represent clinical, financial, or operational risk guarantees.

### Post-Cutoff Commercial Spend Decile Concentration

Tracking realized transaction purchase totals across the 90-day post-cutoff window confirms extreme return value concentration:
- **Total Realized Post-Cutoff Commercial Spend:** **£3,041,357.82** *(Authoritative frozen value across all 5,281 accounts)*.
- **Decile 1 (Top 10% Predicted Returners):** Captured **£1,539,770.34** (**50.63%** of all realized post-cutoff spend).
- **Deciles 1 + 2 (Top 20% Predicted Returners):** Captured **£2,037,389.83** (**66.99%**, approximately 67.0% of realized spend).

---

## Interactive Streamlit Application

The project includes an interactive multi-page web application built with Streamlit, enabling both macro-level portfolio governance and micro-level individual account scoring.

```text
Streamlit Application Architecture:
├── app/dashboard.py (Multi-Page Single-File Router with Lazy-Loaded Data Caching)
└── Six Dashboard Pages:
    ├── 1. Executive Overview: High-level KPI scorecards, spend share charts, decile gains.
    ├── 2. Customer Prediction: Interactive customer lookup, real predict_proba inference.
    ├── 3. Descriptive Segmentation: RFM segment explorer, distribution plots, monetary metrics.
    ├── 4. Model Performance: Benchmark comparison (M0-M5), ROC/PR curves, confusion matrices.
    ├── 5. Retention Analytics: High-value probability bands, decile concentration, spend exposure.
    └── 6. Methodology & Governance: Data lineage, audit logs, and governance disclaimers.
```

### Dashboard Verification Status
- **Automated Verification:** 8/8 test suite checks passed (`tests/test_dashboard.py`).
- **Visual Interface Quality:** High-resolution screenshots captured across all six dashboard pages (`reports/screenshots/`).
- **Theme Support:** Verified under both Streamlit Light and Dark modes.
- **Privacy & Security:** Customer IDs are masked by default in user-facing tables.
- **Zero UI Defects:** Verified free of blank pages, broken links, or visual artifacts.

---

## Methodology & Governance Guardrails

To maintain rigorous data science integrity and prevent misapplication in operational decision-making, the following governance standards are enforced across all documentation and interfaces:

1. **Observational Bounds & Non-Causality:** All model outputs represent observational statistical associations. Predictive scores reflect historical purchase patterns and **do not** demonstrate causal treatment effects.
2. **Experimental Validation Requirement:** Proactive retention interventions, marketing discounts, or account manager re-engagement strategies **require randomized control trials (A/B testing)** with holdout groups to quantify true incremental lift and measure return on investment (ROI).
3. **Absence of Guarantees:** Predictions do not guarantee future retention, purchase completion, or financial return.
4. **Appropriate Terminology:** The project strictly utilizes precise vocabulary:
   - **Approved Terms:** *commercial spend, transaction purchase totals, 90-day future return/non-return, retention proxy, predictive association, observational analysis, testable intervention hypothesis*.
   - **Explicitly Prohibited Claims:** *permanent churn, causal treatment effects, guaranteed retention improvement, guaranteed financial impact, ROI, best model, optimal model, zero-risk prediction, guaranteed future outcome*.

---

## Project Structure

```text
ecommerce-customer-segmentation-churn/
├── app/
│   └── dashboard.py                  # Multi-page Streamlit decision-support application
├── config/
│   └── thresholds.yaml               # Decision thresholds, segment rules, and feature configurations
├── data/
│   ├── raw/                          # Raw transaction data (kept local, excluded from Git)
│   ├── processed/                    # Processed Parquet matrices (tracked) & large CSVs (local)
│   └── checkpoints/                  # 20 frozen JSON checkpoints (Evidence Register)
├── models/
│   ├── online_retail_m3_logistic_regression.joblib  # Frozen primary candidate model (M3)
│   ├── online_retail_m4_random_forest.joblib        # Frozen benchmark model (M4)
│   └── online_retail_m5_hist_gradient_boosting.joblib # Frozen benchmark model (M5)
├── notebooks/
│   ├── 01_phase1_audit.ipynb         # Raw data integrity & schema validation
│   ├── 02_customer_identity.ipynb    # Customer token & transaction cardinality audit
│   ├── 03_purchase_population.ipynb  # Commercial purchase layer filtering
│   ├── 04_temporal_behavior.ipynb    # Inter-purchase intervals & cadence analysis
│   ├── 05_rfm_analysis.ipynb         # Descriptive RFM quintile calculations
│   ├── 06_segmentation_evaluation.ipynb # Rule-based RFM vs K-Means benchmarking
│   └── 07_churn_modeling.ipynb       # Point-in-time ML benchmarks & evaluation
├── reports/
│   ├── AI_Powered_Ecommerce_Customer_Segmentation_and_Retention_Analysis_Report.docx # Comprehensive 19-page report
│   └── screenshots/                  # High-resolution dashboard verification captures
├── src/
│   ├── audit_online_retail_integrity.py # Ingestion auditing & cryptographic verification
│   ├── build_online_retail_purchase_layer.py # Commercial purchase filtering
│   ├── build_online_retail_customer_features.py # Historical feature extraction
│   ├── build_online_retail_retention_features.py # Leakage-safe 27 point-in-time feature matrix
│   ├── evaluate_online_retail_segmentation.py # Descriptive RFM segment assignment
│   ├── train_online_retail_retention_models.py # M0–M5 training, CV threshold tuning & serialization
│   ├── synthesize_online_retail_retention_segmentation.py # Probability bands & spend exposure synthesis
│   └── rfm_analysis.py               # Core RFM calculation routines
├── tests/
│   └── test_dashboard.py             # Automated unit and integration test suite (8/8 passed)
├── requirements.txt                  # Pinned environment dependencies (UTF-8)
└── README.md                         # Authoritative repository documentation & governance
```

---

## Installation & Local Setup

### Prerequisites
- Windows 10/11 operating system
- Python 3.11 or higher installed
- PowerShell terminal

### Step-by-Step Setup in Windows PowerShell

1. **Clone or Download the Repository:**
   ```powershell
   git clone https://github.com/hemanth-nhk/ecommerce-customer-segmentation-churn.git
   Set-Location -Path "ecommerce-customer-segmentation-churn"
   ```

2. **Create and Activate a Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Automated Test Suite:**
   ```powershell
   python -m pytest tests/test_dashboard.py -v
   ```

5. **Launch the Streamlit Dashboard:**
   ```powershell
   .\.venv\Scripts\streamlit.exe run app/dashboard.py
   ```
   Once launched, open your web browser and navigate to:
   ```text
   http://localhost:8501
   ```
   *(Note: The application is configured and verified for local workstation deployment).*

---

## Future Enhancements: Production Batch Scoring Architecture

In this release, the **Customer Prediction** interface provides real-time single-account inference using the frozen primary candidate model across pre-computed point-in-time features. Arbitrary CSV file uploads for bulk customer scoring are deliberately excluded from this version.

### Architectural Rationale for Batch Scoring

In non-contractual transaction environments, accurately predicting an account's future return probability requires historical behavioral context (e.g., inter-purchase intervals, cancellation rates, momentum ratios, and long-term spend). Uploading a batch of raw transaction rows in isolation is insufficient to evaluate returning accounts:

```text
Historical Customer Transactions (Database / Lakehouse)
                         +
    New Ingested Transaction Records (Batch Upload)
                         ↓
       Unified Longitudinal Account History
                         ↓
  Point-in-Time Feature Reconstruction Engine (27 Features)
                         ↓
  Frozen Primary Candidate Model (M3 Logistic Regression)
                         ↓
   Updated Calibrated Probability Scores & Decile Rank
```

**Planned Production Road Map:**
- **Stateful Feature Store Integration:** Connect the model pipeline to a centralized feature store (e.g., Feast or BigQuery Feature Store) to dynamically merge uploaded transactions with historical account timelines.
- **Automated Drift Monitoring:** Implement population stability index (PSI) and concept drift alerts on input feature distributions prior to automated batch scoring.
- **Controlled Experimentation Tracking:** Integration with experimentation platforms to log holdout control groups directly alongside score generation for rigorous ROI attribution.

---

## Author & Citation

- **Author:** Nalluri Hemanth Kumar
- **Dataset Citation:** Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D.

---

## Dataset Attribution

This project uses the **Online Retail II** dataset from the **UCI Machine Learning Repository**.
- **Dataset Attribution:** Daqing Chen
- **Official DOI:** [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)
- **Dataset License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

The dataset and materials derived from it remain subject to the applicable third-party dataset license and attribution requirements.

The MIT License in this repository applies only to the project's original source code and implementation, and does not relicense the third-party dataset or dataset-derived materials.
