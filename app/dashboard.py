"""
Retention Engine - B2B Customer Intelligence & 90-Day Return Prediction Dashboard
Dataset: Online Retail II (UK Commercial Wholesale Transactions)

Evidence-First Production Application:
- Loads frozen Phase 7 Logistic Regression candidate model (models/online_retail_m3_logistic_regression.joblib)
- Live inference across 27 pre-cutoff behavioral & RFM predictors
- Complete benchmark comparison (M0-M5) with PR-AUC primary evaluation
- Descriptive customer segmentation and high-value retention analytics
- Strictly non-causal observational governance and zero Customer ID exposure
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. APPLICATION & PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Retention Engine | Online Retail II",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "online_retail_m3_logistic_regression.joblib")
FEATURE_MATRIX_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "online_retail_retention_features_90d.parquet")
CP06_PATH = os.path.join(PROJECT_ROOT, "data", "checkpoints", "06_online_retail_segmentation_evaluation.json")
CP10_PATH = os.path.join(PROJECT_ROOT, "data", "checkpoints", "10_online_retail_model_benchmark.json")
CP11_PATH = os.path.join(PROJECT_ROOT, "data", "checkpoints", "11_online_retail_retention_synthesis.json")

# ==============================================================================
# 2. DESIGN SYSTEM & CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* Global Content Width & Viewport Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* Clean Enterprise Typography */
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Application Header with Adequate Breathing Room */
    .app-header-container {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.8rem;
        box-sizing: border-box;
    }
    .app-eyebrow {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #2563EB;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
        line-height: 1.2;
    }
    .app-page-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: inherit;
        margin: 0 0 0.45rem 0;
        line-height: 1.25;
    }
    .app-page-subtitle {
        font-size: 0.95rem;
        opacity: 0.75;
        line-height: 1.5;
        margin: 0;
    }
    .meta-badges-col {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        align-items: flex-end;
        justify-content: center;
    }
    .meta-badge-item {
        display: inline-flex;
        align-items: center;
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 6px;
        padding: 0.42rem 0.8rem;
        font-size: 0.82rem;
        color: inherit;
        white-space: nowrap;
        line-height: 1.3;
    }
    .meta-badge-item strong {
        color: inherit;
    }

    /* B2B Metric Cards */
    .b2b-metric-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 1.15rem 1.3rem;
        height: 100%;
        box-sizing: border-box;
    }
    .b2b-metric-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        opacity: 0.7;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .b2b-metric-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: inherit;
        line-height: 1.2;
    }
    .b2b-metric-sub {
        font-size: 0.82rem;
        opacity: 0.7;
        margin-top: 0.4rem;
    }

    /* Sidebar Container (inherits Streamlit theme naturally) */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18) !important;
    }

    /* Sidebar Navigation Section Headers */
    .nav-section-header {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        opacity: 0.6;
        text-transform: uppercase;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        padding-left: 0.35rem;
    }

    /* Sidebar Navigation Buttons - Replaces Radio Buttons Completely */
    section[data-testid="stSidebar"] div[data-testid="stButton"] {
        margin-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.58rem 0.85rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        min-height: 2.4rem !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    /* Inactive Navigation Button */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: transparent !important;
        color: inherit !important;
        opacity: 0.85 !important;
        border: 1px solid transparent !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: rgba(128, 128, 128, 0.1) !important;
        opacity: 1 !important;
        border-color: rgba(128, 128, 128, 0.18) !important;
    }

    /* Active Navigation Button */
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
        background-color: rgba(37, 99, 235, 0.15) !important;
        color: #2563EB !important;
        font-weight: 600 !important;
        border: 1px solid rgba(37, 99, 235, 0.35) !important;
        border-left: 3.5px solid #2563EB !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: rgba(37, 99, 235, 0.25) !important;
        color: #1D4ED8 !important;
        border-left: 3.5px solid #1D4ED8 !important;
    }

    /* Sidebar Status Box */
    .sidebar-status-box {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 6px;
        padding: 0.85rem 0.95rem;
    }
    .status-group {
        margin-bottom: 0.7rem;
    }
    .status-group:last-child {
        margin-bottom: 0;
    }
    .status-label {
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        opacity: 0.65;
        text-transform: uppercase;
    }
    .status-value {
        font-size: 0.84rem;
        font-weight: 600;
        color: inherit;
        margin-top: 0.15rem;
    }
    .status-sub {
        font-size: 0.74rem;
        opacity: 0.7;
    }
    .status-dot {
        color: #2563EB;
        font-size: 0.65rem;
        vertical-align: middle;
        margin-right: 0.25rem;
    }
    .sidebar-divider {
        border-top: 1px solid rgba(128, 128, 128, 0.18);
        margin: 1.15rem 0;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(128, 128, 128, 0.06);
        border-left: 3px solid #2563EB;
        padding: 0.95rem 1.2rem;
        border-radius: 4px;
        font-size: 0.84rem;
        color: inherit;
        margin-top: 1.3rem;
        margin-bottom: 0.9rem;
        line-height: 1.55;
    }
    .disclaimer-box strong {
        color: inherit;
    }

    /* Analytical Band Pills */
    .band-pill-high {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #059669;
        font-weight: 600;
        font-size: 0.86rem;
        padding: 0.4rem 0.85rem;
        border-radius: 6px;
    }
    .band-pill-intermediate {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #D97706;
        font-weight: 600;
        font-size: 0.86rem;
        padding: 0.4rem 0.85rem;
        border-radius: 6px;
    }
    .band-pill-low {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #DC2626;
        font-weight: 600;
        font-size: 0.86rem;
        padding: 0.4rem 0.85rem;
        border-radius: 6px;
    }

    /* Sidebar Brand & Footer Classes */
    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: inherit;
    }
    .sidebar-brand-sub {
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.7;
        margin-top: 2px;
    }
    .sidebar-footer-note {
        font-size: 0.78rem;
        opacity: 0.7;
        line-height: 1.45;
        padding: 0.1rem 0.2rem;
    }
    .sidebar-footer-note strong {
        color: inherit;
    }

    /* Analytical Band Card Typography Classes */
    .band-card-low-text { color: #DC2626; }
    .band-card-mid-text { color: #D97706; }
    .band-card-high-text { color: #059669; }
    .b2b-metric-desc {
        font-size: 0.78rem;
        opacity: 0.75;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. DATA & MODEL LOADERS (CACHED)
# ==============================================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Frozen model artifact not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_matrix():
    if not os.path.exists(FEATURE_MATRIX_PATH):
        st.error(f"Processed feature matrix not found at: {FEATURE_MATRIX_PATH}")
        st.stop()
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    # Mask Customer ID to preserve privacy
    df["masked_account_ref"] = [f"Account Ref #{i+1:04d}" for i in range(len(df))]
    return df


@st.cache_data
def load_checkpoints():
    cp06, cp10, cp11 = {}, {}, {}
    if os.path.exists(CP06_PATH):
        with open(CP06_PATH, "r", encoding="utf-8") as f:
            cp06 = json.load(f)
    if os.path.exists(CP10_PATH):
        with open(CP10_PATH, "r", encoding="utf-8") as f:
            cp10 = json.load(f)
    if os.path.exists(CP11_PATH):
        with open(CP11_PATH, "r", encoding="utf-8") as f:
            cp11 = json.load(f)
    return cp06, cp10, cp11


# Load resources
model_pipeline = load_model()
df_features = load_feature_matrix()
cp06, cp10, cp11 = load_checkpoints()

# 27 predictive feature columns strictly timestamped <= T_obs
PREDICTIVE_FEATURE_COLS = [
    c for c in df_features.columns
    if c not in ["Customer ID", "return_within_90d", "first_purchase_date_pre", "last_purchase_date_pre", "masked_account_ref"]
]


def classify_analytical_band(p: float) -> str:
    if p >= 0.60:
        return "High predicted-return band (p >= 0.60)"
    elif p >= 0.30:
        return "Intermediate predicted-return band (0.30 <= p < 0.60)"
    else:
        return "Low predicted-return band (p < 0.30)"


def render_header(page_title: str, page_subtitle: str):
    st.markdown(f"""
    <div class="app-header-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.5rem;">
            <div style="flex: 1 1 520px; min-width: 320px;">
                <div class="app-eyebrow">RETENTION ENGINE · CUSTOMER INTELLIGENCE</div>
                <div class="app-page-title">{page_title}</div>
                <div class="app-page-subtitle">{page_subtitle}</div>
            </div>
            <div class="meta-badges-col" style="flex: 0 0 auto;">
                <div class="meta-badge-item"><strong>Dataset:</strong>&nbsp;Online Retail II</div>
                <div class="meta-badge-item"><strong>Cutoff:</strong>&nbsp;10 Sep 2011 · 12:50</div>
                <div class="meta-badge-item"><strong>Horizon:</strong>&nbsp;90 Days</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. SIDEBAR NAVIGATION (PROFESSIONAL BUTTONS, ZERO RADIO CIRCLES)
# ==============================================================================
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Overview"

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.6rem 0 1.1rem 0;">
        <div class="sidebar-brand-title">RETENTION ENGINE</div>
        <div class="sidebar-brand-sub">Customer Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-header">ANALYTICS</div>', unsafe_allow_html=True)

    analytics_pages = [
        "Overview",
        "Customer Prediction",
        "Segmentation",
        "Model Performance",
        "Retention Analytics",
    ]
    for p_name in analytics_pages:
        is_active = (st.session_state["current_page"] == p_name)
        if st.button(
            p_name,
            key=f"nav_{p_name}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["current_page"] = p_name
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-header">GOVERNANCE</div>', unsafe_allow_html=True)

    gov_page = "Methodology & Governance"
    is_gov_active = (st.session_state["current_page"] == gov_page)
    if st.button(
        gov_page,
        key=f"nav_{gov_page}",
        type="primary" if is_gov_active else "secondary",
        use_container_width=True,
    ):
        st.session_state["current_page"] = gov_page
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-status-box">
        <div class="status-group">
            <div class="status-label">MODEL STATUS</div>
            <div class="status-value"><span class="status-dot">●</span> Logistic Regression</div>
            <div class="status-sub">Frozen · Phase 7</div>
        </div>
        <div class="status-group">
            <div class="status-label">DATASET</div>
            <div class="status-value"><span class="status-dot">●</span> Online Retail II</div>
        </div>
        <div class="status-group">
            <div class="status-label">CUTOFF</div>
            <div class="status-value">10 Sep 2011 · 12:50</div>
        </div>
        <div class="status-group">
            <div class="status-label">PREDICTION HORIZON</div>
            <div class="status-value">90 Days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer-note">
        <strong>Evidence-first ML</strong><br>
        Non-causal analytics
    </div>
    """, unsafe_allow_html=True)

# Active page resolution
page = st.session_state["current_page"]


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "Overview":
    render_header(
        page_title="Executive Overview",
        page_subtitle="High-level synthesis of 90-day return prediction and customer retention dynamics."
    )

    # 4 Key KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Eligible Customers</div>
            <div class="b2b-metric-val">5,281</div>
            <div class="b2b-metric-sub">Active before 2011-09-10 cutoff</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Observed 90-Day Return Rate</div>
            <div class="b2b-metric-val">43.40%</div>
            <div class="b2b-metric-sub">2,292 returners / 2,989 non-returners</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Pre-Cutoff Commercial Spend</div>
            <div class="b2b-metric-val">£13.93M</div>
            <div class="b2b-metric-sub">£13,930,556.00 verified commercial spend</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Prediction Horizon</div>
            <div class="b2b-metric-val">90 Days</div>
            <div class="b2b-metric-sub">Complete follow-up (ending 2011-12-09)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Model Performance & Predictive Signal Summary")
        st.markdown("""
        - **Primary Discrimination**: The frozen **Logistic Regression** candidate model achieves **PR-AUC 0.7658** on the held-out test set, outperforming the majority baseline (0.4342) by +76.4% and the domain RFM heuristic (0.5772) by +32.7%.
        - **Held-Out Test ROC-AUC**: **0.8003** with a **Brier Score of 0.1792**, showing discrimination and probability-error performance on the held-out benchmark.
        - **Top-Decile Concentration**: Identifies the top 10% of customers with **2.0639x lift** over baseline, capturing 20.70% of all future returners.
        - **Temporal Generalizability**: Out-of-time chronological validation ($C_0 \to C_1$) demonstrated consistent discrimination (PR-AUC 0.7709, ROC-AUC 0.7994), showing similar discrimination in the evaluated forward chronological validation.
        """)

        st.subheader("Key Analytical Observations")
        st.markdown("""
        1. **Recency Feature Dominance**: Feature permutation importance and standardized coefficients confirm that recency of last purchase is the single strongest negative predictor of 90-day future return.
        2. **High-Value Exposure**: 1,850 high-value accounts (`Champions` and `Loyal Customers`) represent **80.35%** of pre-cutoff commercial spend (£11.19M). Within this cohort, **43 accounts** exhibit low predicted return probability ($p < 0.30$), representing **£152.5k** in historical spend.
        3. **Decile Concentration**: Decile ranking concentrates **67.0%** of realized post-cutoff commercial spend in the top 20% of predicted returners (Deciles 1–2).
        """)

    with col_right:
        st.subheader("High-Value Cohort Probability Distribution")
        high_val_labels = ["High Return Band (p >= 0.60)", "Intermediate Band (0.30 <= p < 0.60)", "Low Return Band (p < 0.30)"]
        high_val_counts = [1316, 491, 43]
        fig_pie = px.pie(
            names=high_val_labels,
            values=high_val_counts,
            color=high_val_labels,
            color_discrete_map={
                "High Return Band (p >= 0.60)": "#059669",
                "Intermediate Band (0.30 <= p < 0.60)": "#D97706",
                "Low Return Band (p < 0.30)": "#DC2626"
            },
            hole=0.45,
        )
        fig_pie.update_layout(
            margin=dict(t=15, b=15, l=15, r=15),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.caption("Distribution across 1,850 high-value accounts (£11.19M pre-cutoff spend).")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Governance & Non-Causal Disclosure:</strong> This application generates observational probability estimates 
        of customer repeat-purchase within a 90-day window under observed historical conditions. Predicted probabilities reflect 
        statistical associations, not guaranteed future outcomes or causal intervention effects. All retention outreach 
        requires controlled randomized A/B experimentation.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: CUSTOMER PREDICTION
# ==============================================================================
elif page == "Customer Prediction":
    render_header(
        page_title="Customer Prediction Engine",
        page_subtitle="Live scoring using frozen Logistic Regression pipeline on point-in-time pre-cutoff telemetry."
    )

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        segment_filter = st.selectbox(
            "Filter by Descriptive RFM Segment:",
            ["All Segments"] + sorted(list(df_features["pre_cutoff_rfm_segment"].unique()))
        )

    filtered_df = df_features if segment_filter == "All Segments" else df_features[df_features["pre_cutoff_rfm_segment"] == segment_filter]

    customer_options = [
        f"{row['masked_account_ref']} | Segment: {row['pre_cutoff_rfm_segment']} | Spend: £{row['monetary_total_pre']:,.2f} | Recency: {row['recency_days_pre']:.0f}d"
        for _, row in filtered_df.iterrows()
    ]

    with col_filter2:
        selected_option = st.selectbox(
            f"Select Account to Score ({len(filtered_df):,} accounts available):",
            customer_options,
            index=0 if len(customer_options) > 0 else None
        )

    if selected_option:
        selected_masked_ref = selected_option.split(" | ")[0]
        match_idx = df_features[df_features["masked_account_ref"] == selected_masked_ref].index
        customer_row = df_features.loc[match_idx[0]]

        # Extract features and run actual frozen model inference
        X_cust = df_features.loc[match_idx, PREDICTIVE_FEATURE_COLS]
        raw_prob = float(model_pipeline.predict_proba(X_cust)[0, 1])
        analytical_band = classify_analytical_band(raw_prob)

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.subheader(f"Predictive Assessment: {selected_masked_ref}")

        p_col1, p_col2 = st.columns([1, 1])

        with p_col1:
            st.markdown("#### Predicted 90-Day Return Probability")
            st.metric(
                label="Estimated Probability of Future Commercial Purchase",
                value=f"{raw_prob * 100:.1f}%",
                delta=f"{(raw_prob - 0.4340) * 100:+.1f}% vs Baseline (43.4%)"
            )

            if raw_prob >= 0.60:
                st.markdown(f'<span class="band-pill-high">{analytical_band}</span>', unsafe_allow_html=True)
            elif raw_prob >= 0.30:
                st.markdown(f'<span class="band-pill-intermediate">{analytical_band}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="band-pill-low">{analytical_band}</span>', unsafe_allow_html=True)

            st.markdown(r"""
            <div style="font-size: 0.83rem; color: #64748B; margin-top: 0.85rem; line-height: 1.5;">
                <strong>Prediction Window:</strong> 90 days following observation cutoff (2011-09-10 to 2011-12-09).<br>
                <strong>Concept:</strong> Future return / non-return. A low probability indicates low likelihood of purchase in the 90-day window, <em>NOT permanent churn</em>.
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown("#### Probability Gauge vs Baseline")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=raw_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': "#2563EB"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 30], 'color': '#FEE2E2'},
                        {'range': [30, 60], 'color': '#FEF3C7'},
                        {'range': [60, 100], 'color': '#D1FAE5'},
                    ],
                    'threshold': {
                        'line': {'color': "#0F172A", 'width': 3},
                        'thickness': 0.75,
                        'value': 43.40
                    }
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(t=20, b=10, l=25, r=25))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")
        st.subheader("Pre-Cutoff Customer Profile (Historical Telemetry)")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Recency", f"{customer_row['recency_days_pre']:.0f} days")
        with m2:
            st.metric("Frequency", f"{customer_row['frequency_pre']:.0f} orders")
        with m3:
            st.metric("Commercial Spend", f"£{customer_row['monetary_total_pre']:,.2f}")
        with m4:
            st.metric("Avg Order Value", f"£{customer_row['average_order_value_pre']:,.2f}")
        with m5:
            st.metric("Tenure", f"{customer_row['customer_tenure_days_pre']:.0f} days")
        with m6:
            st.metric("Cancellation Rate", f"{customer_row['cancellation_rate_pre']*100:.1f}%")

        with st.expander("Inspect Complete Feature Vector (27 Predictors)"):
            feat_view = pd.DataFrame({
                "Feature Name": PREDICTIVE_FEATURE_COLS,
                "Value": [str(customer_row[col]) for col in PREDICTIVE_FEATURE_COLS]
            })
            st.dataframe(feat_view, use_container_width=True)

        st.markdown("#### Strategic Retention Playbook Recommendation")
        seg = customer_row["pre_cutoff_rfm_segment"]
        if seg in ["Champions", "Loyal Customers"] and raw_prob < 0.30:
            st.warning("""
            **Priority P1 — High Historical Value & Low Predicted Return Probability**:  
            **Action**: Dedicated commercial account manager contact; customized commercial catalog or terms review. Avoid generic automated email blasts.  
            **Rationale / Testable Hypothesis**: High historical commercial spend combined with low predicted future return probability indicates candidate cohort for testing high-touch retention outreach. (Testing required vs uncontacted holdout).
            """)
        elif seg in ["Champions", "Loyal Customers"] and raw_prob < 0.60:
            st.info("""
            **Priority P2 — High Value & Intermediate Predicted Return Probability**:  
            **Action**: Proactive category replenishment reminders aligned with historical ordering cadence; commercial freight threshold incentives.  
            **Rationale / Testable Hypothesis**: Candidate cohort for testing automated replenishment and freight incentives to evaluate re-order timing relative to an uncontacted holdout.
            """)
        elif raw_prob >= 0.60:
            st.success("""
            **Priority P5 — High-Probability Organic Returner (Organic Maintenance)**:  
            **Action**: Maintain seamless transactional experience, inventory availability, and premium service. Avoid aggressive, margin-diluting discount coupons.  
            **Rationale / Testable Hypothesis**: High observed baseline return rate indicates organic demand; discounts largely subsidize purchases that would occur naturally.
            """)
        elif seg in ["Lost", "Hibernating"] and raw_prob < 0.30:
            st.error("""
            **Priority P4 — Low Historical Value & Low Return Probability (Spend Suppression)**:  
            **Action**: Suppress expensive outbound marketing and paid digital retargeting; rely solely on low-cost automated re-activation workflows.  
            **Rationale / Testable Hypothesis**: Low baseline return rate indicates low responsiveness; suppresses promotional budget waste.
            """)
        else:
            st.info("""
            **Priority P3 — Intermediate Value & Moderate Return Probability**:  
            **Action**: Automated category cross-selling sequences and product bundling recommendations to cement multi-order purchasing habits.  
            **Rationale / Testable Hypothesis**: Mid-tier account nurturing to measure whether second-order conversion can be incrementally elevated.
            """)


# ==============================================================================
# PAGE 3: SEGMENTATION
# ==============================================================================
elif page == "Segmentation":
    render_header(
        page_title="Descriptive Segmentation",
        page_subtitle="Phase 4 RFM segmentation evaluation and descriptive behavioral separation analysis."
    )

    # Executive figures from Checkpoint 06
    seg_m1, seg_m2, seg_m3, seg_m4 = st.columns(4)
    with seg_m1:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Total Full-Period Customers</div>
            <div class="b2b-metric-val">5,878</div>
            <div class="b2b-metric-sub">Phase 4 evaluated population</div>
        </div>
        """, unsafe_allow_html=True)
    with seg_m2:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Total Commercial Spend</div>
            <div class="b2b-metric-val">£17.37M</div>
            <div class="b2b-metric-sub">£17,374,804.25 total observed spend</div>
        </div>
        """, unsafe_allow_html=True)
    with seg_m3:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Primary Segments</div>
            <div class="b2b-metric-val">8 Segments</div>
            <div class="b2b-metric-sub">Rule-based RFM partition</div>
        </div>
        """, unsafe_allow_html=True)
    with seg_m4:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Kruskal-Wallis Tests</div>
            <div class="b2b-metric-val">p < 1e-100</div>
            <div class="b2b-metric-sub">Statistically significant separation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Frozen segment data from Checkpoint 06
    rfm_segments_overview = [
        {"RFM Segment": "Champions", "Customers": 1405, "Share (%)": 23.90, "Observed Spend (£)": 11912395.68, "Spend Share (%)": 68.56, "Median Recency": 8.0, "Median Frequency": 11.0, "Median Monetary (£)": 3593.06},
        {"RFM Segment": "Loyal Customers", "Customers": 755, "Share (%)": 12.84, "Observed Spend (£)": 2045258.26, "Spend Share (%)": 11.77, "Median Recency": 95.1, "Median Frequency": 5.0, "Median Monetary (£)": 1785.13},
        {"RFM Segment": "At Risk", "Customers": 283, "Share (%)": 4.81, "Observed Spend (£)": 1007120.98, "Spend Share (%)": 5.80, "Median Recency": 325.1, "Median Frequency": 6.0, "Median Monetary (£)": 1762.61},
        {"RFM Segment": "Potential Loyalists", "Customers": 709, "Share (%)": 12.06, "Observed Spend (£)": 866751.72, "Spend Share (%)": 4.99, "Median Recency": 23.0, "Median Frequency": 3.0, "Median Monetary (£)": 754.20},
        {"RFM Segment": "Hibernating", "Customers": 928, "Share (%)": 15.79, "Observed Spend (£)": 858240.13, "Spend Share (%)": 4.94, "Median Recency": 391.1, "Median Frequency": 2.0, "Median Monetary (£)": 634.92},
        {"RFM Segment": "About to Sleep", "Customers": 844, "Share (%)": 14.36, "Observed Spend (£)": 360218.28, "Spend Share (%)": 2.07, "Median Recency": 188.9, "Median Frequency": 1.0, "Median Monetary (£)": 303.29},
        {"RFM Segment": "Lost", "Customers": 717, "Share (%)": 12.20, "Observed Spend (£)": 244313.76, "Spend Share (%)": 1.41, "Median Recency": 575.1, "Median Frequency": 1.0, "Median Monetary (£)": 214.80},
        {"RFM Segment": "Recent Customers", "Customers": 237, "Share (%)": 4.03, "Observed Spend (£)": 80505.44, "Spend Share (%)": 0.46, "Median Recency": 29.0, "Median Frequency": 1.0, "Median Monetary (£)": 240.55},
    ]
    df_rfm = pd.DataFrame(rfm_segments_overview)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Observed Commercial Spend by Segment (£)")
        fig_spend = px.bar(
            df_rfm.sort_values("Observed Spend (£)", ascending=True),
            x="Observed Spend (£)",
            y="RFM Segment",
            orientation="h",
            text="Observed Spend (£)",
            color="Observed Spend (£)",
            color_continuous_scale="Blues"
        )
        fig_spend.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        fig_spend.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig_spend, use_container_width=True)

    with col2:
        st.subheader("Customer Distribution by Segment")
        fig_count = px.bar(
            df_rfm.sort_values("Customers", ascending=True),
            x="Customers",
            y="RFM Segment",
            orientation="h",
            text="Customers",
            color="Customers",
            color_continuous_scale="Blues"
        )
        fig_count.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), coloraxis_showscale=False)
        fig_count.update_traces(textposition="outside")
        st.plotly_chart(fig_count, use_container_width=True)

    if "segment_retention_profiles" in cp11:
        st.subheader("Descriptive Agreement: Observed Return Rate vs. Mean Model Predicted Probability")
        seg_data = cp11["segment_retention_profiles"]
        df_seg11 = pd.DataFrame.from_dict(seg_data, orient="index").reset_index().rename(columns={"index": "RFM Segment"})

        fig_agree = go.Figure()
        fig_agree.add_trace(go.Bar(
            name="Observed 90-Day Return Rate (%)",
            x=df_seg11["RFM Segment"],
            y=df_seg11["observed_return_rate_pct"],
            marker_color="#2563EB"
        ))
        fig_agree.add_trace(go.Bar(
            name="Mean Model Predicted Probability (%)",
            x=df_seg11["RFM Segment"],
            y=df_seg11["mean_predicted_return_prob"] * 100,
            marker_color="#10B981"
        ))
        fig_agree.update_layout(
            barmode="group",
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis_title="Percentage (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_agree, use_container_width=True)

    st.subheader("Segment Telemetry Summary Table")
    st.dataframe(df_rfm, use_container_width=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Descriptive Separation Note:</strong> Non-parametric Kruskal-Wallis tests confirmed statistically significant 
        separation across Recency (H=5,160.6), Frequency (H=5,213.0), and Monetary (H=3,826.5) dimensions (all p < 1e-100). 
        This establishes descriptive statistical separation among rule-based groupings; it does NOT imply that segments 
        cause customer purchasing behavior or that this partition is mathematically optimal.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================
elif page == "Model Performance":
    render_header(
        page_title="MODEL PERFORMANCE",
        page_subtitle="Held-out benchmark evaluation of the frozen candidate models."
    )

    benchmark_data = [
        {"Model": "M0 Majority Baseline", "Type": "Baseline", "PR-AUC": 0.4342, "ROC-AUC": 0.5000, "Brier Score": 0.2457, "Precision": "0.0000", "Recall": "0.0000", "F1": "0.0000", "Top-Decile Lift": "N/A (Ties: constant score)", "Top-Decile Gain": "N/A"},
        {"Model": "M1 Stratified Random", "Type": "Baseline", "PR-AUC": 0.4623, "ROC-AUC": 0.5207, "Brier Score": 0.3211, "Precision": "0.4358", "Recall": "0.4858", "F1": "0.4578", "Top-Decile Lift": "1.0211x", "Top-Decile Gain": "10.24%"},
        {"Model": "M2 Frozen RFM Heuristic", "Type": "Baseline", "PR-AUC": 0.5772, "ROC-AUC": 0.6908, "Brier Score": 0.2095, "Precision": "0.6621", "Recall": "0.6275", "F1": "0.6443", "Top-Decile Lift": "1.6076x", "Top-Decile Gain": "16.12%"},
        {"Model": "M3 Logistic Regression", "Type": "Candidate", "PR-AUC": 0.7658, "ROC-AUC": 0.8003, "Brier Score": 0.1792, "Precision": "0.6472", "Recall": "0.7712", "F1": "0.7038", "Top-Decile Lift": "2.0639x", "Top-Decile Gain": "20.70%"},
        {"Model": "M4 Random Forest", "Type": "Candidate", "PR-AUC": 0.7596, "ROC-AUC": 0.7892, "Brier Score": 0.1838, "Precision": "0.6508", "Recall": "0.7146", "F1": "0.6812", "Top-Decile Lift": "2.1290x", "Top-Decile Gain": "21.35%"},
        {"Model": "M5 HistGradientBoosting", "Type": "Candidate", "PR-AUC": 0.7545, "ROC-AUC": 0.7875, "Brier Score": 0.1870, "Precision": "0.5886", "Recall": "0.8105", "F1": "0.6819", "Top-Decile Lift": "2.0639x", "Top-Decile Gain": "20.70%"},
    ]
    df_bm = pd.DataFrame(benchmark_data)

    st.subheader("M0–M5 Model Benchmark Comparison (Frozen Checkpoint 10)")
    st.dataframe(df_bm, use_container_width=True)

    st.markdown(r"""
    <div class="disclaimer-box">
        <strong>Benchmark Evaluation Notes:</strong>
        <ul>
            <li><strong>Primary Metric Emphasis:</strong> PR-AUC evaluates discrimination under class imbalance (43.4% positive prevalence).</li>
            <li><strong>Evaluation Set Isolation:</strong> The 20% stratified held-out test set ($N=1,057$) was used strictly for benchmark evaluation, never for model training or threshold tuning.</li>
            <li><strong>Constant-Score Baseline Top-Decile Lift:</strong> Marked <em>N/A (Ties: constant score)</em> because M0 assigns an identical probability ($p = 0.4339$) to all customers; ranking the top 10% is an arbitrary tie-breaking artifact and not a meaningful predictive ranking.</li>
            <li><strong>Terminology:</strong> Target represents 90-day future return / non-return, <em>NOT permanent churn</em>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 3 Comparison Visualizations Side by Side
    st.subheader("Model Comparison Visualizations: PR-AUC, ROC-AUC, and Brier Score")
    c_pr, c_roc, c_brier = st.columns(3)

    with c_pr:
        st.markdown("**Precision-Recall AUC (Primary)**")
        fig_prauc = px.bar(
            df_bm,
            x="PR-AUC",
            y="Model",
            orientation="h",
            color="Type",
            text="PR-AUC",
            color_discrete_map={"Baseline": "#94A3B8", "Candidate": "#2563EB"}
        )
        fig_prauc.add_vline(x=0.4342, line_dash="dash", line_color="#EF4444", annotation_text="Trivial (0.4342)")
        fig_prauc.update_layout(height=280, margin=dict(t=15, b=15, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_prauc, use_container_width=True)

    with c_roc:
        st.markdown("**ROC-AUC Comparison**")
        fig_roc = px.bar(
            df_bm,
            x="ROC-AUC",
            y="Model",
            orientation="h",
            color="Type",
            text="ROC-AUC",
            color_discrete_map={"Baseline": "#94A3B8", "Candidate": "#2563EB"}
        )
        fig_roc.add_vline(x=0.5000, line_dash="dash", line_color="#EF4444", annotation_text="Random (0.5000)")
        fig_roc.update_layout(height=280, margin=dict(t=15, b=15, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_brier:
        st.markdown("**Brier Score (Lower is Better)**")
        fig_brier = px.bar(
            df_bm,
            x="Brier Score",
            y="Model",
            orientation="h",
            color="Type",
            text="Brier Score",
            color_discrete_map={"Baseline": "#94A3B8", "Candidate": "#10B981"}
        )
        fig_brier.update_layout(height=280, margin=dict(t=15, b=15, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_brier, use_container_width=True)

    st.markdown("---")

    # M3 Operating Point Deep Dive
    st.subheader("Logistic Regression Candidate Operating Point (Threshold = 0.38)")
    st.caption("Threshold t* = 0.38 tuned strictly via 5-fold cross-validation on training data to maximize F1 score; held-out test set remained unaccessed.")

    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.metric("Tuned Threshold (t*)", "0.38", help="Optimized on 5-fold CV to maximize F1")
    with col_m2:
        st.metric("Held-Out Precision", "64.7%", help="Precision: 0.6472 on 20% test set")
    with col_m3:
        st.metric("Held-Out Recall", "77.1%", help="Recall: 0.7712 on 20% test set")
    with col_m4:
        st.metric("Held-Out F1-Score", "0.7038", help="F1-Score on 20% test set")
    with col_m5:
        st.metric("Top-Decile Lift", "2.0639x", help="2.0639x over base rate (20.70% captured)")
    with col_m6:
        st.metric("Test Brier Score", "0.1792", help="Calibration error (lower is better)")

    col_cm, col_calib = st.columns(2)
    with col_cm:
        st.markdown("**Held-Out Test Confusion Matrix (N = 1,057)**")
        cm_data = [[405, 193], [105, 354]]
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Predicted Non-Return (0)", "Predicted Return (1)"],
            y=["Actual Non-Return (0)", "Actual Return (1)"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Held-out test set counts: TP = 354, FN = 105, FP = 193, TN = 405.")

    with col_calib:
        st.markdown("**Probability Quality & Empirical Calibration Diagnostics**")
        st.markdown(r"""
        - **Linear Fit Binned Reliability Slope**: **0.9841** (near-ideal slope 1.0)
        - **Linear Fit Intercept**: **0.0025** (near-zero intercept 0.0)
        - **Test Brier Score**: **0.1792** (vs M0 baseline 0.2457 and M1 0.3211)
        
        <div class="disclaimer-box" style="margin-top: 0.6rem;">
            <strong>Calibration Terminology:</strong> Linear fit to 10 quantile-binned reliability points reflects 
            empirical diagnostics of predicted probability ranking and grouping quality, not formal logistic calibration 
            proof. Brier score reflects calibration, resolution/discrimination, and inherent outcome uncertainty.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Forward Chronological Validation
    st.subheader("Forward Chronological Validation (Cohort 0 -> Cohort 1)")
    col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
    with col_fc1:
        st.metric("Forward PR-AUC", "0.7709", "+0.0051 vs Test")
    with col_fc2:
        st.metric("Forward ROC-AUC", "0.7994", "-0.0009 vs Test")
    with col_fc3:
        st.metric("Forward Top-Decile Lift", "2.1125x", "21.16% captured")
    with col_fc4:
        st.metric("Forward Brier Score", "0.2000", "Stable out-of-time")

    st.markdown(r"""
    To verify that predictive performance is not an artifact of random train/test partitioning, the candidate model was 
    trained strictly on historical **Cohort 0** (cutoff: 2011-06-12, target window: 2011-06-12 to 2011-09-10, $N = 4,977$) 
    and evaluated forward out-of-time on **Cohort 1** (cutoff: 2011-09-10, target window: 2011-09-10 to 2011-12-09, $N = 5,281$):
    - **Temporal Stability**: PR-AUC drift is $+0.0051$ and ROC-AUC drift is $-0.0009$, demonstrating temporal consistency across rolling-origin observation dates.
    - **Evaluation Consistency**: Out-of-time discrimination aligns with the internal stratified test set PR-AUC ($0.7709$ vs $0.7658$), showing similar discrimination in the evaluated forward chronological validation.
    - **Rolling-Origin Overlap**: As audited in Checkpoint 10, $94.24\%$ of Cohort 1 customers overlap with Cohort 0, matching realistic enterprise quarterly rolling-origin production scoring.
    """)

    st.markdown(r"""
    <div class="disclaimer-box">
        <strong>Mandatory Non-Causal Evaluation Disclaimer:</strong> Model benchmark metrics demonstrate predictive discrimination 
        and statistical association under historical observational conditions. They do NOT measure causal uplift, treatment effects, 
        or the business impact of retention interventions. An intervention based on model predictions requires controlled randomized A/B experimentation.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: RETENTION ANALYTICS
# ==============================================================================
elif page == "Retention Analytics":
    render_header(
        page_title="Retention Analytics",
        page_subtitle="Decision-support analytics: high-value cohort concentration and observational commercial spend capture."
    )

    st.subheader("High-Value Cohort Exposure (Champions & Loyal Customers)")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">High-Value Customers</div>
            <div class="b2b-metric-val">1,850</div>
            <div class="b2b-metric-sub">Champions (1,267) + Loyal Customers (583)</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Historical Commercial Spend</div>
            <div class="b2b-metric-val">£11.19M</div>
            <div class="b2b-metric-sub">£11,192,691.82 pre-cutoff spend</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="b2b-metric-card">
            <div class="b2b-metric-label">Cohort Spend Share</div>
            <div class="b2b-metric-val">80.35%</div>
            <div class="b2b-metric-sub">Share of total eligible pre-cutoff spend (£13.93M)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("Descriptive Analytical Bands within High-Value Cohort")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="b2b-metric-card" style="border-left: 4px solid #DC2626;">
            <div class="b2b-metric-label band-card-low-text">Low Return Band (p < 0.30)</div>
            <div class="b2b-metric-val band-card-low-text">43 Accounts</div>
            <div class="b2b-metric-sub">Historical Spend: <strong>£152,487.88</strong></div>
            <div class="b2b-metric-desc">High historical value combined with low predicted return probability. Candidate cohort for testing proactive outreach.</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="b2b-metric-card" style="border-left: 4px solid #D97706;">
            <div class="b2b-metric-label band-card-mid-text">Intermediate Band (0.30 <= p < 0.60)</div>
            <div class="b2b-metric-val band-card-mid-text">491 Accounts</div>
            <div class="b2b-metric-sub">Historical Spend: <strong>£755,081.91</strong></div>
            <div class="b2b-metric-desc">Moderate predicted return probability. Candidate for automated re-order reminders and category replenishment.</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="b2b-metric-card" style="border-left: 4px solid #059669;">
            <div class="b2b-metric-label band-card-high-text">High Return Band (p >= 0.60)</div>
            <div class="b2b-metric-val band-card-high-text">1,316 Accounts</div>
            <div class="b2b-metric-sub">Historical Spend: <strong>£10,285,122.03</strong></div>
            <div class="b2b-metric-desc">High return likelihood. Organic maintenance; avoid margin-diluting discount coupons.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>Mandatory Analytical Band Disclaimer:</strong> These bands are analytical groupings for interpretation and are not 
        validated operational, clinical, financial, or churn-risk thresholds. No claim is made that 0.30 or 0.60 is an optimal intervention threshold.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Prediction Decile Concentration Analysis")

    if "decile_concentration_analysis" in cp11:
        dec_data = cp11["decile_concentration_analysis"]
        df_dec = pd.DataFrame(dec_data)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Observed Return Rate by Predicted Return Decile**")
            fig_dec_ret = px.bar(
                df_dec,
                x="predicted_return_decile",
                y="observed_return_rate_pct",
                text="observed_return_rate_pct",
                labels={"predicted_return_decile": "Decile (1=Highest Predicted Probability)", "observed_return_rate_pct": "Return Rate (%)"},
                color="observed_return_rate_pct",
                color_continuous_scale="Blues"
            )
            fig_dec_ret.update_layout(height=320, margin=dict(t=15, b=15, l=15, r=15), coloraxis_showscale=False)
            fig_dec_ret.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_dec_ret, use_container_width=True)

        with col2:
            st.markdown("**Cumulative Post-Cutoff Commercial Spend Share (%)**")
            fig_dec_spend = px.line(
                df_dec,
                x="predicted_return_decile",
                y="cumulative_spend_share_pct",
                markers=True,
                labels={"predicted_return_decile": "Decile", "cumulative_spend_share_pct": "Cumulative Spend Share (%)"}
            )
            fig_dec_spend.add_hline(y=66.99, line_dash="dash", line_color="#D97706", annotation_text="Deciles 1-2: 67.0% Spend")
            fig_dec_spend.update_layout(height=320, margin=dict(t=15, b=15, l=15, r=15))
            st.plotly_chart(fig_dec_spend, use_container_width=True)

        st.subheader("Decile Ranking Telemetry (10 Cohort Deciles)")
        st.dataframe(
            df_dec[[
                "predicted_return_decile", "customer_count", "mean_predicted_prob",
                "observed_return_rate_pct", "lift_vs_base_rate", "cumulative_gain_pct",
                "realized_post_cutoff_commercial_spend_gbp", "realized_spend_share_pct", "cumulative_spend_share_pct"
            ]].rename(columns={
                "predicted_return_decile": "Decile",
                "customer_count": "Customers",
                "mean_predicted_prob": "Mean Pred Prob",
                "observed_return_rate_pct": "Observed Return (%)",
                "lift_vs_base_rate": "Lift (vs 43.4%)",
                "cumulative_gain_pct": "Cumul Gain (%)",
                "realized_post_cutoff_commercial_spend_gbp": "Realized Spend (£)",
                "realized_spend_share_pct": "Spend Share (%)",
                "cumulative_spend_share_pct": "Cumul Spend Share (%)"
            }),
            use_container_width=True
        )

        st.markdown("""
        <div class="disclaimer-box">
            <strong>Observational Spend Interpretation:</strong> Customers in Deciles 1 and 2 accounted for 67.0% of realized 
            post-cutoff commercial spend (£2.04M of £3.04M). This concentration is a descriptive historical property of the cohort; it provides 
            an empirical basis for testing tiered retention allocation, but does NOT establish causal uplift or guaranteed revenue capture.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: METHODOLOGY & GOVERNANCE
# ==============================================================================
elif page == "Methodology & Governance":
    render_header(
        page_title="Methodology & Governance",
        page_subtitle="Auditability safeguards, point-in-time feature architecture, and non-causal guardrails."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Point-in-Time Prediction Design")
        st.markdown(r"""
        - **Dataset**: Online Retail II (UK commercial wholesale transactions)
        - **Global Observation Cutoff**: $T_{obs} = \text{2011-09-10 12:50:00}$
        - **Forward Prediction Window**: $(T_{obs}, T_{obs} + 90\text{d}]$ (ending 2011-12-09 12:50:00)
        - **Target**: 90-day future return / non-return (binary retention indicator)
        - **Feature Timing**: Pre-cutoff transactions only ($\le T_{obs}$)
        - **Zero Right-Censoring**: 100% complete follow-up for all 5,281 eligible customers
        - **Model**: Logistic Regression (Phase 7, Frozen)
        - **Primary Evaluation Metric**: PR-AUC (Precision-Recall Area Under Curve)
        """)

    with col2:
        st.subheader("Leakage Controls & Engineering Rigor")
        st.markdown(r"""
        - **Point-in-Time Feature Isolation**: All 27 predictors computed strictly from transactions timestamped $\le T_{obs}$.
        - **No Post-Cutoff Features**: Forward transactional telemetry completely masked during feature extraction.
        - **Test Set Isolation**: 80/20 stratified partitioning; held-out test set unaccessed during model & threshold selection.
        - **Threshold Tuning**: Decision threshold $t^* = 0.38$ tuned strictly via 5-fold cross-validation on training data to maximize F1 score.
        - **No Synthetic Augmentation**: SMOTE was evaluated and rejected; real cohort data maintained throughout.
        - **Temporal Validation**: Validated out-of-time via forward chronological evaluation ($C_0 \to C_1$) across calendar periods.
        """)

    st.markdown("---")
    st.subheader("Mandatory Governance & Non-Causal Limitations")

    st.markdown(r"""
    1. **Non-Causal Association**: The predictive model estimates observational statistical association with future return under observed historical conditions. It does NOT estimate the causal treatment effect of an intervention.
    2. **Commercial Spend vs. Revenue**: All transaction figures represent commercial purchase totals in the dataset, not certified accounting revenue.
    3. **Non-Return vs. Permanent Churn**: A customer who does not make a purchase in $(T_{obs}, T_{obs} + 90\text{d}]$ is a non-returner within that 90-day window, NOT permanently churned.
    4. **Analytical Bands vs. Operational Risk**: Probability bands ($p \ge 0.60$, $0.30\text{--}0.60$, $< 0.30$) are descriptive groupings, not validated clinical, financial, or operational thresholds.
    5. **Experimentation Requirement**: All retention playbook strategies must be tested via randomized controlled trials (A/B testing with uncontacted holdout control groups) prior to full deployment.
    6. **ROI Guardrail**: *Intervention effectiveness and ROI require randomized holdout/control testing.*
    """)

    st.markdown("---")
    st.caption("Retention Engine · Online Retail II | Checkpoint 11 Sealed | Audit-Passed")
