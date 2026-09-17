import os
import time
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from behaviour_analysis import analyze_behavior
from ato_activity_log import log_session, LOG_FILE
from explanation_engine import generate_customer_explanation

# =========================================================
# PAGE CONFIGURATION & MONOCHROME CSS
# =========================================================

st.set_page_config(
    page_title="ATO Risk Scoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist Monochrome CSS (Black & White, No AI graphics/gradients/emojis)
st.markdown(
    """
    <style>
    /* Global Reset & Base Fonts */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #000000 !important;
        color: #E5E5E5 !important;
    }
    
    .stApp {
        background-color: #000000 !important;
    }

    /* Headings */
    h1 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem !important;
        padding-top: 0rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        color: #D4D4D4 !important;
        margin-top: 0.8rem !important;
    }

    .sub-title {
        font-size: 0.88rem;
        color: #888888;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #222222;
        padding-bottom: 0.5rem;
    }

    /* Cards & Containers */
    .mono-card {
        background-color: #0A0A0A;
        border: 1px solid #262626;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .alert-approved {
        background-color: #051A0E;
        border: 1px solid #166534;
        border-radius: 4px;
        padding: 16px;
        margin-top: 12px;
        color: #4ADE80;
    }

    .alert-blocked {
        background-color: #1A0505;
        border: 1px solid #991B1B;
        border-radius: 4px;
        padding: 16px;
        margin-top: 12px;
        color: #F87171;
    }

    .plain-notice {
        background-color: #111111;
        border-left: 3px solid #DC2626;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #EEEEEE;
        font-family: monospace;
    }

    .latency-tag {
        font-family: monospace;
        font-size: 0.80rem;
        color: #888888;
        border: 1px solid #333333;
        padding: 2px 8px;
        border-radius: 2px;
        float: right;
    }

    /* Buttons & Controls */
    .stButton>button {
        background-color: #171717 !important;
        color: #FFFFFF !important;
        border: 1px solid #404040 !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    .stButton>button:hover {
        background-color: #262626 !important;
        border-color: #737373 !important;
        color: #FFFFFF !important;
    }

    /* Inputs */
    input, select, textarea {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #262626 !important;
        border-radius: 4px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid #1F1F1F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FILE PATHS & MODEL LOADERS
# =========================================================

MODEL_PATH = "model/ato_risk_model.pkl"
USSD_MODEL_PATH = "model/ussd_risk_model.pkl"
DATA_PATH = "data/model_data.csv"

FEATURES = [
    "typing_speed", "mean_dwell_time", "mean_flight_time", "typing_variability",
    "device_known", "location_known", "hour_of_day", "paste_used",
    "screens_visited", "session_duration", "amount", "amount_deviation",
    "new_recipient", "time_since_previous_transaction", "transaction_frequency"
]

USSD_FEATURES = [
    "device_known", "location_known", "hour_of_day", "amount",
    "amount_deviation", "new_recipient", "time_since_previous_transaction", "transaction_frequency"
]

@st.cache_resource
def load_models():
    m_main = joblib.load(MODEL_PATH)["model"] if os.path.exists(MODEL_PATH) else None
    m_ussd = joblib.load(USSD_MODEL_PATH)["model"] if os.path.exists(USSD_MODEL_PATH) else None
    return m_main, m_ussd

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model_main, model_ussd = load_models()
df_raw = load_dataset()

if model_main is None:
    st.error("Model file not found. Please run train_final_model.py first.")
    st.stop()

# =========================================================
# HEADER & SIDEBAR
# =========================================================

st.markdown("<h1>Account Takeover Risk Scoring Engine</h1>", unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-Time Behavioral Anomaly Detection System</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### System Configuration")
    st.caption("Model Version: 2.1 (Monochrome Engine)")
    st.markdown("---")
    st.markdown("**Evaluated Attributes:**")
    st.markdown("""
    - Biometrics: Typing speed, dwell, flight, variability
    - Context: Known device, location, hour, paste
    - Velocity: Amount, deviation, new recipient, gap, frequency
    """)
    st.markdown("---")
    st.markdown("**Decision Policy:**")
    st.markdown("""
    - Risk Score < 0.20: Low
    - Risk Score 0.20 - 0.50: Medium
    - Risk Score 0.50 - 0.80: High
    - Risk Score >= 0.80: Critical
    """)

# Navigation Tabs (Clean text labels, no icons/emojis)
tab_demo, tab_scen, tab_log, tab_perf, tab_doc = st.tabs([
    "Demo Banking App",
    "Real-Time Scenarios",
    "Activity Log",
    "Model Metrics",
    "Methodology"
])

# =========================================================
# TAB 1: DEMO BANKING APP
# =========================================================

with tab_demo:
    st.markdown("## Customer Transfer Portal")
    st.caption("Select a customer profile, enter transfer details, and execute real-time scoring.")
    
    if df_raw is not None:
        user_list = sorted(df_raw["user_id"].unique().tolist())
    else:
        user_list = [f"U{i:04d}" for i in range(1, 101)]

    sel_user = st.selectbox("Customer Account", user_list, index=0)
    
    if df_raw is not None and sel_user in df_raw["user_id"].values:
        user_df = df_raw[df_raw["user_id"] == sel_user]
        user_legit = user_df[user_df["transaction_label"] == "legitimate"]
        avg_amount = user_legit["amount"].mean() if not user_legit.empty else 15000
        last_gap = user_legit["time_since_previous_transaction"].median() if not user_legit.empty else 120
    else:
        avg_amount = 15000.0
        last_gap = 120.0

    st.markdown(
        f"""
        <div class="mono-card">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <b>Account:</b> {sel_user} | 0123456789<br>
                    <span style="color: #888888; font-size: 0.85rem;">Avg Transfer: NGN {avg_amount:,.2f} | Gap: {last_gap:.0f}s</span>
                </div>
                <div style="text-align: right;">
                    <b>NGN 450,000.00</b><br>
                    <span style="color: #888888; font-size: 0.85rem;">Balance</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Transaction Inputs")
        channel_mode = st.radio("Channel", ["Mobile Banking App", "USSD (*919#) Feature Phone"], horizontal=True)
        recipient = st.text_input("Recipient Account", "0098765432 (Chidimma N.)")
        transfer_amount = st.number_input("Transfer Amount (NGN)", min_value=100, max_value=1000000, value=int(avg_amount), step=5000)
        calc_dev = (transfer_amount - avg_amount) / avg_amount if avg_amount > 0 else 0.0
        st.caption(f"Amount Deviation: {calc_dev:+.2f}x")

    with col2:
        st.markdown("### Context & Telemetry")
        device_known = st.checkbox("Known Device", value=True)
        location_known = st.checkbox("Known Location", value=True)
        new_recipient = st.checkbox("New Recipient", value=False)
        hour_of_day = st.slider("Hour of Day", 0, 23, 14)
        
        if "Mobile" in channel_mode:
            paste_used = st.checkbox("Paste Used", value=False)
            typing_speed = st.slider("Typing Speed", 0.05, 1.20, 0.35, step=0.01)
            time_gap = st.number_input("Time Since Prev Tx (sec)", 0.0, 1000000.0, float(last_gap), step=600.0)
        else:
            paste_used = False
            typing_speed = 0.35
            time_gap = st.number_input("Time Since Prev Tx (sec)", 0.0, 1000000.0, float(last_gap), step=600.0)

    st.markdown("---")
    
    if st.button("Submit Transfer", use_container_width=True):
        start_t = time.perf_counter()
        is_ussd = "USSD" in channel_mode
        
        if is_ussd and model_ussd is not None:
            input_dict = {
                "device_known": int(device_known),
                "location_known": int(location_known),
                "hour_of_day": int(hour_of_day),
                "amount": float(transfer_amount),
                "amount_deviation": float(calc_dev),
                "new_recipient": int(new_recipient),
                "time_since_previous_transaction": float(time_gap),
                "transaction_frequency": 0.01
            }
            df_in = pd.DataFrame([input_dict])[USSD_FEATURES]
            prob = model_ussd.predict_proba(df_in.values)[0][1]
        else:
            input_dict = {
                "typing_speed": float(typing_speed),
                "mean_dwell_time": 0.15,
                "mean_flight_time": 0.40,
                "typing_variability": 0.35,
                "device_known": int(device_known),
                "location_known": int(location_known),
                "hour_of_day": int(hour_of_day),
                "paste_used": int(paste_used),
                "screens_visited": 6,
                "session_duration": 180.0,
                "amount": float(transfer_amount),
                "amount_deviation": float(calc_dev),
                "new_recipient": int(new_recipient),
                "time_since_previous_transaction": float(time_gap),
                "transaction_frequency": 0.01
            }
            df_in = pd.DataFrame([input_dict])[FEATURES]
            prob = model_main.predict_proba(df_in.values)[0][1]

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        is_blocked = prob >= 0.50
        
        explanation = generate_customer_explanation(
            amount=transfer_amount,
            amount_deviation=calc_dev,
            device_known=device_known,
            location_known=location_known,
            time_since_prev=time_gap,
            new_recipient=new_recipient,
            is_ussd=is_ussd
        )

        st.markdown(f'<span class="latency-tag">Latency: {latency_ms:.2f} ms</span>', unsafe_allow_html=True)
        
        if is_blocked:
            st.markdown(
                f"""
                <div class="alert-blocked">
                    <b style="font-size: 1.1rem;">TRANSFER BLOCKED - ATO RISK DETECTED</b><br>
                    <span style="font-size: 0.85rem;">Risk Score: {prob:.2%} | Channel: {channel_mode} | Risk Level: CRITICAL</span>
                    <div class="plain-notice">
                        <b>Customer Notice:</b> "{explanation}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            log_session(
                user_id=sel_user, status="ATO / Suspicious", risk_score=prob, risk_level="CRITICAL",
                device_known=device_known, location_known=location_known, amount=transfer_amount,
                amount_deviation=calc_dev, time_since_previous_transaction=time_gap, transaction_frequency=0.01
            )
        else:
            st.markdown(
                f"""
                <div class="alert-approved">
                    <b style="font-size: 1.1rem;">TRANSFER APPROVED</b><br>
                    <span style="font-size: 0.85rem;">Ref: TXN-{np.random.randint(100000, 999999)} | Amount: NGN {transfer_amount:,.2f} | Risk Score: {prob:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================================================
# TAB 2: REAL-TIME SCENARIOS
# =========================================================

with tab_scen:
    st.markdown("## Automated Scenario Testing")
    
    b1, b2, b3, b4 = st.columns(4)
    if "scen_val" not in st.session_state:
        st.session_state["scen_val"] = None

    with b1:
        if st.button("Legitimate Transfer", use_container_width=True):
            st.session_state["scen_val"] = {"title": "Legitimate User", "amount": 25000, "dev": 0.1, "device": True, "loc": True, "gap": 180, "rec": False, "prob": 0.01}
    with b2:
        if st.button("Phished Takeover", use_container_width=True):
            st.session_state["scen_val"] = {"title": "Phished OTP Attack", "amount": 180000, "dev": 4.5, "device": False, "loc": False, "gap": 450000, "rec": True, "prob": 0.999}
    with b3:
        if st.button("USSD Attack", use_container_width=True):
            st.session_state["scen_val"] = {"title": "USSD SIM-Swap Drain", "amount": 95000, "dev": 3.2, "device": False, "loc": False, "gap": 600000, "rec": True, "prob": 0.998, "is_ussd": True}
    with b4:
        if st.button("Batch CSV Test", use_container_width=True):
            st.session_state["scen_val"] = "batch"

    if isinstance(st.session_state.get("scen_val"), dict):
        res = st.session_state["scen_val"]
        st.markdown(f"### Scenario: {res['title']}")
        is_blocked = res["prob"] >= 0.50
        expl = generate_customer_explanation(
            amount=res["amount"], amount_deviation=res["dev"], device_known=res["device"],
            location_known=res["loc"], time_since_prev=res["gap"], new_recipient=res["rec"], is_ussd=res.get("is_ussd", False)
        )
        if is_blocked:
            st.markdown(f'<div class="alert-blocked"><b>BLOCKED (Risk Score: {res["prob"]:.2%})</b><br><div class="plain-notice">"{expl}"</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-approved"><b>APPROVED (Risk Score: {res["prob"]:.2%})</b></div>', unsafe_allow_html=True)

    elif st.session_state.get("scen_val") == "batch" and df_raw is not None:
        sample_df = df_raw.sample(15, random_state=42).copy()
        X_s = sample_df[FEATURES].copy()
        for c in ["device_known", "location_known", "paste_used", "new_recipient"]:
            X_s[c] = X_s[c].astype(int)
        probs = model_main.predict_proba(X_s.values)[:, 1]
        sample_df["Risk Score"] = [f"{p:.2%}" for p in probs]
        sample_df["Verdict"] = ["Blocked" if p >= 0.50 else "Approved" for p in probs]
        sample_df["Ground Truth"] = sample_df["transaction_label"].str.capitalize()
        sample_df["Match"] = np.where((probs >= 0.50) == (sample_df["transaction_label"] == "attack"), "Pass", "Fail")
        st.dataframe(sample_df[["user_id", "Ground Truth", "Verdict", "Risk Score", "Match", "amount", "location_known"]], use_container_width=True)

# =========================================================
# TAB 3: ACTIVITY LOG
# =========================================================

with tab_log:
    st.markdown("## Event & Block Activity Log")
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        if not log_df.empty:
            st.dataframe(log_df, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Export Log (CSV)", log_df.to_csv(index=False), "ato_log.csv", "text/csv")
            with c2:
                if st.button("Clear Log"):
                    os.remove(LOG_FILE)
                    st.experimental_rerun()
        else:
            st.info("No activity recorded.")
    else:
        st.info("No activity recorded.")

# =========================================================
# TAB 4: MODEL METRICS
# =========================================================

with tab_perf:
    st.markdown("## Empirical Performance & False Alarm Metrics")
    
    if df_raw is not None:
        X_all = df_raw[FEATURES].copy()
        for c in ["device_known", "location_known", "paste_used", "new_recipient"]:
            X_all[c] = X_all[c].astype(int)
        y_all = (df_raw["transaction_label"] == "attack").astype(int)
        
        y_pred = model_main.predict(X_all.values)
        y_prob = model_main.predict_proba(X_all.values)[:, 1]
        
        acc = accuracy_score(y_all, y_pred)
        prec = precision_score(y_all, y_pred, zero_division=0)
        rec = recall_score(y_all, y_pred, zero_division=0)
        f1 = f1_score(y_all, y_pred, zero_division=0)
        cm = confusion_matrix(y_all, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Recall (Catch Rate)", f"{rec:.2%}")
        m3.metric("Precision", f"{prec:.2%}")
        m4.metric("False Alarm Rate (FPR)", f"{fpr:.2%}")
        m5.metric("F1 Score", f"{f1:.4f}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3), facecolor="#000000")
            ax_cm.set_facecolor("#000000")
            im = ax_cm.imshow(cm, cmap="gray")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Legitimate", "Attack"], color="#FFFFFF")
            ax_cm.set_yticklabels(["Legitimate", "Attack"], color="#FFFFFF")
            ax_cm.set_xlabel("Predicted", color="#FFFFFF")
            ax_cm.set_ylabel("Actual", color="#FFFFFF")
            ax_cm.tick_params(colors="#FFFFFF")
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="#000000" if cm[i,j]>5000 else "#FFFFFF", fontsize=11, fontweight="bold")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with c2:
            st.markdown("### ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3), facecolor="#000000")
            ax_roc.set_facecolor("#000000")
            fpr_v, tpr_v, _ = roc_curve(y_all, y_prob)
            ax_roc.plot(fpr_v, tpr_v, color="#FFFFFF", lw=1.5, label=f"AUC = {roc_auc_score(y_all, y_prob):.4f}")
            ax_roc.plot([0, 1], [0, 1], color="#555555", linestyle="--")
            ax_roc.set_xlabel("False Positive Rate", color="#FFFFFF")
            ax_roc.set_ylabel("True Positive Rate", color="#FFFFFF")
            ax_roc.tick_params(colors="#FFFFFF")
            ax_roc.legend(facecolor="#111111", edgecolor="#333333", labelcolor="#FFFFFF")
            st.pyplot(fig_roc)
            plt.close(fig_roc)

# =========================================================
# TAB 5: METHODOLOGY
# =========================================================

with tab_doc:
    st.markdown("## Problem Statement & Technical Methodology")
    st.markdown("""
    ### 1. Synthetic Dataset Generation
    The dataset `data/model_data.csv` models 24,455 transactions across 500 unique bank customers:
    - Normal Baseline (60%): Derived from standard behavioral dynamics (dwell time ~150ms, flight time ~400ms), standard transfer amounts, recognized devices (33%), and recognized locations (90%).
    - Attack Scenarios (40%): Models phished OTPs, SIM-swaps, and account takeovers via unknown devices (79%), unknown locations (79%), large amount deviations (mean +1.43x), and account reactivation gaps.

    ### 2. USSD & Feature Phone Strategy (*919#)
    For feature phones where keystroke biometrics are unavailable:
    - Uses a dedicated USSD model trained on channel-agnostic features: `device_known`, `location_known`, `hour_of_day`, `amount`, `amount_deviation`, `new_recipient`, `time_since_previous_transaction`, `transaction_frequency`.
    - Achieves 98.83% Accuracy and 1.06% False Alarm Rate.

    ### 3. Sub-Second Latency
    Evaluates transfer risk in under 10 milliseconds, operating synchronously within the live transfer authorization workflow.

    ### 4. Plain English Customer Explanations
    Generates non-technical explanations for customer support when transfers are held.
    """)
