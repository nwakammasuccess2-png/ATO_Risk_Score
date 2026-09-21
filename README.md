# 🛡️ Account Takeover (ATO) Risk Scoring Engine
### Real-Time Behavioral Anomaly Detection & Fraud Prevention System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-HistGradientBoostingClassifier-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Proprietary-black.svg)]()

---

## 📌 Overview

The **Account Takeover (ATO) Risk Scoring Engine** is an enterprise-grade, real-time fraud detection and decisioning system designed to protect banking operations against credential stuffing, phishing-driven session hijacking, SIM-swap attacks, and unauthorized fund transfers.

By fusing **keystroke behavioral biometrics**, **device/location context**, and **transactional velocity indicators**, the system assigns an instant risk probability score to each transfer session with **sub-second inference latency (< 10 ms)**.

---

## ✨ Key Features & Capabilities

- **Dual-Channel Architecture**:
  - **Mobile & Web Banking**: Full 15-feature model combining keystroke dynamics (dwell time, flight time, typing speed, variability), session telemetry, and financial velocity.
  - **USSD & Feature Phones (`*919#`)**: Dedicated 8-feature channel-agnostic model tailored for low-resource cellular channels where biometric keystrokes are unavailable.
- **Explainable AI (Plain-English Explanations)**:
  - Generates clear, non-technical customer security notices (e.g., *"Transfer Stopped: Your transfer was paused because it is 4.5x higher than your usual baseline and originated from an unrecognized location"*).
- **Sub-Second Real-Time Scoring**:
  - Inferences execute in **< 10 milliseconds**, allowing synchronous evaluation inline with core banking transfer workflows.
- **Adaptive Risk Tiering Policy**:
  - **Low Risk (`< 0.20`)**: Immediate seamless execution.
  - **Medium Risk (`0.20 - 0.50`)**: Standard transaction monitoring.
  - **High Risk (`0.50 - 0.80`)**: Step-up verification / Step-down hold.
  - **Critical Risk (`≥ 0.80`)**: Instant transfer block + automated incident logging.
- **Auditing & Incident Logging**:
  - Automatically records suspicious sessions to `data/ato_activity_log.csv` for fraud analyst investigation and audit compliance.
- **Interactive Monochrome Command Center**:
  - Streamlit-powered dashboard featuring customer transfer simulation, automated attack scenarios, live confusion matrix & ROC curves, and exportable event audit logs.

---

## 🏗️ System Architecture & Workflow

```
[ Customer Interaction ]
       │
       ├──► Mobile App / Web Portal ──► [ Keystroke Biometrics + Context + Velocity ] ──► Mobile ATO Model (15 Features)
       │                                                                                             │
       └──► USSD Channel (*919#)    ──► [ Device + Location + Velocity ]              ──► USSD ATO Model (8 Features)
                                                                                                     │
                                                                                                     ▼
                                                                                       [ Gradient Boosted Inference ]
                                                                                                     │
                                                                                      ┌──────────────┴──────────────┐
                                                                                      ▼                             ▼
                                                                             [ Risk Score < 0.50 ]        [ Risk Score ≥ 0.50 ]
                                                                                      │                             │
                                                                                      ▼                             ▼
                                                                              ✅ Transfer Approved         ⛔ Transfer Blocked
                                                                                                            ├─ Generate Plain Customer Notice
                                                                                                            └─ Log Incident to Audit Trail
```

---

## 📊 Feature Specifications

### 1. Mobile & Web Banking Features (15 Attributes)

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Keystroke Dynamics** | `typing_speed` | Typing velocity in characters/second |
| | `mean_dwell_time` | Mean key press duration (seconds) |
| | `mean_flight_time` | Mean interval between consecutive keystrokes (seconds) |
| | `typing_variability` | Coefficient of variation in keystroke rhythm |
| **Context & Environment** | `device_known` | Boolean flag (1 = Registered device, 0 = Unrecognized device) |
| | `location_known` | Boolean flag (1 = Familiar location/IP, 0 = Anomalous location) |
| | `hour_of_day` | Hour of transaction initiation (`0` to `23`) |
| | `paste_used` | Clipboard paste indicator (1 = Pasted credentials/amount, 0 = Typed) |
| | `screens_visited` | Number of views/screens navigated during the session |
| | `session_duration` | Total time spent in session (seconds) |
| **Velocity & Financials** | `amount` | Transfer amount in NGN |
| | `amount_deviation` | Ratio of change relative to customer's baseline average |
| | `new_recipient` | Flag indicating transfer to a beneficiary with no prior history |
| | `time_since_previous_transaction` | Inactivity interval since previous account activity (seconds) |
| | `transaction_frequency` | Velocity metric measuring transactions per unit time |

### 2. USSD / Feature Phone Features (8 Attributes)

- `device_known`, `location_known`, `hour_of_day`, `amount`, `amount_deviation`, `new_recipient`, `time_since_previous_transaction`, `transaction_frequency`

---

## 📁 Project Directory Structure

```plaintext
ATO_Risk_score/
│
├── data/
│   ├── model_data.csv              # Synthetic banking dataset (500 users, 24,455 transactions)
│   ├── ato_activity_log.csv        # Real-time incident logs and blocked sessions
│   ├── normal_sessions.csv         # Baseline synthetic normal sessions
│   └── ato_sessions.csv            # Synthetic ATO attack profile sessions
│
├── model/
│   ├── ato_risk_model.pkl          # Serialized HistGradientBoosting model (Full 15 features)
│   └── ussd_risk_model.pkl         # Serialized USSD model (8 channel-agnostic features)
│
├── app.py                          # Interactive Streamlit dashboard & transaction portal
├── train_final_model.py            # Primary model training pipeline with stratified split
├── train_ussd_model.py             # Dedicated USSD / Feature phone model training pipeline
├── train_model.py                  # Baseline model training script
│
├── predict.py                      # CLI inference utility for single session scoring
├── simulate_session.py             # Batch session simulator comparing model vs ground truth
├── behaviour_analysis.py           # Behavioral rule-based anomaly detector & heuristics
├── explanation_engine.py           # Customer-facing plain English explanation generator
├── ato_activity_log.py             # Audit trail recording and CSV logger utility
├── risk_history.py                 # Incident statistics parser for security operations
│
├── evaluate_model.py               # Full dataset model evaluation metrics
├── evaluate_split_model.py         # Stratified test set performance evaluation
├── model_metrics.py                # Train/test split metrics script
│
├── generate_data.py                # Normal banking session synthesizer
├── generate_ato_data.py            # ATO attack behavior synthesizer
│
└── README.md                       # System documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python **3.9+** recommended
- `pip` package manager

### 2. Clone Repository & Create Virtual Environment

```bash
# Navigate to the workspace
cd ATO_Risk_score

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install streamlit scikit-learn pandas numpy joblib matplotlib
```

---

## 🚀 How to Run the Project

### Step 1: Train the Machine Learning Models

Train the primary gradient boosting model and the USSD-specific model:

```bash
# 1. Train the primary model (Mobile & Web Banking)
python train_final_model.py

# 2. Train the USSD / Feature Phone model
python train_ussd_model.py
```

*The models will be saved automatically to `model/ato_risk_model.pkl` and `model/ussd_risk_model.pkl`.*

---

### Step 2: Evaluate Model Performance

Run evaluation scripts to inspect test-set accuracy, precision, recall, ROC-AUC, and false alarm rates:

```bash
# Stratified test-set evaluation
python evaluate_split_model.py

# Full dataset performance metrics
python evaluate_model.py
```

---

### Step 3: Run Command-Line Predictions & Simulations

Test inference on individual sessions or run a batch attack simulation:

```bash
# Single session prediction utility
python predict.py

# Real-time multi-session simulation against attack vectors
python simulate_session.py
```

---

### Step 4: Launch the Interactive Web Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Once launched, open your browser and navigate to the local URL (typically **`http://localhost:8501`**).

---

## 🖥️ Dashboard Walkthrough

The web dashboard is organized into 5 dedicated modules:

1. **Demo Banking App (`Customer Transfer Portal`)**:
   - Choose customer accounts (e.g. `U0001` - `U0500`).
   - Switch between **Mobile Banking App** and **USSD (*919#)** channels.
   - Adjust transaction amount, recipient status, device/location parameters, and keystroke speed.
   - Execute live transfer to observe instant approval or fraud hold with sub-second latency tags.
2. **Real-Time Scenarios (`Automated Attack Testing`)**:
   - Quick one-click execution of **Legitimate Transfers**, **Phished OTP Takeovers**, **USSD SIM-Swap Drains**, and **Batch CSV Testing**.
3. **Activity Log (`Incident Registry`)**:
   - Live view of all blocked ATO sessions.
   - Filter and download audit logs as CSV.
4. **Model Metrics (`Empirical Performance`)**:
   - Confusion Matrix and ROC Curve visualization.
   - Performance metrics: Accuracy, Recall (Catch Rate), Precision, False Alarm Rate (FPR), and F1 Score.
5. **Methodology (`Technical Documentation`)**:
   - Complete synthetic data generation methodology integrating the **KeyRecs keystroke dataset**.

---

## 🛡️ Security Decision Policy

| Risk Score Range | Threat Classification | Automated Decision | System Action |
| :---: | :---: | :---: | :--- |
| **0.00 – 0.19** | `LOW` | **APPROVE** | Transaction processed immediately |
| **0.20 – 0.49** | `MEDIUM` | **APPROVE** | Flagged for background behavioral monitoring |
| **0.50 – 0.79** | `HIGH` | **HOLD / BLOCK** | Step-up OTP or Customer verification required |
| **0.80 – 1.00** | `CRITICAL` | **BLOCK & ALERT** | Instant transfer hold, user alert, and audit logged |

---

## 📄 License & Attribution
Developed for real-time account takeover risk mitigation in modern digital banking environments.
