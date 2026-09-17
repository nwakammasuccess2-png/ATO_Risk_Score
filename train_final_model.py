import os
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==============================
# 1. FEATURES DEFINITION
# ==============================

FEATURES = [
    "typing_speed",
    "mean_dwell_time",
    "mean_flight_time",
    "typing_variability",
    "device_known",
    "location_known",
    "hour_of_day",
    "paste_used",
    "screens_visited",
    "session_duration",
    "amount",
    "amount_deviation",
    "new_recipient",
    "time_since_previous_transaction",
    "transaction_frequency"
]

DATA_PATH = "data/model_data.csv"
MODEL_PATH = "model/ato_risk_model.pkl"

# ==============================
# 2. LOAD DATA
# ==============================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print(f"Total sessions: {len(df)}")
print("Class breakdown:\n", df["transaction_label"].value_counts())

# Preprocess boolean columns
bool_cols = ["device_known", "location_known", "paste_used", "new_recipient"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df[FEATURES]
y = (df["transaction_label"] == "attack").astype(int)

# ==============================
# 3. SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining sessions: {len(X_train)}")
print(f"Testing sessions : {len(X_test)}")

# ==============================
# 4. TRAIN MODEL
# ==============================

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nFinal Gradient Boosting Model trained successfully!")

# ==============================
# 5. MAKE PREDICTIONS & EVALUATE
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# ==============================
# 6. DISPLAY PERFORMANCE RESULTS
# ==============================

print()
print("=" * 50)
print("       FINAL MODEL EVALUATION PERFORMANCE       ")
print("=" * 50)
print(f"Accuracy           : {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision          : {precision:.4f} ({precision:.2%})")
print(f"Recall             : {recall:.4f} ({recall:.2%})")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print(f"False Positive Rate: {fpr:.4f} ({fpr:.2%})")
print("=" * 50)

print("\nCONFUSION MATRIX:")
print(f"True Negatives (Legitimate Correct) : {tn}")
print(f"False Positives (False Alarms)      : {fp}")
print(f"False Negatives (Missed Attacks)    : {fn}")
print(f"True Positives (Attacks Caught)     : {tp}")

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Attack"],
        zero_division=0
    )
)

# ==============================
# 7. SAVE MODEL & METADATA
# ==============================

os.makedirs("model", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)

print("\nFinal model package saved successfully!")
print(f"Location: {MODEL_PATH}")