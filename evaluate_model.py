import os
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

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

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

bool_cols = ["device_known", "location_known", "paste_used", "new_recipient"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df[FEATURES]
y = (df["transaction_label"] == "attack").astype(int)

saved_data = joblib.load(MODEL_PATH)
if isinstance(saved_data, dict):
    model = saved_data["model"]
else:
    model = saved_data

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print("=" * 50)
print("             FULL DATASET MODEL EVALUATION          ")
print("=" * 50)
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print(f"False Positive Rate: {fpr:.4f} ({fpr:.2%})")
print("=" * 50)

print("\nCONFUSION MATRIX:")
print(cm)

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y,
        y_pred,
        target_names=["Legitimate", "Attack"],
        zero_division=0
    )
)