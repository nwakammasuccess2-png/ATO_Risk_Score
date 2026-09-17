import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

saved_data = joblib.load(MODEL_PATH)
if isinstance(saved_data, dict):
    model = saved_data["model"]
else:
    model = saved_data

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print("=" * 50)
print("             MODEL TEST SET METRICS               ")
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