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

# Features available on USSD / Feature Phone transactions (excluding app keystrokes)
USSD_FEATURES = [
    "device_known",
    "location_known",
    "hour_of_day",
    "amount",
    "amount_deviation",
    "new_recipient",
    "time_since_previous_transaction",
    "transaction_frequency"
]

DATA_PATH = "data/model_data.csv"
MODEL_PATH = "model/ussd_risk_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded for USSD model training!")

bool_cols = ["device_known", "location_known", "new_recipient"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df[USSD_FEATURES]
y = (df["transaction_label"] == "attack").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"USSD Training sessions: {len(X_train)}")
print(f"USSD Testing sessions : {len(X_test)}")

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

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

print("\n" + "=" * 50)
print("       USSD / FEATURE PHONE MODEL PERFORMANCE       ")
print("=" * 50)
print(f"Accuracy           : {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision          : {precision:.4f} ({precision:.2%})")
print(f"Recall             : {recall:.4f} ({recall:.2%})")
print(f"F1 Score           : {f1:.4f}")
print(f"ROC-AUC            : {auc:.4f}")
print(f"False Positive Rate: {fpr:.4f} ({fpr:.2%})")
print("=" * 50)

print("\nCONFUSION MATRIX:")
print(cm)

os.makedirs("model", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "features": USSD_FEATURES
    },
    MODEL_PATH
)

print(f"\nUSSD model saved successfully at: {MODEL_PATH}")
