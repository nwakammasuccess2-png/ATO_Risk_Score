import pandas as pd
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier

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
# 2. LOAD & PREPARE DATA
# ==============================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully!")
print("Number of sessions:", len(df))

# Convert boolean columns to integer
bool_cols = ["device_known", "location_known", "paste_used", "new_recipient"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df[FEATURES]
y = (df["transaction_label"] == "attack").astype(int)

# ==============================
# 3. TRAIN CLASSIFIER
# ==============================

print("\nTraining HistGradientBoostingClassifier model...")
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X, y)

print("Model trained successfully!")

# ==============================
# 4. SAVE MODEL & METADATA
# ==============================

os.makedirs("model", exist_ok=True)
joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)

print(f"\nModel saved successfully at: {MODEL_PATH}")