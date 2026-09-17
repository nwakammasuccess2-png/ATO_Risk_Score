import os
import sys
import pandas as pd
import joblib
from behaviour_analysis import analyze_behavior
from explanation_engine import generate_customer_explanation

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = "model/ato_risk_model.pkl"
DATA_PATH = "data/model_data.csv"

if not os.path.exists(MODEL_PATH):
    print("Model file not found! Please run train_final_model.py first.")
    sys.exit(1)

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}!")
    sys.exit(1)

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"] if isinstance(saved_data, dict) else saved_data

FEATURES = [
    "typing_speed", "mean_dwell_time", "mean_flight_time", "typing_variability",
    "device_known", "location_known", "hour_of_day", "paste_used",
    "screens_visited", "session_duration", "amount", "amount_deviation",
    "new_recipient", "time_since_previous_transaction", "transaction_frequency"
]

df = pd.read_csv(DATA_PATH)

legit_sample = df[df["transaction_label"] == "legitimate"].sample(10, random_state=42)
attack_sample = df[df["transaction_label"] == "attack"].sample(10, random_state=42)
sample_df = pd.concat([legit_sample, attack_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)

for col in ["device_known", "location_known", "paste_used", "new_recipient"]:
    sample_df[col] = sample_df[col].astype(int)

X_sample = sample_df[FEATURES]
predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)[:, 1]

sample_df["predicted_label"] = ["Attack" if p == 1 else "Legitimate" for p in predictions]
sample_df["attack_probability"] = probabilities

print("=" * 105)
print("                           REAL-TIME SESSION TAKEOVER SIMULATION                        ")
print("=" * 105)

correct_count = 0
false_alarms = 0

for i, row in sample_df.iterrows():
    actual = row["transaction_label"].capitalize()
    predicted = row["predicted_label"]
    prob = row["attack_probability"]
    
    is_correct = (actual == predicted)
    if is_correct:
        correct_count += 1
    if actual == "Legitimate" and predicted == "Attack":
        false_alarms += 1

    status_icon = "[ALERT] ATTACK DETECTED" if predicted == "Attack" else "[OK] LEGITIMATE"
    match_icon = "✓ MATCH" if is_correct else "X MISMATCH"

    explanation = generate_customer_explanation(
        amount=row["amount"],
        amount_deviation=row["amount_deviation"],
        device_known=bool(row["device_known"]),
        location_known=bool(row["location_known"]),
        time_since_prev=row["time_since_previous_transaction"],
        new_recipient=bool(row["new_recipient"])
    )

    print(f"\n[Session #{i+1:02d}] User: {row['user_id']} | SessionID: {row['session_id']}")
    print(f"  • Ground Truth Label : {actual}")
    print(f"  • Model Prediction   : {status_icon} ({prob:.2%}) [{match_icon}]")
    print(f"  • Key Telemetry      : Device Known={bool(row['device_known'])}, Location Known={bool(row['location_known'])}, "
          f"Amount=₦{row['amount']:,}, Dev={row['amount_deviation']:.2f}x, Gap={row['time_since_previous_transaction']:.0f}s")
    if predicted == "Attack":
        print(f"  • Plain Explanation  : \"{explanation}\"")

print("\n" + "=" * 105)
print(f"SIMULATION SUMMARY: Evaluated {len(sample_df)} real-time sessions.")
print(f"  • Accuracy     : {correct_count / len(sample_df):.2%}")
print(f"  • False Alarms : {false_alarms} / {len(legit_sample)} legitimate sessions ({false_alarms/len(legit_sample):.2%})")
print("=" * 105)