import os
import sys
import joblib
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = "model/ato_risk_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model file not found! Please run train_final_model.py first.")
    sys.exit(1)

saved_data = joblib.load(MODEL_PATH)

if isinstance(saved_data, dict):
    model = saved_data["model"]
    features = saved_data["features"]
else:
    model = saved_data
    features = [
        "typing_speed", "mean_dwell_time", "mean_flight_time", "typing_variability",
        "device_known", "location_known", "hour_of_day", "paste_used",
        "screens_visited", "session_duration", "amount", "amount_deviation",
        "new_recipient", "time_since_previous_transaction", "transaction_frequency"
    ]

print("=" * 50)
print("     ATO RISK SCORING MODEL - PREDICTION UTILITY    ")
print("=" * 50)

# Sample suspicious session (ATO attack profile)
sample_session = pd.DataFrame([{
    "typing_speed": 0.38,
    "mean_dwell_time": 0.16,
    "mean_flight_time": 0.44,
    "typing_variability": 0.48,
    "device_known": 0,           # Unknown device
    "location_known": 0,         # Unknown location
    "hour_of_day": 3,            # 3 AM
    "paste_used": 1,             # Paste used
    "screens_visited": 6,
    "session_duration": 190.0,
    "amount": 45000,
    "amount_deviation": 1.45,    # High deviation
    "new_recipient": 1,          # New recipient
    "time_since_previous_transaction": 450000.0, # Long gap
    "transaction_frequency": 1.5
}])

# Get prediction and probability
prediction = model.predict(sample_session[features])[0]
probabilities = model.predict_proba(sample_session[features])[0]
attack_prob = probabilities[1]

print("\nSample Input Features:")
for col, val in sample_session.iloc[0].items():
    print(f"  • {col:<35}: {val}")

print("\n" + "=" * 50)
print("RESULT:")
print(f"Attack Probability Score : {attack_prob:.4f} ({attack_prob:.2%})")

if attack_prob >= 0.80:
    risk_level = "CRITICAL"
elif attack_prob >= 0.50:
    risk_level = "HIGH"
elif attack_prob >= 0.20:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print(f"Assigned Risk Level      : {risk_level}")

if prediction == 1:
    print("Classification Result    : [ALERT] SUSPICIOUS / ACCOUNT TAKEOVER DETECTED")
else:
    print("Classification Result    : [OK] LEGITIMATE SESSION")
print("=" * 50)