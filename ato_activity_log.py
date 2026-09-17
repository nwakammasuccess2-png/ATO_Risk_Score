import os
import pandas as pd
from datetime import datetime

LOG_FILE = "data/ato_activity_log.csv"

def create_log_file():
    """Create the log file with updated schema if it does not exist."""
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        columns = [
            "timestamp",
            "user_id",
            "status",
            "risk_score",
            "risk_level",
            "device_known",
            "location_known",
            "amount",
            "amount_deviation",
            "time_since_previous_transaction",
            "transaction_frequency"
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

def log_session(
    user_id,
    status,
    risk_score,
    risk_level,
    device_known,
    location_known,
    amount,
    amount_deviation,
    time_since_previous_transaction,
    transaction_frequency
):
    """Save a session alert to the activity log."""
    create_log_file()
    
    new_record = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "status": status,
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "device_known": "Known" if device_known else "Unknown",
        "location_known": "Known" if location_known else "Unknown",
        "amount": round(float(amount), 2),
        "amount_deviation": round(float(amount_deviation), 2),
        "time_since_previous_transaction": round(float(time_since_previous_transaction), 2),
        "transaction_frequency": round(float(transaction_frequency), 4)
    }])
    
    new_record.to_csv(LOG_FILE, mode="a", header=False, index=False)

if __name__ == "__main__":
    log_session(
        user_id="U0001",
        status="ATO / Suspicious",
        risk_score=0.9421,
        risk_level="CRITICAL",
        device_known=False,
        location_known=False,
        amount=85000,
        amount_deviation=2.4,
        time_since_previous_transaction=350000,
        transaction_frequency=1.2
    )
    print(f"Sample ATO activity logged successfully to: {LOG_FILE}")