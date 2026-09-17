import os
import pandas as pd

LOG_FILE = "data/ato_activity_log.csv"

def load_risk_history():
    """Load logged suspicious events."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    return df

def get_statistics(df):
    """Return summary metrics for security dashboard."""
    if df.empty:
        return {
            "total": 0,
            "unknown_device": 0,
            "unknown_location": 0,
            "critical_events": 0,
        }

    unknown_device_count = (df["device_known"] == "Unknown").sum() if "device_known" in df.columns else 0
    unknown_location_count = (df["location_known"] == "Unknown").sum() if "location_known" in df.columns else 0
    critical_events_count = (df["risk_level"] == "CRITICAL").sum() if "risk_level" in df.columns else 0

    return {
        "total": len(df),
        "unknown_device": unknown_device_count,
        "unknown_location": unknown_location_count,
        "critical_events": critical_events_count,
    }

if __name__ == "__main__":
    history = load_risk_history()
    print("=" * 60)
    print("                ATO SECURITY EVENT HISTORY              ")
    print("=" * 60)

    if history.empty:
        print("No suspicious events logged yet.")
    else:
        print(history.to_string(index=False))
        stats = get_statistics(history)
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total Logged Events     : {stats['total']}")
        print(f"Unknown Device Events   : {stats['unknown_device']}")
        print(f"Unknown Location Events : {stats['unknown_location']}")
        print(f"Critical Risk Events    : {stats['critical_events']}")