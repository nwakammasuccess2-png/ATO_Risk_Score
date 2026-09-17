def analyze_behavior(
    typing_speed=0.35,
    mean_dwell_time=0.15,
    mean_flight_time=0.40,
    typing_variability=0.35,
    device_known=True,
    location_known=True,
    hour_of_day=14,
    paste_used=False,
    screens_visited=6,
    session_duration=180.0,
    amount=15000,
    amount_deviation=0.0,
    new_recipient=False,
    time_since_previous_transaction=100.0,
    transaction_frequency=0.01
):
    """
    Analyze session parameters against behavioral heuristic indicators.
    Returns a list of plain text warnings for suspicious anomalies.
    """
    indicators = []

    # 1. Location Anomaly
    if not location_known:
        indicators.append("Access from an unknown / unverified location")

    # 2. Device Anomaly
    if not device_known:
        indicators.append("Session initiated from an unrecognized device")

    # 3. Transaction Amount Deviation
    if amount_deviation >= 1.0:
        indicators.append(f"High transaction amount deviation (+{amount_deviation:.1f}x baseline)")

    # 4. Large Absolute Transaction Amount
    if amount >= 50000:
        indicators.append(f"Unusually large transaction amount (NGN {amount:,.2f})")

    # 5. Long Time Since Previous Transaction (Account Reactivation Anomaly)
    if time_since_previous_transaction >= 100000:
        hours = time_since_previous_transaction / 3600.0
        indicators.append(f"Long inactivity period prior to session ({hours:.1f} hours)")

    # 6. High Transaction Frequency / Velocity
    if transaction_frequency >= 0.5:
        indicators.append(f"High transaction velocity ({transaction_frequency:.2f} tx/min)")

    # 7. New Recipient
    if new_recipient:
        indicators.append("Funds transfer initiated to a new recipient")

    # 8. Paste Detection
    if paste_used:
        indicators.append("Sensitive input was pasted (clipboard usage)")

    # 9. Unusual Hour of Day
    if hour_of_day <= 5 or hour_of_day >= 23:
        indicators.append(f"Session active during unusual hours ({hour_of_day:02d}:00)")

    # 10. Typing Variability Anomaly
    if typing_variability >= 0.65:
        indicators.append(f"High typing variability anomaly ({typing_variability:.2f})")

    # 11. Typing Speed Anomaly
    if typing_speed < 0.20 or typing_speed > 0.70:
        indicators.append(f"Abnormal typing speed ({typing_speed:.2f} chars/sec)")

    return indicators