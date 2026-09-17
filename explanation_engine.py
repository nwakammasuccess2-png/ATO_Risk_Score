def generate_customer_explanation(
    amount=15000,
    amount_deviation=0.0,
    device_known=True,
    location_known=True,
    time_since_prev=100.0,
    new_recipient=False,
    is_ussd=False
):
    """
    Generates a clear, non-technical plain English sentence explaining why a transfer
    was stopped, suitable for a bank customer asking why their transfer was held.
    """
    reasons = []

    # 1. Amount deviation
    if amount_deviation >= 1.5:
        multiplier = amount_deviation + 1.0
        reasons.append(f"is {multiplier:.1f}x higher than your usual transfer amount")
    elif amount >= 75000:
        reasons.append(f"is an unusually large transfer (₦{amount:,.2f})")

    # 2. Location
    if not location_known:
        reasons.append("originated from an unrecognized location")

    # 3. Device (if mobile app)
    if not device_known and not is_ussd:
        reasons.append("was initiated from a new/unregistered device")

    # 4. Long gap (inactivity / reactivation)
    if time_since_prev >= 100000:
        hours = time_since_prev / 3600.0
        reasons.append(f"followed a long period of account inactivity ({hours:.0f} hours)")

    # 5. New recipient
    if new_recipient:
        reasons.append("was directed to a new, unverified recipient")

    # Fallback reason
    if not reasons:
        reasons.append("exhibited behavioral patterns that differ significantly from your account history")

    # Format into a customer-friendly plain sentence
    if len(reasons) == 1:
        reasons_text = reasons[0]
    elif len(reasons) == 2:
        reasons_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reasons_text = f"{', '.join(reasons[:-1])}, and {reasons[-1]}"

    channel_name = "USSD (*919#)" if is_ussd else "Banking App"

    explanation = (
        f"Transfer Stopped: Your {channel_name} transfer of ₦{amount:,.2f} was paused for your security "
        f"because it {reasons_text}."
    )

    return explanation


if __name__ == "__main__":
    # Test sample explanation
    sentence = generate_customer_explanation(
        amount=120000,
        amount_deviation=3.2,
        device_known=False,
        location_known=False,
        new_recipient=True
    )
    print("Sample Plain English Customer Explanation:")
    print(sentence)
