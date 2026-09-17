import pandas as pd
import numpy as np

np.random.seed(123)

number_of_sessions = 1000

data = {
    "transaction_amount": np.clip(
        np.random.normal(50000, 20000, number_of_sessions),
        5000,
        None
    ),

    "transaction_hour": np.random.choice(
        [0, 1, 2, 3, 4, 5, 22, 23],
        number_of_sessions
    ),

    "new_device": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.20, 0.80]
    ),

    "location_changed": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.15, 0.85]
    ),

    "failed_attempts": np.random.poisson(
        3,
        number_of_sessions
    ),

    "transaction_frequency": np.random.poisson(
        8,
        number_of_sessions
    ),

    "account_age_days": np.random.randint(
        90,
        2500,
        number_of_sessions
    ),

    "fingerprint_changed": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.10, 0.90]
    ),

    "password_changed": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.30, 0.70]
    )
}

df = pd.DataFrame(data)

df.to_csv(
    "data/ato_sessions.csv",
    index=False
)

print("ATO behavior dataset generated successfully!")
print()
print("Number of ATO sessions:", len(df))
print()
print("Features:")
print(df.columns.tolist())
print()
print(df.head())
