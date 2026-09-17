import pandas as pd
import numpy as np

np.random.seed(42)

number_of_sessions = 5000

data = {
    "transaction_amount": np.clip(
        np.random.normal(15000, 5000, number_of_sessions),
        500,
        None
    ),

    "transaction_hour": np.clip(
        np.random.normal(13, 3, number_of_sessions),
        0,
        23
    ).astype(int),

    "new_device": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.97, 0.03]
    ),

    "location_changed": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.96, 0.04]
    ),

    "failed_attempts": np.random.poisson(
        0.1,
        number_of_sessions
    ),

    "transaction_frequency": np.random.poisson(
        3,
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
        p=[0.98, 0.02]
    ),

    "password_changed": np.random.choice(
        [0, 1],
        number_of_sessions,
        p=[0.995, 0.005]
    )
}

df = pd.DataFrame(data)

df.to_csv(
    "data/normal_sessions.csv",
    index=False
)

print("Normal banking behavior dataset generated successfully!")
print()
print("Number of sessions:", len(df))
print()
print("Features:")
print(df.columns.tolist())
print()
print(df.head())