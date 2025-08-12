# scripts/gen_data.py
import numpy as np
import pandas as pd
import os

os.makedirs("../data", exist_ok=True)

def create_synthetic_data(num_samples=1_000_000, seed=42):
    np.random.seed(seed)
    age = np.random.randint(18, 90, size=num_samples)
    gender = np.random.choice([0, 1], size=num_samples)  # 0=Male, 1=Female
    booking_lead = np.random.randint(0, 60, size=num_samples)  # lead time in days (bigger range)
    previous_no_shows = np.random.poisson(0.5, size=num_samples)  # skewed low
    sms_reminder = np.random.choice([0, 1], size=num_samples, p=[0.25, 0.75])  # most get SMS
    chronic = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    distance = np.round(np.random.exponential(scale=5.0, size=num_samples) + 0.5, 2)  # many close, some far

    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'BookingLeadTime': booking_lead,
        'PreviousNoShows': np.clip(previous_no_shows, 0, 10),
        'SMSReminderSent': sms_reminder,
        'ChronicConditions': chronic,
        'DistanceToClinic': distance
    })

    # Create a target using a probabilistic rule (more realistic than a strict rule)
    # Base risk
    base_prob = 0.05 + (df['PreviousNoShows'] * 0.15) + (df['BookingLeadTime'] / 100) \
                + (df['SMSReminderSent'] == 0)*0.2 + (df['DistanceToClinic'] > 20)*0.08
    # Age effect (very small)
    base_prob += (df['Age'] < 25) * 0.02
    # Clip probabilities and sample
    base_prob = base_prob.clip(0, 0.95)
    df['NoShow'] = (np.random.rand(len(df)) < base_prob).astype(int)

    return df

if __name__ == "__main__":
    print("Generating 1,000,000 synthetic rows (this may take a few seconds)...")
    df = create_synthetic_data(1_000_000)
    path = "../data/appointments_1m.csv"
    print(f"Saving to {path} (this will take some disk space)...")
    df.to_csv(path, index=False)
    print("Done.")

