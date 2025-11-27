import pandas as pd
import numpy as np
import os

def preprocess_data(df):
    """Clean and engineer features"""
    print("🔧 Preprocessing data...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values (forward fill)
    df = df.fillna(method='ffill')

    # Add combined features
    df['ns_total'] = df['north_count'] + df['south_count']
    df['ew_total'] = df['east_count'] + df['west_count']
    df['total_vehicles'] = df['ns_total'] + df['ew_total']

    # Ratios
    df['ns_ratio'] = df['ns_total'] / (df['total_vehicles'] + 1)
    df['ew_ratio'] = df['ew_total'] / (df['total_vehicles'] + 1)

    # Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Rolling features
    df['total_rolling_mean'] = df['total_vehicles'].rolling(window=6, min_periods=1).mean()
    df['total_rolling_std'] = df['total_vehicles'].rolling(window=6, min_periods=1).std()

    # Density labels
    df['density_label'] = pd.cut(df['total_vehicles'],
                                 bins=[0, 60, 120, float('inf')],
                                 labels=['Low', 'Medium', 'High'])

    # Optimal green times (target)
    df['optimal_ns_green'] = 15 + (df['ns_ratio'] * 45)
    df['optimal_ew_green'] = 15 + (df['ew_ratio'] * 45)

    print(f"✅ Preprocessed {len(df)} records")
    return df


if __name__ == "__main__":
    # Load data (update path if needed)
    input_path = os.path.join("data", "raw", "traffic_data.csv")
    output_path = os.path.join("data", "processed_data.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df_processed = preprocess_data(df)
    df_processed.to_csv(output_path, index=False)
    print(f"📁 Saved to: {output_path}")
