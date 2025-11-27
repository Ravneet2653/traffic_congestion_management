import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import random
import os

def generate_traffic_data(num_days=90):
    """
    Generate realistic synthetic traffic data for 90 days
    """
    print("🚗 Starting data generation...")
    data = []
    start_date = datetime(2024, 1, 1)
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5
        
        for hour in range(24):
            for minute in range(0, 60, 2):  # Every 2 minutes
                timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                
                # Time-based traffic patterns
                if is_weekend:
                    if 10 <= hour <= 12:
                        base_traffic = random.randint(30, 50)
                    elif 18 <= hour <= 21:
                        base_traffic = random.randint(35, 55)
                    else:
                        base_traffic = random.randint(10, 30)
                else:
                    if 7 <= hour <= 9:  # Morning rush
                        base_traffic = random.randint(60, 100)
                    elif 12 <= hour <= 14:  # Lunch
                        base_traffic = random.randint(40, 70)
                    elif 17 <= hour <= 20:  # Evening rush
                        base_traffic = random.randint(70, 110)
                    elif 22 <= hour or hour <= 5:  # Night
                        base_traffic = random.randint(5, 20)
                    else:
                        base_traffic = random.randint(25, 45)
                
                noise = random.randint(-10, 10)
                
                north = max(0, base_traffic + noise + random.randint(-8, 8))
                south = max(0, base_traffic + noise + random.randint(-8, 8))
                east = max(0, base_traffic + noise + random.randint(-8, 8))
                west = max(0, base_traffic + noise + random.randint(-8, 8))
                
                data.append({
                    'timestamp': timestamp,
                    'hour': hour,
                    'minute': minute,
                    'day_of_week': current_date.weekday(),
                    'is_weekend': int(is_weekend),
                    'north_count': north,
                    'south_count': south,
                    'east_count': east,
                    'west_count': west,
                })
        
        if (day + 1) % 10 == 0:
            print(f"  Generated {day + 1}/{num_days} days...")
    
    df = pd.DataFrame(data)
    df['total_vehicles'] = df['north_count'] + df['south_count'] + df['east_count'] + df['west_count']
    df['ns_total'] = df['north_count'] + df['south_count']
    df['ew_total'] = df['east_count'] + df['west_count']
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_traffic_data(90)
    df.to_csv('data/traffic_data.csv', index=False)
    print(f"\n✅ Generated {len(df):,} records")
    print(f"📁 Saved to: data/traffic_data.csv")
    print(f"\n📊 Sample statistics:")
    print(f"  Average vehicles: {df['total_vehicles'].mean():.1f}")
    print(f"  Max vehicles: {df['total_vehicles'].max()}")
    print(f"  Min vehicles: {df['total_vehicles'].min()}")