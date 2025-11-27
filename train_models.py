import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os


def clean_data(df):
    """Remove or fill missing values"""
    print("🧹 Cleaning data...")
    df = df.dropna(subset=['density_label', 'total_vehicles'], how='any')
    df = df.fillna(0)
    print("✅ Missing values handled.")
    return df


def train_classifier(df):
    """Train density classification model"""
    print("\n🎯 Training Classifier...")

    df = clean_data(df)

    features = ['hour', 'day_of_week', 'is_weekend', 'ns_total', 'ew_total',
                'ns_ratio', 'ew_ratio', 'hour_sin', 'hour_cos']

    X = df[features]
    y = df['density_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/classifier.pkl')
    print("✅ Classifier saved")

    return clf


def train_clustering(df):
    """Train clustering model for traffic pattern discovery"""
    print("\n🔍 Training Clustering...")

    df = clean_data(df)

    features = ['hour', 'day_of_week', 'total_vehicles', 'ns_ratio', 'ew_ratio']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['traffic_pattern'] = kmeans.fit_predict(X_scaled)

    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/clustering.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Clustering model saved")

    return df, kmeans


def train_regressor(df):
    """Train regression model for green signal time prediction"""
    print("\n📈 Training Regressor...")

    df = clean_data(df)

    # Ensure clustering was done
    if 'traffic_pattern' not in df.columns:
        df['traffic_pattern'] = 0

    features = ['hour', 'day_of_week', 'is_weekend', 'ns_total', 'ew_total',
                'ns_ratio', 'ew_ratio', 'traffic_pattern', 'hour_sin', 'hour_cos']

    X = df[features]
    y = df['optimal_ns_green']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📈 R² Score: {r2:.3f}")

    joblib.dump(reg, 'models/regressor.pkl')
    print("✅ Regressor saved")

    return reg


if __name__ == "__main__":
    print("🚦 Starting model training pipeline...")
    df = pd.read_csv('data/processed_data.csv')

    print(f"📄 Loaded {len(df)} records")

    # Train all models
    classifier = train_classifier(df)
    df, clustering = train_clustering(df)
    regressor = train_regressor(df)

    # Save final dataset with clusters
    df.to_csv('data/final_data.csv', index=False)
    print("\n🎉 All models trained successfully!")
    print("📁 Models saved in 'models/' folder")
