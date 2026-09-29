"""
Car Price Prediction - Training Pipeline
Trains multiple regression models (Linear Regression, Random Forest, Gradient Boosting),
performs hyperparameter tuning, evaluates performance metrics, and serializes the best model.
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

DATA_PATH = 'car_price_prediction_.csv'
MODEL_PATH = 'car_price_rf_model.joblib'
FEATURES_PATH = 'model_features.json'


def load_and_preprocess_data(file_path=DATA_PATH):
    """Load dataset, handle data types, perform feature engineering, and one-hot encode."""
    print(f"[1/5] Loading dataset from: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"      Initial shape: {df.shape}")

    # Remove duplicates and nulls
    df = df.drop_duplicates()
    df = df.dropna()

    # Feature engineering: calculate Car Age
    current_year = 2026
    df['Car_Age'] = current_year - df['Year']

    # Drop non-predictive Car ID
    df = df.drop('Car ID', axis=1, errors='ignore')

    # Convert numeric datatypes
    df['Year'] = df['Year'].astype(int)
    df['Engine Size'] = df['Engine Size'].astype(float)
    df['Mileage'] = df['Mileage'].astype(int)
    df['Price'] = df['Price'].astype(float)

    # Encode categorical columns
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"      Encoded dataset shape: {df_encoded.shape}")

    return df_encoded, df


def evaluate_model(name, model, X_test, y_test):
    """Compute MAE, MSE, RMSE, and R2 metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {
        "Model": name,
        "MAE ($)": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE ($)": round(rmse, 2),
        "R2 Score": round(r2, 4)
    }


def train_models():
    """Main training routine."""
    df_encoded, raw_df = load_and_preprocess_data()

    # Features and target
    X = df_encoded.drop('Price', axis=1)
    y = df_encoded['Price']

    # Train/test split
    print("[2/5] Splitting data into 80% train and 20% test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Train samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
    print(f"      Test samples:  {X_test.shape[0]}  | Features: {X_test.shape[1]}")

    # Train baseline models
    print("[3/5] Training baseline models (Linear Regression, Random Forest, Gradient Boosting)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)

    results = [
        evaluate_model("Linear Regression", lr, X_test, y_test),
        evaluate_model("Random Forest (Default)", rf, X_test, y_test),
        evaluate_model("Gradient Boosting", gb, X_test, y_test),
    ]

    # Hyperparameter tuning
    print("[4/5] Running GridSearchCV on Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    gs_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    gs_rf.fit(X_train, y_train)

    best_rf = gs_rf.best_estimator_
    print(f"      Best Parameters: {gs_rf.best_params_}")
    print(f"      Best Cross-Validation Score: {gs_rf.best_score_:.4f}")

    results.append(evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test))

    # Print results summary table
    print("\n" + "=" * 65)
    print("               MODEL EVALUATION BENCHMARK SUMMARY               ")
    print("=" * 65)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("=" * 65 + "\n")

    # Serialize best model and feature metadata
    print(f"[5/5] Saving best model to '{MODEL_PATH}'...")
    joblib.dump(best_rf, MODEL_PATH)

    feature_metadata = {
        "feature_names": list(X.columns),
        "categorical_cols": ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model'],
        "brands": sorted(raw_df['Brand'].unique().tolist()),
        "fuel_types": sorted(raw_df['Fuel Type'].unique().tolist()),
        "transmissions": sorted(raw_df['Transmission'].unique().tolist()),
        "conditions": sorted(raw_df['Condition'].unique().tolist()),
        "models": sorted(raw_df['Model'].unique().tolist()),
        "best_params": gs_rf.best_params_,
    }
    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"      Saved feature schema to '{FEATURES_PATH}'.")

    # Quick test inference
    sample_car = X_test.iloc[[0]]
    predicted_val = best_rf.predict(sample_car)[0]
    actual_val = y_test.iloc[0]
    print(f"\n[Test Inference]")
    print(f"  Sample Actual Price:    ${actual_val:,.2f}")
    print(f"  Sample Predicted Price: ${predicted_val:,.2f}")
    print("Training pipeline completed successfully!\n")


if __name__ == '__main__':
    train_models()
