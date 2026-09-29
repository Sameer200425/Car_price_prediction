"""
Car Price Prediction - Inference Pipeline
Predicts vehicle price using the trained Random Forest model.
Supports interactive mode, demo examples, and CLI argument parsing.
"""

import argparse
import json
import os
import sys
import pandas as pd
import joblib

MODEL_PATH = 'car_price_rf_model.joblib'
FEATURES_PATH = 'model_features.json'


def load_artifacts():
    """Load model and feature schema metadata."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run 'train.py' first.")
        sys.exit(1)
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: Metadata file '{FEATURES_PATH}' not found. Please run 'train.py' first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta


def predict_price(car_dict, model, meta):
    """
    Given a dictionary of car attributes:
    Brand, Model, Year, Engine Size, Fuel Type, Transmission, Mileage, Condition
    Returns the predicted price.
    """
    feature_names = meta['feature_names']
    current_year = 2026

    # Base numerical features
    row = {
        'Year': int(car_dict['Year']),
        'Engine Size': float(car_dict['Engine Size']),
        'Mileage': int(car_dict['Mileage']),
        'Car_Age': current_year - int(car_dict['Year'])
    }

    # Initialize all one-hot encoded dummy columns to False/0
    for col in feature_names:
        if col not in row:
            row[col] = False

    # Set active categorical dummy flags
    brand_col = f"Brand_{car_dict['Brand']}"
    if brand_col in row:
        row[brand_col] = True

    fuel_col = f"Fuel Type_{car_dict['Fuel Type']}"
    if fuel_col in row:
        row[fuel_col] = True

    trans_col = f"Transmission_{car_dict['Transmission']}"
    if trans_col in row:
        row[trans_col] = True

    cond_col = f"Condition_{car_dict['Condition']}"
    if cond_col in row:
        row[cond_col] = True

    model_col = f"Model_{car_dict['Model']}"
    if model_col in row:
        row[model_col] = True

    # Assemble into DataFrame with identical column ordering
    input_df = pd.DataFrame([row])[feature_names]
    predicted_price = model.predict(input_df)[0]
    return predicted_price


def run_demo(model, meta):
    """Run demonstration with several realistic vehicle profiles."""
    print("=" * 65)
    print("             CAR PRICE PREDICTOR - DEMO PREDICTIONS             ")
    print("=" * 65)

    sample_cars = [
        {
            "Brand": "Tesla",
            "Model": "Model 3",
            "Year": 2022,
            "Engine Size": 3.0,
            "Fuel Type": "Electric",
            "Transmission": "Automatic",
            "Mileage": 24000,
            "Condition": "Like New"
        },
        {
            "Brand": "BMW",
            "Model": "5 Series",
            "Year": 2018,
            "Engine Size": 4.4,
            "Fuel Type": "Petrol",
            "Transmission": "Automatic",
            "Mileage": 78000,
            "Condition": "Used"
        },
        {
            "Brand": "Toyota",
            "Model": "Prius",
            "Year": 2020,
            "Engine Size": 1.8,
            "Fuel Type": "Hybrid",
            "Transmission": "Automatic",
            "Mileage": 35000,
            "Condition": "New"
        },
        {
            "Brand": "Ford",
            "Model": "Mustang",
            "Year": 2015,
            "Engine Size": 5.0,
            "Fuel Type": "Petrol",
            "Transmission": "Manual",
            "Mileage": 95000,
            "Condition": "Used"
        }
    ]

    for i, car in enumerate(sample_cars, start=1):
        price = predict_price(car, model, meta)
        print(f"\nCar #{i}: {car['Year']} {car['Brand']} {car['Model']}")
        print(f"  Specs:       {car['Engine Size']}L | {car['Fuel Type']} | {car['Transmission']} | {car['Mileage']:,} miles | {car['Condition']}")
        print(f"  >> Estimated Price: ${price:,.2f}")

    print("\n" + "=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Predict car price using trained Random Forest model.")
    parser.add_argument('--brand', type=str, help='Car brand (e.g., Tesla, BMW, Ford, Toyota, etc.)')
    parser.add_argument('--model', type=str, help='Car model (e.g., Model 3, 5 Series, Mustang, Prius, etc.)')
    parser.add_argument('--year', type=int, help='Manufacturing year (e.g., 2020)')
    parser.add_argument('--engine', type=float, help='Engine displacement size in liters (e.g., 2.5)')
    parser.add_argument('--fuel', type=str, help='Fuel type (Petrol, Diesel, Electric, Hybrid)')
    parser.add_argument('--transmission', type=str, help='Transmission (Automatic, Manual)')
    parser.add_argument('--mileage', type=int, help='Odometer mileage (e.g., 45000)')
    parser.add_argument('--condition', type=str, help='Condition (New, Like New, Used)')
    parser.add_argument('--demo', action='store_true', help='Run built-in sample predictions')

    args = parser.parse_args()
    model, meta = load_artifacts()

    if args.demo or not any([args.brand, args.model, args.year]):
        run_demo(model, meta)
        return

    car_input = {
        'Brand': args.brand or 'Tesla',
        'Model': args.model or 'Model 3',
        'Year': args.year or 2020,
        'Engine Size': args.engine or 2.5,
        'Fuel Type': args.fuel or 'Petrol',
        'Transmission': args.transmission or 'Automatic',
        'Mileage': args.mileage or 50000,
        'Condition': args.condition or 'Used'
    }

    price = predict_price(car_input, model, meta)
    print("\n" + "=" * 50)
    print("             PREDICTION RESULT                  ")
    print("=" * 50)
    print(f"Vehicle:         {car_input['Year']} {car_input['Brand']} {car_input['Model']}")
    print(f"Engine:          {car_input['Engine Size']}L {car_input['Fuel Type']}")
    print(f"Transmission:    {car_input['Transmission']}")
    print(f"Mileage:         {car_input['Mileage']:,} miles")
    print(f"Condition:       {car_input['Condition']}")
    print("-" * 50)
    print(f"ESTIMATED PRICE: ${price:,.2f}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
