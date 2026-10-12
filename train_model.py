"""
train_model.py
================
Student Marks Prediction Based on Study Hours - Model Training Script

This project focuses on ONE relationship only:

    Study Hours  -->  Predicted Marks

This script:
1. Loads the dataset.
2. Checks and cleans the data (missing values, duplicates, data types).
3. Performs basic exploratory data analysis (EDA) and saves plots.
4. Separates the feature (Study_Hours) and target (Marks).
5. Splits the data into 80% training and 20% testing.
6. Trains a Linear Regression model.
7. Evaluates the model using MAE, MSE, RMSE, and R2.
8. Prints an Actual vs Predicted comparison table.
9. Saves the trained model using Joblib.

Run with:
    python train_model.py
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration - relative paths so the script works on any machine
# (Windows, macOS, Linux) as long as it is run from the project's root folder.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "student_marks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(MODEL_DIR, "marks_prediction_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_COLUMN = "Study_Hours"
TARGET_COLUMN = "Marks"


def load_dataset():
    """Step 1: Load the dataset, raise a clear error if it is missing."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. "
            "Please make sure 'student_marks.csv' exists inside the 'dataset' folder. "
            "You can regenerate it by running 'python generate_dataset.py'."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Dataset loaded successfully. Shape: {df.shape}")
    return df


def inspect_dataset(df):
    """Step 2: Basic dataset inspection as required by the project spec."""
    print("\n========== DATASET OVERVIEW ==========")
    print("\nFirst 5 rows:")
    print(df.head())

    print(f"\nNumber of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nBasic statistics:")
    print(df.describe())


def preprocess_data(df):
    """Step 3: Minimal, focused preprocessing (only what this project needs)."""
    print("\n========== DATA PREPROCESSING ==========")

    # Missing value handling
    if df.isnull().sum().sum() > 0:
        before = df.shape[0]
        df = df.dropna().reset_index(drop=True)
        print(f"[INFO] Dropped rows with missing values. Rows before: {before}, after: {df.shape[0]}")
    else:
        print("[INFO] No missing values found.")

    # Duplicate removal
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Removed {dup_count} duplicate row(s). New shape: {df.shape}")
    else:
        print("[INFO] No duplicate rows found.")

    # Data type validation - ensure both columns are numeric
    df[FEATURE_COLUMN] = pd.to_numeric(df[FEATURE_COLUMN], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    invalid_rows = df[df[FEATURE_COLUMN].isnull() | df[TARGET_COLUMN].isnull()]
    if len(invalid_rows) > 0:
        print(f"[INFO] Removing {len(invalid_rows)} row(s) with non-numeric values.")
        df = df.dropna().reset_index(drop=True)

    # Basic sanity check on valid ranges
    df = df[(df[FEATURE_COLUMN] >= 0) & (df[FEATURE_COLUMN] <= 24)]
    df = df[(df[TARGET_COLUMN] >= 0) & (df[TARGET_COLUMN] <= 100)]
    df = df.reset_index(drop=True)

    print("[INFO] Preprocessing complete.")
    return df


def run_eda(df):
    """Step 4: Basic EDA plots - scatter plot and marks distribution."""
    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    # Study Hours vs Marks scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(df[FEATURE_COLUMN], df[TARGET_COLUMN], alpha=0.7, color="steelblue")
    plt.title("Study Hours vs Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "study_hours_vs_marks.png"))
    plt.close()
    print("[INFO] Saved plot: study_hours_vs_marks.png")

    # Marks distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET_COLUMN], bins=15, color="teal", edgecolor="white")
    plt.title("Distribution of Marks")
    plt.xlabel("Marks")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "marks_distribution.png"))
    plt.close()
    print("[INFO] Saved plot: marks_distribution.png")


def train_and_evaluate(df):
    """Steps 5-9: split, train, evaluate, plot regression line, save model."""
    print("\n========== MODEL TRAINING & EVALUATION ==========")

    X = df[[FEATURE_COLUMN]]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\n[INFO] Learned relationship: Marks = {model.coef_[0]:.4f} * Study_Hours + {model.intercept_:.4f}")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation on Test Data:")
    print("-" * 40)
    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")
    print("-" * 40)

    # Actual vs Predicted comparison table
    comparison_df = pd.DataFrame({
        "Study_Hours": X_test[FEATURE_COLUMN].values,
        "Actual_Marks": y_test.values,
        "Predicted_Marks": np.round(predictions, 2),
    }).sort_values(by="Study_Hours").reset_index(drop=True)

    print("\nActual vs Predicted Marks (Test Set):")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(os.path.join(BASE_DIR, "actual_vs_predicted.csv"), index=False)
    print("\n[INFO] Saved comparison table to actual_vs_predicted.csv")

    # Regression line plot (full dataset + fitted line)
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.6, color="steelblue", label="Actual data")
    x_line = np.linspace(X[FEATURE_COLUMN].min(), X[FEATURE_COLUMN].max(), 100).reshape(-1, 1)
    y_line = model.predict(pd.DataFrame(x_line, columns=[FEATURE_COLUMN]))
    plt.plot(x_line, y_line, color="crimson", linewidth=2, label="Regression line")
    plt.title("Linear Regression: Study Hours vs Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "regression_line.png"))
    plt.close()
    print("[INFO] Saved plot: regression_line.png")

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    return model, metrics


def save_model(model):
    """Save the trained model using Joblib."""
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")


def main():
    df = load_dataset()
    inspect_dataset(df)
    df = preprocess_data(df)
    run_eda(df)
    model, metrics = train_and_evaluate(df)
    save_model(model)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Final R2 Score: {metrics['R2']:.3f}")
    print("You can now run 'python app.py' to start the web application.")


if __name__ == "__main__":
    main()
