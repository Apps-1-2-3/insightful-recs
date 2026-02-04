"""
ML Model Training Script for Drug Recommendation System

This script trains a Logistic Regression classifier to predict recommended drugs
based on patient features. The model serves as a SUPPLEMENTARY AI layer to the
existing similarity-based recommendation engine.

LEARNING: The model learns patterns from historical EHR data to predict drugs.
EVALUATION: Uses accuracy on a 20% held-out validation split.
ROLE: Complements (does not replace) similarity-based reasoning.

Usage:
    cd backend
    python train_ml_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Configuration
CSV_PATH = "data/ehr_synthetic_max_features.csv"
MODEL_PATH = "ml_model.pkl"
ENCODERS_PATH = "ml_encoders.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COL = "Medication"

def parse_list_field(value: str) -> str:
    """Convert list fields to a normalized string for encoding."""
    if pd.isna(value) or not value:
        return "none"
    items = [item.strip().lower() for item in str(value).replace(";", ",").split(",") if item.strip()]
    return ",".join(sorted(items)) if items else "none"


def prepare_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    if encoders is None:
        encoders = {}

    X = pd.DataFrame()

    # Numeric feature
    if "Age" in df.columns:
        X["age"] = df["Age"].fillna(df["Age"].median()).astype(float) / 100.0

    # Categorical features
    categorical_cols = {
        "Gender": "gender",
        "Blood Type": "blood_type",
        "Medical Condition": "medical_condition"
    }

    for src_col, out_col in categorical_cols.items():
        if src_col in df.columns:
            values = df[src_col].fillna("unknown").astype(str).str.lower()
            if fit:
                encoders[out_col] = LabelEncoder()
                X[out_col] = encoders[out_col].fit_transform(values)
            else:
                known = set(encoders[out_col].classes_)
                values = values.apply(lambda x: x if x in known else "unknown")
                X[out_col] = encoders[out_col].transform(values)

    return X, encoders


def train_model():
    """
    Train and save the ML model.
    
    This trains a Logistic Regression classifier on the EHR dataset.
    The model learns to predict the recommended drug based on patient features.
    """
    print("=" * 60)
    print("Drug Recommendation ML Model Training")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        print("   Please ensure the EHR dataset is placed in the data/ folder")
        return False
    
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} records")
    
    # Check for target column
    if TARGET_COL not in df.columns:
        print(f"❌ Error: '{TARGET_COL}' column not found in dataset")
        return False
    
    # Filter valid targets
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")]
    print(f"📊 Records with valid drug labels: {len(df)}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, encoders = prepare_features(df, fit=True)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Prepare target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL].astype(str))
    encoders["target"] = target_encoder
    print(f"   Number of drug classes: {len(target_encoder.classes_)}")
    
    # Train/test split (80/20)
    print(f"\n📊 Splitting data (80% train, 20% validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Train Logistic Regression model
    # Using Logistic Regression for interpretability and simplicity
    print("\n🧠 Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete")
    
    # Evaluate on validation set
    # EVALUATION METRIC: Accuracy on held-out validation split
    print("\n📈 Evaluating model...")
    val_accuracy = model.score(X_val, y_val)
    train_accuracy = model.score(X_train, y_train)
    
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Save model and encoders
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    print(f"💾 Saving encoders to {ENCODERS_PATH}...")
    joblib.dump(encoders, ENCODERS_PATH)
    
    print("\n" + "=" * 60)
    print("✅ ML MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {MODEL_PATH} (trained Logistic Regression model)")
    print(f"  - {ENCODERS_PATH} (feature encoders)")
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print("\nNote: The ML model serves as a supplementary AI layer.")
    print("      Primary recommendations remain similarity-based.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = train_model()
    exit(0 if success else 1)
