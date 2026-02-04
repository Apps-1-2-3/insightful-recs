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


def parse_list_field(value: str) -> str:
    """Convert list fields to a normalized string for encoding."""
    if pd.isna(value) or not value:
        return "none"
    items = [item.strip().lower() for item in str(value).replace(";", ",").split(",") if item.strip()]
    return ",".join(sorted(items)) if items else "none"


def prepare_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """
    Prepare features for ML model.
    
    Args:
        df: DataFrame with patient data
        encoders: Dictionary of LabelEncoders (None if fitting new)
        fit: Whether to fit encoders (True for training, False for inference)
    
    Returns:
        X: Feature matrix
        encoders: Dictionary of fitted encoders
    """
    if encoders is None:
        encoders = {}
    
    # Select features for ML model
    feature_cols = ["age", "gender", "heart_rate", "blood_type", "symptoms", "medical_history", "allergies"]
    
    # Create a copy for processing
    X = pd.DataFrame()
    
    # Numeric features (normalize)
    if "age" in df.columns:
        X["age"] = df["age"].fillna(45).astype(float) / 100.0  # Normalize to 0-1 range
    
    if "heart_rate" in df.columns:
        X["heart_rate"] = df["heart_rate"].fillna(72).astype(float) / 200.0  # Normalize
    
    # Categorical features (label encode)
    categorical_cols = ["gender", "blood_type"]
    for col in categorical_cols:
        if col in df.columns:
            values = df[col].fillna("unknown").astype(str).str.lower()
            if fit:
                encoders[col] = LabelEncoder()
                X[col] = encoders[col].fit_transform(values)
            else:
                # Handle unseen labels during inference
                known_labels = set(encoders[col].classes_)
                values = values.apply(lambda x: x if x in known_labels else "unknown")
                X[col] = encoders[col].transform(values)
    
    # List-based features (simplified encoding - hash to categories)
    list_cols = ["symptoms", "medical_history", "allergies"]
    for col in list_cols:
        if col in df.columns:
            values = df[col].apply(parse_list_field)
            if fit:
                encoders[col] = LabelEncoder()
                # Limit unique values to prevent overfitting
                unique_vals = values.unique()
                if len(unique_vals) > 100:
                    # Keep top 100 most common, rest as "other"
                    top_vals = values.value_counts().head(100).index.tolist()
                    values = values.apply(lambda x: x if x in top_vals else "other")
                X[col] = encoders[col].fit_transform(values)
            else:
                known_labels = set(encoders[col].classes_)
                values = values.apply(lambda x: x if x in known_labels else "other" if "other" in known_labels else list(known_labels)[0])
                X[col] = encoders[col].transform(values)
    
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
    if "recommended_drug" not in df.columns:
        print("❌ Error: 'recommended_drug' column not found in dataset")
        return False
    
    # Filter valid targets
    df = df[df["recommended_drug"].notna() & (df["recommended_drug"] != "")]
    print(f"📊 Records with valid drug labels: {len(df)}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, encoders = prepare_features(df, fit=True)
    print(f"   Feature matrix shape: {X.shape}")
    
    # Prepare target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["recommended_drug"].astype(str))
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
