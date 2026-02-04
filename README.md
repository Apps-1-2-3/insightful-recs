# Drug Recommendation System with AI/ML

An AI-powered drug recommendation system using similarity-based matching and machine learning.

## Quick Start

### 1. Backend Setup (Python FastAPI)

```bash
# Navigate to backend folder
cd backend

# Install dependencies
pip install -r requirements.txt

# (First time) Train the ML model
python train_ml_model.py

# Start the API server
python main.py
```

The API will be available at `http://localhost:8000` (docs at `/docs`).

### 2. Frontend Setup (React/Vite)

```bash
# In a new terminal, from project root
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server with recommendation engine
│   ├── train_ml_model.py    # ML model training script
│   ├── requirements.txt     # Python dependencies
│   ├── data/
│   │   └── ehr_synthetic_max_features.csv  # EHR dataset (100k records)
│   ├── ml_model.pkl         # Trained ML model (generated)
│   └── ml_encoders.pkl      # Feature encoders (generated)
├── src/                     # React frontend
└── ...
```

## AI/ML Components

### Recommendation Engine (Primary)
- **Similarity-based matching**: Finds similar patients in EHR records
- **SHAP-like explanations**: Shows which features influenced the recommendation

### Machine Learning Model (Supplementary)
- **Algorithm**: Logistic Regression
- **Training**: Run `python train_ml_model.py` to train on EHR data
- **Evaluation**: Accuracy on 80/20 validation split
- **Role**: Provides validation signal; boosts confidence when aligned with similarity-based results

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (includes ML status) |
| `/data/status` | GET | Dataset info |
| `/predict` | POST | Get drug recommendations |

## Data Requirements

Place your EHR CSV file at `backend/data/ehr_synthetic_max_features.csv` with columns:
- `age`, `gender`, `heart_rate`, `blood_type`
- `symptoms`, `medical_history`, `allergies`, `current_medications`
- `recommended_drug` (target for ML training)
