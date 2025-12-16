from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from io import StringIO
import os

# FastAPI app
app = FastAPI(
    title="GitHub Developer Classifier API",
    description="Predicts whether a GitHub user is a Web Developer (0) or ML Developer (1)",
    version="1.0.0"
)

# Load models
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    knn_model = joblib.load(os.path.join(MODELS_DIR, 'knn_k5.pkl'))
    mlp_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'mlp_1layer_64.keras'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    scaler = None
    knn_model = None
    mlp_model = None


class PredictionInput(BaseModel):
    """Input model for prediction endpoint"""
    csv_line: str
    model: str = "mlp"  # pick mlp or knn
    
    class Config:
        json_schema_extra = {
            "example": {
                "csv_line": "0,0,1,0,0,1,0,0,0,0,...",
                "model": "mlp"
            }
        }


class PredictionOutput(BaseModel):
    """Output model for prediction endpoint"""
    probability: float
    prediction: int
    label: str
    model_used: str


@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "GitHub Developer Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "scaler": scaler is not None,
            "knn": knn_model is not None,
            "mlp": mlp_model is not None
        }
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if scaler is None or knn_model is None or mlp_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please run training first.")
    
    try:
        # Parse CSV line
        values = [float(x.strip()) for x in input_data.csv_line.split(',')]
        features = np.array(values).reshape(1, -1)
        
        # Check feature count
        expected_features = scaler.n_features_in_
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_features} features, got {features.shape[1]}"
            )
        
        features_scaled = scaler.transform(features)
        
        if input_data.model.lower() == "knn":
            probability = knn_model.predict_proba(features_scaled)[0, 1]
            model_used = "kNN (k=5)"
        else:
            probability = float(mlp_model.predict(features_scaled, verbose=0)[0, 0])
            model_used = "MLP (1x64)"
        
        prediction = 1 if probability >= 0.5 else 0
        label = "ML Developer" if prediction == 1 else "Web Developer"
        
        return PredictionOutput(
            probability=round(probability, 4),
            prediction=prediction,
            label=label,
            model_used=model_used
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models")
def list_models():
    return {
        "available_models": [
            {"name": "mlp", "description": "MLP Neural Network (1 layer, 64 neurons)"},
            {"name": "knn", "description": "k-Nearest Neighbors (k=5)"}
        ],
        "default": "mlp"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
