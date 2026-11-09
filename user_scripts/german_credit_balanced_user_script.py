"""User script for German credit dataset - Random Forest model."""

import os
from pathlib import Path

import joblib
import numpy as np

_ASSETS = Path(__file__).resolve().parent.parent / "training" / "german_credit_balanced_assets"


def get_data():
    """Return numpy arrays for test and background (already scaled)."""
    data = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    return data["X_test"], data["X_background"]


def load_everything():
    """Load model and processed assets and return a bundle dict."""
    model = joblib.load(os.path.join(_ASSETS, "model_weights.pth"))
    data_assets = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    return {
        "model":         model,
        "scaler":        data_assets.get("scaler"),
        "feature_names": data_assets.get("feature_names"),
    }


def predict(bundle, X):
    """Predict wrapper - returns class-1 probability as a (n, 1) numpy array."""
    model = bundle["model"]
    X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
    proba = model.predict_proba(X_arr)[:, 1]   # probability of class 1
    return proba.reshape(-1, 1)

