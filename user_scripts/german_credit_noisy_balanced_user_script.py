"""User script for German credit dataset - noisy RF model (+ Constant_Zero feature)."""

import os
from pathlib import Path

import joblib
import numpy as np

_ASSETS = Path(__file__).resolve().parent.parent / "training" / "german_credit_noisy_balanced_assets"


def get_data():
    """Return test and background arrays (already scaled + constant zero column appended)."""
    data = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    return data["X_test"], data["X_background"]


def load_everything():
    """Load RF model and assets. Model expects 16 features (15 original + constant zero)."""
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
    proba = model.predict_proba(X_arr)[:, 1]
    return proba.reshape(-1, 1)

