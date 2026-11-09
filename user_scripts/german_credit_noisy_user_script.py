"""User script for German credit dataset with added constant zero feature."""

import os
from pathlib import Path

import joblib
import numpy as np
import torch

from training.german_credit_noisy_trainer import SmallNN

_ASSETS = Path(__file__).resolve().parent.parent / "training" / "german_credit_noisy_assets"


def get_data():
    """Return test and background arrays (already scaled + constant zero column appended)."""
    data = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    return data["X_test"], data["X_background"]


def load_everything():
    """Load model and assets. Model expects 16 features (15 original + constant zero)."""
    model = SmallNN(input_dim=16)
    state = torch.load(os.path.join(_ASSETS, "model_weights.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    data_assets = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    return {
        "model": model,
        "scaler": data_assets.get("scaler"),
        "feature_names": data_assets.get("feature_names"),
    }


def predict(bundle, X):
    """Return predicted positive-class probabilities, shape (n_samples,)."""
    model = bundle["model"]

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    with torch.no_grad():
        out = model.predict_proba(
            torch.from_numpy(X_arr.astype(np.float32))
        ).cpu().numpy()
    return out

