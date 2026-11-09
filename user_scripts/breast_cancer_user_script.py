"""User script for breast cancer dataset."""

import os
from pathlib import Path

import joblib
import numpy as np
import torch

from training.breast_cancer_trainer import SmallNN

_ASSETS = Path(__file__).resolve().parent.parent / "training" / "breast_cancer_assets"


def get_data():
    """Return numpy arrays for test and background (already scaled)."""
    data = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    X_test = data.get("X_test")
    X_background = data.get("X_background")
    return X_test, X_background


def load_everything():
    """Load model and processed assets and return a bundle dict."""
    model = SmallNN(input_dim=10)
    state = torch.load(os.path.join(_ASSETS, "model_weights.pth"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    data_assets = joblib.load(os.path.join(_ASSETS, "processed_data.joblib"))
    scaler = data_assets.get("scaler", None)
    feature_names = data_assets.get("feature_names", None)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def predict(bundle, X):
    """Return predicted positive-class probabilities, shape (n_samples,)."""
    model = bundle["model"]

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    X_t = torch.from_numpy(X_arr.astype(np.float32))

    with torch.no_grad():
        out = model(X_t).cpu().numpy()

    return out

