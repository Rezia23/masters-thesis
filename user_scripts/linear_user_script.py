"""User script for simple linear model (synthetic)."""

import numpy as np

TRUE_WEIGHTS = np.array([10, 5, 0.5, 0.01, 0, -10.0])


def get_data():
    """Generate synthetic data normalized to [0, 1] range."""
    rng = np.random.default_rng(42)

    X_test = rng.uniform(0, 1, size=(100, 6))
    X_background = rng.uniform(0, 1, size=(200, 6))

    return X_test, X_background


def load_everything():
    """Return a bundle with weights and metadata."""
    return {
        "model": TRUE_WEIGHTS,
        "feature_names": [f"Feature_{i}" for i in range(6)],
    }


def predict(bundle, X):
    """Linear prediction with sigmoid normalization to [0, 1]."""
    weights = bundle["model"]

    X_arr = np.atleast_2d(np.array(X))
    raw_output = X_arr @ weights
    normalized_output = 1 / (1 + np.exp(-raw_output))

    return normalized_output.reshape(-1, 1)

