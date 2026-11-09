"""Small helpers shared by the experiment runner and the explainer.

The functions here load user-defined workflow modules, persist feature names
alongside the result CSVs, and select the subset of test samples for which the
model predicts the positive class (those are the only instances worth
explaining for the case-based methods used in this work).
"""

import importlib.util
import os

import numpy as np


def load_workflow(file_path):
    """Import a user workflow module by file path.

    The module must define ``get_data``, ``predict`` and ``load_everything``.
    Returns a dict mapping the keys ``data``, ``predict`` and
    ``load_everything`` to the corresponding functions (or ``None`` when a
    function is absent).
    """
    spec = importlib.util.spec_from_file_location("user_workflow", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load workflow module '{file_path}': {e}")

    return {
        "data": getattr(module, "get_data", None),
        "predict": getattr(module, "predict", None),
        "load_everything": getattr(module, "load_everything", None),
    }


def save_feature_names(workflow_data, folder, num_features):
    """Write the workflow's feature names to ``feature_names.txt`` in ``folder``.

    Falls back to generic ``Feature 0``...``Feature N-1`` if the workflow bundle
    does not provide names.
    """
    feature_names = workflow_data.get(
        "feature_names", [f"Feature {i}" for i in range(num_features)]
    )
    with open(os.path.join(folder, "feature_names.txt"), "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(f"{name}\n")


def get_positive_samples(X_test, predict_fn):
    """Return the rows of ``X_test`` for which ``predict_fn`` outputs > 0.5.

    Raises ``RuntimeError`` if no test sample is predicted positive.
    """
    outputs = predict_fn(X_test)
    probs = outputs.reshape(-1)
    pred_mask = probs > 0.5
    X_test_np = np.asarray(X_test)
    if pred_mask.any():
        return X_test_np[pred_mask]
    raise RuntimeError("No positive predictions in the test set; cannot explain.")