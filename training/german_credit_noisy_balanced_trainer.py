"""Training script for German credit dataset - noisy RF model.

Identical split and preprocessing to german_credit_noisy_trainer.py but uses
a RandomForestClassifier with class_weight='balanced' instead of a NN.
Appends one useless feature: a constant column of 0 (zero variance).

n_jobs=1 - model is called inside a multiprocessing pool (run_experiment_with_pool).
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _append_constant(X, value=0):
    """Append a constant column of zeros to X."""
    col = np.full((X.shape[0], 1), value)
    return np.hstack([X, col])


def run_training_and_save():
    """Train a Random Forest and save assets to disk."""
    save_folder = os.path.join(os.path.dirname(__file__), "german_credit_noisy_balanced_assets")
    os.makedirs(save_folder, exist_ok=True)

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "german_credit", "german.data-numeric"
    )
    raw = np.loadtxt(data_path)
    X = raw[:, :15].astype(np.float64)
    y = (raw[:, -1] == 1).astype(np.float64)
    print(f"Loaded german.data-numeric: {X.shape[0]} samples, 15 features")
    print(f"Class balance - positive: {int(y.sum())}, negative: {int((y == 0).sum())}")

    # Identical split to the other german credit trainers.
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_background, y_train, _ = train_test_split(
        X_train_all, y_train_all, train_size=600, random_state=42, stratify=y_train_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_bg_s    = scaler.transform(X_background)

    # Append constant zero column (useless feature)
    X_train_n = _append_constant(X_train_s)
    X_test_n  = _append_constant(X_test_s)
    X_bg_n    = _append_constant(X_bg_s)

    feature_names = [f"Attr{i+1}" for i in range(15)] + ["Constant_Zero"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,  # must be 1 - model is used inside a multiprocessing pool
    )
    model.fit(X_train_n, y_train)

    y_pred = model.predict(X_test_n)
    accuracy = float((y_pred == y_test).mean())
    n_test_positive = int((y_pred == 1).sum())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    model_path = os.path.join(save_folder, "model_weights.pth")
    joblib.dump(model, model_path)

    joblib.dump(
        {
            "X_test":        X_test_n,
            "X_background":  X_bg_n,
            "scaler":        scaler,
            "feature_names": feature_names,
        },
        os.path.join(save_folder, "processed_data.joblib"),
    )

    info_lines = [
        "=== German Credit Dataset - Noisy RF (balanced) ===",
        f"Model                   : RandomForestClassifier(n_estimators=300, class_weight='balanced')",
        f"Features used           : 16 (15 original + 1 useless)",
        f"Useless feature         : Constant_Zero - constant value 0 (zero variance)",
        f"Train samples           : {len(X_train)}",
        f"Background samples      : {len(X_background)}",
        f"Test samples            : {len(X_test)}",
        f"Test positive predicted : {n_test_positive} / {len(X_test)}",
        f"Test accuracy           : {accuracy * 100:.2f}%",
    ]
    with open(os.path.join(save_folder, "info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"Training complete. Assets saved to {save_folder}/")


if __name__ == "__main__":
    run_training_and_save()
