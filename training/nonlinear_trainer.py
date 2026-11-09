"""Training script for nonlinear synthetic dataset - Random Forest model."""

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_training_and_save():
    """Train a Random Forest on the nonlinear synthetic data and save assets."""
    save_folder = os.path.join(os.path.dirname(__file__), "nonlinear_assets")
    os.makedirs(save_folder, exist_ok=True)

    np.random.seed(42)
    n_samples = 1500
    X = np.random.uniform(-2, 2, (n_samples, 7))
    x1, x2, x3, x4, x5, x6, x7 = (X[:, i] for i in range(7))

    z = (
        2.5 * x1 ** 3
        - 4.0 * x2 ** 2
        + 3.0 * x1 * x3
        + 1.5 * x4
        + 0.8 * x5
        + 0.2 * x6
        + 0.0 * x7
    )
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)

    feature_names = [
        "Cubic_Feat", "Quadratic_Feat", "Interaction_Feat",
        "Linear_High", "Linear_Mid", "Linear_Low", "Useless_Feat",
    ]
    n_features = len(feature_names)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_background, y_train, y_background = train_test_split(
        X_train_all, y_train_all, train_size=800, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_bg_s = scaler.transform(X_background)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    accuracy = float((y_pred == y_test).mean())
    n_test_positive = int((y_pred == 1).sum())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, os.path.join(save_folder, "model_weights.pth"))

    joblib.dump(
        {
            "X_test": X_test_s,
            "X_background": X_bg_s,
            "scaler": scaler,
            "feature_names": feature_names,
            "feature_importances": model.feature_importances_.tolist(),
        },
        os.path.join(save_folder, "processed_data.joblib"),
    )

    info_lines = [
        "=== Nonlinear Synthetic Dataset (Random Forest) ===",
        f"Model               : RandomForestClassifier(n_estimators=300)",
        f"Features used       : {n_features} (Cubic, Quadratic, Interaction, Linear x3, Useless)",
        f"Total samples       : {n_samples}",
        f"Train samples       : {len(X_train)}",
        f"Background samples  : {len(X_background)}",
        f"Test samples        : {len(X_test)}",
        f"Test positive       : {n_test_positive} / {len(X_test)}",
        f"Test accuracy       : {accuracy * 100:.2f}%",
        f"Feature importances : {dict(zip(feature_names, model.feature_importances_))}",
        f"Note                : Feature 7 ({feature_names[-1]}) is mathematically useless",
    ]
    with open(os.path.join(save_folder, "info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"Training complete. Assets saved to {save_folder}/")


if __name__ == "__main__":
    run_training_and_save()