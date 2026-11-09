"""Training script for breast cancer dataset with added noise feature.

Identical to breast_cancer_trainer.py but appends one useless feature:
uniform random noise U(0, 1) - pure noise, no distributional structure.

The noisy column is appended after scaling so the StandardScaler is fitted
only on the original 10 features. The model is then retrained on all 11
features so it must learn to ignore the last one.
"""

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SmallNN(nn.Module):
    """Simple feedforward neural network for binary classification."""

    def __init__(self, input_dim):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def _append_noise(X, rng):
    """Append one column of U(0,1) noise to X."""
    noise = rng.uniform(0, 1, size=(X.shape[0], 1))
    return np.hstack([X, noise])


def run_training_and_save():
    """Train model and save assets to disk."""
    save_folder = os.path.join(os.path.dirname(__file__), "breast_cancer_noisy_assets")
    os.makedirs(save_folder, exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data[:, :10], data.target
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_background, y_train, y_background = train_test_split(
        X_train_all, y_train_all, train_size=355, random_state=42, stratify=y_train_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_bg_s = scaler.transform(X_background)

    rng = np.random.default_rng(0)
    X_train_n = _append_noise(X_train_s, rng)
    X_test_n = _append_noise(X_test_s, rng)
    X_bg_n = _append_noise(X_bg_s, rng)

    n_features = X_train_n.shape[1]
    feature_names = list(data.feature_names[:10]) + ["Noise_Uniform"]

    model = SmallNN(input_dim=n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    X_t = torch.tensor(X_train_n, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for _ in range(50):
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()

    X_test_t = torch.tensor(X_test_n, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        predicted = (outputs >= 0.5).float()
        accuracy = predicted.eq(y_test_t).sum().item() / y_test_t.size(0)
        n_test_positive = int((predicted == 1).sum().item())
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), os.path.join(save_folder, "model_weights.pth"))

    joblib.dump(
        {
            "X_test": X_test_n,
            "X_background": X_bg_n,
            "scaler": scaler,
            "feature_names": feature_names,
        },
        os.path.join(save_folder, "processed_data.joblib"),
    )

    info_lines = [
        "=== Breast Cancer Dataset (+ Useless Feature) ===",
        f"Features used       : {n_features} (10 original + 1 useless)",
        f"Useless feature     : Noise_Uniform  - U(0,1) random noise",
        f"Train samples       : {len(X_train)}",
        f"Background samples  : {len(X_background)}",
        f"Test samples        : {len(X_test)}",
        f"Test positive       : {n_test_positive} / {len(X_test)}",
        f"Test accuracy       : {accuracy * 100:.2f}%",
    ]
    with open(os.path.join(save_folder, "info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    print(f"Training complete. Assets saved to {save_folder}/")


if __name__ == "__main__":
    run_training_and_save()
