"""Training script for breast cancer dataset."""

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


def run_training_and_save():
    """Train model and save assets to disk."""
    save_folder = os.path.join(os.path.dirname(__file__), "breast_cancer_assets")
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

    model = SmallNN(input_dim=10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    X_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for _ in range(50):
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        predicted = (outputs >= 0.5).float()
        accuracy = (predicted.eq(y_test_t).sum().item()) / y_test_t.size(0)
        n_test_positive = int((predicted == 1).sum().item())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    model_path = os.path.join(save_folder, "model_weights.pth")
    torch.save(model.state_dict(), model_path)

    payload = {
        "X_test": X_test_s,
        "X_background": X_bg_s,
        "scaler": scaler,
        "feature_names": data.feature_names[:10]
    }
    data_path = os.path.join(save_folder, "processed_data.joblib")
    joblib.dump(payload, data_path)

    info_lines = [
        "=== Breast Cancer Dataset ===",
        f"Features used       : 10 (first 10 of 30)",
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