"""Training script for German credit dataset."""

import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SmallNN(nn.Module):
    """Simple feedforward neural network for binary classification."""

    def __init__(self, input_dim=15):
        super(SmallNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        """Return probability via sigmoid."""
        return torch.sigmoid(self.net(x))


def run_training_and_save():
    """Train model and save assets to disk."""
    save_folder = os.path.join(os.path.dirname(__file__), "german_credit_assets")
    os.makedirs(save_folder, exist_ok=True)

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "german_credit", "german.data-numeric"
    )
    raw = np.loadtxt(data_path)
    X = raw[:, :15].astype(np.float64)
    y = (raw[:, -1] == 1).astype(np.float64)
    n_features = X.shape[1]
    print(f"Loaded german.data-numeric: {X.shape[0]} samples, {n_features} features")

    feature_names = [f"Attr{i+1}" for i in range(n_features)]

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_background, y_train, y_background = train_test_split(
        X_train_all, y_train_all, train_size=600, random_state=42, stratify=y_train_all
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_bg_s = scaler.transform(X_background)

    model = SmallNN(input_dim=n_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        outputs = model.predict_proba(X_test_t)
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
        "feature_names": feature_names,
    }
    data_path = os.path.join(save_folder, "processed_data.joblib")
    joblib.dump(payload, data_path)

    info_lines = [
        "=== German Credit Dataset ===",
        f"Features used       : {n_features} (first {n_features} of 24)",
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