import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.preprocess import prepare_data


class CreditDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CreditDefaultMLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_pytorch_model(
    file_path: str | None = None,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    save_path: str = "models/pytorch_credit_model.pth",
) -> tuple[
    CreditDefaultMLP,
    dict[str, float],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    X_train, X_test, y_train, y_test, _, _ = prepare_data(file_path=file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(
        f"\n[PyTorch] Running training on device: {device} ({torch.get_num_threads()} threads)"
    )

    train_dataset = CreditDataset(X_train, y_train)
    test_dataset = CreditDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_features = X_train.shape[1]
    model = CreditDefaultMLP(in_features).to(device)

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_auc = 0.0
    patience = 6
    patience_counter = 0

    print(
        f"[PyTorch] Training {epochs} epochs | Batch: {batch_size} | Pos-Weight: {pos_weight.item():.2f}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        model.eval()
        val_probs: list[float] = []
        val_targets: list[float] = []
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)

                probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()
                val_probs.extend(probs)
                val_targets.extend(batch_y.cpu().numpy().flatten().tolist())

        epoch_val_loss = val_loss / len(test_dataset)
        val_probs_arr = np.array(val_probs)
        val_targets_arr = np.array(val_targets)

        val_auc = float(roc_auc_score(val_targets_arr, val_probs_arr))
        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1 or val_auc > best_val_auc:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[PyTorch] Early stopping triggered at Epoch {epoch}. Best Val ROC-AUC: {best_val_auc:.4f}"
                )
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        all_X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_logits = model(all_X_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, test_preds)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, test_preds)),
        "Precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "Recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "F1-Score": float(f1_score(y_test, test_preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, test_probs)),
        "PR-AUC": float(average_precision_score(y_test, test_probs)),
    }

    print(f"\n[PyTorch Final Benchmark] Best Checkpoint: {save_path}")
    print(
        f" -> Accuracy: {metrics['Accuracy'] * 100:.2f}% | Balanced Acc: {metrics['Balanced Accuracy'] * 100:.2f}%"
    )
    print(
        f" -> Precision: {metrics['Precision'] * 100:.2f}% | Recall: {metrics['Recall'] * 100:.2f}% | F1: {metrics['F1-Score']:.4f}"
    )
    print(f" -> ROC-AUC: {metrics['ROC-AUC']:.4f} | PR-AUC: {metrics['PR-AUC']:.4f}")

    return model, metrics, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_pytorch_model()
