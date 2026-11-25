"""
train_bot_rnn.py
----------------
Trains the BotRNN on twitter_human_bots_dataset.csv, evaluates performance,
and saves the trained model weights and vectorizer/scaler.
"""

import os
import math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data.loader import load_bot_dataset
from src.data.preprocessing import preprocess, ACCOUNT_FEATURES
from src.features.temporal_features import build_sequences, SEQ_LEN
from src.models.rnn import BotRNN


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # 1. Load & preprocess data
    df = load_bot_dataset()
    df = preprocess(df)

    X_raw = df[ACCOUNT_FEATURES].values.astype(np.float32)
    y_raw = df["bot_label"].astype(int).values

    print(f"Dataset shape: X={X_raw.shape}, y={y_raw.shape}")
    print(f"Class counts: Human={(y_raw == 0).sum():,}, Bot={(y_raw == 1).sum():,}")

    # 2. Train / Val / Test split (70 / 15 / 15)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )

    # 3. Fit StandardScaler on train ONLY
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # 4. Build sequences
    X_train_seq = build_sequences(X_train_sc, seq_len=SEQ_LEN)
    X_val_seq = build_sequences(X_val_sc, seq_len=SEQ_LEN)
    X_test_seq = build_sequences(X_test_sc, seq_len=SEQ_LEN)

    input_size = X_train_seq.shape[2]
    print(f"Sequence shape: (seq_len={SEQ_LEN}, step_size={input_size})")

    # 5. DataLoaders
    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_seq), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # 6. Model & Loss (weighted loss for class imbalance: human ~25k, bot ~12.5k)
    model = BotRNN(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    class_weights = torch.tensor([1.0, 1.8], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 7. Training loop with early stopping
    epochs = 20
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_acc)
        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

    # 8. Evaluate on test set with best model
    model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print("\n================== TEST SET RESULTS ==================")
    print(f"Accuracy: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Human", "Bot"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("=====================================================\n")

    # 9. Save model, scaler, metadata
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("models/vectorizers", exist_ok=True)

    torch.save(best_weights, "models/saved/bot_rnn_model.pth")
    joblib.dump(scaler, "models/vectorizers/bot_scaler.joblib")
    meta = {
        "input_size": input_size,
        "seq_len": SEQ_LEN,
        "features": ACCOUNT_FEATURES,
    }
    joblib.dump(meta, "models/vectorizers/bot_input_meta.joblib")
    print("Saved all artifacts to models/saved/ and models/vectorizers/")


if __name__ == "__main__":
    train()
