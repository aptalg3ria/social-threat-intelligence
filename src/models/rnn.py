"""
src/models/rnn.py
------------------
RNN (GRU-based) model for bot detection using PyTorch.

Architecture
------------
  Input  : (batch, seq_len, input_size)
  GRU    : 2 stacked layers, hidden_size=64, dropout=0.3
  FC     : hidden_size → 2 (binary classification: human / bot)
  Output : logits; use Softmax/Sigmoid externally for probabilities.

Usage (training, done in the Colab notebook):
  model = BotRNN(input_size=..., hidden_size=64, num_layers=2)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
"""

import torch
import torch.nn as nn


class BotRNN(nn.Module):
    """
    Gated Recurrent Unit (GRU) classifier for bot detection.

    Parameters
    ----------
    input_size  : int — number of features per time step
    hidden_size : int — GRU hidden dimension (default 64)
    num_layers  : int — number of stacked GRU layers (default 2)
    dropout     : float — dropout probability between GRU layers (default 0.3)
    num_classes : int — output classes: 2 (human=0, bot=1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super(BotRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor — shape (batch_size, seq_len, input_size)

        Returns
        -------
        logits : torch.Tensor — shape (batch_size, num_classes)
        """
        # h0 : (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device=x.device
        )

        out, _ = self.gru(x, h0)        # (batch, seq_len, hidden_size)
        out = out[:, -1, :]             # take the last time step
        out = self.dropout(out)
        logits = self.fc(out)           # (batch, num_classes)
        return logits


# ─── Convenience functions ────────────────────────────────────────────────────

def build_model(input_size: int, hidden_size: int = 64,
                num_layers: int = 2, dropout: float = 0.3) -> BotRNN:
    """Instantiate and return a BotRNN model."""
    model = BotRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    return model


def train_one_epoch(model: BotRNN, loader, criterion, optimizer,
                    device: str = "cpu") -> float:
    """
    Run one training epoch.

    Parameters
    ----------
    model     : BotRNN
    loader    : DataLoader — yields (X_batch, y_batch)
    criterion : loss function (e.g. nn.CrossEntropyLoss())
    optimizer : torch optimizer
    device    : 'cuda' or 'cpu'

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(model: BotRNN, loader, criterion,
                   device: str = "cpu"):
    """
    Evaluate the model on a data loader.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def predict_proba(model: BotRNN, X: "torch.Tensor",
                  device: str = "cpu") -> "torch.Tensor":
    """
    Return class probabilities for a batch.

    Parameters
    ----------
    X : torch.Tensor — shape (n, seq_len, input_size)

    Returns
    -------
    proba : torch.Tensor — shape (n, 2)  columns: [P(human), P(bot)]
    """
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        logits = model(X)
        proba = torch.softmax(logits, dim=1)
    return proba
