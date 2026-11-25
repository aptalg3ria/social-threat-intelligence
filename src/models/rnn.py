"""
src/models/rnn.py
------------------
Recurrent Neural Network (GRU-based) for Twitter bot detection using PyTorch.

Architecture:
  - Input: (batch_size, seq_len=5, step_size=3)
  - Stacked GRU layers (hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
  - Fully-connected classification head: 64 -> 32 -> 2
"""

import torch
import torch.nn as nn


class BotRNN(nn.Module):
    """
    Gated Recurrent Unit (GRU) classifier for social media bot detection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
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

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, seq_len, input_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        last_step = out[:, -1, :]
        logits = self.head(last_step)
        return logits


def build_model(input_size: int, hidden_size: int = 64,
                num_layers: int = 2, dropout: float = 0.2) -> BotRNN:
    """Instantiate and return a BotRNN model."""
    return BotRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def predict_proba(model: BotRNN, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Return softmax probabilities for a batch tensor (P(human), P(bot))."""
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        logits = model(X)
        proba = torch.softmax(logits, dim=1)
    return proba
