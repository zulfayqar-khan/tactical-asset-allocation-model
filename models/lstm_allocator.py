"""
models/lstm_allocator.py — LSTM-based portfolio allocator.
Phase 5: Model Build.

Architecture:
  2-layer LSTM, hidden=128, dropout=0.2
  Final hidden state -> Dense(128) -> GELU -> Dropout(0.3) -> Dense(num_assets) -> Softmax

Serves as comparison to validate whether TCN architecture adds value.

Runnable standalone: python models/lstm_allocator.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import math

import torch
import torch.nn as nn
from utils import load_config, set_seeds


class LSTMAllocator(nn.Module):
    """LSTM-based portfolio allocation model.

    Input: (batch, lookback, input_dim)
    Output: (batch, num_assets) weight vector
    """

    def __init__(self, config: Dict[str, Any], num_assets: int, input_dim: int):
        super().__init__()
        self.num_assets = num_assets

        lstm_cfg = config["model"]["lstm"]
        self.max_weight = config["model"]["max_weight"]

        # Learnable logit gain — amplifies differences between assets
        init_temp = config["model"]["softmax_temperature"]
        self._log_gain = nn.Parameter(torch.tensor(math.log(max(1.0 / init_temp, 1.0))))

        hidden_size = lstm_cfg["hidden_size"]
        head_hidden = lstm_cfg["head_hidden"]
        head_dropout = lstm_cfg["head_dropout"]

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_assets),
        )

        # Auxiliary heads for multi-task learning
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_assets),
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_assets),
        )

    @property
    def logit_gain(self) -> torch.Tensor:
        """Learnable gain for logit amplification, constrained >= 1.0."""
        return torch.exp(self._log_gain).clamp(min=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, lookback, input_dim).

        Returns:
            weights: Portfolio weights (batch, num_assets).
        """
        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)

        # Use final hidden state of last layer
        final_hidden = h_n[-1]  # (batch, hidden_size)

        # Portfolio head
        logits = self.head(final_hidden)  # (batch, num_assets)

        # Auxiliary predictions (multi-task)
        self.pred_returns = self.return_head(final_hidden)
        self.pred_vol = self.vol_head(final_hidden)

        # Sigmoid + normalize (each asset gets independent gradient)
        gain = self.logit_gain
        raw_weights = torch.sigmoid(gain * logits)
        weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
        weights = self._enforce_max_weight(weights)

        return weights

    def _enforce_max_weight(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp max weight and renormalize."""
        clamped = torch.clamp(weights, max=self.max_weight)
        return clamped / clamped.sum(dim=-1, keepdim=True)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm_model(config: Dict[str, Any], num_assets: int, input_dim: int) -> LSTMAllocator:
    """Factory function to build the LSTM model from config."""
    return LSTMAllocator(config, num_assets, input_dim)


if __name__ == "__main__":
    config = load_config()
    set_seeds(config)

    num_assets = 30
    num_features = 24
    input_dim = num_assets * num_features
    lookback = config["features"]["lookback_window"]

    model = build_lstm_model(config, num_assets, input_dim)
    print(f"LSTM Allocator built.")
    print(f"  Parameters: {model.count_parameters():,}")

    x = torch.randn(4, lookback, input_dim)
    weights = model(x)
    print(f"  Test output shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum(dim=-1)}")
    print(f"  Weights min: {weights.min().item():.4f}, max: {weights.max().item():.4f}")
    assert (weights >= 0).all() and torch.allclose(weights.sum(dim=-1), torch.ones(4))
    print("  All constraints satisfied.")
