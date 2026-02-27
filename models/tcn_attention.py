"""
models/tcn_attention.py — Temporal Convolutional Network with Attention.
Phase 5: Model Build.

Architecture:
  Input: (batch, lookback=63, num_assets * num_features)
  TCN Block: 4 dilated causal conv layers [1,2,4,8], kernel=3, GELU, residual
  Attention: Single-head self-attention over temporal dimension
  Portfolio Head: Dense(128) -> GELU -> Dropout -> Dense(num_assets) -> Softmax
  Output: Weight vector w_t with w_i >= 0, sum(w) = 1, max(w) <= 0.30

Runnable standalone: python models/tcn_attention.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_config, set_seeds


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation. Ensures no future leakage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x shape: (batch, channels, seq_len)."""
        out = self.conv(x)
        # Remove future padding (causal: only use past)
        out = out[:, :, :x.size(2)]
        out = self.activation(out)
        out = self.dropout(out)
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block: two causal convolutions + residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)

        # Residual projection if channel dimensions differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, channels, seq_len)."""
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.gelu(out + residual)


class TemporalAttention(nn.Module):
    """Single-head self-attention over the temporal dimension.

    Produces an attention-weighted summary of the lookback window.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq_len, hidden_dim).

        Returns: (batch, hidden_dim) — attention-weighted summary.
        """
        Q = self.query(x)  # (batch, seq, hidden)
        K = self.key(x)
        V = self.value(x)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq, seq)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted values
        context = torch.bmm(attn_weights, V)  # (batch, seq, hidden)

        # Take the last time step's context as the summary
        output = context[:, -1, :]  # (batch, hidden)

        return output, attn_weights


class TCNAttentionModel(nn.Module):
    """TCN + Attention portfolio allocation model.

    Input: (batch, lookback, input_dim)
    Output: (batch, num_assets) weight vector
    """

    def __init__(self, config: Dict[str, Any], num_assets: int, input_dim: int):
        super().__init__()
        self.num_assets = num_assets
        self.config = config

        tcn_cfg = config["model"]["tcn"]
        attn_cfg = config["model"]["attention"]
        head_cfg = config["model"]["portfolio_head"]

        hidden = tcn_cfg["hidden_channels"]
        kernel_size = tcn_cfg["kernel_size"]
        dilations = tcn_cfg["dilation_factors"]
        dropout = tcn_cfg["dropout"]
        self.max_weight = config["model"]["max_weight"]

        # Learnable logit gain — amplifies differences between assets
        # Higher gain = more concentrated weights; init from 1/temperature
        init_temp = config["model"]["softmax_temperature"]
        self._log_gain = nn.Parameter(torch.tensor(math.log(max(1.0 / init_temp, 1.0))))

        # TCN layers
        self.tcn_blocks = nn.ModuleList()
        in_ch = input_dim
        for d in dilations:
            self.tcn_blocks.append(TCNBlock(in_ch, hidden, kernel_size, d, dropout))
            in_ch = hidden

        # Attention
        self.attention = TemporalAttention(hidden)

        # Portfolio head
        self.head = nn.Sequential(
            nn.Linear(hidden, head_cfg["hidden_units"]),
            nn.GELU(),
            nn.Dropout(head_cfg["dropout"]),
            nn.Linear(head_cfg["hidden_units"], num_assets),
        )

        # Auxiliary heads for multi-task learning
        self.return_head = nn.Sequential(
            nn.Linear(hidden, head_cfg["hidden_units"]),
            nn.GELU(),
            nn.Dropout(head_cfg["dropout"]),
            nn.Linear(head_cfg["hidden_units"], num_assets),
        )
        self.vol_head = nn.Sequential(
            nn.Linear(hidden, head_cfg["hidden_units"]),
            nn.GELU(),
            nn.Dropout(head_cfg["dropout"]),
            nn.Linear(head_cfg["hidden_units"], num_assets),
        )

    @property
    def logit_gain(self) -> torch.Tensor:
        """Learnable gain for logit amplification, constrained >= 1.0."""
        return torch.exp(self._log_gain).clamp(min=1.0)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, lookback, input_dim).
            return_attention: If True, also return attention weights.

        Returns:
            weights: Portfolio weights (batch, num_assets), sum=1, all >= 0.
            Also stores pred_returns and pred_vol as attributes for multi-task loss.
            attn_weights: (optional) Attention weight matrix.
        """
        # TCN expects (batch, channels, seq_len)
        out = x.transpose(1, 2)  # (batch, input_dim, lookback)

        for block in self.tcn_blocks:
            out = block(out)  # (batch, hidden, lookback)

        # Attention expects (batch, seq_len, hidden)
        out = out.transpose(1, 2)  # (batch, lookback, hidden)
        summary, attn_weights = self.attention(out)  # (batch, hidden)

        # Portfolio head
        logits = self.head(summary)  # (batch, num_assets)

        # Auxiliary predictions (multi-task)
        self.pred_returns = self.return_head(summary)  # (batch, num_assets)
        self.pred_vol = self.vol_head(summary)  # (batch, num_assets)

        # Sigmoid + normalize (inspired by DRL portfolio approach)
        # Each asset gets an independent sigmoid score, then we normalize to sum=1.
        # Unlike softmax, sigmoid gives each asset ~0.25 gradient at center
        # vs softmax's ~1/N = 0.033 for 30 assets — 7.5x stronger signal.
        gain = self.logit_gain
        raw_weights = torch.sigmoid(gain * logits)
        weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

        # Clamp max weight and renormalize
        weights = self._enforce_max_weight(weights)

        if return_attention:
            return weights, attn_weights
        return weights

    def _enforce_max_weight(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp maximum single-asset weight and renormalize.

        Differentiable: uses clamp which has straight-through gradient.
        """
        clamped = torch.clamp(weights, max=self.max_weight)
        # Renormalize to sum to 1
        return clamped / clamped.sum(dim=-1, keepdim=True)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_tcn_model(config: Dict[str, Any], num_assets: int, input_dim: int) -> TCNAttentionModel:
    """Factory function to build the TCN+Attention model from config."""
    model = TCNAttentionModel(config, num_assets, input_dim)
    return model


if __name__ == "__main__":
    config = load_config()
    set_seeds(config)

    num_assets = 30
    num_features = 24
    input_dim = num_assets * num_features  # 720
    lookback = config["features"]["lookback_window"]  # 63

    model = build_tcn_model(config, num_assets, input_dim)
    print(f"TCN+Attention model built.")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Input: (batch, {lookback}, {input_dim})")
    print(f"  Output: (batch, {num_assets})")

    # Test forward pass
    x = torch.randn(4, lookback, input_dim)
    weights, attn = model(x, return_attention=True)
    print(f"  Test output shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum(dim=-1)}")
    print(f"  Weights min: {weights.min().item():.4f}, max: {weights.max().item():.4f}")
    print(f"  Attention shape: {attn.shape}")
    assert (weights >= 0).all(), "Negative weights!"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4)), "Weights don't sum to 1!"
    assert (weights <= config['model']['max_weight'] + 1e-6).all(), "Weight exceeds max!"
    print("  All constraints satisfied.")
