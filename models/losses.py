"""
models/losses.py — Differentiable loss functions for portfolio optimization.
Phase 5: Model Build.

Primary loss: Negative Sharpe Ratio (differentiable)
Augmented loss: -Sharpe + λ_turnover * turnover + λ_concentration * max_weight
                + λ_entropy * (-entropy_bonus)

Transaction costs (10 bps) are included in the training loss.

Runnable standalone: python models/losses.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from utils import load_config, set_seeds


class SharpeRatioLoss(nn.Module):
    """Differentiable negative Sharpe ratio loss with penalty terms.

    L = -Sharpe + λ_turnover * mean_turnover + λ_concentration * mean_max_weight
        - λ_entropy * mean_entropy

    Where:
    - Sharpe = mean(r_portfolio) / (std(r_portfolio) + epsilon)
    - turnover = sum(|w_t - w_{t-1}|) per step
    - Transaction costs = turnover * tc_rate, subtracted from returns
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        loss_cfg = config["loss"]
        port_cfg = config["portfolio"]

        self.epsilon = loss_cfg["epsilon"]
        self.lambda_turnover = loss_cfg["lambda_turnover"]
        self.lambda_concentration = loss_cfg["lambda_concentration"]
        self.lambda_entropy = loss_cfg["lambda_entropy"]
        self.tc_rate = port_cfg["transaction_cost_bps"] / 10000.0

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the augmented Sharpe loss.

        Args:
            weights: Portfolio weights (batch, num_assets).
            returns: Asset returns for the corresponding day (batch, num_assets).
            prev_weights: Previous day's weights (batch, num_assets).
                          If None, assumes equal weight as starting point.

        Returns:
            Dict with 'loss', 'sharpe', 'turnover_penalty', 'concentration_penalty',
            'entropy_bonus', 'portfolio_return'.
        """
        batch_size, num_assets = weights.shape

        # Portfolio returns: r_p = sum(w_i * r_i)
        portfolio_returns = (weights * returns).sum(dim=-1)  # (batch,)

        # Turnover and transaction costs
        if prev_weights is None:
            prev_weights = torch.ones_like(weights) / num_assets

        turnover = torch.abs(weights - prev_weights).sum(dim=-1)  # (batch,)
        tc = turnover * self.tc_rate
        portfolio_returns_after_cost = portfolio_returns - tc

        # Sharpe ratio (over the batch as a rolling window)
        mean_ret = portfolio_returns_after_cost.mean()
        std_ret = portfolio_returns_after_cost.std() + self.epsilon
        sharpe = mean_ret / std_ret

        # Penalty: mean turnover
        turnover_penalty = turnover.mean()

        # Penalty: concentration (mean of max weight per sample)
        concentration_penalty = weights.max(dim=-1).values.mean()

        # Bonus: entropy (higher entropy = more diversified)
        log_weights = torch.log(weights + 1e-10)
        entropy = -(weights * log_weights).sum(dim=-1).mean()
        max_entropy = np.log(num_assets)
        entropy_ratio = entropy / max_entropy  # Normalized [0, 1]

        # Total loss
        loss = (
            -sharpe
            + self.lambda_turnover * turnover_penalty
            + self.lambda_concentration * concentration_penalty
            - self.lambda_entropy * entropy_ratio
        )

        return {
            "loss": loss,
            "sharpe": sharpe.detach(),
            "neg_sharpe": (-sharpe).detach(),
            "turnover_penalty": turnover_penalty.detach(),
            "concentration_penalty": concentration_penalty.detach(),
            "entropy": entropy.detach(),
            "entropy_ratio": entropy_ratio.detach(),
            "mean_return": mean_ret.detach(),
            "portfolio_return": portfolio_returns_after_cost.detach(),
        }


class MeanVarianceUtilityLoss(nn.Module):
    """Mean-variance utility loss: L = -E[r] + gamma * Var(r) + penalties.

    Unlike Sharpe, this has no degenerate gradient region near 1/N.
    The gradient always points toward higher-return, lower-variance portfolios.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        loss_cfg = config["loss"]
        port_cfg = config["portfolio"]

        self.gamma = loss_cfg.get("gamma_risk_aversion", 2.0)
        self.lambda_turnover = loss_cfg["lambda_turnover"]
        self.lambda_concentration = loss_cfg["lambda_concentration"]
        self.tc_rate = port_cfg["transaction_cost_bps"] / 10000.0

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_assets = weights.shape

        portfolio_returns = (weights * returns).sum(dim=-1)

        if prev_weights is None:
            prev_weights = torch.ones_like(weights) / num_assets

        turnover = torch.abs(weights - prev_weights).sum(dim=-1)
        tc = turnover * self.tc_rate
        portfolio_returns_after_cost = portfolio_returns - tc

        # Mean-variance utility
        mean_ret = portfolio_returns_after_cost.mean()
        var_ret = portfolio_returns_after_cost.var()
        utility = mean_ret - self.gamma * var_ret

        # Sharpe for logging (not used in loss)
        std_ret = portfolio_returns_after_cost.std() + 1e-8
        sharpe = mean_ret / std_ret

        turnover_penalty = turnover.mean()
        concentration_penalty = weights.max(dim=-1).values.mean()

        log_weights = torch.log(weights + 1e-10)
        entropy = -(weights * log_weights).sum(dim=-1).mean()
        max_entropy = np.log(num_assets)
        entropy_ratio = entropy / max_entropy

        loss = (
            -utility
            + self.lambda_turnover * turnover_penalty
            + self.lambda_concentration * concentration_penalty
        )

        return {
            "loss": loss,
            "sharpe": sharpe.detach(),
            "neg_sharpe": (-sharpe).detach(),
            "turnover_penalty": turnover_penalty.detach(),
            "concentration_penalty": concentration_penalty.detach(),
            "entropy": entropy.detach(),
            "entropy_ratio": entropy_ratio.detach(),
            "mean_return": mean_ret.detach(),
            "portfolio_return": portfolio_returns_after_cost.detach(),
        }


class ReturnMaxLoss(nn.Module):
    """Simple return maximization loss: L = -E[r_portfolio] + penalties.

    Inspired by DRL portfolio approaches that use raw portfolio value as reward.
    Gives the clearest gradient signal: "put more weight on assets with higher returns."
    Risk adjustment comes from turnover and concentration penalties.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        loss_cfg = config["loss"]
        port_cfg = config["portfolio"]

        self.lambda_turnover = loss_cfg["lambda_turnover"]
        self.lambda_concentration = loss_cfg["lambda_concentration"]
        self.tc_rate = port_cfg["transaction_cost_bps"] / 10000.0

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_assets = weights.shape

        portfolio_returns = (weights * returns).sum(dim=-1)

        if prev_weights is None:
            prev_weights = torch.ones_like(weights) / num_assets

        turnover = torch.abs(weights - prev_weights).sum(dim=-1)
        tc = turnover * self.tc_rate
        portfolio_returns_after_cost = portfolio_returns - tc

        mean_ret = portfolio_returns_after_cost.mean()
        std_ret = portfolio_returns_after_cost.std() + 1e-8
        sharpe = mean_ret / std_ret

        turnover_penalty = turnover.mean()
        concentration_penalty = weights.max(dim=-1).values.mean()

        log_weights = torch.log(weights + 1e-10)
        entropy = -(weights * log_weights).sum(dim=-1).mean()
        max_entropy = np.log(num_assets)
        entropy_ratio = entropy / max_entropy

        loss = (
            -mean_ret
            + self.lambda_turnover * turnover_penalty
            + self.lambda_concentration * concentration_penalty
        )

        return {
            "loss": loss,
            "sharpe": sharpe.detach(),
            "neg_sharpe": (-sharpe).detach(),
            "turnover_penalty": turnover_penalty.detach(),
            "concentration_penalty": concentration_penalty.detach(),
            "entropy": entropy.detach(),
            "entropy_ratio": entropy_ratio.detach(),
            "mean_return": mean_ret.detach(),
            "portfolio_return": portfolio_returns_after_cost.detach(),
        }


class MultiTaskLoss(nn.Module):
    """Wraps a portfolio loss with auxiliary return + volatility prediction losses.

    Total loss = portfolio_loss + alpha_return * MSE(pred_returns, actual_returns)
                 + alpha_vol * MSE(pred_vol, |actual_returns|)

    The auxiliary losses give the backbone meaningful gradient signal about
    which features are predictive, preventing collapse to 1/N.
    """

    def __init__(self, base_loss: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.base_loss = base_loss
        loss_cfg = config["loss"]
        self.alpha_return = loss_cfg.get("alpha_return_pred", 0.5)
        self.alpha_vol = loss_cfg.get("alpha_vol_pred", 0.5)

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        pred_returns: Optional[torch.Tensor] = None,
        pred_vol: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Base portfolio loss
        result = self.base_loss(weights, returns, prev_weights)

        # Auxiliary: return prediction
        if pred_returns is not None:
            return_loss = nn.functional.mse_loss(pred_returns, returns)
            result["return_pred_loss"] = return_loss.detach()
            result["loss"] = result["loss"] + self.alpha_return * return_loss

        # Auxiliary: volatility prediction (target = |returns|)
        if pred_vol is not None:
            vol_target = returns.abs()
            vol_loss = nn.functional.mse_loss(pred_vol, vol_target)
            result["vol_pred_loss"] = vol_loss.detach()
            result["loss"] = result["loss"] + self.alpha_vol * vol_loss

        return result


if __name__ == "__main__":
    config = load_config()
    set_seeds(config)

    num_assets = 30
    batch_size = 64

    loss_fn = SharpeRatioLoss(config)

    # Test with random data
    weights = torch.softmax(torch.randn(batch_size, num_assets), dim=-1)
    returns = torch.randn(batch_size, num_assets) * 0.02
    prev_weights = torch.ones(batch_size, num_assets) / num_assets

    result = loss_fn(weights, returns, prev_weights)

    print("Loss function test:")
    for k, v in result.items():
        if v.dim() == 0:
            print(f"  {k:25s}: {v.item():.6f}")
        else:
            print(f"  {k:25s}: shape {tuple(v.shape)}, mean {v.mean().item():.6f}")

    # Verify gradient flows
    weights.requires_grad_(True)
    result2 = loss_fn(weights, returns, prev_weights)
    result2["loss"].backward()
    print(f"\n  Gradient flows: {weights.grad is not None}")
    print(f"  Grad norm: {weights.grad.norm().item():.6f}")
