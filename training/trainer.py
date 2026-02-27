"""
training/trainer.py — Training loop with early stopping and logging.
Phase 5: Model Build.

Handles:
- Dataset creation from feature tensors and return data
- Training loop with gradient clipping
- Validation with Sharpe monitoring
- Early stopping based on validation Sharpe
- Weight behavior monitoring (entropy, concentration)

Runnable standalone: python training/trainer.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import load_config, set_seeds, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class PortfolioDataset(Dataset):
    """Dataset for portfolio allocation training.

    Each sample contains:
    - x: Feature tensor (lookback, input_dim) for time t
    - returns: Next-day asset returns (num_assets,) — what we optimize against
    - prev_returns: Current-day returns for drift computation
    """

    def __init__(
        self,
        X: np.ndarray,
        returns_wide: pd.DataFrame,
        sample_dates: np.ndarray,
        tickers: List[str],
        forward_days: int = 5,
    ):
        """
        Args:
            X: Feature tensor (num_samples, lookback, input_dim).
            returns_wide: Wide-format returns DataFrame (dates x tickers).
            sample_dates: Array of dates corresponding to each X sample.
            tickers: Ordered list of tickers matching returns_wide columns.
            forward_days: Number of forward trading days to accumulate returns.
                Uses cumulative (compounded) returns over the window.
                Higher values amplify the cross-asset signal relative to noise.
                Default 5 (one trading week) gives ~2.2x SNR improvement.
        """
        self.X = torch.tensor(X, dtype=torch.float32)

        # Align returns with sample dates
        # For each sample at date t, accumulate returns over t+1 to t+forward_days
        # (the model predicts weights at t, executes at t+1 open — no look-ahead)
        returns_aligned = []
        valid_indices = []

        returns_index = returns_wide.index.sort_values()
        returns_dates_set = set(returns_index)
        for i, date in enumerate(sample_dates):
            date_ts = pd.Timestamp(date)
            if date_ts not in returns_dates_set:
                continue
            pos = returns_index.searchsorted(date_ts)
            if pos + forward_days >= len(returns_index):
                continue
            # Cumulative return over next forward_days:
            # (1+r1)*(1+r2)*...*(1+rK) - 1
            cum_ret = np.ones(len(tickers), dtype=np.float64)
            all_valid = True
            for d in range(1, forward_days + 1):
                day_ret = returns_wide.loc[returns_index[pos + d], tickers].values
                if np.any(np.isnan(day_ret)):
                    all_valid = False
                    break
                cum_ret *= (1.0 + day_ret)
            if all_valid:
                returns_aligned.append((cum_ret - 1.0).astype(np.float32))
                valid_indices.append(i)

        self.returns = torch.tensor(
            np.array(returns_aligned), dtype=torch.float32
        )
        self.X = self.X[valid_indices]

        # Store dates for reference
        self.dates = sample_dates[valid_indices]

        logger.info(
            "Dataset: %d samples, input shape %s, returns shape %s, forward_days=%d",
            len(self.X), tuple(self.X.shape[1:]), tuple(self.returns.shape[1:]),
            forward_days,
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.returns[idx]


def create_datasets(
    config: Dict[str, Any],
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> Tuple[PortfolioDataset, PortfolioDataset]:
    """Create train and validation datasets for a given date split.

    Args:
        config: Full config dict.
        train_start: Training period start date.
        train_end: Training period end date.
        val_start: Validation period start date.
        val_end: Validation period end date.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]

    # Load pre-built tensors
    X = np.load(processed_path / "X_tensor.npy")
    sample_dates = np.load(processed_path / "sample_dates.npy", allow_pickle=True)

    # Load returns
    asset_data = pd.read_parquet(processed_path / "asset_data.parquet")
    returns_wide = asset_data.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )
    tickers = sorted(returns_wide.columns.tolist())
    returns_wide = returns_wide[tickers]

    # Split by date — convert to DatetimeIndex for safe comparison
    sample_dates_idx = pd.DatetimeIndex(sample_dates)
    train_mask = np.array(
        (sample_dates_idx >= pd.Timestamp(train_start)) &
        (sample_dates_idx <= pd.Timestamp(train_end))
    )
    val_mask = np.array(
        (sample_dates_idx >= pd.Timestamp(val_start)) &
        (sample_dates_idx <= pd.Timestamp(val_end))
    )

    logger.info("Train samples: %d, Val samples: %d", train_mask.sum(), val_mask.sum())

    forward_days = config["training"].get("forward_days", 5)
    train_ds = PortfolioDataset(X[train_mask], returns_wide, sample_dates[train_mask], tickers, forward_days)
    val_ds = PortfolioDataset(X[val_mask], returns_wide, sample_dates[val_mask], tickers, forward_days)

    return train_ds, val_ds


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Portfolio allocation model.
        dataloader: Training DataLoader.
        loss_fn: SharpeRatioLoss instance.
        optimizer: Optimizer.
        grad_clip: Max gradient norm for clipping.

    Returns:
        Dict of averaged metrics for the epoch.
    """
    model.train()
    epoch_metrics = {
        "loss": 0.0, "sharpe": 0.0, "turnover": 0.0,
        "concentration": 0.0, "entropy_ratio": 0.0,
    }
    n_batches = 0

    prev_weights = None

    for X_batch, returns_batch in dataloader:
        optimizer.zero_grad()

        weights = model(X_batch)

        # Pass auxiliary predictions if model has them and loss supports them
        loss_kwargs = {}
        if hasattr(model, 'pred_returns') and hasattr(loss_fn, 'alpha_return'):
            loss_kwargs['pred_returns'] = model.pred_returns
        if hasattr(model, 'pred_vol') and hasattr(loss_fn, 'alpha_vol'):
            loss_kwargs['pred_vol'] = model.pred_vol

        result = loss_fn(weights, returns_batch, prev_weights, **loss_kwargs)
        loss = result["loss"]

        # Check for NaN
        if torch.isnan(loss):
            logger.warning("NaN loss detected — skipping batch")
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track previous weights for turnover computation
        prev_weights = weights.detach()

        # Accumulate metrics
        epoch_metrics["loss"] += result["loss"].item()
        epoch_metrics["sharpe"] += result["sharpe"].item()
        epoch_metrics["turnover"] += result["turnover_penalty"].item()
        epoch_metrics["concentration"] += result["concentration_penalty"].item()
        epoch_metrics["entropy_ratio"] += result["entropy_ratio"].item()
        n_batches += 1

    if n_batches > 0:
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

    return epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    """Validate the model.

    Processes all validation data sequentially (simulating time series).

    Args:
        model: Portfolio allocation model.
        dataloader: Validation DataLoader.
        loss_fn: SharpeRatioLoss instance.

    Returns:
        Dict of averaged validation metrics.
    """
    model.eval()
    all_returns = []
    all_weights = []
    all_pred_returns = []
    all_pred_vol = []

    for X_batch, returns_batch in dataloader:
        weights = model(X_batch)
        all_weights.append(weights)
        all_returns.append(returns_batch)
        if hasattr(model, 'pred_returns'):
            all_pred_returns.append(model.pred_returns)
        if hasattr(model, 'pred_vol'):
            all_pred_vol.append(model.pred_vol)

    if not all_weights:
        return {"loss": float("inf"), "sharpe": 0.0, "turnover": 0.0,
                "concentration": 0.0, "entropy_ratio": 0.0}

    # Concatenate all batches and compute metrics over full validation period
    all_weights = torch.cat(all_weights, dim=0)
    all_returns_t = torch.cat(all_returns, dim=0)

    # Compute sequential turnover
    prev_w = torch.ones(1, all_weights.shape[1]) / all_weights.shape[1]
    shifted_weights = torch.cat([prev_w, all_weights[:-1]], dim=0)

    # Pass auxiliary predictions if available
    loss_kwargs = {}
    if all_pred_returns and hasattr(loss_fn, 'alpha_return'):
        loss_kwargs['pred_returns'] = torch.cat(all_pred_returns, dim=0)
    if all_pred_vol and hasattr(loss_fn, 'alpha_vol'):
        loss_kwargs['pred_vol'] = torch.cat(all_pred_vol, dim=0)

    result = loss_fn(all_weights, all_returns_t, shifted_weights, **loss_kwargs)

    metrics = {
        "loss": result["loss"].item(),
        "sharpe": result["sharpe"].item(),
        "turnover": result["turnover_penalty"].item(),
        "concentration": result["concentration_penalty"].item(),
        "entropy_ratio": result["entropy_ratio"].item(),
    }

    return metrics


def train_model(
    model: nn.Module,
    train_ds: PortfolioDataset,
    val_ds: PortfolioDataset,
    loss_fn: nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Full training loop with early stopping.

    Args:
        model: Portfolio allocation model.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        loss_fn: Loss function.
        config: Full config dict.

    Returns:
        Dict with 'best_model_state', 'train_history', 'val_history',
        'best_epoch', 'best_val_sharpe'.
    """
    train_cfg = config["training"]
    batch_size = train_cfg["batch_size"]
    max_epochs = train_cfg["max_epochs"]
    patience = train_cfg["early_stopping_patience"]
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    grad_clip = train_cfg["grad_clip_max_norm"]

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_cfg = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_cfg["T_max"], eta_min=sched_cfg["eta_min"],
    )

    # Training state
    best_val_sharpe = -float("inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, grad_clip)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn)

        # Step scheduler
        scheduler.step()

        # Record history
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        # Log progress
        logger.info(
            "Epoch %3d/%d | Train Loss: %.4f Sharpe: %+.3f | "
            "Val Loss: %.4f Sharpe: %+.3f | Entropy: %.3f | "
            "LR: %.1e | %.1fs (total %.0fs)",
            epoch, max_epochs,
            train_metrics["loss"], train_metrics["sharpe"],
            val_metrics["loss"], val_metrics["sharpe"],
            val_metrics["entropy_ratio"],
            optimizer.param_groups[0]["lr"],
            elapsed, total_elapsed,
        )

        # Early stopping on validation Sharpe
        if val_metrics["sharpe"] > best_val_sharpe:
            best_val_sharpe = val_metrics["sharpe"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                "Early stopping at epoch %d. Best val Sharpe: %.4f at epoch %d.",
                epoch, best_val_sharpe, best_epoch,
            )
            break

        # Check for training Sharpe overfit
        if train_metrics["sharpe"] > 3.0 and val_metrics["sharpe"] < 0.5:
            logger.warning(
                "WARNING: Possible overfit — train Sharpe %.3f >> val Sharpe %.3f",
                train_metrics["sharpe"], val_metrics["sharpe"],
            )

        # Check for time budget (CPU kill-switch)
        max_seconds = train_cfg.get("max_seconds_per_run", 1200)
        if total_elapsed > max_seconds:
            logger.warning("Time budget exceeded (%.0fs > %ds). Stopping.", total_elapsed, max_seconds)
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    logger.info(
        "Training complete. Best val Sharpe: %.4f at epoch %d. Total time: %.0fs",
        best_val_sharpe, best_epoch, total_time,
    )

    return {
        "best_model_state": best_model_state,
        "train_history": train_history,
        "val_history": val_history,
        "best_epoch": best_epoch,
        "best_val_sharpe": best_val_sharpe,
        "total_time": total_time,
    }


if __name__ == "__main__":
    config = load_config()
    set_seeds(config)
    ensure_directories(config)

    from models.tcn_attention import build_tcn_model
    from models.losses import SharpeRatioLoss, MeanVarianceUtilityLoss, ReturnMaxLoss, MultiTaskLoss

    splits = config["splits"]

    # Create datasets for initial train/val split
    train_ds, val_ds = create_datasets(
        config,
        train_start=splits["initial_train"]["start"],
        train_end=splits["initial_train"]["end"],
        val_start=splits["validation"]["start"],
        val_end=splits["validation"]["end"],
    )

    # Build model
    num_assets = 30
    input_dim = train_ds.X.shape[-1]
    model = build_tcn_model(config, num_assets, input_dim)
    logger.info("Model parameters: %d", model.count_parameters())

    # Build loss
    loss_type = config["loss"].get("type", "sharpe")
    if loss_type == "mv_utility":
        base_loss = MeanVarianceUtilityLoss(config)
    elif loss_type == "return_max":
        base_loss = ReturnMaxLoss(config)
    else:
        base_loss = SharpeRatioLoss(config)
    loss_fn = MultiTaskLoss(base_loss, config)

    # Train
    result = train_model(model, train_ds, val_ds, loss_fn, config)

    # Verify output weights
    model.eval()
    with torch.no_grad():
        sample_x = train_ds.X[:8]
        weights = model(sample_x)
        logger.info("Sample weights shape: %s", weights.shape)
        logger.info("Weights sum: %s", weights.sum(dim=-1).numpy())
        logger.info("Weights min: %.4f, max: %.4f", weights.min().item(), weights.max().item())
        logger.info("Weight entropy ratio: %.4f",
                     (-(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean() / np.log(num_assets)).item())
