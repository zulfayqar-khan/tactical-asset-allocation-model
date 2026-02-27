"""
features/engineering.py — Feature computation for all 4 feature groups.
Phase 3: Feature Engineering.

Groups:
  A — Price-based (per asset): log returns, SMA ratios, Bollinger, RSI, MACD
  B — Volatility (per asset): realized vol, Garman-Klass, vol ratio
  C — Cross-sectional: rank, z-score, correlation PCA
  D — Market regime (shared): VIX, S&P drawdown, yield curve slope, breadth

All features use strictly backward-looking windows (no look-ahead bias).
Rolling normalization uses a 252-day lookback, clipped to [-3, +3].

Runnable standalone: python features/engineering.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils import load_config, set_seeds, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# GROUP A — Price-Based Features (per asset)
# ============================================================================

def compute_log_returns(
    prices: pd.Series,
    windows: List[int],
) -> pd.DataFrame:
    """Multi-period log returns from adjusted close prices.

    Args:
        prices: Series of adjusted close prices for one ticker.
        windows: List of return periods (e.g., [1, 5, 21]).

    Returns:
        DataFrame with columns like 'logret_1', 'logret_5', etc.
    """
    result = {}
    for w in windows:
        result[f"logret_{w}"] = np.log(prices / prices.shift(w))
    return pd.DataFrame(result, index=prices.index)


def compute_sma_ratios(
    prices: pd.Series,
    windows: List[int],
) -> pd.DataFrame:
    """Price relative to simple moving averages.

    Args:
        prices: Series of adjusted close prices.
        windows: SMA window lengths (e.g., [20, 50, 200]).

    Returns:
        DataFrame with columns like 'sma_ratio_20', etc.
    """
    result = {}
    for w in windows:
        sma = prices.rolling(window=w, min_periods=w).mean()
        result[f"sma_ratio_{w}"] = prices / sma
    return pd.DataFrame(result, index=prices.index)


def compute_bollinger_position(
    prices: pd.Series,
    window: int,
    num_std: float,
) -> pd.Series:
    """Bollinger Band position: (close - lower) / (upper - lower).

    Args:
        prices: Adjusted close prices.
        window: Bollinger Band rolling window.
        num_std: Number of standard deviations for bands.

    Returns:
        Series of Bollinger position values in [0, 1] range (approximately).
    """
    sma = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    position = (prices - lower) / band_width.replace(0, np.nan)
    return position.rename("bollinger_pos")


def compute_rsi(prices: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index (RSI).

    Args:
        prices: Adjusted close prices.
        window: RSI lookback period.

    Returns:
        Series of RSI values in [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


def compute_macd_crossover(
    prices: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> pd.Series:
    """MACD signal line crossover (binary: 1 if MACD > signal, else 0).

    Args:
        prices: Adjusted close prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Binary series (1 = bullish crossover, 0 = bearish).
    """
    ema_fast = prices.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = prices.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    crossover = (macd_line > signal_line).astype(float)
    return crossover.rename("macd_crossover")


def compute_group_a(
    ticker_data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute all Group A (price-based) features for one ticker.

    Args:
        ticker_data: DataFrame with AdjClose column, indexed by Date.
        config: Feature config section from config.yaml.

    Returns:
        DataFrame of Group A features indexed by Date.
    """
    prices = ticker_data["AdjClose"]
    feat_cfg = config["features"]

    parts = []
    parts.append(compute_log_returns(prices, feat_cfg["returns_windows"]))
    parts.append(compute_sma_ratios(prices, feat_cfg["sma_windows"]))
    parts.append(compute_bollinger_position(
        prices, feat_cfg["bollinger_window"], feat_cfg["bollinger_std"],
    ).to_frame())
    parts.append(compute_rsi(prices, feat_cfg["rsi_window"]).to_frame())
    parts.append(compute_macd_crossover(
        prices, feat_cfg["macd_fast"], feat_cfg["macd_slow"], feat_cfg["macd_signal"],
    ).to_frame())

    return pd.concat(parts, axis=1)


# ============================================================================
# GROUP B — Volatility Features (per asset)
# ============================================================================

def compute_realized_volatility(
    log_returns: pd.Series,
    windows: List[int],
) -> pd.DataFrame:
    """Rolling realized volatility (annualized std of log returns).

    Args:
        log_returns: Daily log return series.
        windows: Rolling window lengths (e.g., [21, 63]).

    Returns:
        DataFrame with columns like 'realized_vol_21', etc.
    """
    result = {}
    for w in windows:
        result[f"realized_vol_{w}"] = log_returns.rolling(
            window=w, min_periods=w
        ).std() * np.sqrt(252)
    return pd.DataFrame(result, index=log_returns.index)


def compute_garman_klass(ticker_data: pd.DataFrame, window: int = 21) -> pd.Series:
    """Garman-Klass volatility estimator using OHLC data.

    More efficient than close-to-close volatility as it uses intraday range.

    Args:
        ticker_data: DataFrame with Open, High, Low, Close columns.
        window: Rolling window for averaging.

    Returns:
        Series of Garman-Klass volatility estimates (annualized).
    """
    log_hl = np.log(ticker_data["High"] / ticker_data["Low"])
    log_co = np.log(ticker_data["Close"] / ticker_data["Open"])

    # GK estimator: 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk_vol = np.sqrt(gk.rolling(window=window, min_periods=window).mean() * 252)
    return gk_vol.rename("garman_klass_vol")


def compute_group_b(
    ticker_data: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute all Group B (volatility) features for one ticker.

    Args:
        ticker_data: DataFrame with OHLC + AdjClose, indexed by Date.
        config: Feature config section.

    Returns:
        DataFrame of Group B features indexed by Date.
    """
    feat_cfg = config["features"]
    log_ret_1d = np.log(ticker_data["AdjClose"] / ticker_data["AdjClose"].shift(1))

    short_w = feat_cfg["vol_short_window"]
    long_w = feat_cfg["vol_long_window"]

    parts = []
    vol_df = compute_realized_volatility(log_ret_1d, [short_w, long_w])
    parts.append(vol_df)
    parts.append(compute_garman_klass(ticker_data, short_w).to_frame())

    # Vol ratio: short-term / long-term (regime indicator)
    vol_ratio = vol_df[f"realized_vol_{short_w}"] / vol_df[f"realized_vol_{long_w}"].replace(0, np.nan)
    parts.append(vol_ratio.rename("vol_ratio").to_frame())

    return pd.concat(parts, axis=1)


# ============================================================================
# GROUP C — Cross-Sectional Features
# ============================================================================

def compute_cross_sectional_features(
    returns_wide: pd.DataFrame,
    window: int,
) -> Dict[str, pd.DataFrame]:
    """Compute cross-sectional rank and z-score of rolling returns.

    Args:
        returns_wide: DataFrame with dates as index, tickers as columns,
                      values are simple daily returns.
        window: Rolling window for return aggregation.

    Returns:
        Dict with 'rank' and 'zscore' DataFrames (same shape as input).
    """
    rolling_ret = returns_wide.rolling(window=window, min_periods=window).sum()

    # Cross-sectional rank (0 to 1 range)
    cs_rank = rolling_ret.rank(axis=1, pct=True)

    # Cross-sectional z-score
    cs_mean = rolling_ret.mean(axis=1)
    cs_std = rolling_ret.std(axis=1).replace(0, np.nan)
    cs_zscore = rolling_ret.sub(cs_mean, axis=0).div(cs_std, axis=0)

    return {"rank": cs_rank, "zscore": cs_zscore}


def compute_correlation_pca(
    returns_wide: pd.DataFrame,
    corr_window: int,
    n_components: int,
) -> pd.DataFrame:
    """Rolling pairwise correlation matrix compressed via PCA.

    For each date, compute the rolling correlation matrix of asset returns,
    flatten the upper triangle, and project onto PCA components fit on
    backward-looking data only.

    Args:
        returns_wide: DataFrame (dates x tickers) of daily returns.
        corr_window: Rolling window for correlation estimation.
        n_components: Number of PCA components to keep.

    Returns:
        DataFrame with n_components columns (shared across all assets).
    """
    dates = returns_wide.index
    n_assets = returns_wide.shape[1]
    n_upper = n_assets * (n_assets - 1) // 2

    # Pre-compute upper triangle indices
    upper_idx = np.triu_indices(n_assets, k=1)

    # Compute rolling correlation and extract upper triangle
    corr_features = np.full((len(dates), n_upper), np.nan)

    for i in range(corr_window, len(dates)):
        window_data = returns_wide.iloc[i - corr_window:i]
        corr_matrix = window_data.corr().values
        corr_features[i] = corr_matrix[upper_idx]

    corr_df = pd.DataFrame(corr_features, index=dates)

    # Apply PCA using expanding window (backward-looking only)
    pca_result = np.full((len(dates), n_components), np.nan)
    min_samples = corr_window + 50  # Need enough samples for stable PCA

    for i in range(min_samples, len(dates)):
        valid_data = corr_df.iloc[:i + 1].dropna()
        if len(valid_data) < n_components + 1:
            continue
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(valid_data.values)
        # Transform only the current row
        current = corr_df.iloc[i:i + 1].values
        if not np.any(np.isnan(current)):
            pca_result[i] = pca.transform(current)[0]

    columns = [f"corr_pca_{j}" for j in range(n_components)]
    return pd.DataFrame(pca_result, index=dates, columns=columns)


# ============================================================================
# GROUP D — Market Regime Features (shared across all assets)
# ============================================================================

def compute_group_d(
    market_data: pd.DataFrame,
    asset_returns_wide: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute all Group D (market regime) features.

    Args:
        market_data: DataFrame indexed by Date with columns: spy, vix, tnx, irx.
        asset_returns_wide: Wide-format daily returns (dates x tickers).
        config: Feature config section.

    Returns:
        DataFrame of Group D features indexed by Date.
    """
    feat_cfg = config["features"]
    parts = []

    # VIX level (or substitute: rolling 21-day vol of SPY)
    if "vix" in market_data.columns:
        parts.append(market_data["vix"].rename("vix_level").to_frame())
    else:
        spy_ret = np.log(market_data["spy"] / market_data["spy"].shift(1))
        vix_proxy = spy_ret.rolling(21, min_periods=21).std() * np.sqrt(252) * 100
        parts.append(vix_proxy.rename("vix_level").to_frame())

    # S&P 500 drawdown from rolling peak
    spy = market_data["spy"]
    rolling_peak = spy.expanding(min_periods=1).max()
    drawdown = (spy - rolling_peak) / rolling_peak
    parts.append(drawdown.rename("spy_drawdown").to_frame())

    # Yield curve slope: 10Y - 3M
    if "tnx" in market_data.columns and "irx" in market_data.columns:
        yield_slope = market_data["tnx"] - market_data["irx"]
        parts.append(yield_slope.rename("yield_curve_slope").to_frame())

    # Market breadth: fraction of DJIA stocks with positive 21-day returns
    breadth_window = feat_cfg["market_breadth_window"]
    rolling_rets = asset_returns_wide.rolling(
        window=breadth_window, min_periods=breadth_window
    ).sum()
    breadth = (rolling_rets > 0).mean(axis=1)
    parts.append(breadth.rename("market_breadth").to_frame())

    return pd.concat(parts, axis=1)


# ============================================================================
# ROLLING NORMALIZATION
# ============================================================================

def rolling_normalize(
    df: pd.DataFrame,
    window: int,
    clip_range: float,
) -> pd.DataFrame:
    """Z-score normalize using a backward-looking rolling window.

    Every statistic (mean, std) at time t uses ONLY data from [t-window, t].
    No look-ahead bias.

    Args:
        df: Feature DataFrame to normalize.
        window: Rolling lookback window (e.g., 252 days).
        clip_range: Clip normalized values to [-clip, +clip].

    Returns:
        Normalized DataFrame, same shape as input.
    """
    rolling_mean = df.rolling(window=window, min_periods=window).mean()
    rolling_std = df.rolling(window=window, min_periods=window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    normalized = (df - rolling_mean) / rolling_std
    normalized = normalized.clip(-clip_range, clip_range)
    return normalized


# ============================================================================
# MAIN FEATURE PIPELINE
# ============================================================================

def build_all_features(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the complete feature matrix for all assets.

    Returns features in long format (Date, Ticker, feature columns) and
    Group D market features separately (Date, feature columns).

    Args:
        config: Full config dictionary.

    Returns:
        Tuple of (asset_features DataFrame, regime_features DataFrame).
    """
    feat_cfg = config["features"]
    paths_cfg = config["paths"]
    processed_path = PROJECT_ROOT / paths_cfg["processed_data"]

    # Load processed data
    logger.info("Loading processed data ...")
    asset_data = pd.read_parquet(processed_path / "asset_data.parquet")
    market_data = pd.read_parquet(processed_path / "market_data.parquet")

    tickers = sorted(asset_data["Ticker"].unique())
    n_assets = len(tickers)
    logger.info("Building features for %d assets ...", n_assets)

    # --- Per-asset features (Groups A + B) ---
    all_asset_features = []

    for ticker in tickers:
        tdf = asset_data[asset_data["Ticker"] == ticker].copy()
        tdf = tdf.set_index("Date").sort_index()

        group_a = compute_group_a(tdf, config)
        group_b = compute_group_b(tdf, config)

        combined = pd.concat([group_a, group_b], axis=1)
        combined["Ticker"] = ticker
        all_asset_features.append(combined)

        logger.info("  %s: Group A (%d cols) + Group B (%d cols)",
                     ticker, group_a.shape[1], group_b.shape[1])

    per_asset = pd.concat(all_asset_features)
    per_asset = per_asset.reset_index()
    logger.info("Per-asset features shape: %s", per_asset.shape)

    # --- Cross-sectional features (Group C) ---
    logger.info("Computing Group C (cross-sectional) features ...")
    returns_wide = asset_data.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )

    cs_features = compute_cross_sectional_features(
        returns_wide, feat_cfg["cross_section_window"],
    )

    logger.info("Computing correlation PCA (this may take a minute) ...")
    corr_pca = compute_correlation_pca(
        returns_wide,
        feat_cfg["correlation_window"],
        feat_cfg["pca_components"],
    )

    # Melt cross-sectional features to long format
    cs_rank_long = cs_features["rank"].stack().rename("cs_rank").reset_index()
    cs_rank_long.columns = ["Date", "Ticker", "cs_rank"]

    cs_zscore_long = cs_features["zscore"].stack().rename("cs_zscore").reset_index()
    cs_zscore_long.columns = ["Date", "Ticker", "cs_zscore"]

    # --- Market regime features (Group D) ---
    logger.info("Computing Group D (market regime) features ...")
    regime = compute_group_d(market_data, returns_wide, config)

    # --- Merge everything ---
    logger.info("Merging all feature groups ...")
    features = per_asset.copy()

    # Merge cross-sectional rank and z-score
    features = features.merge(cs_rank_long, on=["Date", "Ticker"], how="left")
    features = features.merge(cs_zscore_long, on=["Date", "Ticker"], how="left")

    # Merge correlation PCA (shared, so merge on Date only)
    features = features.merge(corr_pca.reset_index(), on="Date", how="left")

    # Merge regime features (shared, merge on Date only)
    features = features.merge(regime.reset_index(), on="Date", how="left")

    logger.info("Combined features shape: %s", features.shape)

    # --- Identify feature columns (everything except Date, Ticker) ---
    meta_cols = ["Date", "Ticker"]
    feature_cols = [c for c in features.columns if c not in meta_cols]
    logger.info("Total feature columns: %d", len(feature_cols))

    # --- Rolling normalization (per asset, per feature) ---
    logger.info("Applying rolling normalization (window=%d, clip=±%.1f) ...",
                feat_cfg["normalization_window"], feat_cfg["clip_range"])

    # Normalize per-asset features per ticker
    # Regime/PCA features are shared — normalize across the whole series
    per_asset_feat_cols = [c for c in feature_cols
                          if not c.startswith("corr_pca_")
                          and c not in ["vix_level", "spy_drawdown",
                                        "yield_curve_slope", "market_breadth"]]
    shared_feat_cols = [c for c in feature_cols if c not in per_asset_feat_cols]

    normalized_frames = []
    for ticker in tickers:
        mask = features["Ticker"] == ticker
        tdf = features.loc[mask].copy().set_index("Date").sort_index()

        # Normalize per-asset features
        tdf[per_asset_feat_cols] = rolling_normalize(
            tdf[per_asset_feat_cols],
            feat_cfg["normalization_window"],
            feat_cfg["clip_range"],
        )

        # Normalize shared features (same normalization per ticker view is fine
        # because the values are identical; we normalize the shared series)
        tdf[shared_feat_cols] = rolling_normalize(
            tdf[shared_feat_cols],
            feat_cfg["normalization_window"],
            feat_cfg["clip_range"],
        )

        tdf["Ticker"] = ticker
        normalized_frames.append(tdf.reset_index())

    features_norm = pd.concat(normalized_frames, ignore_index=True)
    features_norm = features_norm.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Drop rows with NaN (warmup period from rolling windows)
    pre_drop = len(features_norm)
    features_norm = features_norm.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info("Dropped %d warmup rows (%.1f%%). Final shape: %s",
                pre_drop - len(features_norm),
                100 * (pre_drop - len(features_norm)) / pre_drop,
                features_norm.shape)

    return features_norm, regime


def build_input_tensors(
    features: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Construct 3D input tensors for the model.

    Shape: (num_samples, lookback_window, num_assets * num_features)

    Each sample at date t contains features from [t-lookback+1, t] for all assets.

    Args:
        features: Normalized feature DataFrame (long format).
        config: Full config dictionary.

    Returns:
        Tuple of (X tensor, dates array, tickers list, feature_columns list).
    """
    feat_cfg = config["features"]
    lookback = feat_cfg["lookback_window"]

    meta_cols = ["Date", "Ticker"]
    feature_cols = [c for c in features.columns if c not in meta_cols]
    tickers = sorted(features["Ticker"].unique())
    n_assets = len(tickers)
    n_features = len(feature_cols)

    logger.info("Building input tensors: lookback=%d, assets=%d, features=%d",
                lookback, n_assets, n_features)

    # Pivot to 3D: for each date, stack all assets' features
    # Shape per date: (num_assets, num_features) -> flatten to (num_assets * num_features)
    dates = sorted(features["Date"].unique())

    # Create a pivot: (dates, tickers, features) -> 3D array
    # First pivot each feature column
    pivoted = {}
    for col in feature_cols:
        pivoted[col] = features.pivot_table(
            index="Date", columns="Ticker", values=col,
        )[tickers]  # Ensure consistent ticker order

    # Stack into 3D array: (num_dates, num_assets, num_features)
    all_dates = sorted(pivoted[feature_cols[0]].index)
    data_3d = np.stack([
        np.column_stack([pivoted[col].loc[all_dates].values[:, a]
                         for col in feature_cols])
        for a in range(n_assets)
    ], axis=1)  # (num_dates, num_assets, num_features)

    # Reshape to (num_dates, num_assets * num_features) for TCN input
    num_dates = data_3d.shape[0]
    data_2d = data_3d.reshape(num_dates, n_assets * n_features)

    # Create rolling windows
    num_samples = num_dates - lookback + 1
    X = np.zeros((num_samples, lookback, n_assets * n_features), dtype=np.float32)
    sample_dates = []

    for i in range(num_samples):
        X[i] = data_2d[i:i + lookback]
        sample_dates.append(all_dates[i + lookback - 1])

    sample_dates = np.array(sample_dates)

    logger.info("Input tensor shape: %s", X.shape)
    logger.info("Date range: %s to %s (%d samples)",
                sample_dates[0], sample_dates[-1], len(sample_dates))

    # Check for NaN in tensor
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.error("WARNING: %d NaN values in input tensor!", nan_count)
    else:
        logger.info("PASS: No NaN values in input tensor.")

    return X, sample_dates, tickers, feature_cols


# ============================================================================
# Standalone execution
# ============================================================================

def run_feature_pipeline() -> None:
    """Execute the full Phase 3 feature engineering pipeline."""
    config = load_config()
    set_seeds(config)
    ensure_directories(config)

    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]

    logger.info("=" * 70)
    logger.info("PHASE 3 — FEATURE ENGINEERING")
    logger.info("=" * 70)

    # Build all features
    features_norm, regime = build_all_features(config)

    # Save normalized features
    feat_path = processed_path / "features_normalized.parquet"
    features_norm.to_parquet(feat_path, index=False)
    logger.info("Saved normalized features to %s", feat_path)

    # Build input tensors
    X, sample_dates, tickers, feature_cols = build_input_tensors(features_norm, config)

    # Save tensors
    np.save(processed_path / "X_tensor.npy", X)
    np.save(processed_path / "sample_dates.npy", sample_dates)
    logger.info("Saved X_tensor.npy (%s) and sample_dates.npy", X.shape)

    # Print feature summary
    meta_cols = ["Date", "Ticker"]
    feat_cols = [c for c in features_norm.columns if c not in meta_cols]

    logger.info("=" * 70)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 70)
    logger.info("  Total features per asset: %d", len(feat_cols))
    logger.info("  Assets: %d", len(tickers))
    logger.info("  Samples: %d", X.shape[0])
    logger.info("  Lookback window: %d", X.shape[1])
    logger.info("  Input dimension: %d (assets × features)", X.shape[2])
    logger.info("  Feature columns: %s", feat_cols)

    # Basic distribution check
    logger.info("=" * 70)
    logger.info("FEATURE DISTRIBUTION CHECK")
    logger.info("=" * 70)
    for col in feat_cols:
        vals = features_norm[col]
        logger.info(
            "  %-25s  mean=%+.3f  std=%.3f  min=%+.3f  max=%+.3f",
            col, vals.mean(), vals.std(), vals.min(), vals.max(),
        )

    logger.info("=" * 70)
    logger.info("Phase 3 feature engineering complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_feature_pipeline()
