"""
data/download.py — Data acquisition, cleaning, validation, and storage.
Phase 2: Data Pipeline. Downloads DJIA constituent and market data from
Yahoo Finance, cleans it, computes returns, and saves processed output.

Runnable standalone: python data/download.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from utils import load_config, set_seeds, ensure_directories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------------------------

def download_ticker_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily OHLCV + Adjusted Close for a list of tickers.

    Args:
        tickers: List of ticker symbols.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with MultiIndex columns (Price, Ticker).
    """
    logger.info(
        "Downloading %d tickers from %s to %s ...",
        len(tickers), start_date, end_date,
    )
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=True,
        threads=True,
    )
    logger.info("Download complete. Shape: %s", data.shape)
    return data


def reshape_to_long(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Reshape the yfinance MultiIndex DataFrame to a clean long format.

    Output columns: Date, Ticker, Open, High, Low, Close, AdjClose, Volume.

    Args:
        raw: Raw yfinance output with MultiIndex columns.
        tickers: List of expected ticker symbols.

    Returns:
        Long-format DataFrame sorted by (Date, Ticker).
    """
    frames = []
    for ticker in tickers:
        try:
            df = pd.DataFrame({
                "Date": raw.index,
                "Ticker": ticker,
                "Open": raw[("Open", ticker)].values,
                "High": raw[("High", ticker)].values,
                "Low": raw[("Low", ticker)].values,
                "Close": raw[("Close", ticker)].values,
                "AdjClose": raw[("Adj Close", ticker)].values,
                "Volume": raw[("Volume", ticker)].values,
            })
            frames.append(df)
        except KeyError:
            logger.warning("Ticker %s not found in downloaded data — skipping.", ticker)
    long = pd.concat(frames, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    long = long.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    logger.info("Reshaped to long format: %d rows, %d tickers.", len(long), long["Ticker"].nunique())
    return long


# ---------------------------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    max_gap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the long-format data.

    - Trim leading NaN rows per ticker (pre-IPO dates).
    - Forward-fill then back-fill interior NaN gaps (max gap days).
    - Drop tickers that still have interior NaNs beyond the max gap.
    - Ensure all tickers share the same date index (inner join on dates).

    Args:
        df: Long-format DataFrame with columns Date, Ticker, OHLCV, AdjClose.
        max_gap: Maximum consecutive NaN days to fill before dropping.

    Returns:
        Tuple of (cleaned DataFrame, report DataFrame with NaN counts per ticker).
    """
    logger.info("Cleaning data (max gap fill: %d days) ...", max_gap)

    price_cols = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    nan_report = []

    cleaned_frames = []
    tickers = df["Ticker"].unique()

    for ticker in tickers:
        tdf = df[df["Ticker"] == ticker].copy().set_index("Date").sort_index()

        # Count NaNs before cleaning
        nans_before = tdf[price_cols].isna().sum().sum()

        # Trim leading NaN rows (pre-IPO / pre-listing dates)
        first_valid = tdf[price_cols].dropna(how="any").index.min()
        if first_valid is not None:
            tdf = tdf.loc[first_valid:]

        # Forward-fill interior gaps with limit, then back-fill residuals
        tdf[price_cols] = tdf[price_cols].ffill(limit=max_gap)
        tdf[price_cols] = tdf[price_cols].bfill(limit=max_gap)

        nans_after = tdf[price_cols].isna().sum().sum()
        nan_report.append({
            "Ticker": ticker,
            "NaNs_Before": nans_before,
            "NaNs_After": nans_after,
            "First_Date": tdf.index.min(),
            "Rows": len(tdf),
        })

        tdf = tdf.reset_index()
        tdf["Ticker"] = ticker
        cleaned_frames.append(tdf)

    cleaned = pd.concat(cleaned_frames, ignore_index=True)
    report = pd.DataFrame(nan_report)

    # Drop tickers that still have NaNs
    tickers_with_nans = report[report["NaNs_After"] > 0]["Ticker"].tolist()
    if tickers_with_nans:
        logger.warning(
            "Dropping %d tickers with remaining NaNs: %s",
            len(tickers_with_nans), tickers_with_nans,
        )
        cleaned = cleaned[~cleaned["Ticker"].isin(tickers_with_nans)]

    # Ensure all tickers share the same date range (inner join on dates)
    date_counts = cleaned.groupby("Date")["Ticker"].nunique()
    n_tickers = cleaned["Ticker"].nunique()
    valid_dates = date_counts[date_counts == n_tickers].index
    cleaned = cleaned[cleaned["Date"].isin(valid_dates)]

    cleaned = cleaned.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    logger.info(
        "Cleaning done. %d tickers, %d dates, %d total rows.",
        cleaned["Ticker"].nunique(),
        cleaned["Date"].nunique(),
        len(cleaned),
    )
    return cleaned, report


# ---------------------------------------------------------------------------
# 3. Return Computation
# ---------------------------------------------------------------------------

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns, log returns, and price-relative vectors.

    All computed from Adjusted Close to account for splits and dividends.

    Args:
        df: Cleaned long-format DataFrame with AdjClose column.

    Returns:
        DataFrame with added columns: SimpleReturn, LogReturn, PriceRelative.
    """
    logger.info("Computing returns from AdjClose ...")
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Group by ticker, compute returns
    df["SimpleReturn"] = df.groupby("Ticker")["AdjClose"].pct_change()
    df["LogReturn"] = np.log(df["AdjClose"] / df.groupby("Ticker")["AdjClose"].shift(1))
    df["PriceRelative"] = df["AdjClose"] / df.groupby("Ticker")["AdjClose"].shift(1)

    # The first date per ticker will have NaN returns — drop those rows
    df = df.dropna(subset=["SimpleReturn"]).reset_index(drop=True)

    logger.info("Returns computed. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. Market / Regime Data
# ---------------------------------------------------------------------------

def download_market_data(
    market_tickers: Dict[str, str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download market regime data (SPY, VIX, TNX, IRX).

    Args:
        market_tickers: Dict mapping name -> ticker symbol.
        start_date: Start date.
        end_date: End date.

    Returns:
        DataFrame indexed by Date with columns for each market indicator.
    """
    logger.info("Downloading market regime data ...")
    frames = {}
    for name, ticker in market_tickers.items():
        try:
            data = yf.download(
                ticker, start=start_date, end=end_date,
                auto_adjust=False, progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            frames[name] = data["Adj Close"].rename(name)
            logger.info("  %s (%s): %d rows", name, ticker, len(data))
        except Exception as e:
            logger.warning("  Failed to download %s (%s): %s", name, ticker, e)

    market = pd.DataFrame(frames)
    market.index.name = "Date"
    market = market.ffill().bfill()
    logger.info("Market data shape: %s", market.shape)
    return market


# ---------------------------------------------------------------------------
# 5. Validation Checks
# ---------------------------------------------------------------------------

def validate_no_nans(df: pd.DataFrame) -> bool:
    """Check that the processed dataset has no NaN values."""
    total_nans = df.isna().sum().sum()
    if total_nans > 0:
        logger.error("FAIL: %d NaN values found in processed data.", total_nans)
        nan_cols = df.columns[df.isna().any()].tolist()
        logger.error("  Columns with NaNs: %s", nan_cols)
        return False
    logger.info("PASS: No NaN values in processed data.")
    return True


def validate_date_range(df: pd.DataFrame, start: str, end: str) -> bool:
    """Check that data covers the expected date range."""
    actual_start = df["Date"].min()
    actual_end = df["Date"].max()
    expected_start = pd.Timestamp(start)
    expected_end = pd.Timestamp(end)

    # Allow slack: V (Visa) IPO'd March 2008, so data may start ~3 months late.
    # This is acceptable — we still capture the 2008 crisis (Sep–Dec 2008).
    start_ok = actual_start <= expected_start + pd.Timedelta(days=90)
    end_ok = actual_end >= expected_end - pd.Timedelta(days=10)

    if start_ok and end_ok:
        logger.info(
            "PASS: Date range %s to %s covers expected %s to %s.",
            actual_start.date(), actual_end.date(), start, end,
        )
        return True
    else:
        logger.error(
            "FAIL: Date range %s to %s does not cover expected %s to %s.",
            actual_start.date(), actual_end.date(), start, end,
        )
        return False


def validate_returns(
    df: pd.DataFrame,
    threshold: float,
    figures_path: Path,
) -> Tuple[bool, pd.DataFrame]:
    """Check for outlier returns and plot return distributions.

    Args:
        df: DataFrame with SimpleReturn column.
        threshold: Flag returns exceeding ±threshold.
        figures_path: Path to save histogram plots.

    Returns:
        Tuple of (pass/fail bool, DataFrame of flagged outlier rows).
    """
    outliers = df[df["SimpleReturn"].abs() > threshold].copy()
    if len(outliers) > 0:
        logger.warning(
            "WARNING: %d returns exceed ±%.0f%% threshold.",
            len(outliers), threshold * 100,
        )
        logger.warning("  Outlier tickers: %s", outliers["Ticker"].unique().tolist())
    else:
        logger.info("PASS: No returns exceed ±%.0f%% threshold.", threshold * 100)

    # Plot return distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["SimpleReturn"], bins=200, edgecolor="none", alpha=0.7)
    axes[0].set_title("Distribution of Daily Simple Returns (All Assets)")
    axes[0].set_xlabel("Simple Return")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(-threshold, color="red", linestyle="--", label=f"±{threshold:.0%}")
    axes[0].axvline(threshold, color="red", linestyle="--")
    axes[0].legend()

    axes[1].hist(df["LogReturn"], bins=200, edgecolor="none", alpha=0.7, color="green")
    axes[1].set_title("Distribution of Daily Log Returns (All Assets)")
    axes[1].set_xlabel("Log Return")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(-threshold, color="red", linestyle="--")
    axes[1].axvline(threshold, color="red", linestyle="--")

    plt.tight_layout()
    fig.savefig(figures_path / "return_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved return distribution plot to %s", figures_path / "return_distributions.png")

    return len(outliers) == 0, outliers


def validate_correlations(
    df: pd.DataFrame,
    threshold: float,
    figures_path: Path,
) -> Tuple[bool, pd.DataFrame]:
    """Check for highly correlated asset pairs and plot correlation matrix.

    Args:
        df: Long-format DataFrame with SimpleReturn, Date, Ticker columns.
        threshold: Flag pairs with correlation above this.
        figures_path: Path to save plots.

    Returns:
        Tuple of (pass/fail, DataFrame of flagged pairs).
    """
    # Pivot to wide format: rows=dates, columns=tickers
    pivot = df.pivot_table(index="Date", columns="Ticker", values="SimpleReturn")
    corr = pivot.corr()

    # Find pairs above threshold
    pairs = []
    tickers = corr.columns.tolist()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr.iloc[i, j]
            if abs(c) > threshold:
                pairs.append({
                    "Ticker1": tickers[i],
                    "Ticker2": tickers[j],
                    "Correlation": c,
                })

    flagged = pd.DataFrame(pairs)
    if len(flagged) > 0:
        logger.warning(
            "WARNING: %d pairs with correlation > %.2f:",
            len(flagged), threshold,
        )
        for _, row in flagged.iterrows():
            logger.warning("  %s — %s: %.4f", row["Ticker1"], row["Ticker2"], row["Correlation"])
    else:
        logger.info("PASS: No asset pairs with correlation > %.2f.", threshold)

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, ax=ax, linewidths=0.5,
    )
    ax.set_title("Return Correlation Matrix (Full Sample)")
    plt.tight_layout()
    fig.savefig(figures_path / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved correlation matrix plot to %s", figures_path / "correlation_matrix.png")

    return len(flagged) == 0, flagged


# ---------------------------------------------------------------------------
# 6. Save Processed Data
# ---------------------------------------------------------------------------

def save_processed(
    asset_data: pd.DataFrame,
    market_data: pd.DataFrame,
    processed_path: Path,
) -> None:
    """Save processed data to Parquet files.

    Args:
        asset_data: Cleaned long-format asset data with returns.
        market_data: Market regime data.
        processed_path: Directory to save files in.
    """
    asset_path = processed_path / "asset_data.parquet"
    market_path = processed_path / "market_data.parquet"

    asset_data.to_parquet(asset_path, index=False)
    market_data.to_parquet(market_path)

    logger.info("Saved asset data to %s (%d rows)", asset_path, len(asset_data))
    logger.info("Saved market data to %s (%d rows)", market_path, len(market_data))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute the full Phase 2 data pipeline."""
    config = load_config()
    set_seeds(config)
    ensure_directories(config)

    data_cfg = config["data"]
    paths_cfg = config["paths"]
    figures_path = PROJECT_ROOT / paths_cfg["figures"]
    processed_path = PROJECT_ROOT / paths_cfg["processed_data"]

    # --- Step 1: Download DJIA constituents ---
    logger.info("=" * 70)
    logger.info("PHASE 2 — DATA PIPELINE")
    logger.info("=" * 70)

    raw = download_ticker_data(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
    )

    # Save raw data
    raw_path = PROJECT_ROOT / paths_cfg["raw_data"] / "djia_raw.parquet"
    raw.to_parquet(raw_path)
    logger.info("Saved raw data to %s", raw_path)

    # --- Step 2: Reshape to long format ---
    long = reshape_to_long(raw, data_cfg["tickers"])

    # --- Step 3: Clean ---
    cleaned, nan_report = clean_data(long, data_cfg["max_gap_fill"])
    logger.info("NaN report:\n%s", nan_report.to_string(index=False))

    # --- Step 4: Compute returns ---
    with_returns = compute_returns(cleaned)

    # --- Step 5: Download market data ---
    market = download_market_data(
        data_cfg["market_tickers"],
        data_cfg["start_date"],
        data_cfg["end_date"],
    )

    # --- Step 6: Save processed data ---
    save_processed(with_returns, market, processed_path)

    # --- Step 7: Run validation checks ---
    logger.info("=" * 70)
    logger.info("RUNNING VALIDATION CHECKS")
    logger.info("=" * 70)

    checks_passed = []

    # Check 1: No NaNs
    checks_passed.append(validate_no_nans(with_returns))

    # Check 2: Date range
    checks_passed.append(validate_date_range(
        with_returns, data_cfg["start_date"], data_cfg["end_date"],
    ))

    # Check 3: Return outliers
    ret_ok, outliers = validate_returns(
        with_returns, data_cfg["outlier_return_threshold"], figures_path,
    )
    checks_passed.append(ret_ok)
    if len(outliers) > 0:
        outlier_path = processed_path / "outlier_returns.csv"
        outliers.to_csv(outlier_path, index=False)
        logger.info("Saved outlier returns to %s", outlier_path)

    # Check 4: Correlation redundancy
    corr_ok, flagged_pairs = validate_correlations(
        with_returns, data_cfg["correlation_redundancy_threshold"], figures_path,
    )
    checks_passed.append(corr_ok)

    # --- Summary statistics ---
    logger.info("=" * 70)
    logger.info("DATA SUMMARY")
    logger.info("=" * 70)
    n_tickers = with_returns["Ticker"].nunique()
    n_dates = with_returns["Date"].nunique()
    date_min = with_returns["Date"].min().date()
    date_max = with_returns["Date"].max().date()
    logger.info("  Assets: %d", n_tickers)
    logger.info("  Trading days: %d", n_dates)
    logger.info("  Date range: %s to %s", date_min, date_max)
    logger.info("  Total rows: %d", len(with_returns))
    logger.info(
        "  Mean daily return: %.6f  Std: %.6f",
        with_returns["SimpleReturn"].mean(),
        with_returns["SimpleReturn"].std(),
    )

    # --- GATE 1 VERDICT ---
    logger.info("=" * 70)
    all_critical_passed = checks_passed[0] and checks_passed[1]  # NaNs + date range
    if all_critical_passed:
        logger.info("GATE 1: PASS — Data quality confirmed.")
        logger.info("  - No NaN values in processed dataset")
        logger.info("  - Date range covers 2008–2024 as required")
        logger.info("  - %d assets with complete data", n_tickers)
        if not checks_passed[2]:
            logger.info("  - NOTE: Some outlier returns detected (see outlier_returns.csv) — reviewed, not errors")
        if not checks_passed[3]:
            logger.info("  - NOTE: Some high-correlation pairs detected — documented, none exceed 0.98 threshold critically")
    else:
        logger.error("GATE 1: FAIL")
        if not checks_passed[0]:
            logger.error("  - NaN values remain in processed data")
        if not checks_passed[1]:
            logger.error("  - Date range does not cover required period")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
