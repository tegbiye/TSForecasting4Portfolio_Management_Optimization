import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Optional: yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Statsmodels for ADF
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

def data_loader(filepath):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"CSV file loaded successfully from {filepath}.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file from {filepath}: {e}")
        return None

def reconstruct_from_two_row_csv(csv_path: str) -> pd.DataFrame:
    """
    The provided uploaded CSV uses a two-row header layout:
    Row 0: field names like Price / Close / High / Open / Volume
    Row 1: corresponding ticker for each column (BND / SPY / TSLA)
    Data starts after a 'Date' label row.

    This function reconstructs a tidy DataFrame with columns like 'TSLA_Close', 'BND_Close', 'SPY_Close'.
    Improved: Dynamically finds the start of data rows by locating the 'Date' label.
    """
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[0] < 4:
        raise ValueError("CSV file doesn't look like the expected two-row header format or is too small.")

    fields = raw.iloc[0].tolist()
    tickers = raw.iloc[1].tolist()

    # Find the row where data starts (look for 'Date' in first column)
    data_start = None
    for r in range(raw.shape[0]):
        if str(raw.iloc[r, 0]).strip() == 'Date':
            data_start = r + 1
            break
    if data_start is None:
        raise ValueError("Could not find 'Date' label row in CSV.")

    # Build column names
    cols = []
    for i, (f, t) in enumerate(zip(fields, tickers)):
        if i == 0:
            cols.append('Date')
        else:
            f_clean = str(f).strip()
            t_clean = str(t).strip()
            cols.append(f"{t_clean}_{f_clean}")

    # Extract data rows
    data = raw.iloc[data_start:].copy()
    data.columns = cols

    # Parse Date
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # Convert numeric columns
    for c in data.columns:
        if c != 'Date':
            data[c] = pd.to_numeric(data[c], errors='coerce')

    # Drop any rows with NaN Date
    data = data.dropna(subset=['Date'])

    # Reindex and sort
    data = data.sort_values('Date').reset_index(drop=True)

    return data


def fetch_with_yfinance(tickers, start, end) -> pd.DataFrame:
    """Fetches Adjusted Close prices using yfinance and returns a tidy DataFrame with Date and tickers as columns."""
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not installed in this environment. Install with `pip install yfinance`.")
    df = yf.download(tickers, start=start, end=end, progress=False, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            adj = df['Adj Close']
        else:
            adj = df['Close']  # fallback
    else:
        if 'Adj Close' in df.columns:
            adj = df[['Adj Close']]
        else:
            adj = df[['Close']]
    adj = adj.rename(columns=lambda c: f"{c}_Close")
    adj = adj.reset_index()
    return adj


def prepare_time_index_and_interpolate(df: pd.DataFrame, close_cols: list, freq='B') -> pd.DataFrame:
    """
    Given a DataFrame with Date column and close columns, set Date as index, set business frequency, and interpolate missing values.
    Returns dataframe with Date index and numeric close columns.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')
    else:
        # assume index is already datetime
        df.index = pd.to_datetime(df.index)

    # Ensure close columns exist
    for c in close_cols:
        if c not in df.columns:
            raise KeyError(f"Expected close column {c} not found in DataFrame columns: {df.columns.tolist()}")

    df = df[close_cols].sort_index()
    df = df.asfreq(freq)  # business-day frequency

    # Interpolate
    df_interp = df.interpolate(method='time')
    df_interp = df_interp.ffill().bfill()
    return df_interp


def compute_returns_and_rolling(df: pd.DataFrame, window=20):
    returns = df.pct_change().dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return returns, rolling_mean, rolling_std


def detect_outliers(returns: pd.DataFrame, z_thresh=3.0):
    z = (returns - returns.mean()) / returns.std()
    mask = (z.abs() > z_thresh)
    outlier_days = returns[mask.any(axis=1)]
    return outlier_days, mask


def compute_var_sharpe(returns: pd.DataFrame, quantiles=(0.05,0.01), rf=0.0):
    VaR = returns.quantile(list(quantiles))
    sharpe_daily = (returns.mean() - rf) / returns.std()
    sharpe_annual = sharpe_daily * np.sqrt(252)
    return VaR, sharpe_annual


def adf_test(series: pd.Series):
    if not STATSMODELS_AVAILABLE:
        raise RuntimeError("statsmodels is required for ADF test. Install with `pip install statsmodels`.")
    res = adfuller(series.dropna())
    return {"adf_stat": res[0], "pvalue": res[1], "usedlag": res[2], "nobs": res[3], "crit_vals": res[4]}


def plot_series(df: pd.DataFrame, outdir: str, prefix: str = ''):
    os.makedirs(outdir, exist_ok=True)
    for col in df.columns:
        plt.figure(figsize=(12,4))
        plt.plot(df.index, df[col])
        plt.title(f"{col} - Time Series")
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.grid(True)
        fname = Path(outdir) / f"{prefix}{col}_timeseries.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()
        plt.close()


def plot_returns(returns: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for col in returns.columns:
        plt.figure(figsize=(12,4))
        plt.plot(returns.index, returns[col])
        plt.title(f"{col} - Daily Returns")
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.grid(True)
        fname = Path(outdir) / f"returns_{col}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()
        plt.close()


def plot_correlation(returns: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', color='black')
    plt.colorbar(im)
    plt.title('Correlation Heatmap of Daily Returns')
    fname = Path(outdir) / "returns_correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()
    plt.close()