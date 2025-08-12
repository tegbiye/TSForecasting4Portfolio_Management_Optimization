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
    plt.close()


def main(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.source == 'csv':
        print(f"Loading CSV from {args.csv_path} ...")
        data_raw = reconstruct_from_two_row_csv(args.csv_path)
        # find close columns
        close_cols = [c for c in data_raw.columns if c.endswith('_Close')]
        if not close_cols:
            raise RuntimeError("No close columns detected in reconstructed CSV. Expected columns like TSLA_Close, SPY_Close, BND_Close")
        # Make tidy df
        tidy = data_raw[['Date'] + close_cols].copy()

    elif args.source == 'yfinance':
        if not YFINANCE_AVAILABLE:
            raise RuntimeError('yfinance not available in environment')
        tickers = args.tickers
        print(f"Fetching {tickers} from yfinance {args.start} to {args.end} ...")
        adj = fetch_with_yfinance(tickers, args.start, args.end)
        # adj columns are like 'TSLA_Close' depending on rename in fetch func
        close_cols = [c for c in adj.columns if c.endswith('_Close')]
        tidy = adj[['Date'] + close_cols].copy()
    else:
        raise ValueError("source must be 'csv' or 'yfinance'")

    # Prepare time index and interpolation
    df_clean = prepare_time_index_and_interpolate(tidy, close_cols)

    # Save tidy cleaned close prices
    tidy_out = outdir / 'tidy_close_prices.csv'
    df_clean.reset_index().to_csv(tidy_out, index=False)
    print('Saved tidy close prices to', tidy_out)

    # Compute returns and rolling stats
    returns, rolling_mean, rolling_std = compute_returns_and_rolling(df_clean, window=args.rolling_window)

    # Save returns
    returns_out = outdir / 'daily_returns.csv'
    returns.reset_index().to_csv(returns_out, index=False)
    print('Saved daily returns to', returns_out)

    # Outliers
    outlier_days, _mask = detect_outliers(returns, z_thresh=args.z_threshold)
    outlier_out = outdir / 'outlier_days.csv'
    outlier_days.reset_index().to_csv(outlier_out, index=False)
    print('Saved outlier days to', outlier_out)

    # Risk metrics
    VaR, sharpe_annual = compute_var_sharpe(returns, rf=args.risk_free_rate)
    VaR.to_csv(outdir / 'VaR.csv')
    sharpe_annual.to_csv(outdir / 'Sharpe_annual.csv')

    # ADF tests
    adf_results = {}
    if STATSMODELS_AVAILABLE:
        for c in df_clean.columns:
            adf_results[c] = adf_test(df_clean[c])
        for c in returns.columns:
            adf_results[f"{c}_returns"] = adf_test(returns[c])
        # Save ADF summary
        with open(outdir / 'adf_results.txt', 'w') as f:
            for k,v in adf_results.items():
                f.write(f"{k}: p={v['pvalue']}, adf_stat={v['adf_stat']}\n")
    else:
        print('statsmodels not available; skipping ADF tests')

    # Save cleaned dataframe for modeling
    df_clean.reset_index().to_csv(outdir / 'clean_close_prices_for_analysis.csv', index=False)

    # Plots
    plot_series(df_clean, outdir, prefix='close_')
    plot_returns(returns, outdir)

    # New: Correlation heatmap
    plot_correlation(returns, outdir)

    # New: Cumulative returns
    cum_returns = (1 + returns).cumprod()
    plot_series(cum_returns, outdir, prefix='cum_returns_')

    # Rolling mean/std example plot for TSLA (if present)
    example_col = None
    for candidate in ['TSLA_Close', 'SPY_Close', 'BND_Close']:
        if candidate in df_clean.columns:
            example_col = candidate
            break
    if example_col:
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df_clean.index, df_clean[example_col].rolling(args.rolling_window).mean(), label=f'{args.rolling_window}d mean')
        ax.plot(df_clean.index, df_clean[example_col].rolling(args.rolling_window).std(), label=f'{args.rolling_window}d std')
        ax.set_title(f"{example_col} rolling mean and std")
        ax.legend()
        fig.savefig(outdir / f"{example_col}_rolling_mean_std.png")
        plt.close(fig)

    # Print a short summary to console
    print('\n=== Summary ===')
    print('Sample means of daily returns:')
    print(returns.mean().to_string())
    print('\nSample std dev of daily returns:')
    print(returns.std().to_string())
    print('\nVaR:')
    print(VaR.to_string())
    print('\nAnnualized Sharpe (rf=' + str(args.risk_free_rate) + '):')
    print(sharpe_annual.to_string())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Finance EDA & Preprocessing Pipeline')
    p.add_argument('--source', choices=['csv','yfinance'], default='csv', help='Data source')
    p.add_argument('--csv_path', type=str, default='/mnt/data/comp_hist_finance_data.csv', help='Path to uploaded CSV (two-row header)')
    p.add_argument('--tickers', nargs='+', default=['TSLA','SPY','BND'], help='Tickers to fetch if using yfinance')
    p.add_argument('--start', type=str, default='2015-01-01', help='Start date for yfinance fetch (YYYY-MM-DD)')
    p.add_argument('--end', type=str, default='2025-08-12', help='End date for yfinance fetch (YYYY-MM-DD)')
    p.add_argument('--output_dir', type=str, default='./finance_eda_output', help='Directory for outputs')
    p.add_argument('--rolling_window', type=int, default=20, help='Window size for rolling stats')
    p.add_argument('--z_threshold', type=float, default=3.0, help='Z-score threshold for outlier detection')
    p.add_argument('--risk_free_rate', type=float, default=0.0, help='Risk-free rate for Sharpe ratio (daily rate)')
    args = p.parse_args()
    main(args)