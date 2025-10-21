"""
research.py - Model Research & Development Utilities
====================================================
Reusable boilerplate code for machine learning research and model development.
This module eliminates copy-paste by centralizing common functions for:

- Data preprocessing and time series aggregation
- Feature engineering for financial data
- Model architecture inspection and debugging
- Visualization and exploratory data analysis
- Training utilities

Import this in all your research notebooks to save time and maintain consistency.

Author: MemLabs
Course: Build a Quant Trading System
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Data manipulation and analysis
import polars as pl                         # Fast dataframes for financial data
from typing import Dict, List, Tuple, Union                     # Type hints for function signatures

# Machine learning framework
import torch                                # PyTorch for neural networks
import torch.nn as nn                       # Neural network modules
import torch.optim as optim                 # Optimization algorithms

# Numerical computing and datetime
import numpy as np                          # Numerical operations
import numpy.typing as npt
from datetime import datetime, timedelta    # Date and time handling

# Visualization
import altair                               # Interactive plotting library
import matplotlib.pyplot as plt

import random
import re
import itertools
from pathlib import Path
from tqdm import tqdm
import os

SEED = 42

# ============================================================================
# TIME SERIES AGGREGATION
# ============================================================================
OHLC_AGGS = [
    # Price statistics (core OHLC data)
    pl.col("price").first().alias("open"),              # Opening price
    pl.col("price").max().alias("high"),                # Highest price
    pl.col("price").min().alias("low"),                 # Lowest price
    pl.col("price").last().alias("close"),              # Closing price (most important)
]


def get_trade_files(directory: str, sym: str) -> List[Path]:
    """
    Get all files in directory that start with '{sym}-trades'.
    
    Args:
        directory: Path to directory to search
        sym: Symbol prefix (e.g., 'BTCUSDT')
    
    Returns:
        List of Path objects matching the pattern
    
    Example:
        >>> files = get_trade_files('./data', 'BTCUSDT')
        >>> # Returns: ['BTCUSDT-trades-2024.csv', 'BTCUSDT-trades-raw.parquet', ...]
    """
    dir_path = Path(directory)
    pattern = f"{sym}-trades*"
    return sorted(dir_path.glob(pattern))


from pathlib import Path
from typing import List, Optional

def load_ohlc_timeseries(sym: str, time_interval: str):
    return load_timeseries(sym, time_interval, OHLC_AGGS)

def load_timeseries(
    sym: str, 
    time_interval: str,
    aggs: List[pl.Expr],
    data_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Load trade CSV files one by one, aggregate to time series, and concatenate.
    
    Args:
        sym: Symbol prefix (e.g., 'BTCUSDT')
        time_interval: Time interval for aggregation (e.g., '1h', '5m')
        aggs: List of aggregation expressions
        data_path: Optional directory path. Defaults to './cache' if not provided
    
    Returns:
        Concatenated time series DataFrame
    
    Example:
        >>> # Use default './cache' directory
        >>> ts = load_ohlc_ts('BTCUSDT', '1h', ohlc_aggs)
        
        >>> # Specify custom directory
        >>> ts = load_ohlc_ts('BTCUSDT', '1h', ohlc_aggs, data_path='./my_data')
    """
    # Default to './cache' if not provided
    if data_path is None:
        data_path = './cache'
    
    files = get_trade_files(data_path, sym)
    
    if not files:
        raise FileNotFoundError(f"No files found for {sym} in {data_path}")
    
    # Process each file and collect results
    ts_list = []
    
    # Add progress bar
    for file in tqdm(files, desc=f"Loading {sym}", unit="file"):
        
        # Load trades from parquet
        trades = pl.read_parquet(file)
        
        # Ensure datetime column exists and is correct type
        if "datetime" not in trades.columns:
            raise ValueError(f"Column 'datetime' not found in {file.name}")
        
        trades = trades.with_columns(
            pl.col("datetime").cast(pl.Datetime)
        ).sort("datetime")
        
        # Aggregate to time series
        ts = trades.group_by_dynamic(
            "datetime",
            every=time_interval,
            offset="0m"
        ).agg(aggs)
        
        ts_list.append(ts)
    
    # Concatenate all time series
    result = pl.concat(ts_list)
    
    # Sort by datetime and remove duplicates if any
    result = result.sort("datetime").unique(subset=["datetime"])
    
    return result


def load_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    agg_cols: Union[pl.Expr,List[pl.Expr]],
    data_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and aggregate trade data for a symbol between start_date and end_date
    into OHLC time series using the given time interval.

    Expects daily files named like:
        {symbol}-trades-YYYY-MM-DD.parquet

    Example filename:
        BTCUSDT-trades-2025-09-22.parquet

    Args:
        sym: Symbol prefix (e.g., 'BTCUSDT')
        time_interval: Aggregation interval (e.g., '1h', '5m')
        start_date: Start datetime (inclusive)
        end_date: End datetime (inclusive)
        data_path: Directory containing cached trade parquet files (default: './cache')

    Returns:
        Polars DataFrame with aggregated OHLC time series for the given range.
    """
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(agg_cols)
            ts_list.append(ts)

        except Exception as e:
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result

def load_ohlc_timeseries_range(
    sym: str,
    time_interval: str,
    start_date: datetime,
    end_date: datetime,
    data_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and aggregate trade data for a symbol between start_date and end_date
    into OHLC time series using the given time interval.

    Expects daily files named like:
        {symbol}-trades-YYYY-MM-DD.parquet

    Example filename:
        BTCUSDT-trades-2025-09-22.parquet

    Args:
        sym: Symbol prefix (e.g., 'BTCUSDT')
        time_interval: Aggregation interval (e.g., '1h', '5m')
        start_date: Start datetime (inclusive)
        end_date: End datetime (inclusive)
        data_path: Directory containing cached trade parquet files (default: './cache')

    Returns:
        Polars DataFrame with aggregated OHLC time series for the given range.
    """
    if data_path is None:
        data_path = "./cache"

    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    ts_list = []
    total_days = (end_date - start_date).days + 1

    for i in tqdm(range(total_days), desc=f"Loading {sym}", unit="day"):
        current_date = start_date + timedelta(days=i)
        file_name = f"{sym}-trades-{current_date.strftime('%Y-%m-%d')}.parquet"
        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):
            tqdm.write(f"[WARNING] Missing file: {file_name}")
            continue

        try:
            trades = pl.read_parquet(file_path)

            if "datetime" not in trades.columns:
                raise ValueError(f"Column 'datetime' not found in {file_name}")

            trades = trades.with_columns(pl.col("datetime").cast(pl.Datetime))

            ts = trades.group_by_dynamic("datetime", every=time_interval, offset="0m").agg(OHLC_AGGS)
            ts_list.append(ts)

        except Exception as e:
            tqdm.write(f"[ERROR] {file_name}: {e}")

    if not ts_list:
        raise ValueError(f"No trade data found for {sym} in range {start_date} to {end_date}")

    result = pl.concat(ts_list).sort("datetime").unique(subset=["datetime"])
    return result


def sharpe_annualization_factor(interval: str,
                                trading_days_per_year: int = 365,
                                trading_hours_per_day: float = 24) -> float:
    """
    Compute annualization factor (sqrt of periods per year) given a return interval.
    
    interval : str
        Frequency string like '1d', '1h', '30m', '15s'.
    trading_days_per_year : int
        Number of trading days in a year (default 252).
    trading_hours_per_day : float
        Number of trading hours in a trading day (default 6.5).
        
    Returns
    -------
    float : annualization factor
    """
    match = re.match(r"(\d+)([dhms])", interval.lower())
    if not match:
        raise ValueError("Interval must be like '1d', '2h', '15m', '30s'")
    
    value, unit = int(match.group(1)), match.group(2)
    
    # periods per year
    if unit == 'd':
        periods = trading_days_per_year / value
    elif unit == 'h':
        periods = trading_days_per_year * (trading_hours_per_day / value)
    elif unit == 'm':
        periods = trading_days_per_year * (trading_hours_per_day * 60 / value)
    elif unit == 's':
        periods = trading_days_per_year * (trading_hours_per_day * 3600 / value)
    else:
        raise ValueError(f"Unsupported unit: {unit}")
    
    return np.sqrt(periods)


def ohlc_timeseries(df: pl.DataFrame, time_interval: str) -> pl.DataFrame:
    """
    Convert tick-level trade data into OHLC (Open, High, Low, Close) bars.
    
    This function aggregates raw trade data into standardized price bars
    with basic volume and trade statistics. If you want to extend this then call regular_timeseries
    
    Args:
        df: DataFrame containing trade data with columns:
            - datetime: Timestamp of each trade
            - price: Execution price
            - quote_qty: Trade size in quote currency (e.g., USDT)
            - is_short: Boolean indicating if trade was a short sale
        time_interval: Aggregation period (e.g., '1m', '5m', '15m', '1h', '1d')
    
    Returns:
        DataFrame with OHLC bars containing:
            - datetime: Bar timestamp
            - open: First price in interval
            - high: Highest price in interval
            - low: Lowest price in interval
            - close: Last price in interval (most important for ML)
            - volume: Total trading volume in quote currency
            - trade_count: Number of individual trades
            - short_ratio: Percentage of trades that were short sales
            - mean_price: Average price (volume-weighted alternative)
    
    Example:
        >>> # Create 15-minute OHLC bars
        >>> bars_15m = ohlc_timeseries(trades_df, '15m')
        >>> 
        >>> # Create hourly bars for longer-term analysis
        >>> bars_1h = ohlc_timeseries(trades_df, '1h')
    """
    # Define aggregation expressions for OHLC calculation

    
    # Use the generic time series aggregation function
    return timeseries(df, time_interval, OHLC_AGGS)

def lag_col_names(col: str, n: int) -> List[str]:
    return [f'{col}_lag_{i}' for i in range(1, n+1)]

def auto_reg_corr_matrx(df, target, max_no_lags) -> pl.DataFrame:
    return df.drop_nulls().select([target]+lag_col_names(target, max_no_lags)).corr()

def log_returns_col(name: str, step_size = 1) -> pl.Expr:
    return (pl.col(name)/pl.col(name).shift(step_size)).log().alias(f'{name}_log_return')

def timeseries(
    df: pl.DataFrame, 
    time_interval: str, 
    aggs: Union[List[pl.Expr],pl.Expr]
) -> pl.DataFrame:
    """
    Generic function for aggregating data into regular time intervals.
    
    This is a flexible time series aggregation framework that can handle
    any custom aggregation expressions. Used as the foundation for OHLC
    bars and other time-based features.
    
    Args:
        df: DataFrame with a 'datetime' column
        time_interval: Aggregation period (Polars duration string)
            Examples: '1m', '5m', '15m', '1h', '4h', '1d'
        aggs: List of Polars expressions defining aggregations to compute
    
    Returns:
        DataFrame with time-aggregated data
    
    Technical Details:
        - Uses left-closed intervals: [start_time, end_time)
        - Bars start at round times (e.g., 09:00, 09:15, 09:30)
        - Missing bars (no trades) are automatically excluded
    
    Example:
        >>> # Custom aggregation for volatility analysis
        >>> custom_aggs = [
        ...     pl.col("price").std().alias("price_volatility"),
        ...     pl.col("volume").sum().alias("total_volume"),
        ... ]
        >>> df_volatility = regular_timeseries(df, '1h', custom_aggs)
    """
    return df.group_by_dynamic(
        "datetime",              # Column to group by (must be datetime type)
        every=time_interval,     # Aggregation frequencyß
        offset="0m"              # No offset (bars align to round times)
    ).agg(aggs)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot(df: pl.DataFrame, col: str, title: str = "") -> altair.Chart:
    """
    Create a smooth density plot for analyzing feature distributions.
    
    Useful for:
    - Understanding data distributions before modeling
    - Detecting outliers and skewness
    - Comparing feature distributions across different time periods
    - Validating data preprocessing steps
    
    Args:
        df: DataFrame containing the column to plot
        col: Name of the column to visualize
        title: Optional chart title (defaults to None)
    
    Returns:
        Altair Chart object (displays automatically in Jupyter)
    
    Example:
        >>> # Plot distribution of returns
        >>> plot(df, 'returns', title='Return Distribution')
        >>> 
        >>> # Plot price changes
        >>> plot(df, 'price_change', title='Price Change Distribution')
    
    Note:
        The density estimation uses kernel density estimation (KDE)
        with basis interpolation for smooth curves.
    """
    return altair.Chart(df).mark_area(
        opacity=0.7,             # Semi-transparent fill
        interpolate='basis'      # Smooth curve interpolation
    ).transform_density(
        col,                     # Column to compute density for
        as_=[col, 'density']     # Output column names
    ).encode(
        x=altair.X(f'{col}:Q', title=col),           # X-axis: feature values
        y=altair.Y('density:Q', title='Density')     # Y-axis: probability density
    ).properties(
        width=600,
        height=400,
        title=title if title else f'Distribution of {col}'
    )

def plot_distribution(data: pl.DataFrame, col: str, label = None, no_bins = 100):
    return altair.Chart(data).mark_bar().encode(
        altair.X(f'{col}:Q', bin=altair.Bin(maxbins=no_bins)),
        y='count()'
    ).properties(
        width=600,
        height=400,
        title=f'Distribution of {label if label else col}'
    ).configure_scale(zero=False).add_params(
        altair.selection_interval(bind='scales')
)    

def plot_static_timeseries(ts: pl.DataFrame, sym: str, col: str, interval_size: str):
    plt.figure(figsize=(12, 6))
    plt.plot(ts['datetime'], ts[col], label=col)  # or whatever column you want
    plt.title(f'{sym} {interval_size} Bars')
    plt.xlabel('time')
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()  


def plot_multiple_lines(
    df: pl.DataFrame, 
    cols_to_plot: List[str], 
    sym: str, 
    width: int = 15, 
    height: int = 6, 
    xlabel_unit: str = "Time Step"
):
    import matplotlib.pyplot as plt 
    """
    Plots multiple columns from a Polars DataFrame on the same axes using Matplotlib.
    The x-axis uses a simple numerical index (since no datetime column is present).
    
    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame containing the columns to plot.
    cols_to_plot : list[str]
        A list of column names to plot (e.g., ['log_return', 'mean']).
    sym : str
        A symbol or identifier for the series (used in the title).
    width : int, default 15
        Width of the plot in inches.
    height : int, default 6
        Height of the plot in inches.
    xlabel_unit : str, default 'Time Step'
        Label for the X-axis (the numerical index).
    """
    
    # 1. Create the numerical index for the x-axis
    x_index = np.arange(len(df))
    
    # 2. Set the figure size (controls the width/height)
    plt.figure(figsize=(width, height))
    
    # 3. Loop through the list of columns and plot each one
    for col in cols_to_plot:
        if col in df.columns:
            # Extract column data as a NumPy array (efficient)
            y_values = df[col].to_numpy()
            
            # Plot the line, using the column name for the label
            plt.plot(x_index, y_values, label=col)
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # 4. Finalize the plot
    
    # Dynamically generate the title based on the symbol and columns
    title_cols = ', '.join(cols_to_plot)
    plt.title(f'{sym} Series: {title_cols}')
    
    plt.xlabel(xlabel_unit)
    plt.ylabel('Value') # Generic Y-label since multiple series are plotted
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    plt.show()  

def plot_dyn_timeseries(ts: pl.DataFrame, sym: str, col: str, time_interval: str ):
    return altair.Chart(ts).mark_line(tooltip=True).encode(
        x="datetime",
        y=col
    ).properties(
        width=800,
        height=400,
        title=f"{sym} {time_interval} {col}"
    ).configure_scale(zero=False).add_selection(
        altair.selection_interval(bind='scales', encodings=['x']),  # Only zoom x-axis
        altair.selection_interval(bind='scales', encodings=['y'])   # Only zoom y-axis
    )    

def to_tensor(x, dtype=None) -> torch.Tensor:
    return torch.tensor(x.to_numpy(), dtype=torch.float32 if dtype is None else dtype)

# ============================================================================
# MODEL ANALYSIS
# ============================================================================

def print_model_complexity_ratio(m1, m1_name, m2, m2_name):
    m1_params = total_model_params(m1)
    m2_params = total_model_params(m2)
    complexity_ratio = m2_params / m1_params

    print(f"Complexity Comparsion:")
    print(f"\t{m2_name} has {complexity_ratio:.1f}x more parameters than {m1_name}")
    print(f"\tParametric difference: {m2_params - m1_params:,} additional parameters")    

def total_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def print_model_info(model: torch.nn.Module, model_name: str) -> None:
    """
    Print detailed information about a PyTorch model's architecture and parameters.
    
    This function helps you understand:
    - Model complexity (number of parameters)
    - Which parameters are trainable vs frozen
    - Overall model architecture
    
    Useful for:
    - Comparing different model architectures
    - Debugging training issues
    - Estimating memory requirements
    - Understanding model capacity
    
    Args:
        model: PyTorch model (nn.Module)
        model_name: Descriptive name for the model (e.g., 'LSTM Predictor')
    
    Returns:
        None (prints to console)
    
    Example:
        >>> model = MyTradingModel(input_size=10, hidden_size=64)
        >>> print_model_info(model, 'Trading LSTM v1')
        
        Output:
        Trading LSTM v1:
          Architecture: MyTradingModel(...)
          Total parameters: 15,234
          Trainable parameters: 15,234
    
    Note:
        Total parameters includes both trainable and frozen parameters.
        For transfer learning, trainable params may be less than total.
    """
    # Count all parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count only parameters that will be updated during training
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    # Print formatted model information
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"\nArchitecture:")
    print(f"  {model}")
    print(f"\nParameter Count:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    
    # Warn if some parameters are frozen
    if total_params != trainable_params:
        frozen_params = total_params - trainable_params
        print(f"  Frozen parameters:     {frozen_params:,}")
        print(f"\n  ⚠️  Note: {frozen_params:,} parameters are frozen")
    
    print(f"{'='*60}\n")

def _prefix_cols(df, prefix):
    return df.rename({col: f"{prefix}_{col}" for col in df.columns})

def _prefix_close_ts(trades, time_interval, prefix):
    return _prefix_cols(ohlc_timeseries(trades, time_interval), prefix)

def compare_ts_corr(x_df, x_prefix, y_df, y_prefix, time_interval, col = 'close'):
    x_col, y_col = f'{x_prefix}_{col}',f'{y_prefix}_{col}'
    joined_ts = pl.concat([
        _prefix_close_ts(x_df, time_interval, x_prefix), 
        _prefix_close_ts(y_df, time_interval, y_prefix)
    ], how="horizontal")
    return joined_ts.select(pl.corr(x_col, y_col)).item()

def log_return_col(col: str) -> str:
    return f"{col}_log_return"

def log_return(col: str, shift_size: int = 1) -> pl.Expr:
    return (pl.col(col)/pl.col(col).shift(shift_size)).log().alias(log_return_col(col))

def lag_cols(col: str, forecast_horizon: str, no_lags: int) -> List[pl.Expr]:
    return [pl.col(col).shift(forecast_horizon * i).alias(f'{col}_lag_{i}') for i in range(1, no_lags + 1)]

def add_lags(df: pl.DataFrame, col: str, max_no_lags: int, forecast_step: int) -> pl.DataFrame:
    return df.with_columns([pl.col(col).shift(i * forecast_step).alias(f'{col}_lag_{i}') for i in range(1, max_no_lags + 1)])

def batch_train_reg(
    model: nn.Module,
    X_train,
    X_test,
    y_train,
    y_test,
    no_epochs: int,
    criterion=None,
    optimizer=None,
    logging=True,
    lr=None
):
    if criterion is None:
        criterion = nn.L1Loss()

    if lr is None:
        lr = 0.0002

    # Default optimizer
    if optimizer is None:
        # Use strong_wolfe line search (more stable)
        optimizer = optim.LBFGS(
            model.parameters(), 
            lr=1,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )

    # Logging model info
    if logging:
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
        print("Model architecture:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} params)")
        print("\nTraining model...")

    train_loss = None
    log_tick_size = max(no_epochs // 10, 1)  # avoid zero division

    # Training loop
    if isinstance(optimizer, torch.optim.LBFGS):
        # LBFGS requires a closure
        for epoch in range(no_epochs):
            def closure():
                optimizer.zero_grad()
                predictions = model(X_train)
                loss = criterion(predictions, y_train)
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                train_loss = criterion(model(X_train), y_train).item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {train_loss:.6f}")

    else:
        # SGD/Adam loop
        for epoch in range(no_epochs):
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if logging and (epoch + 1) % log_tick_size == 0:
                print(f"Epoch [{epoch+1}/{no_epochs}], Loss: {loss.item():.6f}")

    # After training
    if logging:
        print("\nLearned parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}:\n{param.data.numpy()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
        test_loss = criterion(y_hat, y_test)
        if logging:
            print(f'\nTest Loss: {test_loss.item():.6f}, Train Loss: {train_loss:.6f}') 

    return y_hat


def timeseries_train_test_split(df: pl.DataFrame, features, target, test_size=0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = df.drop_nulls()
    X = to_tensor(df[features])
    y = to_tensor(df[target]).reshape(-1, 1)
    X_train, X_test = timeseries_split(X, test_size)
    y_train, y_test = timeseries_split(y, test_size)
    return X_train, X_test, y_train, y_test 

def timeseries_split(t, test_size=0.25):
    """
    Split a tensor or array into train/test sets based on a proportion.

    Parameters
    ----------
    t : torch.Tensor or np.ndarray
        Time series data.
    test_size : float, default 0.25
        Proportion of data to use for testing. Must be between 0 and 1.

    Returns
    -------
    train, test : same type as t
        Train and test splits.

    Raises
    ------
    ValueError
        If test_size is not strictly between 0 and 1.
    """
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1 (got {test_size})")

    split_idx = int(len(t) * (1 - test_size))
    return t[:split_idx], t[split_idx:]


def plot_column(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    """
    Plot a column from a Polars DataFrame using matplotlib.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame
    column_name : str
        Name of the column to plot
    figsize : tuple, default (15, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, uses column name
    xlabel : str, default 'Index'
        X-axis label
    ylabel : str, optional
        Y-axis label. If None, uses column name
    """

    if title is None:
        title = col_name

    chart = df[col_name].plot.line()
    return chart.properties(
        width=800,
        height=400,
        title=title
    )


def plot_columns(df, col_name, figsize=(15, 6), title=None, xlabel='Index'):
    """
    Plot a columns from a Polars DataFrame using matplotlib.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The Polars DataFrame
    column_name : str
        Name of the column to plot
    figsize : tuple, default (15, 6)
        Figure size as (width, height) in inches
    title : str, optional
        Plot title. If None, uses column name
    xlabel : str, default 'Index'
        X-axis label
    ylabel : str, optional
        Y-axis label. If None, uses column name
    """    
    if title is None:
        title = col_name

    chart = df[col_name].plot.line()
    return chart.properties(
        width=800,
        height=400,
        title=title
    )


def model_trade_results(y_true, y_pred) -> pl.DataFrame:
    """Generate trade-level results from model predictions."""

    trade_results =  pl.DataFrame({
        'y_pred': y_pred.squeeze(),
        'y_true': y_true.squeeze()
    }).with_columns([
        (pl.col('y_pred').sign() == pl.col('y_true').sign()).alias('is_won'),
        pl.col('y_pred').sign().alias('position')
    ]).with_columns([
        (pl.col('position') * pl.col('y_true')).alias('trade_log_return')
    ]).with_columns([
        pl.col('trade_log_return').cum_sum().alias('equity_curve')
    ]).with_columns(
        (pl.col('equity_curve')-pl.col('equity_curve').cum_max()).alias('drawdown_log_return'),
    )
    return trade_results


def add_tx_fee(trades: pl.DataFrame, tx_fee: float, name: str):
    tx_fee_col = (pl.col('exit_trade_value') * tx_fee + pl.col('entry_trade_value') * tx_fee).alias(f"tx_fee_{name}")
    return trades.with_columns(tx_fee_col)


def add_tx_fees(trades: pl.DataFrame, maker_fee: float, taker_fee: float):
    trades = add_tx_fee(trades, maker_fee, 'maker')
    trades = add_tx_fee(trades, taker_fee, 'taker')
    return trades  

def add_tx_fees_log(trades: pl.DataFrame, maker_fee, taker_fee):
    return trades.with_columns(
        (pl.col('trade_log_return') + np.log(maker_fee)).alias('trade_log_return_net_maker'),
        (pl.col('trade_log_return') + np.log(taker_fee)).alias('trade_log_return_net_taker'),
    ).with_columns(
        pl.col('trade_log_return_net_maker').cum_sum().alias('equity_curve_net_maker'),
        pl.col('trade_log_return_net_taker').cum_sum().alias('equity_curve_net_taker'),
    )

def eval_model_performance(y_actual, y_pred, feature_names: List[str], target_name: str, annualized_rate: float) -> Dict[str, any]:
    """Calculate performance metrics for the trading model."""
    trade_results = model_trade_results(y_actual, y_pred)
    
    accuracy = trade_results['is_won'].mean()
    avg_win = trade_results.filter(pl.col('is_won'))['trade_log_return'].mean()
    avg_loss = trade_results.filter(~pl.col('is_won'))['trade_log_return'].mean()
    expected_value = accuracy * avg_win + (1 - accuracy) * avg_loss
    drawdown = (trade_results['equity_curve'] - trade_results['equity_curve'].cum_max())
    max_drawdown = drawdown.min()
    sharpe = trade_results['trade_log_return'].mean() / trade_results['trade_log_return'].std() if trade_results['trade_log_return'].std() > 0 else 0
    annualized_sharpe = sharpe * annualized_rate
    equity_trough = trade_results['equity_curve'].min()
    equity_peak = trade_results['equity_curve'].max()
    total_log_return = trade_results['trade_log_return'].sum()
    std = trade_results['trade_log_return'].std()
    return {
        'features': ','.join(list(feature_names)),
        'target': target_name,
        'no_trades': len(trade_results),
        'win_rate': accuracy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'best_trade': trade_results['trade_log_return'].max(),
        'worst_trade': trade_results['trade_log_return'].min(),
        'ev': expected_value,
        'std': std,
        'total_log_return': total_log_return,
        'compound_return': np.exp(total_log_return),
        'max_drawdown': max_drawdown,
        'equity_trough': equity_trough,
        'equity_peak': equity_peak,
        'sharpe': annualized_sharpe,
    }

def train_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss = None, optimizer = None, no_epochs = None, log = False, lr = None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(),dtype=torch.float32).reshape(-1, 1)

    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr = lr, logging = log)

    

def benchmark_reg_model(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, annualized_rate, test_size=0.25, loss = None, optimizer = None, no_epochs = None, log = False, lr = None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(),dtype=torch.float32).reshape(-1, 1)
    
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr = lr, logging = log)
    
    perf = eval_model_performance(y_test, y_hat, features, target, annualized_rate)
    
    weights, biases = get_linear_params(model)
    perf['weights'] = str(weights)
    perf['biases'] = str(biases)
    
    return perf


def learn_model_trades(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss = None, optimizer = None, no_epochs = None, log = False, lr = None):
    df = df.drop_nulls()
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(),dtype=torch.float32).reshape(-1, 1)
    
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, criterion=loss, optimizer=optimizer, lr = lr, logging = log)

    return model_trade_results(y_test, y_hat)
        
def learn_model_trade_pnl(df: pl.DataFrame, features: List[str], target: str, model: nn.Module, test_size=0.25, loss = None, optimizer = None, no_epochs = None, log = False, lr = None):
    df_train, df_test = timeseries_split(df, test_size=test_size)
    if no_epochs is None:
        no_epochs = 6000
    
    X_train, y_train = torch.tensor(df_train[features].to_numpy(), dtype=torch.float32), torch.tensor(df_train[target].to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test, y_test = torch.tensor(df_test[features].to_numpy(), dtype=torch.float32), torch.tensor(df_test[target].to_numpy(),dtype=torch.float32).reshape(-1, 1)
    
    y_hat = batch_train_reg(model, X_train, X_test, y_train, y_test, no_epochs, loss, optimizer, lr = lr, logging = log)

    trade_results = model_trade_results(y_test, y_hat)

      

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(42)  # ensures same init every time
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_linear_params(model: nn.Module) -> tuple[np.ndarray, float]:
    """Extract weights and bias from LinearModel as (w, b)."""
    weight = model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.linear.bias.detach().cpu().numpy().item()
    return weight, bias

def add_log_return_features(df: pl.DataFrame, col: str, forecast_horizon: int, max_no_lags = None):
    if max_no_lags is None:
        max_no_lags = 0
    df = df.with_columns(log_return(col, forecast_horizon))
    if max_no_lags > 0:
        df = add_lags(df, log_return_col('close'), max_no_lags, forecast_horizon)
    return df

def benchmark_linear_models(ts: pl.DataFrame, target: str, feature_pool: List[str], annualized_rate: int, max_no_features: int = 1, no_epochs = 200, loss = None, test_size=0.25) -> pl.DataFrame:
    import models
    
    ts = ts.drop_nulls()

    benchmarks = []
    fs = []
    for i in range(1, max_no_features+1):
        fs += list(itertools.combinations(feature_pool, i))
    
    for features in fs:
        m = models.LinearModel(len(features))
        m.apply(init_weights)
        benchmarks.append(benchmark_reg_model(ts, list(features), target, m, annualized_rate, no_epochs=no_epochs, loss=loss, test_size=test_size))

    benchmark = pl.DataFrame(benchmarks)
    return benchmark.sort('sharpe', descending=True)  


# print out our learned params
def print_model_params(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:\n{param.data.numpy()}")  


def add_model_predictions(test_trades: pl.DataFrame, model: nn.Module, features: Union[str, List[str]]) -> pl.DataFrame:
    if type(features) != list:
        features = [features]
    X_test = torch.tensor(test_trades[features].to_numpy(), dtype=torch.float32)
    y_hat = model(X_test)
    s = pl.Series('y_hat', model(X_test).detach().cpu().numpy().squeeze())
    return test_trades.with_columns(s)


def add_trade_log_returns(trades: pl.DataFrame, pre_trade_values: Union[List[float],npt.NDArray[np.float32]], tx_fee: float, initial_capital: float) -> pl.DataFrame:
    # add directional signal to indicate if we're going long or short
    trades = trades.with_columns(pl.col('y_hat').sign().alias('dir_signal'))
    # calculate trade log return
    trades = trades.with_columns((pl.col('close_log_return') * pl.col('dir_signal')).alias('trade_log_return'))
    # calculate the cumulative sum of the trade log returns - this is the equity curves in log space
    trades = trades.with_columns(pl.col('trade_log_return').cum_sum().alias('cum_trade_log_return'))
    trades = trades.with_columns(
        # add pre trade values
        pre_trade_values.alias('pre_trade_value'),
        # add post trade values
        (pre_trade_values * pl.col('trade_log_return').exp()).alias('post_trade_value'),
        # add trade qty
        (pre_trade_values / pl.col('open')).alias('trade_qty'),
    )
    
    trades = trades.with_columns(
        # add signed trade quantities (the main output of our strategy)
        (pl.col('trade_qty') * pl.col('dir_signal')).alias('signed_trade_qty'),
        # add trade gross pnl
        (pl.col('post_trade_value') - pl.col('pre_trade_value')).alias('trade_gross_pnl')
        # add tx fees
        (pl.col('pre_trade_value') * tx_fee + pl.col('post_trade_value') * tx_fee).alias('tx_fees')
    )
    trades = trades.with_columns(
        # calculate each trade's profit after fees (net)
        (pl.col('trade_gross_pnl')-pl.col('tx_fees')).alias('trade_net_pnl')
    )
    trades = trades.with_columns(
        # calculate equity curve for gross profit
        (initial_capital + pl.col('trade_gross_pnl').cum_sum()).alias('equity_curve_gross')
        # calculate equity curve for net profit
        (initial_capital + pl.col('trade_net_pnl').cum_sum()).alias('equity_curve_net')        
    )

def add_equity_curve(trades: pl.DataFrame, initial_capital: float, col_name: str, suffix: str) -> pl.DataFrame:
    return trades.with_columns(
        (initial_capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')
    )

def add_compounding_trades(trades, capital, leverage, maker_fee, taker_fee):
    lev_capital = capital * leverage
    # calculate entry and exit trade value and size
    trades = trades.with_columns(
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).shift().fill_null(lev_capital).alias('entry_trade_value'),
        ((pl.col('cum_trade_log_return').exp()) * lev_capital).alias('exit_trade_value'),
    ).with_columns(
        (pl.col('entry_trade_value') / pl.col('open') * pl.col('dir_signal')).alias('signed_trade_qty'),
        (pl.col('exit_trade_value')-pl.col('entry_trade_value')).alias('trade_gross_pnl'),
    )
    # add transaction fee
    trades = add_tx_fees(trades, maker_fee, taker_fee)
    # add net trade pnl
    trades = trades.with_columns(
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_taker')).alias('trade_net_taker_pnl'),
        (pl.col('trade_gross_pnl') - pl.col('tx_fee_maker')).alias('trade_net_maker_pnl'),
    )
    trades = add_equity_curve(trades, capital, 'trade_gross_pnl', 'gross')
    # add net equity curves (both taker and maker)  
    trades = add_equity_curve(trades, capital, 'trade_net_taker_pnl', 'taker')
    trades = add_equity_curve(trades, capital, 'trade_net_maker_pnl', 'maker')
    return trades

