from typing import List
import requests
import zipfile
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
from tqdm import tqdm

import research

MAKER_FEE = 0.000450
TAKER_FEE = 0.000450

def download_and_unzip(symbol: str, date: str | datetime,
                       download_dir: str = "data", cache_dir: str = "cache") -> pl.DataFrame:
    """
    Download and unzip Binance futures trade data for a given symbol and date.
    Caches results as parquet files to avoid repeated downloads.
    """
    # Normalize date to string
    date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{symbol}-trades-{date_str}.parquet"

    if cache_path.exists():
        return pl.read_parquet(cache_path)

    url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)
    zip_path = download_dir / f"{symbol}-trades-{date_str}.zip"

    # Download zip
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(download_dir)

    csv_path = download_dir / f"{symbol}-trades-{date_str}.csv"

    # Load into Polars
    df = pl.read_csv(
        csv_path,
        schema={
            "id": pl.Int64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "quoteQty": pl.Float64,
            "time": pl.Int64,
            "isBuyerMaker": pl.Boolean,
        }
    ).with_columns(
        pl.from_epoch("time", time_unit="ms").alias("datetime")
    )

    # Cache and clean
    df.write_parquet(cache_path)
    zip_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)

    return df


def download_date_range(symbol: str, start_date: str | datetime, end_date: str | datetime,
                        download_dir: str = "data", cache_dir: str = "cache") -> list[pl.DataFrame]:
    """
    Download trade data for a range of dates with a progress bar.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    num_days = (end_date - start_date).days + 1

    for i in tqdm(range(num_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            download_and_unzip(symbol, current_date, download_dir, cache_dir)
        except Exception as e:
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")



def download_trades(symbol: str, no_days: int,
                    download_dir: str = "data", cache_dir: str = "cache", return_trades=False) -> pl.DataFrame:
    """
    Download trades for the last N days up to yesterday with a progress bar.
    """
    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday - timedelta(days=no_days - 1)

    dfs = []
    for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            if return_trades:
                dfs.append(download_and_unzip(symbol, current_date, download_dir, cache_dir))
            else:
                download_and_unzip(symbol, current_date, download_dir, cache_dir)
        except Exception as e:
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
    
    return pl.concat(dfs) if return_trades else None


def download_ohlc_timeseries(symbol: str, no_days: int, time_interval: str, download_dir: str = "data", cache_dir: str = "cache") -> pl.DataFrame:
    """
    Download trades for the last N days up to yesterday with a progress bar.
    """
    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday - timedelta(days=no_days - 1)

    time_series = []
    for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            trades = download_and_unzip(symbol, current_date, download_dir, cache_dir)
            time_series.append(research.timeseries(trades, time_interval, research.OHLC_AGGS))
        except Exception as e:
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
    return pl.concat(time_series)


def download_timeseries(symbol: str, no_days: int, time_interval: str, aggs: List[pl.Expr], download_dir: str = "data", cache_dir: str = "cache") -> pl.DataFrame:
    """
    Download trades for the last N days up to yesterday with a progress bar.
    """
    yesterday = datetime.now() - timedelta(days=1)
    start_date = yesterday - timedelta(days=no_days - 1)

    time_series = []
    for i in tqdm(range(no_days), desc=f"Downloading {symbol}"):
        current_date = start_date + timedelta(days=i)
        try:
            trades = download_and_unzip(symbol, current_date, download_dir, cache_dir)
            time_series.append(research.timeseries(trades, time_interval, aggs))
        except Exception as e:
            tqdm.write(f"[ERROR] {symbol} {current_date.date()}: {e}")
    return pl.concat(time_series)