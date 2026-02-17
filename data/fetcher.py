"""
Data fetching module — downloads intraday data via yfinance and
resamples to required timeframes.
"""

import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import pytz
import yfinance as yf

from config import INSTRUMENTS, NY_TZ

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, start: str, end: str, interval: str) -> str:
    safe = symbol.replace("=", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}_{interval}.pkl")


def fetch_intraday(
    symbol: str,
    ticker: str,
    start: str,
    end: str,
    interval: str = "1m",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV data for a single instrument.
    yfinance limits 1m data to last 30 days per request, so we chunk.
    Returns DataFrame indexed by NY-timezone datetime.
    """
    cp = _cache_path(ticker, start, end, interval)
    if use_cache and os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)

    ny = pytz.timezone(NY_TZ)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    # yfinance 1m data: max 7 days per request
    chunk_days = 7 if interval == "1m" else 59
    frames = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            df = yf.download(
                ticker,
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception as e:
            print(f"  Warning: chunk {cursor.date()}-{chunk_end.date()} failed: {e}")
        cursor = chunk_end

    if not frames:
        print(f"  No data fetched for {symbol} ({ticker})")
        return pd.DataFrame()

    data = pd.concat(frames)
    data = data[~data.index.duplicated(keep="first")]
    data.sort_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Localise to NY time
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC").tz_convert(ny)
    else:
        data.index = data.index.tz_convert(ny)

    if use_cache:
        with open(cp, "wb") as f:
            pickle.dump(data, f)

    return data


def resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to 5-minute bars."""
    if df_1m.empty:
        return df_1m
    return df_1m.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])


def fetch_all_instruments(start: str, end: str) -> dict:
    """
    Fetch 1-min data for all configured instruments.
    Returns dict: {symbol: {"1m": df_1m, "5m": df_5m}}
    """
    result = {}
    for symbol, ticker in INSTRUMENTS.items():
        print(f"Fetching {symbol} ({ticker})...")
        df_1m = fetch_intraday(symbol, ticker, start, end, interval="1m")
        df_5m = resample_to_5min(df_1m)
        result[symbol] = {"1m": df_1m, "5m": df_5m}
        print(f"  {symbol}: {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    return result
