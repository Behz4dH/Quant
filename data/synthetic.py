"""
Synthetic Data Generator
========================
Generates realistic 1-minute OHLCV data for backtesting when live data
is unavailable. Models intraday patterns including:
  - Opening volatility spike at 9:30 NY
  - Trending sessions with institutional impulse moves
  - Fair Value Gap structures (3-candle imbalances)
  - Retracement behavior after impulse moves
  - Realistic volume profiles
"""

import numpy as np
import pandas as pd
import pytz

from config import NY_TZ

# Instrument characteristics for realistic simulation
SYNTH_PROFILES = {
    "EURUSD": {
        "base_price": 1.0850,
        "daily_vol": 0.005,
        "tick_size": 0.00001,
        "avg_spread": 0.00010,
        "volume_base": 1000,
    },
    "GBPUSD": {
        "base_price": 1.2650,
        "daily_vol": 0.006,
        "tick_size": 0.00001,
        "avg_spread": 0.00012,
        "volume_base": 800,
    },
    "NASDAQ": {
        "base_price": 21500.0,
        "daily_vol": 0.012,
        "tick_size": 0.25,
        "avg_spread": 0.50,
        "volume_base": 5000,
    },
    "DOW": {
        "base_price": 44000.0,
        "daily_vol": 0.008,
        "tick_size": 1.0,
        "avg_spread": 1.0,
        "volume_base": 3000,
    },
}


def _intraday_volume_profile(n_minutes: int) -> np.ndarray:
    """U-shaped intraday volume (high at open/close, low midday)."""
    x = np.linspace(0, 1, n_minutes)
    profile = 2.0 * (x - 0.5) ** 2 + 0.5
    profile[:10] *= 2.5  # opening spike
    profile[-10:] *= 1.5  # closing spike
    return profile / profile.mean()


def _generate_day(
    profile: dict,
    date: pd.Timestamp,
    seed_price: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Generate one day of 1-minute OHLCV data (9:30 to 16:00 NY).

    Models a realistic session:
    - Phase 1 (9:30-9:35): Opening range — contained, builds up
    - Phase 2 (9:35-9:50): Breakout + impulse with FVG gaps
    - Phase 3 (9:50-10:15): Retracement back toward FVG zones
    - Phase 4 (10:15-16:00): Continuation or mean reversion
    """
    ny = pytz.timezone(NY_TZ)
    start = ny.localize(pd.Timestamp(date.year, date.month, date.day, 9, 30))
    end = ny.localize(pd.Timestamp(date.year, date.month, date.day, 16, 0))

    n_minutes = int((end - start).total_seconds() / 60)  # 390
    times = pd.date_range(start, periods=n_minutes, freq="1min")
    tick = profile["tick_size"]
    base_vol = profile["daily_vol"] / np.sqrt(n_minutes)

    # Decide session type and direction
    session_trend = rng.choice([-1, 1])  # daily bias
    trend_strength = rng.uniform(0.4, 1.5)  # how strong the trend is
    # ~60% of days are trending, ~40% are range-bound
    is_trending = rng.random() < 0.60

    prices = np.zeros(n_minutes)
    prices[0] = seed_price

    # ── Phase 1: Opening Range (0-5 min) — tight consolidation ──
    or_vol = base_vol * 0.8
    for i in range(1, min(5, n_minutes)):
        prices[i] = prices[i - 1] * (1 + rng.normal(0, or_vol))

    # ── Phase 2: Breakout (5-20 min) — impulse move with gaps ──
    if is_trending:
        impulse_mag = base_vol * trend_strength * 3.0
        for i in range(5, min(20, n_minutes)):
            drift = session_trend * impulse_mag * 0.3
            noise = rng.normal(0, base_vol * 1.5)
            # Create 2-3 impulse candles (FVG generators)
            if i in [7, 8, 9, 12, 13]:
                drift *= 2.5  # strong impulse candles
                noise *= 0.3  # less noise, more direction
            prices[i] = prices[i - 1] * (1 + drift + noise)
    else:
        for i in range(5, min(20, n_minutes)):
            prices[i] = prices[i - 1] * (1 + rng.normal(0, base_vol * 1.2))

    # ── Phase 3: Retracement (20-45 min) — pullback toward FVG ──
    if is_trending:
        retrace_depth = rng.uniform(0.3, 0.7)  # retrace 30-70% of impulse
        impulse_move = prices[19] - prices[4]
        retrace_target = prices[19] - impulse_move * retrace_depth

        for i in range(20, min(45, n_minutes)):
            progress = (i - 20) / 25
            # Gradual pullback toward retrace target
            target_pull = (retrace_target - prices[i - 1]) * 0.08
            noise = rng.normal(0, base_vol * 0.8)
            prices[i] = prices[i - 1] * (1 + noise) + target_pull

            # Add rejection/engulfing behavior near FVG zone (~65% of trending days)
            if progress > 0.5 and rng.random() < 0.04:
                # Sharp reversal candle (engulfing)
                prices[i] = prices[i - 1] * (1 + session_trend * base_vol * 2.5)
    else:
        for i in range(20, min(45, n_minutes)):
            prices[i] = prices[i - 1] * (1 + rng.normal(0, base_vol * 0.9))

    # ── Phase 4: Continuation / Range (45-390 min) ──
    if is_trending:
        # 60% chance of continuation, 40% range-bound rest of day
        continues = rng.random() < 0.60
        for i in range(45, n_minutes):
            if continues:
                drift = session_trend * base_vol * 0.15 * trend_strength
            else:
                drift = 0
            # Gradually reduce volatility
            decay = max(0.4, 1 - (i - 45) / (n_minutes - 45) * 0.5)
            noise = rng.normal(0, base_vol * decay)
            # Mild mean reversion
            mean_price = prices[max(0, i - 20):i].mean() if i > 20 else prices[i - 1]
            mr = (mean_price - prices[i - 1]) * 0.01
            prices[i] = prices[i - 1] * (1 + drift + noise) + mr
    else:
        for i in range(45, n_minutes):
            mean_price = prices[max(0, i - 30):i].mean() if i > 30 else prices[i - 1]
            mr = (mean_price - prices[i - 1]) * 0.02
            noise = rng.normal(0, base_vol * 0.7)
            prices[i] = prices[i - 1] * (1 + noise) + mr

    # Round to tick
    close_prices = np.round(prices / tick) * tick

    # Generate OHLC
    vol_profile = _intraday_volume_profile(n_minutes)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = seed_price

    wick_size = np.abs(rng.normal(0, base_vol * seed_price * 0.3, n_minutes))
    highs = np.maximum(open_prices, close_prices) + wick_size
    lows = np.minimum(open_prices, close_prices) - np.abs(rng.normal(0, base_vol * seed_price * 0.3, n_minutes))

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(open_prices, close_prices))
    lows = np.minimum(lows, np.minimum(open_prices, close_prices))
    highs = np.round(highs / tick) * tick
    lows = np.round(lows / tick) * tick

    volume = (profile["volume_base"] * vol_profile * rng.uniform(0.5, 1.5, n_minutes)).astype(int)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": highs,
        "Low": lows,
        "Close": close_prices,
        "Volume": volume,
    }, index=times)

    return df, close_prices[-1]


def generate_synthetic_data(
    symbol: str,
    start: str,
    end: str,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic 1-min and 5-min data for a symbol over a date range.
    Returns {"1m": df_1m, "5m": df_5m}.
    """
    profile = SYNTH_PROFILES.get(symbol)
    if profile is None:
        raise ValueError(f"No synthetic profile for {symbol}")

    rng = np.random.default_rng(seed + hash(symbol) % 10000)

    dates = pd.bdate_range(start, end)
    all_frames = []
    current_price = profile["base_price"]

    for date in dates:
        df_day, current_price = _generate_day(profile, date, current_price, rng)
        all_frames.append(df_day)

    if not all_frames:
        return {"1m": pd.DataFrame(), "5m": pd.DataFrame()}

    df_1m = pd.concat(all_frames)

    df_5m = df_1m.resample("5min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])

    return {"1m": df_1m, "5m": df_5m}


def generate_all_synthetic(start: str, end: str, seed: int = 42) -> dict:
    """Generate synthetic data for all instruments."""
    result = {}
    for symbol in SYNTH_PROFILES:
        print(f"Generating synthetic data for {symbol}...")
        data = generate_synthetic_data(symbol, start, end, seed)
        print(f"  {symbol}: {len(data['1m'])} 1m bars, {len(data['5m'])} 5m bars")
        result[symbol] = data
    return result
