"""
FVG Opening Range Strategy — Configuration
"""

# Instruments and their yfinance tickers
INSTRUMENTS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "NASDAQ": "NQ=F",
    "DOW": "YM=F",
}

# Instrument metadata
INSTRUMENT_META = {
    "EURUSD": {"pip": 0.0001, "point_value": 100_000, "type": "forex"},
    "GBPUSD": {"pip": 0.0001, "point_value": 100_000, "type": "forex"},
    "NASDAQ": {"pip": 0.25,   "point_value": 20,      "type": "index"},
    "DOW":    {"pip": 1.0,    "point_value": 5,        "type": "index"},
}

# Opening range window (New York time)
OR_START_HOUR = 9
OR_START_MIN = 30
OR_END_HOUR = 9
OR_END_MIN = 35
NY_TZ = "America/New_York"

# Strategy defaults
DEFAULT_PARAMS = {
    "fvg_min_body_pct": 0.5,       # min body % of candle range for FVG candles
    "fvg_min_gap_atr_mult": 0.3,   # min gap size as multiple of ATR
    "retracement_min_pct": 0.25,   # price must retrace at least 25% into FVG
    "retracement_max_pct": 1.0,    # price must not exceed 100% of FVG
    "engulf_min_body_ratio": 1.2,  # engulfing body must be >= 1.2x prior body
    "atr_period": 14,
    "sl_atr_mult": 1.5,            # stop loss = ATR * mult
    "tp_rr_ratio": 2.0,            # take profit = risk * RR ratio
    "max_daily_trades": 2,
    "session_end_hour": 16,        # close all trades by 4pm NY
    "session_end_min": 0,
    "lookback_candles": 30,        # how many 1-min candles after OR to scan
}

# Backtest settings
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2025-12-31"
INITIAL_CAPITAL = 100_000
RISK_PER_TRADE_PCT = 1.0  # risk 1% of capital per trade

# Parameter grid for variant testing
PARAM_VARIANTS = {
    "conservative": {
        "fvg_min_gap_atr_mult": 0.5,
        "retracement_min_pct": 0.4,
        "retracement_max_pct": 0.8,
        "engulf_min_body_ratio": 1.5,
        "sl_atr_mult": 0.5,           # tight SL anchored to FVG zone
        "tp_rr_ratio": 3.0,           # higher RR for selectivity
        "lookback_candles": 20,
        "max_daily_trades": 1,
    },
    "balanced": {
        "fvg_min_gap_atr_mult": 0.3,
        "retracement_min_pct": 0.25,
        "retracement_max_pct": 1.0,
        "engulf_min_body_ratio": 1.2,
        "sl_atr_mult": 0.8,
        "tp_rr_ratio": 2.0,
        "lookback_candles": 30,
        "max_daily_trades": 2,
    },
    "aggressive": {
        "fvg_min_gap_atr_mult": 0.2,
        "retracement_min_pct": 0.15,
        "retracement_max_pct": 1.2,
        "engulf_min_body_ratio": 1.0,
        "sl_atr_mult": 0.5,
        "tp_rr_ratio": 1.5,
        "lookback_candles": 40,
        "max_daily_trades": 2,
    },
}
