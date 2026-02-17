#!/usr/bin/env python3
"""
FVG Opening Range Strategy — Main Runner
==========================================
Fetches data, runs strategy across all instruments and parameter variants,
executes backtests, and generates a full comparison report.

Usage:
    python main.py                    # Run with default date range
    python main.py --start 2025-01-01 --end 2025-06-30
    python main.py --symbols EURUSD NASDAQ
    python main.py --synthetic         # Use synthetic data (for testing)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BACKTEST_END,
    BACKTEST_START,
    DEFAULT_PARAMS,
    INITIAL_CAPITAL,
    INSTRUMENTS,
    PARAM_VARIANTS,
    RISK_PER_TRADE_PCT,
)
from strategies.fvg_opening_range import FVGOpeningRangeStrategy
from backtest.engine import BacktestEngine
from backtest.report import generate_full_report


def _fetch_data(symbols, start, end, use_synthetic):
    """Fetch data from yfinance or generate synthetic data."""
    if use_synthetic:
        from data.synthetic import generate_all_synthetic
        print("Using SYNTHETIC data for backtesting")
        return generate_all_synthetic(start, end)
    else:
        from data.fetcher import fetch_all_instruments
        all_data = fetch_all_instruments(start, end)
        # Fall back to synthetic if all instruments returned empty
        has_data = any(
            not all_data[s]["1m"].empty
            for s in all_data
            if s in symbols or symbols is None
        )
        if not has_data:
            print("\nLive data unavailable — falling back to synthetic data")
            from data.synthetic import generate_all_synthetic
            return generate_all_synthetic(start, end)
        return all_data


def run_backtest(
    symbols: list = None,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    use_synthetic: bool = False,
):
    """
    Main entry point: fetch data -> run strategy -> backtest -> report.
    """
    if symbols is None:
        symbols = list(INSTRUMENTS.keys())

    print(f"\nFVG Opening Range Strategy Backtest")
    print(f"Period: {start} to {end}")
    print(f"Instruments: {', '.join(symbols)}")
    print(f"Variants: {', '.join(PARAM_VARIANTS.keys())}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Risk per Trade: {RISK_PER_TRADE_PCT}%")
    print("-" * 60)

    # Fetch data
    all_data = _fetch_data(symbols, start, end, use_synthetic)

    # Run strategy + backtest for each symbol x variant
    all_results = []

    for symbol in symbols:
        if symbol not in all_data:
            print(f"\nSkipping {symbol} — no data available")
            continue

        data = all_data[symbol]
        df_1m = data["1m"]
        df_5m = data["5m"]

        if df_1m.empty:
            print(f"\nSkipping {symbol} — empty dataset")
            continue

        print(f"\n{'=' * 40}")
        print(f"Processing {symbol}: {len(df_1m)} 1-min bars")
        print(f"{'=' * 40}")

        for variant_name, variant_params in PARAM_VARIANTS.items():
            print(f"\n  Running variant: {variant_name}...")

            # Merge default params with variant overrides
            params = {**DEFAULT_PARAMS, **variant_params}
            strategy = FVGOpeningRangeStrategy(params)
            signals = strategy.run(df_1m, df_5m)

            print(f"    Signals generated: {len(signals)}")

            # Run backtest
            engine = BacktestEngine(symbol, INITIAL_CAPITAL, RISK_PER_TRADE_PCT)
            result = engine.simulate(signals, df_1m, variant_name)
            all_results.append(result)

            print(f"    Trades executed: {result.total_trades}")
            if result.total_trades > 0:
                print(f"    Win rate: {result.win_rate:.1f}%")
                print(f"    Return: {result.total_return_pct:.2f}%")

    # Generate full report
    if all_results:
        best = generate_full_report(all_results)
        return all_results, best
    else:
        print("\nNo results to report — check data availability.")
        return [], None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FVG Opening Range Strategy Backtest")
    parser.add_argument("--start", default=BACKTEST_START, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=BACKTEST_END, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to test")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    results, best = run_backtest(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        use_synthetic=args.synthetic,
    )
