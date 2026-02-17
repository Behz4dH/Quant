"""
Backtesting Engine
==================
Simulates trade execution from strategy signals against 1-min price data.
Tracks equity curve, drawdowns, and per-trade statistics.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

from config import INITIAL_CAPITAL, INSTRUMENT_META, NY_TZ, RISK_PER_TRADE_PCT


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pct: float
    result: str  # "win", "loss", "breakeven", "timeout"
    risk_amount: float
    r_multiple: float


@dataclass
class BacktestResult:
    """Full backtest output."""
    symbol: str
    variant: str
    trades: List[Trade]
    equity_curve: pd.Series
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    avg_r_multiple: float
    sharpe_ratio: float
    expectancy: float


class BacktestEngine:
    """Simulates trades bar-by-bar on 1-min data."""

    def __init__(
        self,
        symbol: str,
        initial_capital: float = INITIAL_CAPITAL,
        risk_pct: float = RISK_PER_TRADE_PCT,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.meta = INSTRUMENT_META.get(symbol, {})

    def simulate(
        self, signals: list, df_1m: pd.DataFrame, variant_name: str = "default"
    ) -> BacktestResult:
        """
        Walk through 1-min bars and simulate each signal's trade.
        """
        trades: List[Trade] = []
        equity = self.initial_capital
        equity_history = {}
        ny = pytz.timezone(NY_TZ)

        for sig in signals:
            risk_amount = equity * (self.risk_pct / 100.0)
            entry_risk = abs(sig.entry_price - sig.stop_loss)
            if entry_risk <= 0:
                continue

            # Position sizing based on risk
            position_size = risk_amount / entry_risk

            # Get candles from entry to end of session
            session_end_time = sig.timestamp.normalize() + pd.Timedelta(hours=16)
            trade_candles = df_1m.loc[
                (df_1m.index > sig.timestamp)
                & (df_1m.index <= session_end_time)
            ]

            if trade_candles.empty:
                continue

            exit_price = None
            exit_time = None
            result = "timeout"

            for _, bar in trade_candles.iterrows():
                if sig.direction == "long":
                    # Check stop loss hit
                    if bar["Low"] <= sig.stop_loss:
                        exit_price = sig.stop_loss
                        exit_time = bar.name
                        result = "loss"
                        break
                    # Check take profit hit
                    if bar["High"] >= sig.take_profit:
                        exit_price = sig.take_profit
                        exit_time = bar.name
                        result = "win"
                        break
                else:  # short
                    if bar["High"] >= sig.stop_loss:
                        exit_price = sig.stop_loss
                        exit_time = bar.name
                        result = "loss"
                        break
                    if bar["Low"] <= sig.take_profit:
                        exit_price = sig.take_profit
                        exit_time = bar.name
                        result = "win"
                        break

            # Session timeout — close at last available price
            if exit_price is None:
                exit_price = trade_candles.iloc[-1]["Close"]
                exit_time = trade_candles.index[-1]

            # Calculate P&L
            if sig.direction == "long":
                pnl = (exit_price - sig.entry_price) * position_size
            else:
                pnl = (sig.entry_price - exit_price) * position_size

            pnl_pct = pnl / equity * 100
            r_multiple = pnl / risk_amount if risk_amount > 0 else 0

            equity += pnl
            equity_history[exit_time] = equity

            trades.append(Trade(
                entry_time=sig.timestamp,
                exit_time=exit_time,
                direction=sig.direction,
                entry_price=sig.entry_price,
                exit_price=exit_price,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                pnl=pnl,
                pnl_pct=pnl_pct,
                result=result,
                risk_amount=risk_amount,
                r_multiple=r_multiple,
            ))

        return self._build_result(trades, equity_history, equity, variant_name)

    def _build_result(
        self,
        trades: List[Trade],
        equity_history: dict,
        final_equity: float,
        variant_name: str,
    ) -> BacktestResult:
        """Compute summary statistics."""
        total = len(trades)
        winners = [t for t in trades if t.result == "win"]
        losers = [t for t in trades if t.result == "loss"]
        n_win = len(winners)
        n_loss = len(losers)

        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        win_rate = n_win / total * 100 if total > 0 else 0

        r_multiples = [t.r_multiple for t in trades]
        avg_r = np.mean(r_multiples) if r_multiples else 0

        # Equity curve
        eq_series = pd.Series(equity_history, dtype=float)
        if not eq_series.empty:
            # Use the earliest trade time minus 1 minute as the start point
            start_time = eq_series.index.min() - pd.Timedelta(minutes=1)
        else:
            start_time = pd.Timestamp.now(tz=pytz.timezone(NY_TZ))
        eq_series = pd.concat([
            pd.Series({start_time: self.initial_capital}), eq_series
        ]).sort_index()

        # Max drawdown
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100
        max_dd = drawdown.min() if len(drawdown) > 0 else 0

        # Sharpe (daily returns approximation)
        if len(trades) > 1:
            returns = pd.Series([t.pnl_pct / 100 for t in trades])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Expectancy
        if total > 0:
            expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        else:
            expectancy = 0

        total_ret = (final_equity - self.initial_capital) / self.initial_capital * 100

        return BacktestResult(
            symbol=self.symbol,
            variant=variant_name,
            trades=trades,
            equity_curve=eq_series,
            initial_capital=self.initial_capital,
            final_capital=final_equity,
            total_return_pct=total_ret,
            total_trades=total,
            winners=n_win,
            losers=n_loss,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd,
            avg_r_multiple=avg_r,
            sharpe_ratio=sharpe,
            expectancy=expectancy,
        )
