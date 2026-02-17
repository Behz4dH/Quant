"""
Reporting module — generates summary tables, equity curves, and comparison charts.
"""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tabulate import tabulate

from backtest.engine import BacktestResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def print_summary_table(results: List[BacktestResult]):
    """Print a formatted comparison table of all backtest variants."""
    headers = [
        "Symbol", "Variant", "Trades", "Win%", "PF",
        "Avg R", "Max DD%", "Return%", "Sharpe", "Expectancy",
    ]
    rows = []
    for r in results:
        rows.append([
            r.symbol,
            r.variant,
            r.total_trades,
            f"{r.win_rate:.1f}",
            f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "∞",
            f"{r.avg_r_multiple:.2f}",
            f"{r.max_drawdown_pct:.2f}",
            f"{r.total_return_pct:.2f}",
            f"{r.sharpe_ratio:.2f}",
            f"{r.expectancy:.2f}",
        ])

    print("\n" + "=" * 90)
    print("FVG OPENING RANGE STRATEGY — BACKTEST RESULTS")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_trade_log(result: BacktestResult, max_rows: int = 20):
    """Print individual trade details."""
    headers = ["#", "Entry Time", "Dir", "Entry", "Exit", "SL", "TP", "P&L", "R", "Result"]
    rows = []
    for i, t in enumerate(result.trades[:max_rows], 1):
        rows.append([
            i,
            t.entry_time.strftime("%Y-%m-%d %H:%M"),
            t.direction,
            f"{t.entry_price:.5f}",
            f"{t.exit_price:.5f}",
            f"{t.stop_loss:.5f}",
            f"{t.take_profit:.5f}",
            f"{t.pnl:.2f}",
            f"{t.r_multiple:.2f}R",
            t.result,
        ])

    print(f"\n--- Trade Log: {result.symbol} [{result.variant}] ---")
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    if len(result.trades) > max_rows:
        print(f"  ... and {len(result.trades) - max_rows} more trades")


def plot_equity_curves(results: List[BacktestResult], filename: str = "equity_curves.png"):
    """Plot equity curves for all variants on one chart per symbol."""
    symbols = sorted(set(r.symbol for r in results))
    fig, axes = plt.subplots(len(symbols), 1, figsize=(14, 5 * len(symbols)), squeeze=False)

    for idx, symbol in enumerate(symbols):
        ax = axes[idx][0]
        sym_results = [r for r in results if r.symbol == symbol]

        for r in sym_results:
            if len(r.equity_curve) > 1:
                ax.plot(r.equity_curve.index, r.equity_curve.values,
                        label=f"{r.variant} ({r.total_return_pct:.1f}%)")

        ax.set_title(f"{symbol} — Equity Curve", fontsize=13, fontweight="bold")
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100_000, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nEquity curves saved to: {path}")


def plot_win_loss_distribution(results: List[BacktestResult], filename: str = "win_loss_dist.png"):
    """Bar chart of win/loss counts per variant."""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"{r.symbol}\n{r.variant}" for r in results]
    wins = [r.winners for r in results]
    losses = [r.losers for r in results]
    timeouts = [r.total_trades - r.winners - r.losers for r in results]

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, wins, width, label="Wins", color="#2ecc71")
    ax.bar(x, losses, width, label="Losses", color="#e74c3c")
    ax.bar(x + width, timeouts, width, label="Timeouts", color="#95a5a6")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Number of Trades")
    ax.set_title("Win / Loss / Timeout Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Win/Loss distribution saved to: {path}")


def find_best_variant(results: List[BacktestResult]) -> BacktestResult:
    """
    Score each variant using a weighted composite:
      - Profit Factor (25%)
      - Win Rate (20%)
      - Risk-adjusted Return / Max DD (25%)
      - Sharpe Ratio (20%)
      - Average R Multiple (10%)
    Returns the best overall result.
    """
    if not results:
        return None

    scored = []
    for r in results:
        if r.total_trades == 0:
            scored.append((r, 0))
            continue

        pf_score = min(r.profit_factor, 5) / 5  # cap at 5
        wr_score = r.win_rate / 100
        dd_score = 1 - min(abs(r.max_drawdown_pct), 50) / 50  # less DD = better
        sharpe_score = min(max(r.sharpe_ratio, 0), 3) / 3
        r_score = min(max(r.avg_r_multiple, 0), 3) / 3

        composite = (
            0.25 * pf_score
            + 0.20 * wr_score
            + 0.25 * dd_score
            + 0.20 * sharpe_score
            + 0.10 * r_score
        )
        scored.append((r, composite))

    scored.sort(key=lambda x: x[1], reverse=True)

    best = scored[0][0]
    print(f"\n{'=' * 60}")
    print(f"BEST VARIANT: {best.symbol} — {best.variant}")
    print(f"  Win Rate:       {best.win_rate:.1f}%")
    print(f"  Profit Factor:  {best.profit_factor:.2f}")
    print(f"  Total Return:   {best.total_return_pct:.2f}%")
    print(f"  Max Drawdown:   {best.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:   {best.sharpe_ratio:.2f}")
    print(f"  Avg R Multiple: {best.avg_r_multiple:.2f}")
    print(f"  Total Trades:   {best.total_trades}")
    print(f"{'=' * 60}")

    return best


def generate_full_report(results: List[BacktestResult]):
    """Generate the complete backtest report."""
    print_summary_table(results)

    # Print trade logs for variants with trades
    for r in results:
        if r.total_trades > 0:
            print_trade_log(r)

    plot_equity_curves(results)
    plot_win_loss_distribution(results)
    best = find_best_variant(results)
    return best
