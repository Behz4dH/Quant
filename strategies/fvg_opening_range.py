"""
FVG Opening Range Strategy
===========================
Multi-timeframe strategy:
  1. Mark High/Low of the 9:30-9:35 NY 5-min candle (Opening Range)
  2. On 1-min chart, detect Fair Value Gaps (FVGs) that show directional bias
  3. Wait for price to retrace into the FVG zone
  4. Enter on an engulfing rejection candle off the FVG zone

Key filter: Direction is confirmed by OR breakout — only long if price
has traded above OR high (buyers in control), only short if price
has traded below OR low (sellers in control).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pytz

from config import (
    DEFAULT_PARAMS,
    NY_TZ,
    OR_END_HOUR,
    OR_END_MIN,
    OR_START_HOUR,
    OR_START_MIN,
)


@dataclass
class FVG:
    """Represents a Fair Value Gap."""
    direction: str          # "bullish" or "bearish"
    top: float              # upper boundary of the gap
    bottom: float           # lower boundary of the gap
    timestamp: pd.Timestamp
    candle_idx: int         # index in the 1-min DataFrame
    filled: bool = False


@dataclass
class Signal:
    """Represents a trade entry signal."""
    timestamp: pd.Timestamp
    direction: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    fvg: FVG
    or_high: float
    or_low: float
    atr: float


class FVGOpeningRangeStrategy:
    """
    Core strategy logic.  Operates on a single day's 1-min and 5-min data.
    """

    def __init__(self, params: Optional[dict] = None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    # ------------------------------------------------------------------
    # Step 1: Opening Range
    # ------------------------------------------------------------------
    def get_opening_range(self, df_5m: pd.DataFrame, date: pd.Timestamp):
        """
        Return (or_high, or_low) for the 9:30-9:35 candle on the given date.
        """
        ny = pytz.timezone(NY_TZ)
        day_start = ny.localize(
            pd.Timestamp(date.year, date.month, date.day, OR_START_HOUR, OR_START_MIN)
        )
        day_end = ny.localize(
            pd.Timestamp(date.year, date.month, date.day, OR_END_HOUR, OR_END_MIN)
        )

        mask = (df_5m.index >= day_start) & (df_5m.index < day_end)
        or_candles = df_5m.loc[mask]

        if or_candles.empty:
            return None, None

        or_high = or_candles["High"].max()
        or_low = or_candles["Low"].min()
        return or_high, or_low

    # ------------------------------------------------------------------
    # Step 2: Determine OR breakout direction
    # ------------------------------------------------------------------
    def get_or_bias(
        self, df_1m: pd.DataFrame, or_high: float, or_low: float, date: pd.Timestamp
    ) -> Optional[str]:
        """
        After the OR candle closes, check which side price breaks first.
        Returns "bullish", "bearish", or None if no clear break.
        """
        ny = pytz.timezone(NY_TZ)
        or_end = ny.localize(
            pd.Timestamp(date.year, date.month, date.day, OR_END_HOUR, OR_END_MIN)
        )
        # Look at first 15 candles after OR for a breakout
        after_or = df_1m.loc[df_1m.index >= or_end].head(15)

        for _, bar in after_or.iterrows():
            if bar["Close"] > or_high:
                return "bullish"
            if bar["Close"] < or_low:
                return "bearish"

        return None

    # ------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------
    def compute_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Compute Average True Range on 1-min data."""
        period = period or self.p["atr_period"]
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ------------------------------------------------------------------
    # Step 3: Fair Value Gap Detection (1-min chart)
    # ------------------------------------------------------------------
    def detect_fvgs(
        self,
        df_1m: pd.DataFrame,
        or_high: float,
        or_low: float,
        bias: str,
        date: pd.Timestamp,
    ) -> List[FVG]:
        """
        Scan 1-min candles after the opening range for Fair Value Gaps
        that align with the OR breakout bias.

        Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
            → only valid when bias is bullish

        Bearish FVG: candle[i-2].low > candle[i].high  (gap down)
            → only valid when bias is bearish
        """
        ny = pytz.timezone(NY_TZ)
        or_end = ny.localize(
            pd.Timestamp(date.year, date.month, date.day, OR_END_HOUR, OR_END_MIN)
        )
        session_end = ny.localize(
            pd.Timestamp(
                date.year, date.month, date.day,
                self.p["session_end_hour"], self.p["session_end_min"],
            )
        )

        mask = (df_1m.index >= or_end) & (df_1m.index < session_end)
        candles = df_1m.loc[mask].copy()

        if len(candles) < 3:
            return []

        atr = self.compute_atr(df_1m)
        min_gap = self.p["fvg_min_gap_atr_mult"]
        fvgs = []

        # Only scan the first N candles after OR
        scan_limit = min(len(candles), self.p["lookback_candles"])

        for i in range(2, scan_limit):
            idx = candles.index[i]
            atr_val = atr.loc[:idx].iloc[-1] if len(atr.loc[:idx]) > 0 else np.nan
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            c0 = candles.iloc[i - 2]  # first candle of the pattern
            c1 = candles.iloc[i - 1]  # middle candle (the impulse)
            c2 = candles.iloc[i]      # third candle

            # Bullish FVG: gap between candle 0's high and candle 2's low
            if bias == "bullish":
                gap_bull = c2["Low"] - c0["High"]
                if gap_bull > 0 and gap_bull >= atr_val * min_gap:
                    mid_body = c1["Close"] - c1["Open"]
                    if mid_body > 0:  # impulse candle must be bullish
                        fvgs.append(FVG(
                            direction="bullish",
                            top=c2["Low"],
                            bottom=c0["High"],
                            timestamp=candles.index[i - 1],
                            candle_idx=i - 1,
                        ))

            # Bearish FVG: gap between candle 2's high and candle 0's low
            if bias == "bearish":
                gap_bear = c0["Low"] - c2["High"]
                if gap_bear > 0 and gap_bear >= atr_val * min_gap:
                    mid_body = c1["Open"] - c1["Close"]
                    if mid_body > 0:  # impulse candle must be bearish
                        fvgs.append(FVG(
                            direction="bearish",
                            top=c0["Low"],
                            bottom=c2["High"],
                            timestamp=candles.index[i - 1],
                            candle_idx=i - 1,
                        ))

        return fvgs

    # ------------------------------------------------------------------
    # Step 4 & 5: Retracement + Engulfing Rejection
    # ------------------------------------------------------------------
    def find_signals(
        self,
        df_1m: pd.DataFrame,
        fvgs: List[FVG],
        or_high: float,
        or_low: float,
        date: pd.Timestamp,
    ) -> List[Signal]:
        """
        For each detected FVG, look for:
          a) Price retracing into the FVG zone
          b) An engulfing candle rejecting off the zone -> entry signal

        Uses the FVG zone midpoint as the ideal entry area.
        Stop loss below/above the FVG zone + ATR buffer.
        """
        ny = pytz.timezone(NY_TZ)
        session_end = ny.localize(
            pd.Timestamp(
                date.year, date.month, date.day,
                self.p["session_end_hour"], self.p["session_end_min"],
            )
        )

        atr = self.compute_atr(df_1m)
        signals = []

        for fvg in fvgs:
            if fvg.filled:
                continue

            # Scan candles after the FVG was formed
            after_fvg = df_1m.loc[
                (df_1m.index > fvg.timestamp) & (df_1m.index < session_end)
            ]
            if len(after_fvg) < 2:
                continue

            zone_size = fvg.top - fvg.bottom
            if zone_size <= 0:
                continue

            for j in range(1, len(after_fvg)):
                prev = after_fvg.iloc[j - 1]
                curr = after_fvg.iloc[j]
                curr_time = after_fvg.index[j]

                current_atr = atr.loc[:curr_time].iloc[-1] if len(atr.loc[:curr_time]) > 0 else np.nan
                if pd.isna(current_atr) or current_atr <= 0:
                    continue

                if fvg.direction == "bullish":
                    # Price must dip INTO the FVG zone (wick or body)
                    prev_entered_zone = prev["Low"] <= fvg.top and prev["Low"] >= fvg.bottom
                    curr_entered_zone = curr["Low"] <= fvg.top and curr["Low"] >= fvg.bottom

                    if not (prev_entered_zone or curr_entered_zone):
                        # Zone fully broken → invalidate
                        if prev["Close"] < fvg.bottom:
                            fvg.filled = True
                            break
                        continue

                    # Engulfing bullish rejection:
                    # - Current candle is bullish
                    # - Current body engulfs prior candle's body
                    # - Current close above FVG top (rejection bounce)
                    prev_body = abs(prev["Close"] - prev["Open"])
                    curr_body = abs(curr["Close"] - curr["Open"])
                    is_bullish = curr["Close"] > curr["Open"]

                    if prev_body == 0:
                        prev_body = current_atr * 0.01  # avoid div by zero

                    engulf = (
                        is_bullish
                        and curr_body >= prev_body * self.p["engulf_min_body_ratio"]
                        and curr["Close"] >= fvg.top  # bounced out of zone
                        and curr["Low"] >= fvg.bottom - current_atr * 0.2  # didn't blow through
                    )

                    if engulf:
                        # SL below FVG zone bottom with ATR buffer
                        sl = fvg.bottom - current_atr * self.p["sl_atr_mult"]
                        risk = curr["Close"] - sl
                        if risk <= 0:
                            continue
                        tp = curr["Close"] + risk * self.p["tp_rr_ratio"]

                        signals.append(Signal(
                            timestamp=curr_time,
                            direction="long",
                            entry_price=curr["Close"],
                            stop_loss=sl,
                            take_profit=tp,
                            fvg=fvg,
                            or_high=or_high,
                            or_low=or_low,
                            atr=current_atr,
                        ))
                        fvg.filled = True
                        break

                elif fvg.direction == "bearish":
                    # Price must rally INTO the FVG zone
                    prev_entered_zone = prev["High"] >= fvg.bottom and prev["High"] <= fvg.top
                    curr_entered_zone = curr["High"] >= fvg.bottom and curr["High"] <= fvg.top

                    if not (prev_entered_zone or curr_entered_zone):
                        if prev["Close"] > fvg.top:
                            fvg.filled = True
                            break
                        continue

                    prev_body = abs(prev["Close"] - prev["Open"])
                    curr_body = abs(curr["Close"] - curr["Open"])
                    is_bearish = curr["Close"] < curr["Open"]

                    if prev_body == 0:
                        prev_body = current_atr * 0.01

                    engulf = (
                        is_bearish
                        and curr_body >= prev_body * self.p["engulf_min_body_ratio"]
                        and curr["Close"] <= fvg.bottom  # rejected back down
                        and curr["High"] <= fvg.top + current_atr * 0.2
                    )

                    if engulf:
                        sl = fvg.top + current_atr * self.p["sl_atr_mult"]
                        risk = sl - curr["Close"]
                        if risk <= 0:
                            continue
                        tp = curr["Close"] - risk * self.p["tp_rr_ratio"]

                        signals.append(Signal(
                            timestamp=curr_time,
                            direction="short",
                            entry_price=curr["Close"],
                            stop_loss=sl,
                            take_profit=tp,
                            fvg=fvg,
                            or_high=or_high,
                            or_low=or_low,
                            atr=current_atr,
                        ))
                        fvg.filled = True
                        break

        return signals

    # ------------------------------------------------------------------
    # Run full strategy for one day
    # ------------------------------------------------------------------
    def run_day(
        self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, date: pd.Timestamp
    ) -> List[Signal]:
        """Execute full strategy pipeline for a single trading day."""
        # Step 1: Get Opening Range
        or_high, or_low = self.get_opening_range(df_5m, date)
        if or_high is None:
            return []

        # Step 2: Determine directional bias from OR breakout
        bias = self.get_or_bias(df_1m, or_high, or_low, date)
        if bias is None:
            return []  # no clear direction — sit out

        # Step 3: Find FVGs aligned with bias
        fvgs = self.detect_fvgs(df_1m, or_high, or_low, bias, date)
        if not fvgs:
            return []

        # Step 4-5: Find retracement + engulfing signals
        signals = self.find_signals(df_1m, fvgs, or_high, or_low, date)

        # Limit signals per day
        return signals[: self.p["max_daily_trades"]]

    # ------------------------------------------------------------------
    # Run strategy across all trading days
    # ------------------------------------------------------------------
    def run(
        self, df_1m: pd.DataFrame, df_5m: pd.DataFrame
    ) -> List[Signal]:
        """Run strategy over all available trading days."""
        if df_1m.empty:
            return []

        dates = df_1m.index.normalize().unique()
        all_signals = []

        for date in dates:
            day_signals = self.run_day(df_1m, df_5m, date)
            all_signals.extend(day_signals)

        return all_signals
