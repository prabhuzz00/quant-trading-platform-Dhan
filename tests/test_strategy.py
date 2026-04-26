"""Tests for example trading strategies."""

import pandas as pd
import pytest

from src.strategy.example_strategies import (
    BollingerBandsStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
)


def _make_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes}
    )


class TestMovingAverageCrossStrategy:
    def _strategy(self) -> MovingAverageCrossStrategy:
        return MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
            fast_period=3,
            slow_period=5,
            quantity=10,
        )

    def test_no_signal_before_enough_bars(self):
        strat = self._strategy()
        df = _make_df([100.0, 101.0, 102.0])
        for i in range(len(df)):
            signal = strat.generate_signals(df.iloc[[i]])
            assert signal is None

    def test_buy_signal_on_golden_cross(self):
        strat = self._strategy()
        # Declining prices → fast MA < slow MA
        prices = [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
        # Rising prices → fast MA crosses above slow MA
        prices += [105.0, 110.0, 115.0, 120.0, 125.0]

        last_signal = None
        for i, p in enumerate(prices):
            df = _make_df([p])
            sig = strat.generate_signals(df)
            if sig:
                last_signal = sig

        assert last_signal is not None
        assert last_signal["action"] == "BUY"
        assert last_signal["quantity"] == 10

    def test_sell_signal_on_death_cross(self):
        strat = self._strategy()
        # Rising prices first (position goes long)
        rising = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
        for p in rising:
            strat.generate_signals(_make_df([p]))

        # Falling prices → death cross
        falling = [120.0, 110.0, 100.0, 90.0, 80.0, 70.0]
        last_signal = None
        for p in falling:
            sig = strat.generate_signals(_make_df([p]))
            if sig:
                last_signal = sig

        assert last_signal is not None
        assert last_signal["action"] == "SELL"


class TestRSIStrategy:
    def _strategy(self) -> RSIStrategy:
        return RSIStrategy(
            symbol="TEST",
            security_id="9999",
            rsi_period=5,
            oversold=30.0,
            overbought=70.0,
            quantity=5,
        )

    def test_no_signal_before_enough_bars(self):
        strat = self._strategy()
        for p in [100.0, 101.0, 102.0]:
            assert strat.generate_signals(_make_df([p])) is None

    def test_buy_signal_when_oversold(self):
        strat = self._strategy()
        # Sharp decline to trigger oversold RSI
        prices = [100.0, 95.0, 88.0, 80.0, 70.0, 58.0, 44.0]
        signals = []
        for p in prices:
            sig = strat.generate_signals(_make_df([p]))
            if sig:
                signals.append(sig)

        assert any(s["action"] == "BUY" for s in signals)

    def test_sell_signal_when_overbought(self):
        strat = self._strategy()
        # Sharp rise to trigger overbought RSI
        prices = [50.0, 60.0, 72.0, 86.0, 102.0, 120.0, 140.0]
        signals = []
        for p in prices:
            sig = strat.generate_signals(_make_df([p]))
            if sig:
                signals.append(sig)

        assert any(s["action"] == "SELL" for s in signals)


class TestBollingerBandsStrategy:
    def _strategy(self) -> BollingerBandsStrategy:
        return BollingerBandsStrategy(
            symbol="TEST",
            security_id="9999",
            period=5,
            num_std=1.5,
            quantity=3,
        )

    def test_no_signal_before_period(self):
        strat = self._strategy()
        for p in [100.0, 100.0, 100.0]:
            assert strat.generate_signals(_make_df([p])) is None

    def test_buy_signal_below_lower_band(self):
        strat = self._strategy()
        # Stable baseline then sharp dip
        baseline = [100.0] * 5
        for p in baseline:
            strat.generate_signals(_make_df([p]))

        sig = strat.generate_signals(_make_df([80.0]))
        assert sig is not None
        assert sig["action"] == "BUY"

    def test_sell_signal_above_upper_band(self):
        strat = self._strategy()
        baseline = [100.0] * 5
        for p in baseline:
            strat.generate_signals(_make_df([p]))

        sig = strat.generate_signals(_make_df([120.0]))
        assert sig is not None
        assert sig["action"] == "SELL"
