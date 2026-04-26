"""Tests for the PortfolioManager and Backtester."""

import pandas as pd
import pytest

from src.backtesting.backtester import Backtester
from src.portfolio.portfolio_manager import PortfolioManager
from src.strategy.example_strategies import MovingAverageCrossStrategy


class TestPortfolioManager:
    def _pm(self, capital: float = 100_000.0) -> PortfolioManager:
        return PortfolioManager(
            initial_capital=capital,
            max_position_size=0.10,
            max_portfolio_risk=0.50,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

    def test_initial_equity_equals_capital(self):
        pm = self._pm()
        assert pm.equity() == 100_000.0

    def test_buy_reduces_cash(self):
        pm = self._pm()
        pm.update_position("RELIANCE", "BUY", 10, 500.0)
        assert pm.cash == pytest.approx(100_000.0 - 10 * 500.0)

    def test_sell_increases_cash(self):
        pm = self._pm()
        pm.update_position("RELIANCE", "BUY", 10, 500.0)
        pm.update_position("RELIANCE", "SELL", 10, 600.0)
        # After full exit cash = 100 000 - 5000 + 6000 = 101 000
        assert pm.cash == pytest.approx(101_000.0)

    def test_position_tracked_after_buy(self):
        pm = self._pm()
        pm.update_position("RELIANCE", "BUY", 5, 400.0)
        pos = pm.get_position("RELIANCE")
        assert pos is not None
        assert pos["quantity"] == 5
        assert pos["avg_price"] == pytest.approx(400.0)

    def test_position_removed_after_full_sell(self):
        pm = self._pm()
        pm.update_position("RELIANCE", "BUY", 5, 400.0)
        pm.update_position("RELIANCE", "SELL", 5, 420.0)
        assert pm.get_position("RELIANCE") is None

    def test_trade_history_after_round_trip(self):
        pm = self._pm()
        pm.update_position("RELIANCE", "BUY", 10, 500.0)
        pm.update_position("RELIANCE", "SELL", 10, 550.0)
        history = pm.get_trade_history()
        assert len(history) == 1
        assert history[0]["pnl"] == pytest.approx(500.0)

    def test_can_trade_passes_within_limits(self):
        pm = self._pm(100_000.0)
        assert pm.can_trade("RELIANCE", "BUY", 10, 500.0) is True

    def test_can_trade_fails_above_position_limit(self):
        pm = self._pm(10_000.0)
        # 10 000 * 0.10 = 1 000 max value; order value = 5 * 500 = 2 500
        assert pm.can_trade("RELIANCE", "BUY", 5, 500.0) is False

    def test_max_order_quantity(self):
        pm = self._pm(100_000.0)
        # equity=100000, max_position_size=0.10 → max value=10000
        # price=500 → max qty = 10000//500 = 20
        assert pm.max_order_quantity("RELIANCE", 500.0) == 20

    def test_performance_summary_keys(self):
        pm = self._pm()
        summary = pm.performance_summary()
        for key in (
            "initial_capital",
            "current_equity",
            "cash",
            "total_realised_pnl",
            "total_trades",
            "win_rate",
            "return_pct",
        ):
            assert key in summary

    def test_equity_with_open_position(self):
        pm = self._pm(100_000.0)
        pm.update_position("RELIANCE", "BUY", 10, 500.0)
        # MTM at 600 → equity = 95 000 + 10*600 = 101 000
        eq = pm.equity({"RELIANCE": 600.0})
        assert eq == pytest.approx(101_000.0)


class TestBacktester:
    def _make_data(self) -> pd.DataFrame:
        """Generate synthetic price data with a detectable MA crossover."""
        closes = (
            [100.0] * 10          # flat baseline
            + [c for c in range(101, 140)]   # rising → golden cross
            + [c for c in range(138, 100, -1)]  # falling → death cross
        )
        index = pd.date_range("2023-01-01", periods=len(closes), freq="D")
        return pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000] * len(closes),
            },
            index=index,
        )

    def test_backtest_runs_and_returns_summary(self):
        data = self._make_data()
        strategy = MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
            fast_period=5,
            slow_period=10,
            quantity=1,
        )
        bt = Backtester(strategy=strategy, data=data, initial_capital=100_000.0)
        summary = bt.run()

        assert "initial_capital" in summary
        assert "current_equity" in summary
        assert "total_trades" in summary

    def test_equity_curve_has_correct_length(self):
        data = self._make_data()
        strategy = MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
            fast_period=5,
            slow_period=10,
            quantity=1,
        )
        bt = Backtester(strategy=strategy, data=data, initial_capital=100_000.0)
        bt.run()

        curve = bt.equity_curve()
        assert len(curve) == len(data)

    def test_empty_data_raises_value_error(self):
        strategy = MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
        )
        with pytest.raises(ValueError, match="empty"):
            Backtester(strategy=strategy, data=pd.DataFrame())

    def test_missing_close_column_raises_value_error(self):
        strategy = MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
        )
        data = pd.DataFrame({"open": [100.0], "high": [101.0]})
        with pytest.raises(ValueError, match="close"):
            Backtester(strategy=strategy, data=data)

    def test_trade_history_returns_dataframe(self):
        data = self._make_data()
        strategy = MovingAverageCrossStrategy(
            symbol="TEST",
            security_id="9999",
            fast_period=5,
            slow_period=10,
            quantity=1,
        )
        bt = Backtester(strategy=strategy, data=data, initial_capital=100_000.0)
        bt.run()

        trades = bt.trade_history()
        assert isinstance(trades, pd.DataFrame)
