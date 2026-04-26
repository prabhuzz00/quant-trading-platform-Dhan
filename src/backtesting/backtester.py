"""Event-driven backtesting engine."""

from typing import Any

import pandas as pd

from src.portfolio.portfolio_manager import PortfolioManager
from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    """Run a strategy against historical OHLCV data and report results.

    Parameters
    ----------
    strategy : BaseStrategy
        An instantiated (but not yet started) strategy object.
    data : pd.DataFrame
        Historical price data with at minimum a ``close`` column and a
        DatetimeIndex.
    initial_capital : float
        Starting cash balance (default 100 000).
    commission : float
        Commission as a fraction of trade value (default 0.0003 = 0.03 %).
    slippage : float
        Slippage as a fraction of the execution price (default 0.0001).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission: float = 0.0003,
        slippage: float = 0.0001,
    ) -> None:
        if data.empty:
            raise ValueError("Backtester received an empty DataFrame.")

        required = {"close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Data is missing required columns: {missing}")

        self.strategy = strategy
        self.data = data.copy()
        self.commission = commission
        self.slippage = slippage

        self.portfolio = PortfolioManager(initial_capital=initial_capital)
        self.strategy.attach_portfolio(self.portfolio)

        self._equity_curve: list[dict] = []

    def run(self) -> dict:
        """Execute the backtest and return a performance summary dict."""
        logger.info(
            "Backtesting '%s' on %d bars …", self.strategy.name, len(self.data)
        )
        self.strategy.on_start()

        for timestamp, bar in self.data.iterrows():
            signal = self.strategy.generate_signals(bar.to_frame().T)
            if signal is not None:
                self._process_signal(signal, bar)

            current_price = float(bar["close"])
            symbol = getattr(self.strategy, "symbol", "UNKNOWN")
            self._equity_curve.append(
                {
                    "datetime": timestamp,
                    "equity": self.portfolio.equity({symbol: current_price}),
                    "cash": self.portfolio.cash,
                }
            )

        self.strategy.on_stop()

        summary = self.portfolio.performance_summary()
        logger.info("Backtest complete: %s", summary)
        return summary

    def _process_signal(self, signal: dict, bar: pd.Series) -> None:
        """Validate risk limits and record the trade in the portfolio."""
        action = signal["action"]
        quantity = signal["quantity"]
        raw_price = signal.get("price") or float(bar["close"])

        # Apply slippage
        if action == "BUY":
            exec_price = raw_price * (1 + self.slippage)
        else:
            exec_price = raw_price * (1 - self.slippage)

        commission_cost = exec_price * quantity * self.commission
        symbol = signal.get("symbol", "UNKNOWN")

        if not self.portfolio.can_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=exec_price,
        ):
            logger.debug("Signal rejected by risk manager: %s", signal)
            return

        self.portfolio.update_position(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=exec_price,
            commission=commission_cost,
        )

    def equity_curve(self) -> pd.DataFrame:
        """Return the equity curve as a DataFrame indexed by datetime."""
        if not self._equity_curve:
            return pd.DataFrame(columns=["datetime", "equity", "cash"])
        df = pd.DataFrame(self._equity_curve)
        df.set_index("datetime", inplace=True)
        return df

    def trade_history(self) -> pd.DataFrame:
        """Return all closed trades as a DataFrame."""
        trades = self.portfolio.get_trade_history()
        if not trades:
            return pd.DataFrame(columns=["symbol", "entry_price", "exit_price", "quantity", "pnl"])
        return pd.DataFrame(trades)
