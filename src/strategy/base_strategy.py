"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Base class that every strategy must inherit from.

    Sub-classes **must** implement :meth:`generate_signals`.  They may
    optionally override :meth:`on_start`, :meth:`on_stop`, and
    :meth:`on_bar`.

    Parameters
    ----------
    name:
        Human-readable strategy name used in logs and reports.
    params:
        Arbitrary key/value configuration passed through to the strategy
        (e.g. moving-average windows, RSI thresholds …).
    """

    def __init__(self, name: str = "Strategy", params: dict | None = None) -> None:
        self.name = name
        self.params: dict = params or {}
        self._broker: Any = None
        self._portfolio: Any = None

    def attach_broker(self, broker: Any) -> None:
        """Attach a live or paper-trade broker instance."""
        self._broker = broker

    def attach_portfolio(self, portfolio: Any) -> None:
        """Attach a portfolio manager instance."""
        self._portfolio = portfolio

    def on_start(self) -> None:
        """Called once before the strategy begins receiving bars."""

    def on_stop(self) -> None:
        """Called once after all bars have been processed."""

    def on_bar(self, bar: pd.Series) -> None:
        """Called for each new price bar during live or back-testing.

        The default implementation delegates to :meth:`generate_signals`.
        Override for custom per-bar logic.
        """
        signal = self.generate_signals(bar.to_frame().T)
        if signal and self._broker is not None:
            self._execute_signal(signal, bar)

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Compute trading signals from price data.

        Parameters
        ----------
        data:
            DataFrame slice available to the strategy (current bar and all
            historical bars up to this point).

        Returns
        -------
        dict or None
            A signal dict with at minimum the keys ``symbol``,
            ``security_id``, ``action`` (``"BUY"`` | ``"SELL"``),
            ``quantity``, and ``price``.  Return ``None`` when there is no
            actionable signal.
        """

    def _execute_signal(self, signal: dict, bar: pd.Series) -> None:
        """Place an order via the attached broker for *signal*."""
        if self._broker is None:
            logger.warning("No broker attached – cannot execute signal %s", signal)
            return

        order = self._broker.place_order(
            security_id=signal.get("security_id", ""),
            exchange_segment=signal.get("exchange_segment", "NSE_EQ"),
            transaction_type=signal["action"],
            quantity=signal["quantity"],
            order_type=signal.get("order_type", "MARKET"),
            product_type=signal.get("product_type", "INTRADAY"),
            price=signal.get("price", 0.0),
        )
        logger.info("Executed signal %s -> order %s", signal, order)
