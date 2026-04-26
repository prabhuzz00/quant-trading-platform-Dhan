"""Portfolio management module."""

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """Track positions, realised/unrealised P&L and enforce risk limits.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance (default 100 000).
    max_position_size : float
        Maximum fraction of portfolio equity allocated to a single position
        (default 0.05 = 5 %).
    max_portfolio_risk : float
        Maximum total exposure relative to equity (default 0.20 = 20 %).
    stop_loss_pct : float
        Default stop-loss percentage from entry price (default 0.02 = 2 %).
    take_profit_pct : float
        Default take-profit percentage from entry price (default 0.04 = 4 %).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_position_size: float = 0.05,
        max_portfolio_risk: float = 0.20,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # symbol -> {quantity, avg_price, side}
        self._positions: dict[str, dict] = {}
        # list of closed trade records
        self._trade_history: list[dict] = []

    def equity(self, current_prices: dict[str, float] | None = None) -> float:
        """Return total equity (cash + mark-to-market position value)."""
        mtm = 0.0
        if current_prices:
            for symbol, pos in self._positions.items():
                price = current_prices.get(symbol, pos["avg_price"])
                mtm += pos["quantity"] * price * (1 if pos["side"] == "BUY" else -1)
        return self.cash + mtm

    def max_order_quantity(
        self, symbol: str, price: float, current_prices: dict[str, float] | None = None
    ) -> int:
        """Return the maximum number of shares that can be bought/sold for
        *symbol* without breaching position-size limits.
        """
        eq = self.equity(current_prices)
        max_value = eq * self.max_position_size
        if price <= 0:
            return 0
        return max(0, int(max_value // price))

    def can_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        current_prices: dict[str, float] | None = None,
    ) -> bool:
        """Return ``True`` if the proposed order passes all risk checks."""
        eq = self.equity(current_prices)
        order_value = quantity * price

        if order_value > eq * self.max_position_size:
            logger.warning(
                "Risk check FAILED for %s %s %d @ %.2f: exceeds max position size",
                action,
                symbol,
                quantity,
                price,
            )
            return False

        total_exposure = sum(
            p["quantity"] * p["avg_price"] for p in self._positions.values()
        )
        if (total_exposure + order_value) > eq * self.max_portfolio_risk:
            logger.warning(
                "Risk check FAILED for %s %s: exceeds max portfolio risk",
                action,
                symbol,
            )
            return False

        return True

    def update_position(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
    ) -> None:
        """Record a filled trade and update cash/positions accordingly."""
        trade_value = quantity * price
        cost = trade_value + commission

        if action == "BUY":
            self.cash -= cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["quantity"] + quantity
                pos["avg_price"] = (
                    (pos["quantity"] * pos["avg_price"] + trade_value) / total_qty
                )
                pos["quantity"] = total_qty
            else:
                self._positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": price,
                    "side": "BUY",
                    "stop_loss": price * (1 - self.stop_loss_pct),
                    "take_profit": price * (1 + self.take_profit_pct),
                }
        elif action == "SELL":
            self.cash += trade_value - commission
            if symbol in self._positions:
                pos = self._positions[symbol]
                if pos["quantity"] <= quantity:
                    realised_pnl = (price - pos["avg_price"]) * pos["quantity"]
                    self._trade_history.append(
                        {
                            "symbol": symbol,
                            "entry_price": pos["avg_price"],
                            "exit_price": price,
                            "quantity": pos["quantity"],
                            "pnl": realised_pnl,
                        }
                    )
                    del self._positions[symbol]
                else:
                    pos["quantity"] -= quantity

        logger.info(
            "Position update: %s %s %d @ %.2f  cash=%.2f",
            action,
            symbol,
            quantity,
            price,
            self.cash,
        )

    def get_position(self, symbol: str) -> dict | None:
        """Return current position dict for *symbol*, or ``None``."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, dict]:
        """Return a copy of all open positions."""
        return dict(self._positions)

    def get_trade_history(self) -> list[dict]:
        """Return the list of closed-trade records."""
        return list(self._trade_history)

    def performance_summary(
        self, current_prices: dict[str, float] | None = None
    ) -> dict:
        """Return a summary of key performance metrics."""
        eq = self.equity(current_prices)
        total_pnl = sum(t["pnl"] for t in self._trade_history)
        wins = [t for t in self._trade_history if t["pnl"] > 0]
        losses = [t for t in self._trade_history if t["pnl"] <= 0]
        win_rate = len(wins) / len(self._trade_history) if self._trade_history else 0.0

        return {
            "initial_capital": self.initial_capital,
            "current_equity": round(eq, 2),
            "cash": round(self.cash, 2),
            "total_realised_pnl": round(total_pnl, 2),
            "total_trades": len(self._trade_history),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 2),
            "return_pct": round((eq - self.initial_capital) / self.initial_capital * 100, 2),
        }
