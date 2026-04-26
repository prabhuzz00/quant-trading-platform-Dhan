"""Ready-to-use example trading strategies."""

import pandas as pd

from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MovingAverageCrossStrategy(BaseStrategy):
    """Simple dual moving-average crossover strategy.

    Generates a **BUY** signal when the fast MA crosses above the slow MA and
    a **SELL** signal when the fast MA crosses below the slow MA.

    Parameters
    ----------
    fast_period : int
        Look-back window for the fast moving average (default 20).
    slow_period : int
        Look-back window for the slow moving average (default 50).
    quantity : int
        Number of shares per order (default 1).
    """

    def __init__(
        self,
        symbol: str,
        security_id: str,
        fast_period: int = 20,
        slow_period: int = 50,
        quantity: int = 1,
        exchange_segment: str = "NSE_EQ",
    ) -> None:
        super().__init__(
            name="MovingAverageCross",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "quantity": quantity,
            },
        )
        self.symbol = symbol
        self.security_id = security_id
        self.exchange_segment = exchange_segment
        self._history: list[float] = []
        self._position: int = 0  # +1 long, -1 short, 0 flat

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._history.append(close)

        fast = self.params["fast_period"]
        slow = self.params["slow_period"]

        if len(self._history) < slow + 1:
            return None

        fast_ma_now = sum(self._history[-fast:]) / fast
        slow_ma_now = sum(self._history[-slow:]) / slow
        fast_ma_prev = sum(self._history[-fast - 1 : -1]) / fast
        slow_ma_prev = sum(self._history[-slow - 1 : -1]) / slow

        signal: dict | None = None

        if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
            if self._position <= 0:
                signal = self._make_signal("BUY", close)
                self._position = 1
                logger.info(
                    "%s BUY signal at %.2f (fast_ma=%.2f, slow_ma=%.2f)",
                    self.symbol,
                    close,
                    fast_ma_now,
                    slow_ma_now,
                )

        elif fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
            if self._position >= 0:
                signal = self._make_signal("SELL", close)
                self._position = -1
                logger.info(
                    "%s SELL signal at %.2f (fast_ma=%.2f, slow_ma=%.2f)",
                    self.symbol,
                    close,
                    fast_ma_now,
                    slow_ma_now,
                )

        return signal

    def _make_signal(self, action: str, price: float) -> dict:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_segment": self.exchange_segment,
            "action": action,
            "quantity": self.params["quantity"],
            "price": price,
            "order_type": "MARKET",
            "product_type": "INTRADAY",
        }


class RSIStrategy(BaseStrategy):
    """RSI mean-reversion strategy.

    Generates a **BUY** signal when RSI drops below *oversold* and a
    **SELL** signal when RSI rises above *overbought*.

    Parameters
    ----------
    rsi_period : int
        RSI look-back period (default 14).
    oversold : float
        RSI level below which the instrument is considered oversold (default 30).
    overbought : float
        RSI level above which the instrument is considered overbought (default 70).
    quantity : int
        Shares per order (default 1).
    """

    def __init__(
        self,
        symbol: str,
        security_id: str,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        quantity: int = 1,
        exchange_segment: str = "NSE_EQ",
    ) -> None:
        super().__init__(
            name="RSI",
            params={
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
                "quantity": quantity,
            },
        )
        self.symbol = symbol
        self.security_id = security_id
        self.exchange_segment = exchange_segment
        self._history: list[float] = []
        self._position: int = 0

    def _compute_rsi(self) -> float:
        period = self.params["rsi_period"]
        prices = self._history[-(period + 1) :]
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._history.append(close)

        period = self.params["rsi_period"]
        if len(self._history) < period + 1:
            return None

        rsi = self._compute_rsi()
        signal: dict | None = None

        if rsi < self.params["oversold"] and self._position <= 0:
            signal = self._make_signal("BUY", close)
            self._position = 1
            logger.info("%s RSI BUY at %.2f (RSI=%.2f)", self.symbol, close, rsi)

        elif rsi > self.params["overbought"] and self._position >= 0:
            signal = self._make_signal("SELL", close)
            self._position = -1
            logger.info("%s RSI SELL at %.2f (RSI=%.2f)", self.symbol, close, rsi)

        return signal

    def _make_signal(self, action: str, price: float) -> dict:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_segment": self.exchange_segment,
            "action": action,
            "quantity": self.params["quantity"],
            "price": price,
            "order_type": "MARKET",
            "product_type": "INTRADAY",
        }


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands breakout strategy.

    Generates a **BUY** signal when price closes below the lower band and a
    **SELL** signal when price closes above the upper band.

    Parameters
    ----------
    period : int
        Look-back window for the middle band (SMA), default 20.
    num_std : float
        Number of standard deviations for the band width, default 2.0.
    quantity : int
        Shares per order (default 1).
    """

    def __init__(
        self,
        symbol: str,
        security_id: str,
        period: int = 20,
        num_std: float = 2.0,
        quantity: int = 1,
        exchange_segment: str = "NSE_EQ",
    ) -> None:
        super().__init__(
            name="BollingerBands",
            params={"period": period, "num_std": num_std, "quantity": quantity},
        )
        self.symbol = symbol
        self.security_id = security_id
        self.exchange_segment = exchange_segment
        self._history: list[float] = []
        self._position: int = 0

    def _compute_bands(self) -> tuple[float, float, float]:
        period = self.params["period"]
        prices = self._history[-period:]
        middle = sum(prices) / period
        variance = sum((p - middle) ** 2 for p in prices) / period
        std = variance ** 0.5
        width = self.params["num_std"] * std
        return middle - width, middle, middle + width

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._history.append(close)

        if len(self._history) < self.params["period"]:
            return None

        lower, _middle, upper = self._compute_bands()
        signal: dict | None = None

        if close < lower and self._position <= 0:
            signal = self._make_signal("BUY", close)
            self._position = 1
            logger.info(
                "%s BB BUY at %.2f (lower=%.2f)", self.symbol, close, lower
            )

        elif close > upper and self._position >= 0:
            signal = self._make_signal("SELL", close)
            self._position = -1
            logger.info(
                "%s BB SELL at %.2f (upper=%.2f)", self.symbol, close, upper
            )

        return signal

    def _make_signal(self, action: str, price: float) -> dict:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_segment": self.exchange_segment,
            "action": action,
            "quantity": self.params["quantity"],
            "price": price,
            "order_type": "MARKET",
            "product_type": "INTRADAY",
        }
