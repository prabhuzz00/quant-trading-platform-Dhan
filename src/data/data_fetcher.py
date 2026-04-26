"""Market data fetching utilities."""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """Fetch OHLCV and live market data from Dhan or fallback to yfinance.

    When a ``DhanBroker`` is provided and has an active API connection,
    Dhan's historical-data endpoint is used.  This works regardless of
    whether the broker is in paper-trade or live mode – only *order
    execution* is paper-simulated, but market data is always real.

    If no broker is supplied or no API connection is available, the fetcher
    falls back to *yfinance* (must be installed separately).
    """

    def __init__(self, broker: Any | None = None) -> None:
        self.broker = broker

    def get_historical_data(
        self,
        symbol: str,
        security_id: str = "",
        exchange_segment: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        from_date: str | None = None,
        to_date: str | None = None,
        interval: int = 1,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [open, high, low, close, volume].

        *from_date* and *to_date* should be ISO-8601 strings (``YYYY-MM-DD``).
        If omitted, the last 365 days are used.

        Parameters
        ----------
        interval:
            Candle interval in minutes (default 1).  Passed to
            ``intraday_minute_data``.  Use ``0`` for daily candles via
            ``historical_daily_data``.
        """
        today = date.today()
        if to_date is None:
            to_date = today.isoformat()
        if from_date is None:
            from_date = (today - timedelta(days=365)).isoformat()

        # Use Dhan whenever the broker has an active API connection
        if self.broker is not None and self.broker._dhan is not None and security_id:
            return self._fetch_from_dhan(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
            )

        return self._fetch_from_yfinance(symbol, from_date, to_date)

    def _fetch_from_dhan(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str,
        from_date: str,
        to_date: str,
        interval: int = 1,
    ) -> pd.DataFrame:
        """Use the Dhan intraday-minute-data API (dhanhq >= 2.1.0)."""
        try:
            if interval == 0:
                # Daily candles
                response = self.broker._dhan.historical_daily_data(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument_type=instrument_type,
                    from_date=from_date,
                    to_date=to_date,
                )
            else:
                response = self.broker._dhan.intraday_minute_data(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    instrument_type=instrument_type,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval,
                )
            data = response.get("data", {})
            df = pd.DataFrame(
                {
                    "open": data.get("open", []),
                    "high": data.get("high", []),
                    "low": data.get("low", []),
                    "close": data.get("close", []),
                    "volume": data.get("volume", []),
                },
                index=pd.to_datetime(data.get("timestamp", [])),
            )
            df.index.name = "datetime"
            logger.info(
                "Fetched %d rows from Dhan for security_id=%s", len(df), security_id
            )
            return df
        except Exception as exc:
            logger.warning("Dhan data fetch failed (%s), falling back to yfinance", exc)
            return pd.DataFrame()

    def _fetch_from_yfinance(
        self, symbol: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """Fallback: download daily OHLCV data from Yahoo Finance."""
        try:
            import yfinance as yf  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "yfinance is required as a data fallback. "
                "Install it with: pip install yfinance"
            ) from exc

        ticker_symbol = symbol if "." in symbol else f"{symbol}.NS"
        df = yf.download(
            ticker_symbol,
            start=from_date,
            end=to_date,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning("No data returned by yfinance for %s", ticker_symbol)
            return df

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "datetime"
        logger.info(
            "Fetched %d rows from yfinance for %s", len(df), ticker_symbol
        )
        return df[["open", "high", "low", "close", "volume"]]

    def get_live_ltp(self, security_id: str, exchange_segment: str) -> float:
        """Return the live last traded price for a security.

        Delegates to :meth:`~src.broker.dhan_broker.DhanBroker.get_ltp`.
        Returns ``0.0`` when no broker is attached.
        """
        if self.broker is None:
            return 0.0
        return self.broker.get_ltp(
            security_id=security_id,
            exchange_segment=exchange_segment,
        )

