"""Market data fetching utilities."""

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Dhan Data API supports up to 5 years of daily historical data.
_MAX_DAILY_HISTORY_DAYS: int = 5 * 365  # 1,825 days

# Intraday data is subject to Dhan API limits (typically 60–90 days per request
# depending on the candle interval).  Use a conservative default.
_MAX_INTRADAY_HISTORY_DAYS: int = 60


class DataFetcher:
    """Fetch OHLCV and live market data using Dhan's market data APIs.

    **Historical data** is fetched via the Dhan REST endpoints:

    * ``historical_daily_data`` – daily OHLCV candles (up to 5 years)
    * ``intraday_minute_data``  – intraday minute candles (up to 60 days)

    Use ``interval=0`` to request daily candles.  Any positive integer is
    treated as a minute-candle interval.

    **Real-time LTP** is resolved in priority order:

    1. :class:`~src.data.market_streamer.MarketDataStreamer` (WebSocket,
       lowest latency) – when attached via ``streamer=`` argument.
    2. :class:`~src.broker.dhan_broker.DhanBroker` REST ``ticker_data``
       (REST polling) – when a ``broker=`` with live API connection is given.
    3. ``0.0`` – when neither source is available.

    If no broker is supplied or no API connection is available, the
    historical-data fallback is *yfinance* (must be installed separately).

    Parameters
    ----------
    broker:
        Optional :class:`~src.broker.dhan_broker.DhanBroker` instance.
        Used for REST-based historical data and LTP polling.
    streamer:
        Optional :class:`~src.data.market_streamer.MarketDataStreamer`
        instance.  When provided and running, real-time LTP queries are
        served from the WebSocket tick cache instead of polling the REST API.
    """

    def __init__(
        self,
        broker: Any | None = None,
        streamer: Any | None = None,
    ) -> None:
        self.broker = broker
        self.streamer = streamer

    def get_historical_data(
        self,
        symbol: str,
        security_id: str = "",
        exchange_segment: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        from_date: str | None = None,
        to_date: str | None = None,
        interval: int = 0,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [open, high, low, close, volume].

        *from_date* and *to_date* should be ISO-8601 strings (``YYYY-MM-DD``).
        When omitted the defaults depend on *interval*:

        * ``interval=0`` (daily candles) – last 5 years (Dhan Data API limit)
        * ``interval>0`` (minute candles) – last 60 days

        Parameters
        ----------
        symbol:
            NSE ticker symbol used as fallback for yfinance (e.g.
            ``"RELIANCE"``).  Ignored when Dhan credentials are available.
        security_id:
            Dhan string security ID (e.g. ``"2885"``).  Required for the
            Dhan data path.
        exchange_segment:
            Exchange segment string (e.g. ``"NSE_EQ"``, ``"NSE_FNO"``).
        instrument_type:
            ``"EQUITY"``, ``"INDEX"``, ``"FUTIDX"``, ``"OPTIDX"``, etc.
        from_date, to_date:
            Date range in ``YYYY-MM-DD`` format.
        interval:
            Candle interval in minutes.  Use ``0`` for daily candles via
            ``historical_daily_data`` (supports up to 5 years).  Any positive
            integer uses ``intraday_minute_data`` (up to 60 days).
        """
        today = date.today()
        if to_date is None:
            to_date = today.isoformat()
        if from_date is None:
            default_days = (
                _MAX_DAILY_HISTORY_DAYS if interval == 0 else _MAX_INTRADAY_HISTORY_DAYS
            )
            from_date = (today - timedelta(days=default_days)).isoformat()

        # Dhan Data API is the primary source whenever credentials are present
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

        Resolution order:

        1. :class:`~src.data.market_streamer.MarketDataStreamer` WebSocket
           tick cache (lowest latency) – when *streamer* is running.
        2. :class:`~src.broker.dhan_broker.DhanBroker` REST ``ticker_data``
           polling – when *broker* has an active API connection.
        3. ``0.0`` – when neither source is available.

        Parameters
        ----------
        security_id:
            Dhan string security ID.
        exchange_segment:
            Segment string (e.g. ``"NSE_EQ"``).
        """
        # Prefer WebSocket tick cache (real-time, lower latency)
        if self.streamer is not None and self.streamer.is_running():
            ltp = self.streamer.get_ltp(
                security_id=security_id,
                exchange_segment=exchange_segment,
            )
            if ltp > 0.0:
                return ltp

        # Fall back to REST polling via DhanBroker
        if self.broker is not None:
            return self.broker.get_ltp(
                security_id=security_id,
                exchange_segment=exchange_segment,
            )

        return 0.0

