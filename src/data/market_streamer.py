"""Real-time WebSocket market data streamer using Dhan's MarketFeed API.

The :class:`MarketDataStreamer` wraps :class:`dhanhq.marketfeed.MarketFeed`
to provide a clean subscribe / start / stop interface with a thread-safe
tick cache and optional user callbacks.

Subscription types
------------------
.. data:: TICKER
    LTP + last-trade timestamp (``MarketFeed.Ticker = 15``).

.. data:: QUOTE
    LTP, OHLC, volume, total buy/sell quantity
    (``MarketFeed.Quote = 17``).

.. data:: FULL
    Everything in QUOTE plus 5-level market depth and open interest
    (``MarketFeed.Full = 21``).

Exchange segments (string → MarketFeed int)
-------------------------------------------
+----------------+-----+
| DhanBroker str | int |
+================+=====+
| IDX_I          |   0 |
| NSE_EQ         |   1 |
| NSE_FNO        |   2 |
| NSE_CURR       |   3 |
| BSE_EQ         |   4 |
| MCX_COMM       |   5 |
| BSE_CURR       |   7 |
| BSE_FNO        |   8 |
+----------------+-----+
"""

import threading
from typing import Any, Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Subscription type constants (mirrored from dhanhq.marketfeed.MarketFeed)
TICKER: int = 15  #: LTP + timestamp only
QUOTE: int = 17   #: LTP, OHLC, volume, bid/ask quantity
FULL: int = 21    #: QUOTE + 5-level market depth + open interest

# Map DhanBroker exchange-segment strings → MarketFeed integer segment codes
_SEGMENT_MAP: dict[str, int] = {
    "IDX_I": 0,
    "NSE_EQ": 1,
    "NSE_FNO": 2,
    "NSE_CURR": 3,
    "BSE_EQ": 4,
    "MCX_COMM": 5,
    "BSE_CURR": 7,
    "BSE_FNO": 8,
}

# Reverse map: MarketFeed integer → DhanBroker string
_SEGMENT_REVERSE_MAP: dict[int, str] = {v: k for k, v in _SEGMENT_MAP.items()}


class MarketDataStreamer:
    """Stream real-time market data from Dhan's WebSocket feed.

    Uses :class:`dhanhq.marketfeed.MarketFeed` (dhanhq >= 2.1.0) to
    subscribe to live tick data for one or more instruments.  A background
    daemon thread runs the WebSocket event loop so the calling code is not
    blocked.

    Parameters
    ----------
    dhan_context:
        A :class:`dhanhq.DhanContext` instance that holds ``client_id`` and
        ``access_token``.
    instruments:
        Initial list of instruments to subscribe.  Each element is a tuple
        of **either**

        * ``(exchange_segment, security_id)`` – subscription type defaults
          to :data:`TICKER` (15), or
        * ``(exchange_segment, security_id, subscription_type)``

        where *exchange_segment* is a ``DhanBroker`` segment string
        (e.g. ``"NSE_EQ"``) **or** the raw integer code, and *security_id*
        is the Dhan string security ID (e.g. ``"2885"``).
    on_tick:
        Optional callback ``(data: dict) -> None`` invoked for every
        incoming tick.  The dict structure mirrors the parsed packet from
        :class:`~dhanhq.marketfeed.MarketFeed`:

        * Ticker – ``type, exchange_segment, security_id, LTP, LTT``
        * Quote  – adds ``LTQ, avg_price, volume, total_buy_quantity,
          total_sell_quantity, open, close, high, low``
        * Full   – adds ``OI, oi_day_high, oi_day_low, depth``
          (``depth`` is a list of 5 bid/ask level dicts)

    Examples
    --------
    ::

        from dhanhq import DhanContext
        from src.data.market_streamer import MarketDataStreamer, QUOTE

        ctx = DhanContext(client_id, access_token)
        streamer = MarketDataStreamer(
            dhan_context=ctx,
            instruments=[("NSE_EQ", "2885", QUOTE)],   # RELIANCE
            on_tick=lambda d: print(d),
        )
        streamer.start()

        # Poll the cached LTP at any time
        ltp = streamer.get_ltp("2885", "NSE_EQ")

        # Dynamically add another instrument
        streamer.subscribe("1333", "NSE_EQ", FULL)          # HDFC Bank

        # Retrieve 5-level market depth (requires FULL subscription)
        depth = streamer.get_depth("1333", "NSE_EQ")

        streamer.stop()
    """

    #: Subscription type: LTP + timestamp
    TICKER: int = TICKER
    #: Subscription type: LTP, OHLC, volume
    QUOTE: int = QUOTE
    #: Subscription type: Quote + market depth + OI
    FULL: int = FULL

    def __init__(
        self,
        dhan_context: Any,
        instruments: list[tuple] | None = None,
        on_tick: Callable[[dict], None] | None = None,
    ) -> None:
        self._dhan_context = dhan_context
        self._on_tick = on_tick
        self._feed: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Cache: (security_id_str, exchange_segment_str) → latest tick dict
        self._ticks: dict[tuple[str, str], dict] = {}
        self._instruments: list[tuple[int, str, int]] = []

        for inst in (instruments or []):
            normalized = self._normalize_instrument(inst)
            if normalized not in self._instruments:
                self._instruments.append(normalized)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def subscribe(
        self,
        security_id: str,
        exchange_segment: str,
        subscription_type: int = TICKER,
    ) -> None:
        """Subscribe to a single instrument.

        Safe to call before :meth:`start` (the instrument will be included
        in the initial subscription) **or** after :meth:`start` (the
        instrument is added to the live feed without reconnecting).

        Parameters
        ----------
        security_id:
            Dhan string security ID (e.g. ``"2885"`` for RELIANCE).
        exchange_segment:
            Segment string as used by :class:`~src.broker.dhan_broker.DhanBroker`
            (e.g. ``"NSE_EQ"``, ``"NSE_FNO"``, ``"IDX_I"``).
        subscription_type:
            :data:`TICKER` (15), :data:`QUOTE` (17), or :data:`FULL` (21).
        """
        inst = self._normalize_instrument((exchange_segment, security_id, subscription_type))
        if inst not in self._instruments:
            self._instruments.append(inst)
        if self._feed is not None:
            self._feed.subscribe_symbols([inst])
            logger.info(
                "Subscribed to %s / %s (type=%d) on live feed.",
                exchange_segment,
                security_id,
                subscription_type,
            )

    def unsubscribe(
        self,
        security_id: str,
        exchange_segment: str,
        subscription_type: int = TICKER,
    ) -> None:
        """Unsubscribe from a single instrument.

        Parameters are the same as :meth:`subscribe`.
        """
        inst = self._normalize_instrument((exchange_segment, security_id, subscription_type))
        if inst in self._instruments:
            self._instruments.remove(inst)
        if self._feed is not None:
            self._feed.unsubscribe_symbols([inst])
            logger.info(
                "Unsubscribed from %s / %s (type=%d) on live feed.",
                exchange_segment,
                security_id,
                subscription_type,
            )

    def start(self) -> None:
        """Connect to Dhan's WebSocket feed and start streaming in background.

        The WebSocket runs in a background daemon thread so the caller is
        not blocked.  All subscribed instruments begin receiving live ticks
        immediately after the handshake completes.

        Raises
        ------
        ImportError
            When ``dhanhq`` is not installed.
        RuntimeError
            When the streamer is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MarketDataStreamer is already running.")
            return

        try:
            from dhanhq import marketfeed  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "dhanhq>=2.1.0 is required for real-time streaming. "
                "Install with: pip install 'dhanhq>=2.1.0'"
            ) from exc

        # Use a placeholder Ticker instrument when none are configured so
        # the MarketFeed object can be constructed (it validates non-empty list).
        instruments = self._instruments or [(marketfeed.MarketFeed.NSE, "0", self.TICKER)]

        self._feed = marketfeed.MarketFeed(
            dhan_context=self._dhan_context,
            instruments=instruments,
            on_message=self._on_message,
            on_error=self._on_error,
        )
        self._thread = self._feed.start()
        logger.info(
            "MarketDataStreamer started – subscribed to %d instrument(s).",
            len(self._instruments),
        )

    def stop(self) -> None:
        """Stop the WebSocket feed and release resources."""
        if self._feed is not None:
            try:
                self._feed.close_connection()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing MarketDataStreamer: %s", exc)
            self._feed = None
        self._thread = None
        logger.info("MarketDataStreamer stopped.")

    def is_running(self) -> bool:
        """Return ``True`` when the background feed thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    #  Cached data access                                                  #
    # ------------------------------------------------------------------ #

    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        """Return the latest cached LTP for a security.

        Parameters
        ----------
        security_id:
            Dhan string security ID.
        exchange_segment:
            Segment string *or* the raw integer segment code.

        Returns
        -------
        float
            Last traded price, or ``0.0`` if no tick has been received yet.
        """
        seg_key = self._segment_key(exchange_segment)
        with self._lock:
            tick = self._ticks.get((security_id, seg_key), {})
        try:
            return float(tick.get("LTP", 0.0))
        except (ValueError, TypeError):
            return 0.0

    def get_tick(self, security_id: str, exchange_segment: str) -> dict:
        """Return the full latest tick dict for a security.

        Returns an empty ``dict`` when no tick has been received yet.
        """
        seg_key = self._segment_key(exchange_segment)
        with self._lock:
            return dict(self._ticks.get((security_id, seg_key), {}))

    def get_depth(self, security_id: str, exchange_segment: str) -> list[dict]:
        """Return the 5-level market depth for a security.

        Market depth is only populated when the instrument was subscribed
        with :data:`FULL` (21).

        Returns
        -------
        list[dict]
            Up to 5 dicts with keys ``bid_price``, ``bid_quantity``,
            ``bid_orders``, ``ask_price``, ``ask_quantity``, ``ask_orders``.
            Empty list when no depth data is available.
        """
        return self.get_tick(security_id, exchange_segment).get("depth", [])

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_instrument(inst: tuple) -> tuple[int, str, int]:
        """Return ``(segment_int, security_id_str, subscription_type_int)``."""
        if len(inst) == 2:
            exchange, sec_id = inst
            sub_type = TICKER
        else:
            exchange, sec_id, sub_type = inst

        if isinstance(exchange, str):
            exchange = _SEGMENT_MAP.get(exchange, 1)

        return (int(exchange), str(sec_id), int(sub_type))

    @staticmethod
    def _segment_key(exchange_segment: str | int) -> str:
        """Normalise an exchange segment to a string key for tick cache lookups."""
        if isinstance(exchange_segment, int):
            return _SEGMENT_REVERSE_MAP.get(exchange_segment, str(exchange_segment))
        return exchange_segment

    def _on_message(self, feed: Any, data: dict) -> None:
        """Internal callback: cache tick and forward to user callback."""
        if not data:
            return

        sec_id = str(data.get("security_id", ""))
        seg_int = data.get("exchange_segment")
        seg_key = self._segment_key(seg_int) if seg_int is not None else ""

        if sec_id:
            with self._lock:
                self._ticks[(sec_id, seg_key)] = data

        if self._on_tick is not None:
            try:
                self._on_tick(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_tick callback raised an exception: %s", exc)

    def _on_error(self, feed: Any, error: Exception) -> None:
        """Internal error callback."""
        logger.error("MarketDataStreamer WebSocket error: %s", error)
