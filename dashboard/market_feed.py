"""Process-wide singleton WebSocket market feed using Dhan Live Market Feed.

All modules should use :func:`get_streamer` to obtain the shared instance and
:func:`get_ltp` as a convenient drop-in replacement for the REST-based
``DhanBroker.get_ltp()``.

Usage::

    from dashboard.market_feed import get_streamer, get_ltp, subscribe

    # Subscribe instruments before or after the streamer starts
    subscribe("13",     "IDX_I",    TICKER)   # NIFTY index
    subscribe("488290", "MCX_COMM", TICKER)   # Crude Oil futures

    # Start once (e.g. inside app startup or trading_start)
    start_feed(broker)

    # Read LTP anywhere
    price = get_ltp("13", "IDX_I")
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from src.data.market_streamer import MarketDataStreamer, TICKER, QUOTE, FULL  # noqa: F401

logger = logging.getLogger(__name__)

_streamer: MarketDataStreamer | None = None
_streamer_lock = threading.Lock()

# Instruments pending subscription before the streamer is started
_pending: list[tuple[str, str, int]] = []


def get_streamer() -> MarketDataStreamer | None:
    """Return the running streamer singleton, or None if not yet started."""
    return _streamer


def is_running() -> bool:
    """Return True when the WebSocket feed is connected and streaming."""
    return _streamer is not None and _streamer.is_running()


def subscribe(security_id: str, exchange_segment: str, sub_type: int = TICKER) -> None:
    """Subscribe an instrument to the live feed.

    Safe to call before :func:`start_feed` – the instrument is queued and
    sent as soon as the connection is established.  Safe to call after start –
    the subscription is sent immediately over the live connection.
    """
    global _pending  # noqa: PLW0603
    with _streamer_lock:
        if _streamer is not None and _streamer.is_running():
            _streamer.subscribe(security_id, exchange_segment, sub_type)
        else:
            entry = (security_id, exchange_segment, sub_type)
            if entry not in _pending:
                _pending.append(entry)
                logger.debug("Queued subscription: %s/%s type=%d", exchange_segment, security_id, sub_type)


def get_ltp(security_id: str, exchange_segment: str) -> float:
    """Return the WebSocket-streamed LTP for a security.

    Falls back to 0.0 when the streamer is not running or no tick has been
    received yet for this security.
    """
    with _streamer_lock:
        s = _streamer
    if s is None:
        return 0.0
    return s.get_ltp(security_id, exchange_segment)


def get_tick(security_id: str, exchange_segment: str) -> dict:
    """Return the full latest tick dict (LTP, OHLC, OI, depth …)."""
    with _streamer_lock:
        s = _streamer
    if s is None:
        return {}
    return s.get_tick(security_id, exchange_segment)


def start_feed(broker: Any) -> None:
    """Start (or restart) the WebSocket feed using *broker*'s credentials.

    If the feed is already running this is a no-op.  Call
    :func:`stop_feed` first if you want to reconnect with different
    credentials.
    """
    global _streamer, _pending  # noqa: PLW0603

    with _streamer_lock:
        if _streamer is not None and _streamer.is_running():
            logger.info("market_feed: WebSocket feed already running – skipping start.")
            # Still flush any newly queued subscriptions
            for sec_id, seg, sub_type in _pending:
                try:
                    _streamer.subscribe(sec_id, seg, sub_type)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("market_feed: subscribe error for %s/%s: %s", seg, sec_id, exc)
            _pending.clear()
            return

        if broker is None or broker._dhan is None:
            logger.warning("market_feed: no broker credentials – cannot start WebSocket feed.")
            return

        try:
            from dhanhq import DhanContext  # type: ignore[import-untyped]
        except ImportError:
            logger.error("market_feed: dhanhq not installed – cannot start WebSocket feed.")
            return

        dhan_context = DhanContext(broker.client_id, broker.access_token)

        # Build initial instrument list from pending queue
        init_instruments = [
            (seg, sec_id, sub_type)
            for sec_id, seg, sub_type in _pending
        ]
        _pending.clear()

        try:
            _streamer = MarketDataStreamer(
                dhan_context=dhan_context,
                instruments=init_instruments if init_instruments else None,
                on_tick=_on_tick,
            )
            _streamer.start()
            logger.info(
                "market_feed: WebSocket feed started with %d initial instruments.",
                len(init_instruments),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("market_feed: failed to start WebSocket feed: %s", exc)
            _streamer = None


def stop_feed() -> None:
    """Stop the WebSocket feed."""
    global _streamer  # noqa: PLW0603
    with _streamer_lock:
        if _streamer is not None:
            try:
                _streamer.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("market_feed: error stopping feed: %s", exc)
            _streamer = None
            logger.info("market_feed: WebSocket feed stopped.")


def _on_tick(data: dict) -> None:
    """Global tick handler – extend here for cross-cutting concerns."""
    # Currently a no-op; the MarketDataStreamer already caches the tick.
    pass
