"""Background trading engine.

Runs a 60-second polling loop that:
  1. Fetches the latest market prices for every enabled strategy.
  2. Calls ``generate_signals`` on each strategy instance.
  3. Places orders (or records paper trades) when a signal fires.
  4. Writes every executed trade to the trade journal.

Strategy instances are kept alive between ticks so that stateful strategies
(e.g. EMA Crossover) can accumulate their internal price history correctly.
When trading starts, each strategy is pre-warmed with the most recent
1-minute historical candles so the EMAs/RSI/Bollinger are primed from the
very first real tick.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: TradingEngine | None = None
_engine_lock = threading.Lock()

# Throttle crude auto-detection: only attempt once per 5 minutes per strategy
_crude_autodetect_at: dict[str, float] = {}
_CRUDE_AUTODETECT_COOLDOWN = 300  # seconds


def get_engine() -> "TradingEngine":
    """Return the process-wide singleton :class:`TradingEngine`."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TradingEngine()
    return _engine


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TradingEngine:
    """Runs strategy execution in a background daemon thread.

    Usage::

        engine = get_engine()
        engine.start(broker, paper_trade=True)   # spawns background thread
        ...
        engine.stop()
    """

    #: Polling interval in seconds – 1 s for real-time price updates
    TICK_INTERVAL: int = 1

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Live strategy instances — kept between ticks to preserve state
        self._instances: dict[str, Any] = {}
        self._broker: Any = None
        self._paper_trade: bool = True

    # ------------------------------------------------------------------ #
    #  Public control API                                                  #
    # ------------------------------------------------------------------ #

    def start(self, broker: Any, paper_trade: bool = True) -> None:
        """Start (or update) the trading loop.

        Parameters
        ----------
        broker:
            A connected :class:`~src.broker.dhan_broker.DhanBroker` instance.
        paper_trade:
            When ``True`` trades are only recorded in the journal and not
            sent to the exchange.
        """
        with _engine_lock:
            self._broker = broker
            self._paper_trade = paper_trade

            if self._thread and self._thread.is_alive():
                logger.info("Trading engine already running; updated broker/paper_trade settings.")
                return

            # Reset any leftover stop signal
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="TradingEngine"
            )
            self._thread.start()
            logger.info("Trading engine started (paper_trade=%s).", paper_trade)

    def stop(self) -> None:
        """Signal the trading loop to exit after the current tick."""
        self._stop_event.set()
        logger.info("Trading engine stop requested.")

    def is_running(self) -> bool:
        """Return ``True`` when the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def invalidate_instance(self, strategy_id: str) -> None:
        """Drop a cached instance so it is rebuilt on the next tick.

        Call this after params are changed via the UI.
        """
        self._instances.pop(strategy_id, None)
        logger.debug("Invalidated cached instance for %s.", strategy_id)

    def invalidate_all(self) -> None:
        """Drop all cached instances (e.g. when trading is restarted)."""
        self._instances.clear()
        logger.debug("All strategy instances invalidated.")

    def get_instance(self, strategy_id: str) -> Any:
        """Return the live strategy instance for *strategy_id*, or None."""
        return self._instances.get(strategy_id)

    # ------------------------------------------------------------------ #
    #  Internal loop                                                       #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        logger.info("Trading loop thread started.")
        while not self._stop_event.is_set():
            try:
                from dashboard.strategy_manager import get_trading_active

                if get_trading_active():
                    self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.error("Trading loop error: %s", exc, exc_info=True)
            # Sleep until the next tick (or until stop() is called)
            self._stop_event.wait(self.TICK_INTERVAL)
        logger.info("Trading loop thread stopped.")

    # ------------------------------------------------------------------ #
    #  Single tick                                                         #
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        """Execute one trading tick for all enabled strategies."""
        from dashboard.strategy_manager import (
            STRATEGY_CATALOG,
            build_strategy_instance,
            get_all_strategies,
        )
        from dashboard.regime_finder import auto_refresh_regime, auto_refresh_crude_regime
        from dashboard import market_feed as _mf

        broker = self._broker

        def _ltp(sec_id: str, seg: str) -> float:
            """Get LTP: WebSocket cache first, REST fallback."""
            price = _mf.get_ltp(sec_id, seg)
            if price > 0:
                return price
            if broker is not None and broker._dhan is not None:
                return broker.get_ltp(sec_id, seg)
            return 0.0

        # ---- Auto-refresh NIFTY50 regime ----
        if broker is not None and broker._dhan is not None:
            try:
                auto_refresh_regime(broker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NIFTY regime refresh failed: %s", exc)

        # Fetch NIFTY50 spot price once — reused by all NIFTY option strategies
        nifty_spot: float = _ltp("13", "IDX_I")
        logger.debug("NIFTY50 spot for this tick: %.2f", nifty_spot)

        for strategy_data in get_all_strategies():
            if not strategy_data.get("enabled", False):
                continue

            sid = strategy_data["id"]
            params = strategy_data["params"]

            # --- get or build instance ---
            instance = self._instances.get(sid)
            if instance is None:
                instance = build_strategy_instance(sid, params)
                if instance is None:
                    logger.debug("No builder for strategy %s; skipping.", sid)
                    continue
                if broker is not None:
                    instance.attach_broker(broker)
                # Pre-warm with historical data before the first live tick
                self._prewarm(instance, sid, params)
                self._instances[sid] = instance

            # --- determine close price for this tick ---
            asset_type = STRATEGY_CATALOG.get(sid, {}).get("asset_type", "equity")
            if asset_type == "options":
                close_price = nifty_spot
                if close_price <= 0:
                    logger.debug("Skipping %s – no NIFTY50 spot available.", sid)
                    if hasattr(instance, "_log"):
                        instance._log("Waiting for NIFTY50 LTP — check Dhan credentials & connection", "debug")
                    continue
                tick_df = pd.DataFrame({"close": [close_price]})
            elif asset_type == "crude_options":
                futures_sec_id = str(params.get("futures_security_id", "488290"))
                futures_seg    = str(params.get("under_exchange_segment", "MCX_COMM"))
                crude_spot: float = _ltp(futures_sec_id, futures_seg)
                if crude_spot > 0:
                    logger.info("[CRUDE TICK] %s  sec=%s  LTP=%.2f", sid, futures_sec_id, crude_spot)
                    try:
                        auto_refresh_crude_regime(broker, security_id=futures_sec_id, exchange_segment=futures_seg)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Crude regime refresh skipped: %s", exc)
                if crude_spot <= 0:
                    logger.info("[CRUDE TICK] %s  sec=%s  no LTP available – skipping tick", sid, futures_sec_id)
                    # ---- Auto-detect current contract if this one is expired/invalid ----
                    import time as _time
                    last_attempt = _crude_autodetect_at.get(sid, 0.0)
                    if _time.time() - last_attempt >= _CRUDE_AUTODETECT_COOLDOWN:
                        _crude_autodetect_at[sid] = _time.time()
                        if hasattr(instance, "_log"):
                            instance._log(
                                f"⚙️ sec={futures_sec_id} returned no LTP — auto-detecting current MCX Crude contract …",
                                "debug",
                            )
                        try:
                            from src.broker.dhan_broker import detect_crude_futures_security_id
                            from dashboard.strategy_manager import update_params
                            detected = detect_crude_futures_security_id()
                            if detected and detected["security_id"] and detected["security_id"] != futures_sec_id:
                                new_sid = detected["security_id"]
                                logger.info(
                                    "[CRUDE AUTO-DETECT] Updated %s: old=%s  new=%s  expiry=%s  name=%s",
                                    sid, futures_sec_id, new_sid, detected["expiry"], detected["display_name"],
                                )
                                update_params(sid, {
                                    "futures_security_id": new_sid,
                                    "under_security_id": int(new_sid),
                                })
                                self.invalidate_instance(sid)
                                if hasattr(instance, "_log"):
                                    instance._log(
                                        f"✅ Auto-detected new contract: {detected['display_name']}  "
                                        f"sec={new_sid}  expiry={detected['expiry']} — restarting strategy …",
                                        "signal",
                                    )
                            elif detected and detected["security_id"] == futures_sec_id:
                                if hasattr(instance, "_log"):
                                    instance._log(
                                        f"⚠️ Scrip master confirms sec={futures_sec_id} ({detected['display_name']}) "
                                        f"is the current contract — LTP failure may be a credentials or API issue",
                                        "debug",
                                    )
                            elif not detected:
                                if hasattr(instance, "_log"):
                                    instance._log(
                                        "❌ Auto-detect failed — check internet connection or Dhan scrip master URL",
                                        "debug",
                                    )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Crude auto-detect error: %s", exc)
                            if hasattr(instance, "_log"):
                                instance._log(f"❌ Auto-detect error: {exc}", "debug")
                    else:
                        if hasattr(instance, "_log"):
                            instance._log(
                                f"Waiting for Crude Oil LTP (sec={futures_sec_id}) — "
                                f"next auto-detect in {int(_CRUDE_AUTODETECT_COOLDOWN - (_time.time() - last_attempt))}s",
                                "debug",
                            )
                    continue
                tick_df = pd.DataFrame({"close": [crude_spot]})
            else:
                # Equity: use the strategy symbol's own LTP
                sec_id = str(params.get("security_id", ""))
                seg = str(params.get("exchange_segment", "NSE_EQ"))
                ltp: float = _ltp(sec_id, seg) if sec_id else 0.0
                if ltp <= 0:
                    logger.debug("Skipping %s – no LTP for security_id=%s.", sid, sec_id)
                    if hasattr(instance, "_log"):
                        instance._log(f"Waiting for LTP (security_id={sec_id}, segment={seg}) — check Dhan credentials & connection", "debug")
                    continue
                tick_df = pd.DataFrame({"close": [ltp]})

            # --- call generate_signals ---
            try:
                signal = instance.generate_signals(tick_df)
            except Exception as exc:  # noqa: BLE001
                logger.error("generate_signals error for %s: %s", sid, exc)
                continue

            if signal is None:
                continue

            logger.info(
                "Signal from %s: %s %s qty=%s @ %.2f",
                sid,
                signal.get("action"),
                signal.get("symbol"),
                signal.get("quantity"),
                float(signal.get("price", 0)),
            )
            self._execute_signal(sid, strategy_data["name"], signal, instance)

    # ------------------------------------------------------------------ #
    #  Signal execution                                                    #
    # ------------------------------------------------------------------ #

    def _execute_signal(
        self, strategy_id: str, strategy_name: str, signal: dict, instance: Any = None
    ) -> None:
        """Place the order or record a paper trade, then journal it."""
        from dashboard.risk_manager import check_risk_limits
        from dashboard.trade_journal import record_trade_entry

        broker = self._broker
        action = str(signal.get("action", "BUY")).upper()
        security_id = str(signal.get("security_id", ""))
        symbol = str(signal.get("symbol", security_id))
        quantity = int(signal.get("quantity", 1))
        price = float(signal.get("price", 0.0))
        exchange_segment = str(signal.get("exchange_segment", "NSE_FNO"))
        order_type = str(signal.get("order_type", "MARKET"))
        product_type = str(signal.get("product_type", "INTRADAY"))
        option_type = str(signal.get("option_type", ""))
        strike = signal.get("strike", 0.0)

        def _strategy_log(msg: str) -> None:
            if instance is not None and hasattr(instance, "_log"):
                try:
                    instance._log(msg, "signal")
                except Exception:
                    pass

        # --- risk gate ---
        trade_value = price * quantity
        risk_result = check_risk_limits(
            strategy_id=strategy_id, trade_value=trade_value
        )
        if not risk_result.get("allowed", True):
            reason = risk_result.get("reason", "")
            logger.warning(
                "Risk limit blocked %s signal for %s: %s",
                action, strategy_id, reason,
            )
            _strategy_log(f"BLOCKED by risk manager: {reason}")
            return

        # --- place order or paper-trade ---
        if self._paper_trade or broker is None or broker._dhan is None:
            logger.info(
                "[PAPER] %s %s x%d @ %.2f (strategy=%s)",
                action, symbol, quantity, price, strategy_id,
            )
            _strategy_log(
                f"[PAPER TRADE] {action} {option_type or ''} "
                f"strike={strike} sec={security_id} x{quantity} @ ₹{price:.2f}"
            )
        else:
            try:
                broker.place_order(
                    security_id=security_id,
                    exchange_segment=exchange_segment,
                    transaction_type=action,
                    quantity=quantity,
                    order_type=order_type,
                    product_type=product_type,
                    price=price if order_type == "LIMIT" else 0,
                )
                logger.info(
                    "[LIVE] Placed %s %s x%d for strategy=%s",
                    action, symbol, quantity, strategy_id,
                )
                _strategy_log(
                    f"[LIVE ORDER] {action} {option_type or ''} "
                    f"strike={strike} sec={security_id} x{quantity} @ ₹{price:.2f}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Order placement failed for %s: %s", strategy_id, exc
                )
                _strategy_log(f"ORDER FAILED: {exc}")
                return

        # --- SL / target from order management settings ---
        from dashboard.order_manager import get_sl_and_target  # noqa: PLC0415
        from dashboard.regime_finder import get_current_regime  # noqa: PLC0415

        sl_target = get_sl_and_target(entry_price=price, action=action)
        sl_price     = sl_target.get("sl_price",     0.0)
        target_price = sl_target.get("target_price", 0.0)

        # --- journal ---
        try:
            record_trade_entry(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                symbol=symbol,
                security_id=security_id,
                action=action,
                quantity=quantity,
                entry_price=price,
                option_type=option_type,
                exchange_segment=exchange_segment,
                sl_price=sl_price,
                target_price=target_price,
                regime=get_current_regime(),
                notes=f"Auto-trade | {'Paper' if self._paper_trade else 'Live'}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to journal trade for %s: %s", strategy_id, exc)

    # ------------------------------------------------------------------ #
    #  Historical pre-warm                                                 #
    # ------------------------------------------------------------------ #

    def _prewarm(
        self, instance: Any, strategy_id: str, params: dict
    ) -> None:
        """Seed the strategy with recent 1-minute historical candles.

        This ensures indicators (EMA, RSI, Bollinger) are properly primed
        before the first live signal is evaluated.
        """
        from dashboard.strategy_manager import STRATEGY_CATALOG
        from src.data.data_fetcher import DataFetcher

        broker = self._broker
        if broker is None:
            return

        asset_type = STRATEGY_CATALOG.get(strategy_id, {}).get("asset_type", "equity")
        fetcher = DataFetcher(broker=broker)

        try:
            if asset_type == "options":
                slow_period = int(params.get("slow_period", 21))
                n_bars = slow_period + 10
                hist = fetcher.get_historical_data(
                    symbol="NIFTY50",
                    security_id="13",
                    exchange_segment="IDX_I",
                    instrument_type="INDEX",
                    interval=1,
                )
                if hist.empty or "close" not in hist.columns:
                    logger.debug("No 1-min history for NIFTY50 to pre-warm %s.", strategy_id)
                    return
                warm_closes = hist["close"].tail(n_bars).tolist()
            elif asset_type == "crude_options":
                slow_period = int(params.get("slow_period", 21))
                n_bars = slow_period + 10
                futures_sec_id = str(params.get("futures_security_id", "488290"))
                futures_seg    = str(params.get("under_exchange_segment", "MCX_COMM"))
                logger.info("[CRUDE PREWARM] %s  fetching %d 1-min bars  sec=%s", strategy_id, n_bars, futures_sec_id)
                hist = fetcher.get_historical_data(
                    symbol="CRUDE_OIL",
                    security_id=futures_sec_id,
                    exchange_segment=futures_seg,
                    instrument_type="FUTCOM",
                    interval=1,
                )
                if hist.empty or "close" not in hist.columns:
                    logger.info("[CRUDE PREWARM] %s  no 1-min history for sec=%s – starting cold", strategy_id, futures_sec_id)
                    return
                warm_closes = hist["close"].tail(n_bars).tolist()
            else:
                slow_period = int(
                    params.get("slow_period", params.get("period", 50))
                )
                n_bars = slow_period + 10
                hist = fetcher.get_historical_data(
                    symbol=str(params.get("symbol", "")),
                    security_id=str(params.get("security_id", "")),
                    exchange_segment=str(params.get("exchange_segment", "NSE_EQ")),
                    instrument_type="EQUITY",
                    interval=1,
                )
                if hist.empty or "close" not in hist.columns:
                    logger.debug("No 1-min history to pre-warm %s.", strategy_id)
                    return
                warm_closes = hist["close"].tail(n_bars).tolist()

            # Feed bars directly into _prices (candle-based strategies) or
            # via generate_signals (legacy indicator strategies).
            # Direct population avoids the minute-boundary tracker being
            # confused by rapid historical-data ingestion within a single
            # real-world minute.
            if hasattr(instance, "_prices") and hasattr(instance, "_current_minute"):
                # EMA-style candle-based strategy — populate price history directly
                for close in warm_closes[:-1]:
                    instance._prices.append(float(close))
                logger.info(
                    "[PREWARM] %s  loaded %d historical candle closes → EMAs primed.",
                    strategy_id, len(instance._prices),
                )
            else:
                # Legacy/indicator strategy — feed via generate_signals
                saved_position = getattr(instance, "_position", None)
                for close in warm_closes[:-1]:
                    try:
                        instance.generate_signals(pd.DataFrame({"close": [float(close)]}))
                    except Exception:  # noqa: BLE001
                        pass
                if saved_position is not None and hasattr(instance, "_position"):
                    instance._position = saved_position
                logger.info(
                    "[PREWARM] %s  fed %d historical 1-min bars → ready.",
                    strategy_id, len(warm_closes),
                )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-warm skipped for %s: %s", strategy_id, exc)
