"""Dhan broker integration module."""

import json
import os
import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Credentials file written by the dashboard Credentials UI
_CREDENTIALS_FILE = Path(__file__).resolve().parent.parent.parent / "dashboard" / "data" / "credentials.json"

# Rate-limit repeated get_ltp warnings: only log once per 60 s per security
_ltp_warn_at: dict[str, float] = {}

# Maps exchange segment → instrument_type used by intraday_minute_data
_SEGMENT_TO_INSTRUMENT: dict[str, str] = {
    "MCX_COMM":     "FUTCOM",
    "IDX_I":        "INDEX",
    "NSE_EQ":       "EQUITY",
    "BSE_EQ":       "EQUITY",
    "NSE_FNO":      "FUTSTK",
    "BSE_FNO":      "FUTSTK",
    "NSE_CURRENCY": "FUTCUR",
    "BSE_CURRENCY": "FUTCUR",
}


class DhanBroker:
    """Wrapper around the Dhan trading API (dhanhq >= 2.1.0).

    Provides methods for placing/modifying/cancelling orders, fetching
    positions, holdings, market quotes, and option chain data.

    When *paper_trade* is ``True`` (the default) all order operations are
    simulated locally so you can test strategies without risking real capital.
    Data-fetching methods (option chain, LTP, market quotes) always attempt to
    connect to the live Dhan API when credentials are present, even in
    paper-trade mode.
    """

    # Exchange segments
    NSE_EQ = "NSE_EQ"
    BSE_EQ = "BSE_EQ"
    NSE_FNO = "NSE_FNO"
    BSE_FNO = "BSE_FNO"
    MCX = "MCX_COMM"
    NSE_CURRENCY = "NSE_CURRENCY"
    IDX_I = "IDX_I"  # Index segment for option chain underlying

    # Order types
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"

    # Transaction types
    BUY = "BUY"
    SELL = "SELL"

    # Product types
    INTRADAY = "INTRADAY"
    DELIVERY = "CNC"
    MARGIN = "MARGIN"

    def __init__(
        self,
        client_id: str | None = None,
        access_token: str | None = None,
        paper_trade: bool = True,
    ) -> None:
        self.client_id = client_id or os.getenv("DHAN_CLIENT_ID", "")
        self.access_token = access_token or os.getenv("DHAN_ACCESS_TOKEN", "")

        # Fall back to persisted credentials written by the dashboard UI
        if not self.client_id or not self.access_token:
            self._load_credentials_file()

        self.paper_trade = paper_trade

        self._paper_orders: list[dict] = []
        self._order_counter = 1

        self._dhan: Any = None
        # Always connect when credentials are available – data methods (option
        # chain, LTP, quotes) work in both paper-trade and live modes.
        if self.client_id and self.access_token:
            self._connect()

        mode = "paper-trade" if paper_trade else "live"
        logger.info("DhanBroker initialised in %s mode (client_id=%s)", mode, self.client_id)

    def _load_credentials_file(self) -> None:
        """Load client_id and access_token from the dashboard credentials file.

        Falls back to ``config/config.yaml`` when the JSON credentials file
        does not contain valid credentials.
        """
        try:
            if _CREDENTIALS_FILE.exists():
                with _CREDENTIALS_FILE.open() as f:
                    data = json.load(f)
                file_client_id = data.get("client_id", "")
                file_access_token = data.get("access_token", "")
                if file_client_id and file_access_token:
                    self.client_id = file_client_id
                    self.access_token = file_access_token
                    logger.info("Loaded credentials from %s", _CREDENTIALS_FILE)
                    return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read credentials file (%s): %s", _CREDENTIALS_FILE, exc)
            logger.warning("Could not load credentials from credentials file.")

        # Last resort: try config/config.yaml (broker.client_id / broker.access_token)
        try:
            import yaml  # already in requirements
            _config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            if _config_path.exists():
                with _config_path.open() as f:
                    cfg = yaml.safe_load(f) or {}
                broker_cfg = cfg.get("broker", {})
                cfg_client_id = str(broker_cfg.get("client_id", "")).strip()
                cfg_access_token = str(broker_cfg.get("access_token", "")).strip()
                if cfg_client_id and cfg_access_token:
                    self.client_id = cfg_client_id
                    self.access_token = cfg_access_token
                    logger.info("Loaded credentials from config.yaml")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read config.yaml for credentials: %s", exc)

    def _connect(self) -> None:
        """Establish a connection to the Dhan API using DhanContext."""
        try:
            from dhanhq import DhanContext, dhanhq  # type: ignore[import-untyped]

            dhan_context = DhanContext(self.client_id, self.access_token)
            self._dhan = dhanhq(dhan_context)
            logger.info("Connected to Dhan API (v2)")
        except ImportError as exc:
            raise ImportError(
                "dhanhq>=2.1.0 is required. Install with: pip install 'dhanhq>=2.1.0'"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize Dhan API connection: %s", exc)
            self._dhan = None

    # ------------------------------------------------------------------ #
    #  Order management                                                    #
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "LIMIT",
        product_type: str = "INTRADAY",
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> dict:
        """Place a buy or sell order.

        Returns a dict with ``order_id`` and ``status`` keys.
        """
        if self.paper_trade:
            return self._paper_place_order(
                security_id=security_id,
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product_type=product_type,
                price=price,
            )

        logger.info(
            "place_order payload → security_id=%s  exchange_segment=%s  "
            "transaction_type=%s  quantity=%s  order_type=%s  product_type=%s  price=%s",
            security_id, exchange_segment, transaction_type,
            quantity, order_type, product_type, price,
        )
        response = self._dhan.place_order(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            trigger_price=trigger_price,
        )
        logger.info("Order response: %s", response)

        # dhanhq returns {'status': 'success'|'failure', 'remarks': ..., 'data': ...}
        if response.get("status") != "success":
            remarks = response.get("remarks", {})
            if isinstance(remarks, dict):
                msg = remarks.get("error_message") or remarks.get("error_type") or str(remarks)
            else:
                msg = str(remarks) or "Unknown error"
            raise RuntimeError(f"Dhan rejected order: {msg}")

        # Normalise so callers can always use response["order_id"]
        data = response.get("data") or {}
        if isinstance(data, dict) and "orderId" in data:
            response["order_id"] = data["orderId"]

        return response

    def place_option_order(
        self,
        security_id: str,
        transaction_type: str,
        quantity: int,
        order_type: str = "MARKET",
        product_type: str = "INTRADAY",
        price: float = 0.0,
        trigger_price: float = 0.0,
        exchange_segment: str = "NSE_FNO",
    ) -> dict:
        """Convenience method for placing F&O (option) orders.

        Defaults *exchange_segment* to ``NSE_FNO`` and *order_type* to
        ``MARKET``.  All other arguments are the same as :meth:`place_order`.
        """
        return self.place_order(
            security_id=security_id,
            exchange_segment=exchange_segment,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product_type=product_type,
            price=price,
            trigger_price=trigger_price,
        )

    def _paper_place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        product_type: str,
        price: float,
    ) -> dict:
        """Simulate order placement for paper trading."""
        order_id = f"PAPER-{self._order_counter:05d}"
        self._order_counter += 1
        order = {
            "order_id": order_id,
            "security_id": security_id,
            "exchange_segment": exchange_segment,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product_type": product_type,
            "price": price,
            "status": "TRADED",
        }
        self._paper_orders.append(order)
        logger.info("Paper order placed: %s", order)
        return {"order_id": order_id, "status": "TRADED"}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order by *order_id*."""
        if self.paper_trade:
            logger.info("Paper cancel order: %s", order_id)
            return {"order_id": order_id, "status": "CANCELLED"}

        response = self._dhan.cancel_order(order_id)
        logger.info("Order cancelled: %s", response)
        return response

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        order_type: str | None = None,
    ) -> dict:
        """Modify an existing open order."""
        if self.paper_trade:
            logger.info("Paper modify order: %s", order_id)
            return {"order_id": order_id, "status": "MODIFIED"}

        kwargs: dict[str, Any] = {"order_id": order_id}
        if quantity is not None:
            kwargs["quantity"] = quantity
        if price is not None:
            kwargs["price"] = price
        if trigger_price is not None:
            kwargs["trigger_price"] = trigger_price
        if order_type is not None:
            kwargs["order_type"] = order_type

        response = self._dhan.modify_order(**kwargs)
        logger.info("Order modified: %s", response)
        return response

    # ------------------------------------------------------------------ #
    #  Account / portfolio queries                                         #
    # ------------------------------------------------------------------ #

    def get_order_list(self) -> list[dict]:
        """Return a list of today's orders."""
        if self.paper_trade:
            return list(self._paper_orders)

        response = self._dhan.get_order_list()
        return response.get("data", [])

    def get_positions(self) -> list[dict]:
        """Return current open positions."""
        if self.paper_trade:
            return []

        response = self._dhan.get_positions()
        return response.get("data", [])

    def get_holdings(self) -> list[dict]:
        """Return long-term holdings."""
        if self.paper_trade:
            return []

        response = self._dhan.get_holdings()
        return response.get("data", [])

    def get_fund_limits(self) -> dict:
        """Return available margin / fund details."""
        if self.paper_trade:
            return {"availableBalance": 0, "sodLimit": 0}

        response = self._dhan.get_fund_limits()
        return response.get("data", {})

    # ------------------------------------------------------------------ #
    #  Live market data (available in both paper-trade and live modes)    #
    # ------------------------------------------------------------------ #

    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        """Return the last traded price for a security via the Dhan REST API.

        Tries three methods in order:
        1. ``ticker_data``       — LTP market feed (fastest)
        2. ``ohlc_data``         — OHLC snapshot (fallback when LTP feed is unavailable)
        3. ``intraday_minute_data`` — last 1-min candle close (fallback for MCX / F&O)

        Returns ``0.0`` when no API connection is available or all methods fail.
        """
        if self._dhan is None:
            return 0.0

        def _throttled_warn(msg: str) -> None:
            """Log *msg* at WARNING level at most once every 60 s per security."""
            key = f"{security_id}:{exchange_segment}"
            now = time.monotonic()
            if now - _ltp_warn_at.get(key, 0.0) >= 60.0:
                _ltp_warn_at[key] = now
                logger.warning(msg)

        def _price_from_records(records: list, source: str) -> float:
            """Extract LTP from a list of market-data records."""
            if not records:
                return 0.0
            rec = records[0]
            price = float(rec.get("last_price") or rec.get("close") or 0.0)
            if price > 0:
                logger.debug("get_ltp [%s] %s/%s → %.2f", source, security_id, exchange_segment, price)
            return price

        # ------------------------------------------------------------------ #
        # Method 1: ticker_data (LTP feed)                                   #
        # ------------------------------------------------------------------ #
        try:
            response = self._dhan.ticker_data(
                securities={exchange_segment: [int(security_id)]}
            )
            if isinstance(response, dict):
                data = response.get("data", {})
                if isinstance(data, dict):
                    price = _price_from_records(data.get(exchange_segment, []), "ticker")
                    if price > 0:
                        return price
                    # data dict present but empty → segment name mismatch, try other keys
                    for records in data.values():
                        if isinstance(records, list):
                            price = _price_from_records(records, "ticker-alt")
                            if price > 0:
                                return price
                # data is not a dict (e.g. '' on API error) → fall through to next method
        except Exception as exc:
            logger.debug("get_ltp ticker_data exception for %s/%s: %s", security_id, exchange_segment, exc)

        # ------------------------------------------------------------------ #
        # Method 2: ohlc_data (snapshot)                                     #
        # ------------------------------------------------------------------ #
        try:
            response = self._dhan.ohlc_data(
                securities={exchange_segment: [int(security_id)]}
            )
            if isinstance(response, dict):
                data = response.get("data", {})
                if isinstance(data, dict):
                    price = _price_from_records(data.get(exchange_segment, []), "ohlc")
                    if price > 0:
                        return price
                    for records in data.values():
                        if isinstance(records, list):
                            price = _price_from_records(records, "ohlc-alt")
                            if price > 0:
                                return price
        except Exception as exc:
            logger.debug("get_ltp ohlc_data exception for %s/%s: %s", security_id, exchange_segment, exc)

        # ------------------------------------------------------------------ #
        # Method 3: intraday_minute_data — last 1-min candle close           #
        # Used by regime_finder for MCX/INDEX; works when feed APIs fail.    #
        # ------------------------------------------------------------------ #
        instrument_type = _SEGMENT_TO_INSTRUMENT.get(exchange_segment, "EQUITY")
        try:
            import datetime as _dt
            today = _dt.date.today().isoformat()
            response = self._dhan.intraday_minute_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=today,
                to_date=today,
                interval=1,
            )
            if isinstance(response, dict):
                data = response.get("data", {})
                if isinstance(data, dict):
                    closes = data.get("close", [])
                    if closes:
                        price = float(closes[-1])
                        if price > 0:
                            logger.debug(
                                "get_ltp [intraday] %s/%s → %.2f (last candle close)",
                                security_id, exchange_segment, price,
                            )
                            return price
        except Exception as exc:
            logger.debug("get_ltp intraday exception for %s/%s: %s", security_id, exchange_segment, exc)

        _throttled_warn(
            f"get_ltp: all methods failed for {security_id} ({exchange_segment} / {instrument_type}) "
            "— check credentials, market hours, and subscription"
        )
        return 0.0

    def get_market_quote(
        self,
        securities: dict[str, list[int]],
        mode: str = "ticker",
    ) -> dict:
        """Return market quote data for a basket of securities.

        Parameters
        ----------
        securities:
            Mapping of exchange-segment → list of integer security IDs,
            e.g. ``{"NSE_FNO": [52175, 52176]}``.
        mode:
            ``"ticker"`` (LTP only), ``"ohlc"`` (OHLC snapshot), or
            ``"quote"`` (full packet including depth, OI, etc.).

        Returns the raw response dict from the Dhan API, or an empty dict
        when no API connection is available.
        """
        if self._dhan is None:
            logger.warning("get_market_quote called without API connection.")
            return {}

        try:
            if mode == "ohlc":
                return self._dhan.ohlc_data(securities=securities)
            if mode == "quote":
                return self._dhan.quote_data(securities=securities)
            return self._dhan.ticker_data(securities=securities)
        except Exception as exc:
            logger.warning("get_market_quote failed: %s", exc)
            return {}

    # ------------------------------------------------------------------ #
    #  Option chain                                                        #
    # ------------------------------------------------------------------ #

    def get_expiry_list(
        self,
        under_security_id: int,
        under_exchange_segment: str = "IDX_I",
    ) -> list[str]:
        """Return available expiry dates for an underlying.

        Parameters
        ----------
        under_security_id:
            Dhan security ID of the underlying (e.g. 13 for Nifty 50).
        under_exchange_segment:
            Exchange segment of the underlying (default ``"IDX_I"`` for
            index; use ``"NSE_EQ"`` for equities).

        Returns a list of expiry date strings (``YYYY-MM-DD``), or an empty
        list when no API connection is available.
        """
        if self._dhan is None:
            logger.warning("get_expiry_list called without API connection.")
            return []

        try:
            response = self._dhan.expiry_list(
                under_security_id=under_security_id,
                under_exchange_segment=under_exchange_segment,
            )
            if not isinstance(response, dict):
                logger.warning(
                    "get_expiry_list unexpected response (%s) for seg=%s: %s",
                    type(response).__name__, under_exchange_segment, response,
                )
                return []
            # dhanhq >= 2.x: response["data"]["data"] is the list of dates
            outer = response.get("data", {})
            if not isinstance(outer, dict):
                return []
            inner = outer.get("data", None)
            if isinstance(inner, list):
                return inner
            # Fallback for older SDK versions that used "ExpiryDate"
            return outer.get("ExpiryDate", [])
        except Exception as exc:
            logger.warning("get_expiry_list failed: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Real-time WebSocket streaming                                      #
    # ------------------------------------------------------------------ #

    def create_streamer(
        self,
        instruments: list | None = None,
        on_tick=None,
    ):
        """Create a :class:`~src.data.market_streamer.MarketDataStreamer`.

        The streamer uses Dhan's WebSocket market feed (``MarketFeed``) for
        real-time price data with lower latency than the REST ``ticker_data``
        endpoint.  It supports :data:`~src.data.market_streamer.TICKER`,
        :data:`~src.data.market_streamer.QUOTE`, and
        :data:`~src.data.market_streamer.FULL` packet types.

        Parameters
        ----------
        instruments:
            Initial list of instrument tuples
            ``(exchange_segment, security_id[, subscription_type])``.
            *exchange_segment* should be a ``DhanBroker`` segment string
            such as ``"NSE_EQ"``.  *subscription_type* defaults to ``TICKER``
            (15) when omitted.
        on_tick:
            Optional callback ``(data: dict) -> None`` called for every
            incoming tick.

        Returns
        -------
        :class:`~src.data.market_streamer.MarketDataStreamer`
            A streamer instance.  Call :meth:`~MarketDataStreamer.start` to
            begin receiving ticks.

        Raises
        ------
        RuntimeError
            When no API connection is available (credentials are missing).

        Examples
        --------
        ::

            from src.broker.dhan_broker import DhanBroker
            from src.data.market_streamer import QUOTE

            broker = DhanBroker()  # reads credentials from .env
            streamer = broker.create_streamer(
                instruments=[("NSE_EQ", "2885", QUOTE)],  # RELIANCE
                on_tick=lambda d: print(d["LTP"]),
            )
            streamer.start()
        """
        if self._dhan is None:
            raise RuntimeError(
                "No Dhan API connection available. "
                "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to use the streamer."
            )

        from dhanhq import DhanContext  # type: ignore[import-untyped]
        from src.data.market_streamer import MarketDataStreamer

        dhan_context = DhanContext(self.client_id, self.access_token)
        return MarketDataStreamer(
            dhan_context=dhan_context,
            instruments=instruments,
            on_tick=on_tick,
        )

    def get_option_chain(
        self,
        under_security_id: int,
        under_exchange_segment: str,
        expiry: str,
    ) -> dict:
        """Fetch the full option chain for an underlying and expiry.

        Parameters
        ----------
        under_security_id:
            Dhan security ID of the underlying (e.g. 13 for Nifty 50,
            25 for BankNifty).
        under_exchange_segment:
            Exchange segment of the underlying (e.g. ``"IDX_I"``).
        expiry:
            Expiry date string in ``YYYY-MM-DD`` format.

        Returns the raw ``data`` dict from the Dhan API, which contains
        ``oc_data`` (list of strike-level records with call/put details),
        ``last_price`` (spot LTP), and ``expiry_list``.  Returns an empty
        dict when no API connection is available.
        """
        if self._dhan is None:
            logger.warning("get_option_chain called without API connection.")
            return {}

        try:
            response = self._dhan.option_chain(
                under_security_id=under_security_id,
                under_exchange_segment=under_exchange_segment,
                expiry=expiry,
            )
            if not isinstance(response, dict):
                logger.warning(
                    "get_option_chain unexpected response (%s) for seg=%s: %s",
                    type(response).__name__, under_exchange_segment, response,
                )
                return {}
            # dhanhq >= 2.x wraps server JSON in {"status":…, "data": <server_json>}.
            # The server JSON is {"data": {"last_price":…, "oc":{…}}, "status":"success"}.
            # So the actual chain data is at response["data"]["data"].
            outer = response.get("data", {})
            if not isinstance(outer, dict):
                logger.warning(
                    "get_option_chain: outer 'data' is %s (expected dict) for "
                    "under_id=%s expiry=%s — full response: %s",
                    type(outer).__name__, under_security_id, expiry, response,
                )
                return {}
            inner = outer.get("data", None)
            if isinstance(inner, dict):
                # Detect API-level error inside a 200 response
                if inner.get("status") not in (None, "success", "SUCCESS"):
                    logger.warning(
                        "get_option_chain: API error in inner data for "
                        "under_id=%s expiry=%s: %s",
                        under_security_id, expiry, inner,
                    )
                    return {}
                return inner
            # Fallback: outer IS the chain dict (older SDK or different wrapping)
            if not outer.get("oc") and not outer.get("oc_data"):
                logger.warning(
                    "get_option_chain: unrecognised response structure for "
                    "under_id=%s expiry=%s — outer keys: %s",
                    under_security_id, expiry, list(outer.keys()),
                )
                return {}
            return outer
        except Exception as exc:
            logger.warning("get_option_chain failed: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Standalone utility — MCX Crude Oil contract auto-detection
# ---------------------------------------------------------------------------

_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


def detect_crude_futures_security_id() -> dict | None:
    """Download Dhan's instrument master CSV and return the near-month MCX Crude Oil
    FUTCOM contract details, or ``None`` on failure.

    Returns a dict with keys:
        security_id  – Dhan security ID string for the near-month futures contract
        expiry       – ISO date string, e.g. "2026-06-17"
        display_name – Human-readable label, e.g. "CRUDEOIL JUN FUT"
    """
    import csv
    import datetime
    import io
    import urllib.request

    try:
        req = urllib.request.Request(
            _SCRIP_MASTER_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.error("Scrip master download failed: %s", exc)
        return None

    today = datetime.date.today()
    best: dict | None = None
    try:
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            if row.get("SEM_EXM_EXCH_ID", "").strip() != "MCX":
                continue
            if row.get("SEM_INSTRUMENT_NAME", "").strip() != "FUTCOM":
                continue
            sym = row.get("SM_SYMBOL_NAME", "").strip().upper()
            trd = row.get("SEM_TRADING_SYMBOL", "").strip().upper()
            if "CRUDE" not in sym and "CRUDE" not in trd:
                continue
            exp_str = row.get("SEM_EXPIRY_DATE", "").strip()[:10]
            try:
                exp_date = datetime.date.fromisoformat(exp_str)
            except ValueError:
                continue
            if exp_date < today:
                continue
            if best is None or exp_date < best["exp_date"]:
                best = {
                    "security_id":  row.get("SEM_SMST_SECURITY_ID", "").strip(),
                    "expiry":       exp_date.isoformat(),
                    "exp_date":     exp_date,
                    "display_name": (
                        row.get("SEM_CUSTOM_SYMBOL", "").strip()
                        or trd
                    ),
                }
    except Exception as exc:  # noqa: BLE001
        logger.error("Scrip master parse failed: %s", exc)
        return None

    if not best or not best["security_id"]:
        logger.warning("No active MCX CRUDEOIL FUTCOM found in instrument master.")
        return None

    return {
        "security_id":  best["security_id"],
        "expiry":       best["expiry"],
        "display_name": best["display_name"],
    }

