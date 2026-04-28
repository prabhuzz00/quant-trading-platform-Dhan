"""Dhan broker integration module."""

import json
import os
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Credentials file written by the dashboard Credentials UI
_CREDENTIALS_FILE = Path(__file__).resolve().parent.parent.parent / "dashboard" / "data" / "credentials.json"


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
        """Load client_id and access_token from the dashboard credentials file."""
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read credentials file: %s", exc)

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
        logger.info("Order placed: %s", response)
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

        Returns ``0.0`` when no API connection is available.
        """
        if self._dhan is None:
            return 0.0

        try:
            response = self._dhan.ticker_data(
                securities={exchange_segment: [int(security_id)]}
            )
            data = response.get("data", {})
            # ticker_data returns a list of records under the segment key
            records = data.get(exchange_segment, [])
            if records:
                return float(records[0].get("last_price", 0.0))
        except Exception as exc:
            logger.warning("get_ltp failed for %s: %s", security_id, exc)
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
            data = response.get("data", {})
            return data.get("ExpiryDate", [])
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
            return response.get("data", {})
        except Exception as exc:
            logger.warning("get_option_chain failed: %s", exc)
            return {}

