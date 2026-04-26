"""Dhan broker integration module."""

import os
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DhanBroker:
    """Wrapper around the Dhan trading API.

    Provides methods for placing/modifying/cancelling orders, fetching
    positions, holdings, and market quotes.  When *paper_trade* is ``True``
    (the default) all order operations are simulated locally so you can test
    strategies without risking real capital.
    """

    # Supported exchange segments (mirrors Dhan constants)
    NSE_EQ = "NSE_EQ"
    BSE_EQ = "BSE_EQ"
    NSE_FNO = "NSE_FNO"
    BSE_FNO = "BSE_FNO"
    MCX = "MCX_COMM"
    NSE_CURRENCY = "NSE_CURRENCY"

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
        self.paper_trade = paper_trade

        self._paper_orders: list[dict] = []
        self._order_counter = 1

        self._dhan: Any = None
        if not paper_trade:
            self._connect()

        mode = "paper-trade" if paper_trade else "live"
        logger.info("DhanBroker initialised in %s mode (client_id=%s)", mode, self.client_id)

    def _connect(self) -> None:
        """Establish a live connection to the Dhan API."""
        try:
            from dhanhq import dhanhq  # type: ignore[import-untyped]

            self._dhan = dhanhq(self.client_id, self.access_token)
            logger.info("Connected to Dhan API")
        except ImportError as exc:
            raise ImportError(
                "dhanhq package is required for live trading. "
                "Install it with: pip install dhanhq"
            ) from exc

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

    def get_ltp(self, security_id: str, exchange_segment: str) -> float:
        """Return the last traded price for a security.

        Returns 0.0 in paper-trade mode (caller should supply price from data
        feed instead).
        """
        if self.paper_trade:
            return 0.0

        response = self._dhan.get_ltp_data(
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type="EQUITY",
        )
        data = response.get("data", {})
        return float(data.get("lastTradedPrice", 0.0))
