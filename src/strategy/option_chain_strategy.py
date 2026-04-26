"""Option chain trading strategies using the Dhan API."""

from abc import abstractmethod
from typing import Any

import pandas as pd

from src.data.option_chain import OptionChainFetcher
from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionChainStrategy(BaseStrategy):
    """Abstract base class for strategies that trade options using the Dhan API.

    Sub-classes must implement :meth:`generate_signals` (inherited from
    :class:`~src.strategy.base_strategy.BaseStrategy`) and may use the
    :attr:`chain_fetcher` attribute to access live option chain data.

    Parameters
    ----------
    name:
        Human-readable strategy name.
    under_security_id:
        Dhan security ID of the underlying (e.g. 13 for Nifty 50).
    under_exchange_segment:
        Exchange segment for the underlying (``"IDX_I"`` for indices).
    exchange_segment:
        Exchange segment for placing option orders (default ``"NSE_FNO"``).
    quantity:
        Number of lots per leg (default 1).
    product_type:
        Order product type (``"INTRADAY"`` or ``"CNC"``).
    params:
        Additional strategy parameters.
    """

    def __init__(
        self,
        name: str,
        under_security_id: int,
        under_exchange_segment: str = "IDX_I",
        exchange_segment: str = "NSE_FNO",
        quantity: int = 1,
        product_type: str = "INTRADAY",
        params: dict | None = None,
    ) -> None:
        super().__init__(name=name, params=params)
        self.under_security_id = under_security_id
        self.under_exchange_segment = under_exchange_segment
        self.exchange_segment = exchange_segment
        self.quantity = quantity
        self.product_type = product_type
        self.chain_fetcher: OptionChainFetcher | None = None

    def attach_broker(self, broker: Any) -> None:
        """Attach a broker and create the associated OptionChainFetcher."""
        super().attach_broker(broker)
        self.chain_fetcher = OptionChainFetcher(broker)

    def _get_nearest_expiry(self) -> str:
        """Return the nearest available expiry date for the underlying.

        Returns an empty string when no expiry list is available.
        """
        if self.chain_fetcher is None:
            return ""
        expiries = self.chain_fetcher.get_expiry_list(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
        )
        return expiries[0] if expiries else ""

    def _make_option_signal(
        self,
        action: str,
        security_id: str,
        price: float,
        option_type: str = "CE",
        strike: float = 0.0,
    ) -> dict:
        """Build a signal dict for an option order."""
        return {
            "symbol": f"{self.under_security_id}-{option_type}-{strike}",
            "security_id": security_id,
            "exchange_segment": self.exchange_segment,
            "action": action,
            "quantity": self.quantity,
            "price": price,
            "order_type": "MARKET",
            "product_type": self.product_type,
        }


class ShortStraddleStrategy(OptionChainStrategy):
    """Short straddle: sell ATM call + ATM put to collect premium.

    Entry logic:
      - Fetch the current option chain and identify the ATM strike.
      - If both call IV and put IV exceed *min_iv_threshold*, sell both the
        ATM call and the ATM put (two separate orders).
      - Exits are managed externally (e.g. by a stop-loss overlay or a
        separate exit strategy).

    This strategy is intended for backtesting entry signals only.  In live
    trading the caller is responsible for managing exits.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price used to identify the ATM strike in
        paper-trade / backtest mode.
    min_iv_threshold:
        Minimum implied volatility (%) on both legs before initiating the
        straddle (default 15.0).
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        min_iv_threshold: float = 15.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="ShortStraddle",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"min_iv_threshold": min_iv_threshold},
        )
        self.spot_price = spot_price
        self._position_open: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a short straddle entry signal.

        Because a straddle requires two orders, this method places the put
        leg order directly via the attached broker and returns the call leg
        signal (the caller / backtester records the call side).

        Returns ``None`` when no entry conditions are met.
        """
        if self._position_open:
            return None

        if self.chain_fetcher is None:
            logger.warning(
                "%s: no broker/chain_fetcher attached; cannot fetch live option chain.",
                self.name,
            )
            return None

        expiry = self._get_nearest_expiry()
        if not expiry:
            logger.warning("%s: could not determine expiry date.", self.name)
            return None

        chain = self.chain_fetcher.get_option_chain(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )
        if chain.empty:
            return None

        # Use live spot or the provided estimate
        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        min_iv = self.params["min_iv_threshold"]
        call_iv = atm["call"]["iv"]
        put_iv = atm["put"]["iv"]

        if call_iv < min_iv or put_iv < min_iv:
            logger.info(
                "%s: IV below threshold (call=%.1f%%, put=%.1f%%, min=%.1f%%); no signal.",
                self.name,
                call_iv,
                put_iv,
                min_iv,
            )
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry signal at strike=%.2f (call_iv=%.1f%%, put_iv=%.1f%%)",
            self.name,
            strike,
            call_iv,
            put_iv,
        )

        # Place put leg immediately if broker is attached
        if self._broker is not None:
            put_signal = self._make_option_signal(
                action="SELL",
                security_id=atm["put"]["security_id"],
                price=atm["put"]["ltp"],
                option_type="PE",
                strike=strike,
            )
            # Use a minimal bar Series with the put LTP as the reference price
            put_bar = pd.Series({"close": atm["put"]["ltp"]})
            self._execute_signal(put_signal, put_bar)

        self._position_open = True

        # Return call leg as the primary signal
        return self._make_option_signal(
            action="SELL",
            security_id=atm["call"]["security_id"],
            price=atm["call"]["ltp"],
            option_type="CE",
            strike=strike,
        )


class PCRStrategy(OptionChainStrategy):
    """Directional strategy based on the Put-Call Ratio (PCR).

    Entry logic:
      - Fetch the option chain and compute the PCR (put OI / call OI).
      - PCR ≥ *bullish_pcr* → market is bearishly positioned; buy ATM calls
        (contrarian: market likely to rise).
      - PCR ≤ *bearish_pcr* → market is bullishly positioned; buy ATM puts
        (contrarian: market likely to fall).

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for ATM strike selection.
    bullish_pcr:
        PCR threshold above which a bullish signal (buy call) is generated
        (default 1.5).
    bearish_pcr:
        PCR threshold below which a bearish signal (buy put) is generated
        (default 0.5).
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        bullish_pcr: float = 1.5,
        bearish_pcr: float = 0.5,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="PCRStrategy",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"bullish_pcr": bullish_pcr, "bearish_pcr": bearish_pcr},
        )
        self.spot_price = spot_price
        self._position: int = 0  # +1 long call, -1 long put, 0 flat

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a directional option signal based on PCR.

        Returns a ``BUY`` signal dict for the ATM call (bullish) or ATM put
        (bearish), or ``None`` when no threshold is breached or a position is
        already open.
        """
        if self.chain_fetcher is None:
            logger.warning(
                "%s: no broker/chain_fetcher attached; cannot fetch live option chain.",
                self.name,
            )
            return None

        expiry = self._get_nearest_expiry()
        if not expiry:
            return None

        chain = self.chain_fetcher.get_option_chain(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )
        if chain.empty:
            return None

        pcr = self.chain_fetcher.calculate_pcr(chain)

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )
        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        strike = atm["strike_price"]
        bullish_threshold = self.params["bullish_pcr"]
        bearish_threshold = self.params["bearish_pcr"]

        if pcr >= bullish_threshold and self._position <= 0:
            logger.info(
                "%s: PCR=%.3f >= %.2f → BUY ATM call at strike=%.2f",
                self.name,
                pcr,
                bullish_threshold,
                strike,
            )
            self._position = 1
            return self._make_option_signal(
                action="BUY",
                security_id=atm["call"]["security_id"],
                price=atm["call"]["ltp"],
                option_type="CE",
                strike=strike,
            )

        if pcr <= bearish_threshold and self._position >= 0:
            logger.info(
                "%s: PCR=%.3f <= %.2f → BUY ATM put at strike=%.2f",
                self.name,
                pcr,
                bearish_threshold,
                strike,
            )
            self._position = -1
            return self._make_option_signal(
                action="BUY",
                security_id=atm["put"]["security_id"],
                price=atm["put"]["ltp"],
                option_type="PE",
                strike=strike,
            )

        return None
