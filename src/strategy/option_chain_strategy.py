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

    def _get_otm_strike(
        self, chain: pd.DataFrame, spot: float, option_type: str, otm_distance: int = 1
    ) -> dict:
        """Return the n-th OTM strike record for CE (above ATM) or PE (below ATM).

        Returns a dict with keys ``security_id``, ``ltp``, ``iv``, and
        ``strike_price``.  Returns an empty dict when the requested strike is
        not available.

        Parameters
        ----------
        chain:
            DataFrame produced by :class:`~src.data.option_chain.OptionChainFetcher`.
        spot:
            Current underlying spot price used to find the ATM strike.
        option_type:
            ``"CE"`` to search above ATM, ``"PE"`` to search below ATM.
        otm_distance:
            How many strikes away from ATM to select (default 1 = first OTM).
        """
        near_strikes = self.chain_fetcher.get_strikes_near_atm(chain, spot, n_strikes=otm_distance)
        if near_strikes.empty:
            return {}
        atm_strike = chain["strike_price"].iloc[
            int((chain["strike_price"] - spot).abs().argmin())
        ]
        if option_type == "CE":
            above = near_strikes[near_strikes["strike_price"] > atm_strike].sort_values("strike_price")
            if len(above) < otm_distance:
                return {}
            row = above.iloc[otm_distance - 1]
            return {
                "security_id": str(row.get("call_security_id", "")),
                "ltp": float(row.get("call_ltp", 0.0)),
                "iv": float(row.get("call_iv", 0.0)),
                "strike_price": float(row["strike_price"]),
            }
        else:
            below = near_strikes[near_strikes["strike_price"] < atm_strike].sort_values(
                "strike_price", ascending=False
            )
            if len(below) < otm_distance:
                return {}
            row = below.iloc[otm_distance - 1]
            return {
                "security_id": str(row.get("put_security_id", "")),
                "ltp": float(row.get("put_ltp", 0.0)),
                "iv": float(row.get("put_iv", 0.0)),
                "strike_price": float(row["strike_price"]),
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


# ---------------------------------------------------------------------------
# Long volatility strategies
# ---------------------------------------------------------------------------


class LongStraddleStrategy(OptionChainStrategy):
    """Long straddle: buy ATM call + ATM put to profit from large moves.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for ATM strike selection.
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
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="LongStraddle",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
        )
        self.spot_price = spot_price
        self._position_open: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a long straddle entry signal.

        Places the ATM put leg directly via the broker and returns the ATM
        call leg signal.  Returns ``None`` when a position is already open or
        no chain data is available.
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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        strike = atm["strike_price"]
        logger.info("%s: entry signal at ATM strike=%.2f", self.name, strike)

        if self._broker is not None:
            pe_signal = self._make_option_signal(
                action="BUY",
                security_id=atm["put"]["security_id"],
                price=atm["put"]["ltp"],
                option_type="PE",
                strike=strike,
            )
            self._execute_signal(pe_signal, pd.Series({"close": atm["put"]["ltp"]}))

        self._position_open = True

        return self._make_option_signal(
            action="BUY",
            security_id=atm["call"]["security_id"],
            price=atm["call"]["ltp"],
            option_type="CE",
            strike=strike,
        )


class LongStrangleStrategy(OptionChainStrategy):
    """Long strangle: buy OTM call + OTM put to profit from large moves at lower cost.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    otm_distance:
        Number of strikes away from ATM for OTM selection (default 1).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
        otm_distance: int = 1,
    ) -> None:
        super().__init__(
            name="LongStrangle",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"otm_distance": otm_distance},
        )
        self.spot_price = spot_price
        self._position_open: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a long strangle entry signal.

        Places the OTM put leg directly via the broker and returns the OTM
        call leg signal.  Returns ``None`` when a position is already open or
        the required OTM strikes are not available.
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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        otm_distance = self.params["otm_distance"]
        otm_ce = self._get_otm_strike(chain, spot, "CE", otm_distance)
        otm_pe = self._get_otm_strike(chain, spot, "PE", otm_distance)
        if not otm_ce or not otm_pe:
            return None

        logger.info(
            "%s: entry signal OTM CE strike=%.2f OTM PE strike=%.2f",
            self.name,
            otm_ce["strike_price"],
            otm_pe["strike_price"],
        )

        if self._broker is not None:
            pe_signal = self._make_option_signal(
                action="BUY",
                security_id=otm_pe["security_id"],
                price=otm_pe["ltp"],
                option_type="PE",
                strike=otm_pe["strike_price"],
            )
            self._execute_signal(pe_signal, pd.Series({"close": otm_pe["ltp"]}))

        self._position_open = True

        return self._make_option_signal(
            action="BUY",
            security_id=otm_ce["security_id"],
            price=otm_ce["ltp"],
            option_type="CE",
            strike=otm_ce["strike_price"],
        )


# ---------------------------------------------------------------------------
# Debit spread strategies
# ---------------------------------------------------------------------------


class BullCallSpreadStrategy(OptionChainStrategy):
    """Bull call spread: buy ATM call + sell OTM call (debit, bullish).

    Profits when the underlying rises.  Maximum profit is capped at the
    difference between the two strikes minus the net debit paid.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    otm_distance:
        Number of strikes above ATM for the short call leg (default 1).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
        otm_distance: int = 1,
    ) -> None:
        super().__init__(
            name="BullCallSpread",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"otm_distance": otm_distance},
        )
        self.spot_price = spot_price
        self._position: int = 0

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a bull call spread entry signal.

        Sells the OTM call leg directly and returns the ATM call buy signal.
        Returns ``None`` when a position is already open or strikes are
        unavailable.
        """
        if self._position != 0:
            return None

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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        otm_ce = self._get_otm_strike(chain, spot, "CE", self.params["otm_distance"])
        if not otm_ce:
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry BUY CE strike=%.2f SELL CE strike=%.2f",
            self.name, strike, otm_ce["strike_price"],
        )

        if self._broker is not None:
            sell_signal = self._make_option_signal(
                action="SELL",
                security_id=otm_ce["security_id"],
                price=otm_ce["ltp"],
                option_type="CE",
                strike=otm_ce["strike_price"],
            )
            self._execute_signal(sell_signal, pd.Series({"close": otm_ce["ltp"]}))

        self._position = 1

        return self._make_option_signal(
            action="BUY",
            security_id=atm["call"]["security_id"],
            price=atm["call"]["ltp"],
            option_type="CE",
            strike=strike,
        )


class BearPutSpreadStrategy(OptionChainStrategy):
    """Bear put spread: buy ATM put + sell OTM put (debit, bearish).

    Profits when the underlying falls.  Maximum profit is capped at the
    difference between the two strikes minus the net debit paid.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    otm_distance:
        Number of strikes below ATM for the short put leg (default 1).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
        otm_distance: int = 1,
    ) -> None:
        super().__init__(
            name="BearPutSpread",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"otm_distance": otm_distance},
        )
        self.spot_price = spot_price
        self._position: int = 0

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a bear put spread entry signal.

        Sells the OTM put leg directly and returns the ATM put buy signal.
        Returns ``None`` when a position is already open or strikes are
        unavailable.
        """
        if self._position != 0:
            return None

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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        otm_pe = self._get_otm_strike(chain, spot, "PE", self.params["otm_distance"])
        if not otm_pe:
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry BUY PE strike=%.2f SELL PE strike=%.2f",
            self.name, strike, otm_pe["strike_price"],
        )

        if self._broker is not None:
            sell_signal = self._make_option_signal(
                action="SELL",
                security_id=otm_pe["security_id"],
                price=otm_pe["ltp"],
                option_type="PE",
                strike=otm_pe["strike_price"],
            )
            self._execute_signal(sell_signal, pd.Series({"close": otm_pe["ltp"]}))

        self._position = -1

        return self._make_option_signal(
            action="BUY",
            security_id=atm["put"]["security_id"],
            price=atm["put"]["ltp"],
            option_type="PE",
            strike=strike,
        )


# ---------------------------------------------------------------------------
# Credit spread strategies
# ---------------------------------------------------------------------------


class BullPutSpreadStrategy(OptionChainStrategy):
    """Bull put spread: sell ATM put + buy OTM put (credit, bullish).

    Collects net premium when the underlying stays above the short put strike.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    otm_distance:
        Number of strikes below ATM for the long put hedge (default 1).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
        otm_distance: int = 1,
    ) -> None:
        super().__init__(
            name="BullPutSpread",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"otm_distance": otm_distance},
        )
        self.spot_price = spot_price
        self._position: int = 0

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a bull put spread entry signal.

        Buys the OTM put hedge leg directly and returns the ATM put sell signal.
        Returns ``None`` when a position is already open or strikes are
        unavailable.
        """
        if self._position != 0:
            return None

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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        otm_pe = self._get_otm_strike(chain, spot, "PE", self.params["otm_distance"])
        if not otm_pe:
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry SELL PE strike=%.2f BUY PE strike=%.2f",
            self.name, strike, otm_pe["strike_price"],
        )

        if self._broker is not None:
            buy_signal = self._make_option_signal(
                action="BUY",
                security_id=otm_pe["security_id"],
                price=otm_pe["ltp"],
                option_type="PE",
                strike=otm_pe["strike_price"],
            )
            self._execute_signal(buy_signal, pd.Series({"close": otm_pe["ltp"]}))

        self._position = 1

        return self._make_option_signal(
            action="SELL",
            security_id=atm["put"]["security_id"],
            price=atm["put"]["ltp"],
            option_type="PE",
            strike=strike,
        )


class BearCallSpreadStrategy(OptionChainStrategy):
    """Bear call spread: sell ATM call + buy OTM call (credit, bearish).

    Collects net premium when the underlying stays below the short call strike.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    quantity:
        Number of lots per leg (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    otm_distance:
        Number of strikes above ATM for the long call hedge (default 1).
    """

    def __init__(
        self,
        under_security_id: int = OptionChainFetcher.NIFTY50_ID,
        under_exchange_segment: str = "IDX_I",
        spot_price: float = 0.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
        otm_distance: int = 1,
    ) -> None:
        super().__init__(
            name="BearCallSpread",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"otm_distance": otm_distance},
        )
        self.spot_price = spot_price
        self._position: int = 0

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate a bear call spread entry signal.

        Buys the OTM call hedge leg directly and returns the ATM call sell signal.
        Returns ``None`` when a position is already open or strikes are
        unavailable.
        """
        if self._position != 0:
            return None

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

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        otm_ce = self._get_otm_strike(chain, spot, "CE", self.params["otm_distance"])
        if not otm_ce:
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry SELL CE strike=%.2f BUY CE strike=%.2f",
            self.name, strike, otm_ce["strike_price"],
        )

        if self._broker is not None:
            buy_signal = self._make_option_signal(
                action="BUY",
                security_id=otm_ce["security_id"],
                price=otm_ce["ltp"],
                option_type="CE",
                strike=otm_ce["strike_price"],
            )
            self._execute_signal(buy_signal, pd.Series({"close": otm_ce["ltp"]}))

        self._position = -1

        return self._make_option_signal(
            action="SELL",
            security_id=atm["call"]["security_id"],
            price=atm["call"]["ltp"],
            option_type="CE",
            strike=strike,
        )


# ---------------------------------------------------------------------------
# Multi-leg neutral strategies
# ---------------------------------------------------------------------------


class IronCondorStrategy(OptionChainStrategy):
    """Iron condor: sell OTM CE + buy further OTM CE + sell OTM PE + buy further OTM PE.

    Collects premium in a range-bound market.  IV must exceed *min_iv_threshold*
    before the strategy enters.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    short_otm_distance:
        Strikes away from ATM for the short legs (default 1).
    long_otm_distance:
        Strikes away from ATM for the long (wing) legs (default 2).
    min_iv_threshold:
        Minimum ATM IV (%) required to enter (default 15.0).
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
        short_otm_distance: int = 1,
        long_otm_distance: int = 2,
        min_iv_threshold: float = 15.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="IronCondor",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "short_otm_distance": short_otm_distance,
                "long_otm_distance": long_otm_distance,
                "min_iv_threshold": min_iv_threshold,
            },
        )
        self.spot_price = spot_price
        self._position_open: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate an iron condor entry signal.

        Places three legs directly (buy long CE wing, sell short PE, buy long
        PE wing) and returns the short CE sell signal.  Returns ``None`` when a
        position is already open, IV is below threshold, or strikes are
        unavailable.
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
            return None

        chain = self.chain_fetcher.get_option_chain(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )
        if chain.empty:
            return None

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        min_iv = self.params["min_iv_threshold"]
        if atm["call"]["iv"] < min_iv or atm["put"]["iv"] < min_iv:
            logger.info(
                "%s: IV below threshold (call=%.1f%%, put=%.1f%%, min=%.1f%%); no signal.",
                self.name, atm["call"]["iv"], atm["put"]["iv"], min_iv,
            )
            return None

        short_d = self.params["short_otm_distance"]
        long_d = self.params["long_otm_distance"]

        short_ce = self._get_otm_strike(chain, spot, "CE", short_d)
        long_ce = self._get_otm_strike(chain, spot, "CE", long_d)
        short_pe = self._get_otm_strike(chain, spot, "PE", short_d)
        long_pe = self._get_otm_strike(chain, spot, "PE", long_d)

        if not all([short_ce, long_ce, short_pe, long_pe]):
            return None

        logger.info(
            "%s: entry short_CE=%.2f long_CE=%.2f short_PE=%.2f long_PE=%.2f",
            self.name,
            short_ce["strike_price"], long_ce["strike_price"],
            short_pe["strike_price"], long_pe["strike_price"],
        )

        if self._broker is not None:
            for sig_args in [
                ("BUY", long_ce["security_id"], long_ce["ltp"], "CE", long_ce["strike_price"]),
                ("SELL", short_pe["security_id"], short_pe["ltp"], "PE", short_pe["strike_price"]),
                ("BUY", long_pe["security_id"], long_pe["ltp"], "PE", long_pe["strike_price"]),
            ]:
                s = self._make_option_signal(*sig_args)
                self._execute_signal(s, pd.Series({"close": sig_args[2]}))

        self._position_open = True

        return self._make_option_signal(
            action="SELL",
            security_id=short_ce["security_id"],
            price=short_ce["ltp"],
            option_type="CE",
            strike=short_ce["strike_price"],
        )


class IronButterflyStrategy(OptionChainStrategy):
    """Iron butterfly: sell ATM CE + sell ATM PE + buy OTM CE wing + buy OTM PE wing.

    A limited-risk variant of the short straddle.  IV must exceed
    *min_iv_threshold* before the strategy enters.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the underlying (default 13 = Nifty 50).
    under_exchange_segment:
        Exchange segment of the underlying (default ``"IDX_I"``).
    spot_price:
        Known or estimated spot price for strike selection.
    wing_distance:
        Number of strikes away from ATM for the long wing legs (default 1).
    min_iv_threshold:
        Minimum ATM IV (%) required to enter (default 15.0).
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
        wing_distance: int = 1,
        min_iv_threshold: float = 15.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="IronButterfly",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "wing_distance": wing_distance,
                "min_iv_threshold": min_iv_threshold,
            },
        )
        self.spot_price = spot_price
        self._position_open: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Generate an iron butterfly entry signal.

        Places three legs directly (sell ATM PE, buy OTM CE wing, buy OTM PE
        wing) and returns the ATM CE sell signal.  Returns ``None`` when a
        position is already open, IV is below threshold, or strikes are
        unavailable.
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
            return None

        chain = self.chain_fetcher.get_option_chain(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )
        if chain.empty:
            return None

        spot = self.spot_price or self.chain_fetcher.get_spot_price(
            under_security_id=self.under_security_id,
            under_exchange_segment=self.under_exchange_segment,
            expiry=expiry,
        )

        atm = self.chain_fetcher.get_atm_options(chain, spot)
        if not atm:
            return None

        min_iv = self.params["min_iv_threshold"]
        if atm["call"]["iv"] < min_iv or atm["put"]["iv"] < min_iv:
            logger.info(
                "%s: IV below threshold (call=%.1f%%, put=%.1f%%, min=%.1f%%); no signal.",
                self.name, atm["call"]["iv"], atm["put"]["iv"], min_iv,
            )
            return None

        wing_d = self.params["wing_distance"]
        wing_ce = self._get_otm_strike(chain, spot, "CE", wing_d)
        wing_pe = self._get_otm_strike(chain, spot, "PE", wing_d)

        if not wing_ce or not wing_pe:
            return None

        strike = atm["strike_price"]
        logger.info(
            "%s: entry ATM strike=%.2f wing_CE=%.2f wing_PE=%.2f",
            self.name, strike, wing_ce["strike_price"], wing_pe["strike_price"],
        )

        if self._broker is not None:
            for sig_args in [
                ("SELL", atm["put"]["security_id"], atm["put"]["ltp"], "PE", strike),
                ("BUY", wing_ce["security_id"], wing_ce["ltp"], "CE", wing_ce["strike_price"]),
                ("BUY", wing_pe["security_id"], wing_pe["ltp"], "PE", wing_pe["strike_price"]),
            ]:
                s = self._make_option_signal(*sig_args)
                self._execute_signal(s, pd.Series({"close": sig_args[2]}))

        self._position_open = True

        return self._make_option_signal(
            action="SELL",
            security_id=atm["call"]["security_id"],
            price=atm["call"]["ltp"],
            option_type="CE",
            strike=strike,
        )

