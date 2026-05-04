"""EMA Crossover (9/21) option strategy using NIFTY50 Futures for signal detection."""

import pandas as pd

from src.strategy.option_chain_strategy import OptionChainStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ema(prices: list[float], period: int) -> float:
    """Compute EMA of the last *period* values in *prices*."""
    if len(prices) < period:
        return sum(prices) / len(prices)
    k = 2.0 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


class EMACrossoverNiftyStrategy(OptionChainStrategy):
    """EMA Crossover (9/21) on NIFTY50 Futures → ATM option trade.

    Generates a **BUY** signal (trade ATM Call) when the fast EMA crosses
    above the slow EMA and a **SELL** signal (trade ATM Put) when the fast
    EMA crosses below the slow EMA.  All price data is sourced from the
    NIFTY50 Futures contract; orders are placed on the nearest-expiry ATM
    option at the current LTP.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the NIFTY50 index underlying (default 13).
    under_exchange_segment:
        Exchange segment for the underlying (default ``"IDX_I"``).
    futures_security_id:
        Security ID of the NIFTY50 Futures contract used for EMA
        computation (default ``"13"``).
    fast_period:
        EMA look-back for the fast line (default 9).
    slow_period:
        EMA look-back for the slow line (default 21).
    quantity:
        Number of lots per order (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    """

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        futures_security_id: str = "13",
        fast_period: int = 9,
        slow_period: int = 21,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="EMA Crossover NIFTY (9/21)",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            exchange_segment="NSE_FNO",
            quantity=quantity,
            product_type=product_type,
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "quantity": quantity,
                "futures_security_id": futures_security_id,
            },
        )
        self.futures_security_id = futures_security_id
        self._prices: list[float] = []
        self._position: int = 0  # +1 = long call, -1 = long put, 0 = flat

    # ------------------------------------------------------------------ #
    #  Signal generation                                                   #
    # ------------------------------------------------------------------ #

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Compute EMA crossover signal and return an ATM option order dict.

        Parameters
        ----------
        data:
            DataFrame with at least a ``close`` column containing NIFTY50
            Futures price data.

        Returns
        -------
        dict or None
            Order signal targeting the ATM Call (BUY signal) or ATM Put
            (SELL signal), or ``None`` when there is no crossover or not
            enough data.
        """
        close = float(data["close"].iloc[-1])
        self._prices.append(close)

        fast = self.params["fast_period"]
        slow = self.params["slow_period"]

        if len(self._prices) < slow + 1:
            bars_needed = slow + 1 - len(self._prices)
            self._log(
                f"Warming up — need {bars_needed} more bars "
                f"(have {len(self._prices)}/{slow + 1})",
                "debug",
            )
            return None

        # EMA values at current and previous bar
        fast_now  = _ema(self._prices,      fast)
        slow_now  = _ema(self._prices,      slow)
        fast_prev = _ema(self._prices[:-1], fast)
        slow_prev = _ema(self._prices[:-1], slow)
        distance  = fast_now - slow_now

        # Per-tick indicator log (always emitted once warmed)
        pos_label = {1: "LONG CE", -1: "LONG PE", 0: "FLAT"}.get(self._position, "FLAT")
        self._log(
            f"price={close:.2f}  EMA{fast}={fast_now:.2f}  EMA{slow}={slow_now:.2f}"
            f"  dist={distance:+.2f}  pos={pos_label}"
        )

        signal: dict | None = None

        # Golden cross → buy ATM call
        if fast_prev <= slow_prev and fast_now > slow_now:
            if self._position <= 0:
                atm = self._get_atm_option(close, "CE")
                if atm:
                    signal = atm
                    self._position = 1
                    self._log(
                        f"🟢 BULLISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed ABOVE "
                        f"EMA{slow}={slow_now:.2f} at price={close:.2f} → BUY ATM CE "
                        f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                        f"@ ₹{atm.get('price', 0):.2f}",
                        "signal",
                    )
                    logger.info(
                        "EMA golden cross at %.2f – buying ATM CE %s @ %.2f",
                        close, atm.get("security_id"), atm.get("price"),
                    )

        # Death cross → buy ATM put
        elif fast_prev >= slow_prev and fast_now < slow_now:
            if self._position >= 0:
                atm = self._get_atm_option(close, "PE")
                if atm:
                    signal = atm
                    self._position = -1
                    self._log(
                        f"🔴 BEARISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed BELOW "
                        f"EMA{slow}={slow_now:.2f} at price={close:.2f} → BUY ATM PE "
                        f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                        f"@ ₹{atm.get('price', 0):.2f}",
                        "signal",
                    )
                    logger.info(
                        "EMA death cross at %.2f – buying ATM PE %s @ %.2f",
                        close, atm.get("security_id"), atm.get("price"),
                    )

        return signal

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_atm_option(self, spot_price: float, option_type: str) -> dict | None:
        """Fetch the ATM option and return an order signal dict, or None."""
        if self.chain_fetcher is None:
            logger.warning("No chain_fetcher attached – cannot fetch ATM option.")
            return None

        try:
            expiry = self._get_nearest_expiry()
            if not expiry:
                logger.warning("No expiry available for NIFTY50.")
                return None

            chain = self.chain_fetcher.get_option_chain(
                under_security_id=self.under_security_id,
                under_exchange_segment=self.under_exchange_segment,
                expiry=expiry,
            )
            if chain.empty:
                logger.warning("Empty option chain for expiry %s.", expiry)
                return None

            atm = self.chain_fetcher.get_atm_options(chain, spot_price)
            if not atm:
                return None

            if option_type == "CE":
                sec_id = atm["call"]["security_id"]
                ltp    = atm["call"]["ltp"]
            else:
                sec_id = atm["put"]["security_id"]
                ltp    = atm["put"]["ltp"]

            return self._make_option_signal(
                action="BUY",
                security_id=sec_id,
                price=ltp,
                option_type=option_type,
                strike=float(atm["strike_price"]),
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch ATM option: %s", exc)
            return None


# ---------------------------------------------------------------------------
# MCX Crude Oil EMA Crossover Strategy
# ---------------------------------------------------------------------------


class EMACrossoverCrudeStrategy(OptionChainStrategy):
    """EMA Crossover (9/21) on MCX Crude Oil Futures → ATM option trade.

    Identical crossover logic to :class:`EMACrossoverNiftyStrategy` but
    targets MCX Crude Oil Futures for signal generation and MCX Crude Oil
    Options for order placement.

    Parameters
    ----------
    under_security_id:
        Dhan security ID of the Crude Oil near-month futures.
        Update this to the current active contract each month.
    under_exchange_segment:
        Exchange segment for the underlying (default ``"MCX_COMM"``).
    futures_security_id:
        Security ID used to fetch LTP for EMA computation.
    fast_period / slow_period:
        EMA periods (default 9 / 21).
    quantity:
        Number of lots per order (default 1).
    product_type:
        ``"INTRADAY"`` or ``"CNC"`` (default ``"INTRADAY"``).
    """

    def __init__(
        self,
        under_security_id: int = 488290,
        under_exchange_segment: str = "MCX_COMM",
        futures_security_id: str = "488290",
        fast_period: int = 9,
        slow_period: int = 21,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="EMA Crossover Crude Oil (9/21)",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            exchange_segment="MCX_OPT",
            quantity=quantity,
            product_type=product_type,
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "quantity": quantity,
                "futures_security_id": futures_security_id,
            },
        )
        self.futures_security_id = futures_security_id
        self._prices: list[float] = []
        self._position: int = 0  # +1 = long call, -1 = long put, 0 = flat

    def generate_signals(self, data: "pd.DataFrame") -> dict | None:  # noqa: F821
        """Compute EMA crossover signal for Crude Oil and return ATM option order."""
        close = float(data["close"].iloc[-1])
        self._prices.append(close)

        fast = self.params["fast_period"]
        slow = self.params["slow_period"]

        if len(self._prices) < slow + 1:
            bars_needed = slow + 1 - len(self._prices)
            msg = (
                f"Warming up — need {bars_needed} more bars "
                f"(have {len(self._prices)}/{slow + 1})"
            )
            self._log(msg, "debug")
            logger.debug("[CRUDE EMA] %s", msg)
            return None

        fast_now  = _ema(self._prices,      fast)
        slow_now  = _ema(self._prices,      slow)
        fast_prev = _ema(self._prices[:-1], fast)
        slow_prev = _ema(self._prices[:-1], slow)
        distance  = fast_now - slow_now

        pos_label = {1: "LONG CE", -1: "LONG PE", 0: "FLAT"}.get(self._position, "FLAT")
        tick_msg = (
            f"CrudeOil={close:.2f}  EMA{fast}={fast_now:.2f}  EMA{slow}={slow_now:.2f}"
            f"  dist={distance:+.2f}  pos={pos_label}"
        )
        self._log(tick_msg)
        logger.info("[CRUDE EMA] %s", tick_msg)

        signal: dict | None = None

        if fast_prev <= slow_prev and fast_now > slow_now:
            if self._position <= 0:
                atm = self._get_atm_option(close, "CE")
                if atm:
                    signal = atm
                    self._position = 1
                    self._log(
                        f"🟢 BULLISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed ABOVE "
                        f"EMA{slow}={slow_now:.2f} at price={close:.2f} → BUY ATM CE "
                        f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                        f"@ ₹{atm.get('price', 0):.2f}",
                        "signal",
                    )
                    logger.info(
                        "Crude EMA golden cross at %.2f – buying ATM CE %s @ %.2f",
                        close, atm.get("security_id"), atm.get("price"),
                    )

        elif fast_prev >= slow_prev and fast_now < slow_now:
            if self._position >= 0:
                atm = self._get_atm_option(close, "PE")
                if atm:
                    signal = atm
                    self._position = -1
                    self._log(
                        f"🔴 BEARISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed BELOW "
                        f"EMA{slow}={slow_now:.2f} at price={close:.2f} → BUY ATM PE "
                        f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                        f"@ ₹{atm.get('price', 0):.2f}",
                        "signal",
                    )
                    logger.info(
                        "Crude EMA death cross at %.2f – buying ATM PE %s @ %.2f",
                        close, atm.get("security_id"), atm.get("price"),
                    )

        return signal

    def _get_atm_option(self, spot_price: float, option_type: str) -> dict | None:
        """Fetch nearest ATM Crude Oil option or None."""
        if self.chain_fetcher is None:
            logger.warning("No chain_fetcher – cannot fetch ATM crude option.")
            return None
        try:
            expiry = self._get_nearest_expiry()
            if not expiry:
                logger.warning("No expiry available for Crude Oil.")
                return None
            chain = self.chain_fetcher.get_option_chain(
                under_security_id=self.under_security_id,
                under_exchange_segment=self.under_exchange_segment,
                expiry=expiry,
            )
            if chain.empty:
                logger.warning("Empty Crude Oil option chain for expiry %s.", expiry)
                return None
            atm = self.chain_fetcher.get_atm_options(chain, spot_price)
            if not atm:
                return None
            if option_type == "CE":
                sec_id, ltp = atm["call"]["security_id"], atm["call"]["ltp"]
            else:
                sec_id, ltp = atm["put"]["security_id"], atm["put"]["ltp"]
            return self._make_option_signal(
                action="BUY",
                security_id=sec_id,
                price=ltp,
                option_type=option_type,
                strike=float(atm["strike_price"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch ATM crude option: %s", exc)
            return None
