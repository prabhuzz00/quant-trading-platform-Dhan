"""EMA Crossover (9/21) option strategy using NIFTY50 Futures for signal detection."""

import pandas as pd
from datetime import datetime

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
        self._prices: list[float] = []     # completed 1-min candle closes only
        self._position: int = 0            # +1 = long CE, -1 = long PE, 0 = flat

        # Candle-close tracking (live mode)
        self._current_minute: int = -1         # hour*60+min of last tick
        self._current_candle_close: float = 0.0  # last tick in current candle
        self._pending_signal: str | None = None  # "CE" or "PE" — from closed candle

    # ------------------------------------------------------------------ #
    #  Signal generation                                                   #
    # ------------------------------------------------------------------ #

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        """Candle-close EMA crossover.

        Each live tick updates the in-progress candle.  When the clock ticks
        into a new minute the previous candle is committed to ``_prices``,
        EMAs are evaluated on the closed bar, and — if a crossover is
        confirmed — a ``_pending_signal`` is set.  The signal is *executed*
        at the very first tick of the next candle (i.e. the candle open),
        so fake intra-candle crosses are completely filtered out.
        """
        close = float(data["close"].iloc[-1])
        fast  = self.params["fast_period"]
        slow  = self.params["slow_period"]

        now = datetime.now()
        current_minute = now.hour * 60 + now.minute

        # ---- detect minute boundary ----
        new_candle = (self._current_minute >= 0) and (current_minute != self._current_minute)

        if new_candle:
            # Commit the previous candle's close
            candle_close = self._current_candle_close
            self._prices.append(candle_close)

            if len(self._prices) < slow + 1:
                bars_needed = slow + 1 - len(self._prices)
                self._log(
                    f"[WARMUP] candle closed @ ₹{candle_close:.2f}  "
                    f"bars {len(self._prices)}/{slow + 1}  ({bars_needed} more needed)",
                    "debug",
                )
            else:
                fast_now  = _ema(self._prices,      fast)
                slow_now  = _ema(self._prices,      slow)
                fast_prev = _ema(self._prices[:-1], fast)
                slow_prev = _ema(self._prices[:-1], slow)
                dist = fast_now - slow_now
                pos_label = {1: "LONG CE", -1: "LONG PE", 0: "FLAT"}.get(self._position, "FLAT")

                self._log(
                    f"[CANDLE] close=₹{candle_close:.2f}  EMA{fast}={fast_now:.2f}  "
                    f"EMA{slow}={slow_now:.2f}  dist={dist:+.2f}  pos={pos_label}"
                )

                # Only enter a new trade when completely flat
                if self._position == 0:
                    if fast_prev <= slow_prev and fast_now > slow_now:
                        self._pending_signal = "CE"
                        self._log(
                            f"🟢 BULLISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed ABOVE "
                            f"EMA{slow}={slow_now:.2f} — will enter CE on next candle open …",
                            "signal",
                        )
                    elif fast_prev >= slow_prev and fast_now < slow_now:
                        self._pending_signal = "PE"
                        self._log(
                            f"🔴 BEARISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed BELOW "
                            f"EMA{slow}={slow_now:.2f} — will enter PE on next candle open …",
                            "signal",
                        )

        # ---- advance candle tracker ----
        self._current_minute = current_minute
        self._current_candle_close = close

        # ---- execute pending signal at the open of the new candle ----
        if new_candle and self._pending_signal is not None:
            opt_type = self._pending_signal
            self._pending_signal = None
            self._log(
                f"⏩ NEW CANDLE OPEN @ ₹{close:.2f} — entering ATM {opt_type} …",
                "signal",
            )
            atm = self._get_atm_option(close, opt_type)
            if atm:
                self._position = 1 if opt_type == "CE" else -1
                self._log(
                    f"{'🟢' if opt_type == 'CE' else '🔴'} ORDER READY | BUY ATM {opt_type} "
                    f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                    f"@ ₹{atm.get('price', 0):.2f}",
                    "signal",
                )
                logger.info(
                    "NIFTY EMA %s crossover → ATM %s %s @ %.2f",
                    "bullish" if opt_type == "CE" else "bearish",
                    opt_type, atm.get("security_id"), atm.get("price"),
                )
                return atm
            else:
                self._log(
                    f"⚠️ Crossover confirmed but ATM {opt_type} fetch failed — see [ATM] entries above",
                    "debug",
                )

        return None

    def get_live_indicators(self) -> dict:
        """Return current EMA values for real-time dashboard display."""
        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        if len(self._prices) < slow:
            return {}
        ema_fast = _ema(self._prices, fast)
        ema_slow = _ema(self._prices, slow)
        return {
            f"EMA{fast}": round(ema_fast, 2),
            f"EMA{slow}": round(ema_slow, 2),
            "dist": round(ema_fast - ema_slow, 2),
            "price": round(self._prices[-1], 2),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_atm_option(self, spot_price: float, option_type: str) -> dict | None:
        """Fetch the ATM option and return an order signal dict, or None."""
        if self.chain_fetcher is None:
            self._log("[ATM] No broker/chain-fetcher — start trading or check credentials", "debug")
            logger.warning("No chain_fetcher attached – cannot fetch ATM option.")
            return None

        try:
            self._log(f"[ATM] Fetching {option_type} option chain (spot={spot_price:.2f}) …", "debug")
            expiry, chain = self._get_chain_with_fallback()
            if not expiry:
                self._log("[ATM] ❌ No expiry found — verify under_security_id & exchange_segment", "debug")
                logger.warning("No expiry available for NIFTY50.")
                return None
            if chain.empty:
                self._log(f"[ATM] ❌ Empty option chain for expiry {expiry} (all expiries tried)", "debug")
                logger.warning("Empty option chain for all tried expiries (nearest: %s).", expiry)
                return None

            self._log(f"[ATM] Using expiry {expiry}  under_id={self.under_security_id}  seg={self.under_exchange_segment}", "debug")
            self._log(f"[ATM] Chain OK  {len(chain)} strikes  spot={spot_price:.2f}", "debug")
            atm = self.chain_fetcher.get_atm_options(chain, spot_price)
            if not atm:
                self._log("[ATM] ❌ Could not find ATM strike in chain", "debug")
                return None

            strike = float(atm["strike_price"])
            if option_type == "CE":
                sec_id = atm["call"]["security_id"]
                ltp    = atm["call"]["ltp"]
            else:
                sec_id = atm["put"]["security_id"]
                ltp    = atm["put"]["ltp"]

            self._log(f"[ATM] ✅ {option_type} strike={strike}  sec={sec_id}  LTP=₹{ltp:.2f}", "debug")
            return self._make_option_signal(
                action="BUY",
                security_id=sec_id,
                price=ltp,
                option_type=option_type,
                strike=strike,
            )

        except Exception as exc:  # noqa: BLE001
            self._log(f"[ATM] ❌ Exception: {exc}", "debug")
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
            exchange_segment="MCX_COMM",
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
        self._prices: list[float] = []     # completed 1-min candle closes only
        self._position: int = 0            # +1 = long CE, -1 = long PE, 0 = flat

        # Candle-close tracking (live mode)
        self._current_minute: int = -1
        self._current_candle_close: float = 0.0
        self._pending_signal: str | None = None  # "CE" or "PE"

    def generate_signals(self, data: "pd.DataFrame") -> dict | None:  # noqa: F821
        """Candle-close EMA crossover for Crude Oil.

        Ticks are aggregated into 1-minute candles.  A crossover is only
        confirmed when a candle **closes** (minute boundary), and the entry
        is placed at the **open of the following candle** — filtering out
        all intra-candle noise.
        """
        close = float(data["close"].iloc[-1])
        fast  = self.params["fast_period"]
        slow  = self.params["slow_period"]

        now = datetime.now()
        current_minute = now.hour * 60 + now.minute

        # ---- detect minute boundary ----
        new_candle = (self._current_minute >= 0) and (current_minute != self._current_minute)

        if new_candle:
            candle_close = self._current_candle_close
            self._prices.append(candle_close)

            if len(self._prices) < slow + 1:
                bars_needed = slow + 1 - len(self._prices)
                self._log(
                    f"[WARMUP] candle closed @ ₹{candle_close:.2f}  "
                    f"bars {len(self._prices)}/{slow + 1}  ({bars_needed} more needed)",
                    "debug",
                )
                logger.debug("[CRUDE EMA WARMUP] %d/%d bars", len(self._prices), slow + 1)
            else:
                fast_now  = _ema(self._prices,      fast)
                slow_now  = _ema(self._prices,      slow)
                fast_prev = _ema(self._prices[:-1], fast)
                slow_prev = _ema(self._prices[:-1], slow)
                dist = fast_now - slow_now
                pos_label = {1: "LONG CE", -1: "LONG PE", 0: "FLAT"}.get(self._position, "FLAT")

                tick_msg = (
                    f"[CANDLE] close=₹{candle_close:.2f}  EMA{fast}={fast_now:.2f}  "
                    f"EMA{slow}={slow_now:.2f}  dist={dist:+.2f}  pos={pos_label}"
                )
                self._log(tick_msg)
                logger.info("[CRUDE EMA] %s", tick_msg)

                # Only enter a new trade when completely flat
                if self._position == 0:
                    if fast_prev <= slow_prev and fast_now > slow_now:
                        self._pending_signal = "CE"
                        self._log(
                            f"🟢 BULLISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed ABOVE "
                            f"EMA{slow}={slow_now:.2f} — will enter CE on next candle open …",
                            "signal",
                        )
                    elif fast_prev >= slow_prev and fast_now < slow_now:
                        self._pending_signal = "PE"
                        self._log(
                            f"🔴 BEARISH CROSSOVER | EMA{fast}={fast_now:.2f} crossed BELOW "
                            f"EMA{slow}={slow_now:.2f} — will enter PE on next candle open …",
                            "signal",
                        )

        # ---- advance candle tracker ----
        self._current_minute = current_minute
        self._current_candle_close = close

        # ---- execute pending signal at the open of the new candle ----
        if new_candle and self._pending_signal is not None:
            opt_type = self._pending_signal
            self._pending_signal = None
            self._log(
                f"⏩ NEW CANDLE OPEN @ ₹{close:.2f} — entering ATM {opt_type} …",
                "signal",
            )
            atm = self._get_atm_option(close, opt_type)
            if atm:
                self._position = 1 if opt_type == "CE" else -1
                self._log(
                    f"{'🟢' if opt_type == 'CE' else '🔴'} ORDER READY | BUY ATM {opt_type} "
                    f"sec={atm.get('security_id')} strike={atm.get('strike')} "
                    f"@ ₹{atm.get('price', 0):.2f}",
                    "signal",
                )
                logger.info(
                    "Crude EMA %s crossover → ATM %s %s @ %.2f",
                    "bullish" if opt_type == "CE" else "bearish",
                    opt_type, atm.get("security_id"), atm.get("price"),
                )
                return atm
            else:
                self._log(
                    f"⚠️ Crossover confirmed but ATM {opt_type} fetch failed — see [ATM] entries above",
                    "debug",
                )

        return None

    def get_live_indicators(self) -> dict:
        """Return current EMA values for real-time dashboard display."""
        fast = self.params["fast_period"]
        slow = self.params["slow_period"]
        if len(self._prices) < slow:
            return {}
        ema_fast = _ema(self._prices, fast)
        ema_slow = _ema(self._prices, slow)
        return {
            f"EMA{fast}": round(ema_fast, 2),
            f"EMA{slow}": round(ema_slow, 2),
            "dist": round(ema_fast - ema_slow, 2),
            "price": round(self._prices[-1], 2),
        }

    def _get_atm_option(self, spot_price: float, option_type: str) -> dict | None:
        """Fetch nearest ATM Crude Oil option or None."""
        if self.chain_fetcher is None:
            msg = "[ATM] No broker/chain-fetcher — start trading or check credentials"
            self._log(msg, "debug")
            logger.warning("No chain_fetcher – cannot fetch ATM crude option.")
            return None
        try:
            self._log(f"[ATM] Fetching {option_type} option chain (spot={spot_price:.2f}) …", "debug")
            expiry, chain = self._get_chain_with_fallback()
            if not expiry:
                msg = "[ATM] ❌ No expiry found for Crude Oil — verify security_id & exchange_segment in parameters"
                self._log(msg, "debug")
                logger.warning("No expiry available for Crude Oil.")
                return None
            if chain.empty:
                msg = f"[ATM] ❌ Empty option chain for expiry {expiry} (all expiries tried) — contract may be expired or security_id wrong"
                self._log(msg, "debug")
                logger.warning("Empty Crude Oil option chain for all tried expiries (nearest: %s).", expiry)
                return None
            self._log(f"[ATM] Using expiry {expiry}  under_id={self.under_security_id}  seg={self.under_exchange_segment}", "debug")
            self._log(f"[ATM] Chain OK  {len(chain)} strikes  spot={spot_price:.2f}", "debug")
            atm = self.chain_fetcher.get_atm_options(chain, spot_price)
            if not atm:
                self._log("[ATM] ❌ Could not find ATM strike in chain", "debug")
                return None
            strike = float(atm["strike_price"])
            if option_type == "CE":
                sec_id = atm["call"]["security_id"]
                ltp    = atm["call"]["ltp"]
            else:
                sec_id = atm["put"]["security_id"]
                ltp    = atm["put"]["ltp"]
            self._log(
                f"[ATM] ✅ {option_type} strike={strike}  sec={sec_id}  LTP=₹{ltp:.2f}",
                "debug",
            )
            return self._make_option_signal(
                action="BUY",
                security_id=sec_id,
                price=ltp,
                option_type=option_type,
                strike=strike,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"[ATM] ❌ Exception fetching option: {exc}", "debug")
            logger.error("Failed to fetch ATM crude option: %s", exc)
            return None
