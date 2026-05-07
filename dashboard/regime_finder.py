"""Market regime detection using EMA, ADX and ATR indicators."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------
TRENDING_UP   = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
SIDEWAYS      = "SIDEWAYS"
VOLATILE      = "VOLATILE"
UNKNOWN       = "UNKNOWN"


class RegimeFinder:
    """Classifies the current market into one of four regimes.

    Feed price bars with :meth:`update` and read the current regime via
    :meth:`get_regime` or :meth:`get_regime_details`.

    Regime logic
    ------------
    1. If ``ATR / price > 2 %`` → **VOLATILE**
    2. If ``ADX > 25``:
       - ``EMA20 > EMA50`` → **TRENDING_UP**
       - ``EMA20 < EMA50`` → **TRENDING_DOWN**
    3. Otherwise → **SIDEWAYS**

    Parameters
    ----------
    lookback:
        Maximum price history to retain (default 200).
    """

    def __init__(self, lookback: int = 200) -> None:
        self._lookback = lookback
        self._closes: list[float] = []
        self._highs:  list[float] = []
        self._lows:   list[float] = []

    # ------------------------------------------------------------------ #
    #  Feed                                                                #
    # ------------------------------------------------------------------ #

    def update(
        self,
        price: float,
        high: float = 0.0,
        low:  float = 0.0,
    ) -> None:
        """Feed one price bar.

        Parameters
        ----------
        price:
            Closing price (required).
        high, low:
            Bar high and low (used for ATR).  When both are 0 the close
            price is used as a proxy so ATR collapses to zero.
        """
        h = high if high else price
        lo = low  if low  else price
        self._closes.append(price)
        self._highs.append(h)
        self._lows.append(lo)
        # Keep bounded
        if len(self._closes) > self._lookback:
            self._closes = self._closes[-self._lookback:]
            self._highs  = self._highs[-self._lookback:]
            self._lows   = self._lows[-self._lookback:]

    def reset(self) -> None:
        """Clear all accumulated price data."""
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()

    def seed_from_dataframe(self, df: "pd.DataFrame") -> None:  # noqa: F821
        """Reset and feed an OHLCV DataFrame into the finder.

        Parameters
        ----------
        df:
            DataFrame with at least a ``close`` column.  ``high`` and
            ``low`` are used when present; otherwise the close is used as
            a proxy so ATR collapses to zero.
        """
        self.reset()
        for _, row in df.iterrows():
            close = float(row.get("close", row.get("Close", 0.0)))
            high  = float(row.get("high",  row.get("High",  close)))
            low   = float(row.get("low",   row.get("Low",   close)))
            if close > 0:
                self.update(close, high, low)

    # ------------------------------------------------------------------ #
    #  Regime queries                                                      #
    # ------------------------------------------------------------------ #

    def get_regime(self) -> str:
        """Return the current regime label."""
        return self.get_regime_details()["regime"]

    def get_regime_details(self) -> dict:
        """Return a dict with all indicator values and the regime label."""
        n = len(self._closes)
        if n < 30:
            return {
                "regime": UNKNOWN,
                "ema20": 0.0, "ema50": 0.0,
                "adx": 0.0, "atr": 0.0, "atr_ratio": 0.0,
                "price": self._closes[-1] if self._closes else 0.0,
                "bars": n, "confidence": 0,
            }

        price   = self._closes[-1]
        ema20   = self._ema(20)
        ema50   = self._ema(50)
        adx     = self._adx(14)
        atr     = self._atr(14)
        atr_ratio = (atr / price * 100) if price else 0.0

        # Determine regime
        if atr_ratio > 2.0:
            regime = VOLATILE
            confidence = min(100, int(atr_ratio * 30))
        elif adx > 25:
            regime = TRENDING_UP if ema20 >= ema50 else TRENDING_DOWN
            confidence = min(100, int((adx - 25) * 4))
        else:
            regime = SIDEWAYS
            confidence = min(100, int((25 - adx) * 4))

        return {
            "regime":    regime,
            "ema20":     round(ema20, 2),
            "ema50":     round(ema50, 2),
            "adx":       round(adx, 2),
            "atr":       round(atr, 2),
            "atr_ratio": round(atr_ratio, 2),
            "price":     round(price, 2),
            "bars":      n,
            "confidence": confidence,
        }

    def is_favorable_for_options_selling(self) -> bool:
        """Return True when regime is SIDEWAYS (best for premium selling)."""
        return self.get_regime() == SIDEWAYS

    def is_favorable_for_options_buying(self) -> bool:
        """Return True when the market is trending (best for directional buys)."""
        return self.get_regime() in (TRENDING_UP, TRENDING_DOWN)

    # ------------------------------------------------------------------ #
    #  Indicator helpers                                                   #
    # ------------------------------------------------------------------ #

    def _ema(self, period: int) -> float:
        prices = self._closes
        if len(prices) < period:
            return sum(prices) / len(prices)
        k   = 2.0 / (period + 1)
        val = prices[0]
        for p in prices[1:]:
            val = p * k + val * (1 - k)
        return val

    def _atr(self, period: int = 14) -> float:
        closes = self._closes
        highs  = self._highs
        lows   = self._lows
        n = len(closes)
        if n < 2:
            return 0.0
        trs: list[float] = []
        for i in range(1, n):
            tr = max(
                highs[i]  - lows[i],
                abs(highs[i]  - closes[i - 1]),
                abs(lows[i]   - closes[i - 1]),
            )
            trs.append(tr)
        period = min(period, len(trs))
        return sum(trs[-period:]) / period

    def _adx(self, period: int = 14) -> float:
        """ADX using Wilder's smoothing method."""
        closes = self._closes
        highs  = self._highs
        lows   = self._lows
        n = len(closes)
        if n < period * 2 + 1:
            return 0.0

        tr_list: list[float] = []
        pdm_list: list[float] = []
        mdm_list: list[float] = []
        for i in range(1, n):
            up_move   = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i]  - closes[i - 1]),
            ))
            pdm_list.append(up_move   if up_move > down_move and up_move > 0   else 0.0)
            mdm_list.append(down_move if down_move > up_move and down_move > 0 else 0.0)

        def _wilder_sum(values: list[float], p: int) -> list[float]:
            """Wilder smoothing keeping scale of raw values (ATR/DM variant)."""
            if len(values) < p:
                return []
            result = [sum(values[:p])]
            for v in values[p:]:
                result.append(result[-1] - result[-1] / p + v)
            return result

        def _wilder_avg(values: list[float], p: int) -> list[float]:
            """Wilder smoothing using initial average (DX/ADX variant)."""
            if len(values) < p:
                return []
            result = [sum(values[:p]) / p]
            for v in values[p:]:
                result.append(result[-1] - result[-1] / p + v / p)
            return result

        str14  = _wilder_sum(tr_list,  period)
        spdm14 = _wilder_sum(pdm_list, period)
        smdm14 = _wilder_sum(mdm_list, period)

        dx_list: list[float] = []
        for s, pdm, mdm in zip(str14, spdm14, smdm14):
            if s == 0:
                dx_list.append(0.0)
                continue
            pdi = pdm / s * 100
            mdi = mdm / s * 100
            dx_list.append(abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) else 0.0)

        if not dx_list:
            return 0.0
        adx_list = _wilder_avg(dx_list, period)
        return min(100.0, adx_list[-1]) if adx_list else 0.0


# ---------------------------------------------------------------------------
# Module-level singletons  (NIFTY50 + Crude Oil)
# ---------------------------------------------------------------------------

_finder       = RegimeFinder()   # NIFTY50
_crude_finder = RegimeFinder()   # MCX Crude Oil


# ---- NIFTY50 helpers -------------------------------------------------------

def get_current_regime() -> str:
    """Return the current market regime from the module singleton."""
    return _finder.get_regime()


def update_regime(price: float, high: float = 0.0, low: float = 0.0) -> None:
    """Feed a new price bar to the module-level regime finder."""
    _finder.update(price, high, low)


def get_regime_details() -> dict:
    """Return full regime details from the module singleton."""
    return _finder.get_regime_details()


# ---- Crude Oil helpers -----------------------------------------------------

def get_crude_regime_details() -> dict:
    """Return full regime details for the Crude Oil singleton."""
    d = _crude_finder.get_regime_details()
    d["instrument"] = "CRUDE_OIL"
    return d


def update_crude_regime(price: float, high: float = 0.0, low: float = 0.0) -> None:
    """Feed a new Crude Oil price bar to the Crude Oil regime finder."""
    _crude_finder.update(price, high, low)


def auto_refresh_crude_regime(
    broker: object,
    security_id: str = "488290",
    exchange_segment: str = "MCX_COMM",
    n_bars: int = 100,
) -> dict:
    """Fetch recent Crude Oil Futures 1-minute bars and re-seed the Crude finder.

    Parameters
    ----------
    broker:
        A connected :class:`~src.broker.dhan_broker.DhanBroker` instance.
    security_id:
        Dhan security ID for the MCX Crude Oil near-month futures contract.
        This changes each month; update via strategy params.
    exchange_segment:
        Exchange segment (default ``"MCX_COMM"``).
    n_bars:
        How many most-recent 1-minute bars to use (default 15).

    Returns
    -------
    dict
        Current Crude Oil regime details after refresh.
    """
    from src.data.data_fetcher import DataFetcher  # noqa: PLC0415

    try:
        fetcher = DataFetcher(broker=broker)
        df = fetcher.get_historical_data(
            symbol="CRUDE_OIL",
            security_id=security_id,
            exchange_segment=exchange_segment,
            instrument_type="FUTCOM",
            interval=1,  # 1-minute candles
        )
        if df.empty or "close" not in df.columns:
            logger.warning(
                "auto_refresh_crude_regime: no 1-min data for sec_id=%s, regime unchanged.",
                security_id,
            )
            return get_crude_regime_details()

        df = df.tail(n_bars)
        _crude_finder.seed_from_dataframe(df)
        details = get_crude_regime_details()
        logger.info(
            "Crude Oil regime refreshed from %d 1-min bars → %s (ADX=%.1f, ATR_ratio=%.2f%%)",
            len(df),
            details["regime"],
            details["adx"],
            details["atr_ratio"],
        )
        return details

    except Exception as exc:  # noqa: BLE001
        logger.error("auto_refresh_crude_regime failed: %s", exc)
        return get_crude_regime_details()



def auto_refresh_regime(broker: object, n_bars: int = 100) -> dict:
    """Fetch recent NIFTY50 1-minute bars and re-seed the regime finder.

    Uses ``security_id="13"`` (NIFTY50 index) with ``exchange_segment="IDX_I"``
    and 1-minute candles (``interval=1``).  Silently falls back to the last
    known regime when the API is unavailable.

    Parameters
    ----------
    broker:
        A connected :class:`~src.broker.dhan_broker.DhanBroker` instance.
    n_bars:
        How many most-recent 1-minute bars to use (default 15).

    Returns
    -------
    dict
        Current regime details after refresh (same shape as
        :func:`get_regime_details`).
    """
    import pandas as pd  # noqa: PLC0415
    from src.data.data_fetcher import DataFetcher  # noqa: PLC0415

    try:
        fetcher = DataFetcher(broker=broker)
        df = fetcher.get_historical_data(
            symbol="NIFTY50",
            security_id="13",
            exchange_segment="IDX_I",
            instrument_type="INDEX",
            interval=1,  # 1-minute candles
        )
        if df.empty or "close" not in df.columns:
            logger.warning("auto_refresh_regime: no 1-min data returned, regime unchanged.")
            return get_regime_details()

        df = df.tail(n_bars)
        _finder.seed_from_dataframe(df)
        details = _finder.get_regime_details()
        logger.info(
            "Regime auto-refreshed from %d 1-min bars → %s (ADX=%.1f, ATR_ratio=%.2f%%)",
            len(df),
            details["regime"],
            details["adx"],
            details["atr_ratio"],
        )
        return details

    except Exception as exc:  # noqa: BLE001
        logger.error("auto_refresh_regime failed: %s", exc)
        return get_regime_details()
