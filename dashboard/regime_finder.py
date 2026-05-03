"""Market regime detection using EMA, ADX and ATR indicators."""

from __future__ import annotations

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

    # ------------------------------------------------------------------ #
    #  Regime queries                                                      #
    # ------------------------------------------------------------------ #

    def get_regime(self) -> str:
        """Return the current regime label."""
        return self.get_regime_details()["regime"]

    def get_regime_details(self) -> dict:
        """Return a dict with all indicator values and the regime label."""
        n = len(self._closes)
        if n < 10:
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
        """Simplified ADX approximation using the DM method."""
        closes = self._closes
        highs  = self._highs
        lows   = self._lows
        n = len(closes)
        if n < period + 2:
            return 0.0
        plus_dm_list:  list[float] = []
        minus_dm_list: list[float] = []
        tr_list:       list[float] = []
        for i in range(1, n):
            up_move   = highs[i]  - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm_list.append(max(up_move,   0) if up_move   > down_move else 0)
            minus_dm_list.append(max(down_move, 0) if down_move > up_move   else 0)
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i]  - closes[i - 1]),
                abs(lows[i]   - closes[i - 1]),
            ))

        # Smooth with simple average over last `period` values
        p = min(period, len(tr_list))
        atr_s   = sum(tr_list[-p:])   / p
        pdm_s   = sum(plus_dm_list[-p:])  / p
        mdm_s   = sum(minus_dm_list[-p:]) / p

        if atr_s == 0:
            return 0.0
        pdi = pdm_s / atr_s * 100
        mdi = mdm_s / atr_s * 100
        dx  = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) else 0.0
        return dx


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_finder = RegimeFinder()


def get_current_regime() -> str:
    """Return the current market regime from the module singleton."""
    return _finder.get_regime()


def update_regime(price: float, high: float = 0.0, low: float = 0.0) -> None:
    """Feed a new price bar to the module-level regime finder."""
    _finder.update(price, high, low)


def get_regime_details() -> dict:
    """Return full regime details from the module singleton."""
    return _finder.get_regime_details()
