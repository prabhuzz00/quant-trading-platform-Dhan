"""10 directional option-buying strategies using NIFTY50 Futures price signals.

Each strategy:
  - Accepts 1-min close prices from NIFTY50 Futures.
  - Computes a technical indicator signal.
  - On crossover/trigger: fetches the ATM strike from the live option chain.
  - Returns a BUY ATM CE (bullish) or BUY ATM PE (bearish) signal.
  - Exposes get_live_indicators() for real-time dashboard display.
"""

import math

import pandas as pd

from src.strategy.option_chain_strategy import OptionChainStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure-Python indicator helpers (no pandas/ta dependency for speed)
# ---------------------------------------------------------------------------


def _ema(prices: list[float], period: int) -> float:
    if not prices:
        return 0.0
    if len(prices) < period:
        return sum(prices) / len(prices)
    k = 2.0 / (period + 1)
    v = prices[0]
    for p in prices[1:]:
        v = p * k + v * (1 - k)
    return v


def _sma(prices: list[float], period: int) -> float:
    if not prices:
        return 0.0
    window = prices[-period:] if len(prices) >= period else prices
    return sum(window) / len(window)


def _std(prices: list[float], period: int) -> float:
    if len(prices) < 2:
        return 0.0
    window = prices[-period:] if len(prices) >= period else prices
    mean = sum(window) / len(window)
    return math.sqrt(sum((p - mean) ** 2 for p in window) / len(window))


def _rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        d = prices[i] - prices[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1 + ag / al)


def _dema(prices: list[float], period: int) -> float:
    """Double EMA = 2*EMA(n) - EMA(EMA(n))."""
    e1 = _ema(prices, period)
    # Build EMA series to compute EMA-of-EMA
    k = 2.0 / (period + 1)
    ema_series: list[float] = []
    v = prices[0]
    for p in prices:
        v = p * k + v * (1 - k)
        ema_series.append(v)
    e2 = _ema(ema_series, period)
    return 2 * e1 - e2


def _hma(prices: list[float], period: int) -> float:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""

    def _wma(p: list[float], n: int) -> float:
        window = p[-n:] if len(p) >= n else p
        weights = list(range(1, len(window) + 1))
        denom = sum(weights)
        return sum(w * v for w, v in zip(weights, window)) / denom if denom else 0.0

    half = max(1, period // 2)
    sq = max(1, int(math.sqrt(period)))
    wma_half = _wma(prices, half)
    wma_full = _wma(prices, period)
    raw = 2 * wma_half - wma_full
    # Approximate: use single raw value at the tip
    return raw


# ---------------------------------------------------------------------------
# Common ATM-option fetcher mixin
# ---------------------------------------------------------------------------


class _NiftyOptionBase(OptionChainStrategy):
    """Shared helpers for all Nifty option-buying strategies."""

    def __init__(
        self,
        name: str,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        quantity: int = 1,
        product_type: str = "INTRADAY",
        params: dict | None = None,
    ) -> None:
        super().__init__(
            name=name,
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            exchange_segment="NSE_FNO",
            quantity=quantity,
            product_type=product_type,
            params=params,
        )
        self._prices: list[float] = []
        self._position: int = 0  # +1 long CE, -1 long PE, 0 flat

    def _warming_up(self, required: int, close: float) -> bool:
        """Return True and write a debug log when not yet warmed up."""
        n = len(self._prices)
        if n < required + 1:
            need = required + 1 - n
            self._log(
                f"[WARMUP] ₹{close:.2f}  bars {n}/{required + 1}  "
                f"({need} more needed)",
                "debug",
            )
            return True
        return False

    def _fetch_atm(self, spot: float, option_type: str) -> dict | None:
        if self.chain_fetcher is None:
            self._log(
                "[ATM] No option-chain fetcher — credentials not set or trading not started",
                "debug",
            )
            return None
        try:
            self._log(f"[ATM] Fetching {option_type} chain  spot={spot:.2f} …", "debug")
            expiry, chain = self._get_chain_with_fallback()
            if not expiry:
                self._log(
                    "[ATM] ❌ No expiry found — verify under_security_id & exchange_segment",
                    "debug",
                )
                return None
            if chain.empty:
                self._log(
                    f"[ATM] ❌ Empty option chain for expiry {expiry} (all expiries tried) — contract may be expired",
                    "debug",
                )
                return None
            atm = self.chain_fetcher.get_atm_options(chain, spot)
            if not atm:
                self._log("[ATM] ❌ Could not find ATM strike in chain", "debug")
                return None
            strike = float(atm["strike_price"])
            if option_type == "CE":
                sec_id, ltp = atm["call"]["security_id"], atm["call"]["ltp"]
            else:
                sec_id, ltp = atm["put"]["security_id"], atm["put"]["ltp"]
            self._log(f"[ATM] ✅ {option_type} strike={strike}  sec={sec_id}  LTP=₹{ltp:.2f}", "debug")
            return self._make_option_signal(
                "BUY", sec_id, ltp, option_type, strike
            )
        except Exception as exc:
            self._log(f"[ATM] ❌ Exception: {exc}", "debug")
            logger.error("[%s] ATM fetch failed: %s", self.name, exc)
            return None

    def get_live_indicators(self) -> dict:
        """Return current indicator values for dashboard display. Override in subclasses."""
        return {}


# ---------------------------------------------------------------------------
# 1. MACD Crossover (12, 26, 9)
# ---------------------------------------------------------------------------


class MACDCrossoverStrategy(_NiftyOptionBase):
    """MACD(12,26,9) line crosses signal line → buy ATM CE or PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="MACD Crossover NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "quantity": quantity,
            },
        )
        self._macd_history: list[float] = []

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        fast = int(self.params.get("fast_period", 12))
        slow = int(self.params.get("slow_period", 26))
        sig_p = int(self.params.get("signal_period", 9))

        if self._warming_up(slow, close):
            return None

        macd_now  = _ema(self._prices,      fast) - _ema(self._prices,      slow)
        macd_prev = _ema(self._prices[:-1], fast) - _ema(self._prices[:-1], slow)
        self._macd_history.append(macd_now)

        if len(self._macd_history) < sig_p + 1:
            self._log(
                f"[WARMUP] ₹{close:.2f}  MACD history {len(self._macd_history)}/{sig_p + 1}",
                "debug",
            )
            return None

        sig_now  = _ema(self._macd_history,      sig_p)
        sig_prev = _ema(self._macd_history[:-1], sig_p)
        hist = macd_now - sig_now

        self._log(
            f"price={close:.2f}  MACD={macd_now:.2f}  Signal={sig_now:.2f}  Hist={hist:+.2f}"
        )

        if macd_prev <= sig_prev and macd_now > sig_now and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 MACD Bull Cross → BUY ATM CE @ ₹{atm['price']:.2f}", "signal")
                return atm

        elif macd_prev >= sig_prev and macd_now < sig_now and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 MACD Bear Cross → BUY ATM PE @ ₹{atm['price']:.2f}", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        if len(self._prices) < 27:
            return {}
        fast = int(self.params.get("fast_period", 12))
        slow = int(self.params.get("slow_period", 26))
        sig_p = int(self.params.get("signal_period", 9))
        macd = _ema(self._prices, fast) - _ema(self._prices, slow)
        signal_val = _ema(self._macd_history, sig_p) if len(self._macd_history) >= sig_p else 0.0
        return {
            "MACD": round(macd, 2),
            "Signal": round(signal_val, 2),
            "Histogram": round(macd - signal_val, 2),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 2. RSI Extreme Reversal (14)
# ---------------------------------------------------------------------------


class RSIReversalStrategy(_NiftyOptionBase):
    """RSI < oversold → buy CE (bounce expected); RSI > overbought → buy PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="RSI Reversal NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought,
                "quantity": quantity,
            },
        )
        self._prev_rsi: float = 50.0

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period = int(self.params.get("rsi_period", 14))
        os = float(self.params.get("oversold", 30.0))
        ob = float(self.params.get("overbought", 70.0))

        if self._warming_up(period + 1, close):
            return None

        rsi_now  = _rsi(self._prices,      period)
        rsi_prev = _rsi(self._prices[:-1], period)
        self._log(
            f"price={close:.2f}  RSI({period})={rsi_now:.1f}  OS={os}  OB={ob}"
        )
        signal = None

        # Crosses back above oversold from below → bullish
        if rsi_prev < os and rsi_now >= os and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 RSI crossed above {os} → BUY ATM CE", "signal")
                signal = atm

        # Crosses back below overbought from above → bearish
        elif rsi_prev > ob and rsi_now <= ob and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 RSI crossed below {ob} → BUY ATM PE", "signal")
                signal = atm

        self._prev_rsi = rsi_now
        return signal

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("rsi_period", 14))
        if len(self._prices) < period + 2:
            return {}
        rsi = _rsi(self._prices, period)
        return {
            f"RSI({period})": round(rsi, 1),
            "oversold": self.params.get("oversold", 30.0),
            "overbought": self.params.get("overbought", 70.0),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 3. Bollinger Band Breakout (20, 2σ)
# ---------------------------------------------------------------------------


class BollingerBreakoutStrategy(_NiftyOptionBase):
    """Price breaks above upper BB → buy CE; breaks below lower BB → buy PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        period: int = 20,
        std_dev: float = 2.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Bollinger Breakout NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"period": period, "std_dev": std_dev, "quantity": quantity},
        )

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period = int(self.params.get("period", 20))
        mult = float(self.params.get("std_dev", 2.0))

        if self._warming_up(period, close):
            return None

        mid   = _sma(self._prices, period)
        sd    = _std(self._prices, period)
        upper = mid + mult * sd
        lower = mid - mult * sd
        prev  = self._prices[-2] if len(self._prices) >= 2 else close

        self._log(
            f"price={close:.2f}  BB upper={upper:.2f}  mid={mid:.2f}  lower={lower:.2f}"
        )

        if prev <= upper and close > upper and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 BB Breakout above {upper:.2f} → BUY ATM CE", "signal")
                return atm

        elif prev >= lower and close < lower and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 BB Breakdown below {lower:.2f} → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("period", 20))
        mult   = float(self.params.get("std_dev", 2.0))
        if len(self._prices) < period:
            return {}
        mid   = _sma(self._prices, period)
        sd    = _std(self._prices, period)
        return {
            "BB_Upper": round(mid + mult * sd, 2),
            "BB_Mid":   round(mid, 2),
            "BB_Lower": round(mid - mult * sd, 2),
            "price":    round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 4. Momentum Rate-of-Change (10)
# ---------------------------------------------------------------------------


class MomentumROCStrategy(_NiftyOptionBase):
    """ROC(10) > +threshold → buy CE; ROC < -threshold → buy PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        roc_period: int = 10,
        roc_threshold: float = 0.5,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Momentum ROC NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "roc_period": roc_period,
                "roc_threshold": roc_threshold,
                "quantity": quantity,
            },
        )

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period = int(self.params.get("roc_period", 10))
        thresh = float(self.params.get("roc_threshold", 0.5))

        if self._warming_up(period + 1, close):
            return None

        roc_now  = (self._prices[-1] - self._prices[-period - 1]) / self._prices[-period - 1] * 100
        roc_prev = (self._prices[-2] - self._prices[-period - 2]) / self._prices[-period - 2] * 100
        self._log(f"price={close:.2f}  ROC({period})={roc_now:.3f}%  thresh=±{thresh}%")

        if roc_prev <= thresh and roc_now > thresh and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 ROC > +{thresh}% → BUY ATM CE", "signal")
                return atm

        elif roc_prev >= -thresh and roc_now < -thresh and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 ROC < -{thresh}% → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("roc_period", 10))
        if len(self._prices) < period + 1:
            return {}
        roc = (self._prices[-1] - self._prices[-period - 1]) / self._prices[-period - 1] * 100
        return {
            f"ROC({period})": round(roc, 3),
            "threshold": self.params.get("roc_threshold", 0.5),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 5. Triple EMA Trend Alignment (5, 13, 21)
# ---------------------------------------------------------------------------


class TripleEMATrendStrategy(_NiftyOptionBase):
    """EMA5>EMA13>EMA21 → CE; EMA5<EMA13<EMA21 → PE (trend alignment)."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        fast_period: int = 5,
        mid_period: int = 13,
        slow_period: int = 21,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Triple EMA Trend NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "fast_period": fast_period,
                "mid_period": mid_period,
                "slow_period": slow_period,
                "quantity": quantity,
            },
        )

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        f = int(self.params.get("fast_period", 5))
        m = int(self.params.get("mid_period", 13))
        s = int(self.params.get("slow_period", 21))

        if self._warming_up(s, close):
            return None

        ef_now  = _ema(self._prices,      f)
        em_now  = _ema(self._prices,      m)
        es_now  = _ema(self._prices,      s)
        ef_prev = _ema(self._prices[:-1], f)
        em_prev = _ema(self._prices[:-1], m)
        es_prev = _ema(self._prices[:-1], s)

        bull_now  = ef_now  > em_now  > es_now
        bull_prev = ef_prev > em_prev > es_prev
        bear_now  = ef_now  < em_now  < es_now
        bear_prev = ef_prev < em_prev < es_prev

        dist = ef_now - es_now
        self._log(
            f"price={close:.2f}  EMA{f}={ef_now:.2f}  EMA{m}={em_now:.2f}"
            f"  EMA{s}={es_now:.2f}  dist={dist:+.2f}"
        )

        if not bull_prev and bull_now and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 Triple EMA aligned BULL → BUY ATM CE", "signal")
                return atm

        elif not bear_prev and bear_now and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 Triple EMA aligned BEAR → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        f = int(self.params.get("fast_period", 5))
        m = int(self.params.get("mid_period", 13))
        s = int(self.params.get("slow_period", 21))
        if len(self._prices) < s:
            return {}
        ef = _ema(self._prices, f)
        em = _ema(self._prices, m)
        es = _ema(self._prices, s)
        return {
            f"EMA{f}": round(ef, 2),
            f"EMA{m}": round(em, 2),
            f"EMA{s}": round(es, 2),
            "dist_fast_slow": round(ef - es, 2),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 6. Stochastic RSI Crossover
# ---------------------------------------------------------------------------


class StochRSICrossoverStrategy(_NiftyOptionBase):
    """Stochastic applied to RSI: %K crosses %D in extreme zones → CE/PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Stoch RSI NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "rsi_period": rsi_period,
                "stoch_period": stoch_period,
                "smooth_k": smooth_k,
                "oversold": oversold,
                "overbought": overbought,
                "quantity": quantity,
            },
        )
        self._rsi_history: list[float] = []
        self._k_history:   list[float] = []

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        rp   = int(self.params.get("rsi_period", 14))
        sp   = int(self.params.get("stoch_period", 14))
        sk   = int(self.params.get("smooth_k", 3))
        os   = float(self.params.get("oversold", 20.0))
        ob   = float(self.params.get("overbought", 80.0))

        if self._warming_up(rp + sp + sk + 1, close):
            return None

        rsi_val = _rsi(self._prices, rp)
        self._rsi_history.append(rsi_val)

        if len(self._rsi_history) < sp:
            self._log(
                f"[WARMUP] ₹{close:.2f}  RSI history {len(self._rsi_history)}/{sp}",
                "debug",
            )
            return None

        rsi_min = min(self._rsi_history[-sp:])
        rsi_max = max(self._rsi_history[-sp:])
        rng = rsi_max - rsi_min
        raw_k = (rsi_val - rsi_min) / rng * 100 if rng > 0 else 50.0
        self._k_history.append(raw_k)

        if len(self._k_history) < sk + 1:
            self._log(
                f"[WARMUP] ₹{close:.2f}  StochK history {len(self._k_history)}/{sk + 1}",
                "debug",
            )
            return None

        k_now  = _sma(self._k_history,      sk)
        k_prev = _sma(self._k_history[:-1], sk)
        d_now  = _sma(self._k_history[-sk:], sk)

        self._log(
            f"price={close:.2f}  RSI={rsi_val:.1f}  StochK={k_now:.1f}  StochD={d_now:.1f}"
        )

        if k_prev < os and k_now >= os and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 StochRSI crossed out of oversold → BUY ATM CE", "signal")
                return atm

        elif k_prev > ob and k_now <= ob and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 StochRSI crossed out of overbought → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        rp = int(self.params.get("rsi_period", 14))
        sk = int(self.params.get("smooth_k", 3))
        if len(self._prices) < rp + 2 or len(self._k_history) < sk:
            return {}
        rsi_val = _rsi(self._prices, rp)
        k_now   = _sma(self._k_history, sk)
        d_now   = _sma(self._k_history[-sk:], sk)
        return {
            "RSI": round(rsi_val, 1),
            "StochK": round(k_now, 1),
            "StochD": round(d_now, 1),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 7. DEMA Crossover (9, 21)
# ---------------------------------------------------------------------------


class DEMACrossoverStrategy(_NiftyOptionBase):
    """Double-smoothed EMA crossover: fast DEMA crosses slow DEMA → CE/PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        fast_period: int = 9,
        slow_period: int = 21,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="DEMA Crossover NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "quantity": quantity,
            },
        )

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        f = int(self.params.get("fast_period", 9))
        s = int(self.params.get("slow_period", 21))

        if self._warming_up(s * 2, close):
            return None

        df_now  = _dema(self._prices,      f)
        ds_now  = _dema(self._prices,      s)
        df_prev = _dema(self._prices[:-1], f)
        ds_prev = _dema(self._prices[:-1], s)
        dist    = df_now - ds_now

        self._log(
            f"price={close:.2f}  DEMA{f}={df_now:.2f}  DEMA{s}={ds_now:.2f}  dist={dist:+.2f}"
        )

        if df_prev <= ds_prev and df_now > ds_now and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 DEMA Bull Cross → BUY ATM CE", "signal")
                return atm

        elif df_prev >= ds_prev and df_now < ds_now and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 DEMA Bear Cross → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        f = int(self.params.get("fast_period", 9))
        s = int(self.params.get("slow_period", 21))
        if len(self._prices) < s * 2:
            return {}
        df = _dema(self._prices, f)
        ds = _dema(self._prices, s)
        return {
            f"DEMA{f}": round(df, 2),
            f"DEMA{s}": round(ds, 2),
            "dist": round(df - ds, 2),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 8. Hull Moving Average Direction Flip (21)
# ---------------------------------------------------------------------------


class HullMAFlipStrategy(_NiftyOptionBase):
    """HMA(21) slope flips positive → buy CE; negative → buy PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        period: int = 21,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Hull MA Flip NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"period": period, "quantity": quantity},
        )
        self._hma_history: list[float] = []

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period = int(self.params.get("period", 21))

        if self._warming_up(period + 1, close):
            return None

        hma_now  = _hma(self._prices,      period)
        hma_prev = _hma(self._prices[:-1], period)
        self._hma_history.append(hma_now)

        slope_now  = hma_now  - (self._hma_history[-2] if len(self._hma_history) >= 2 else hma_now)
        slope_prev = (self._hma_history[-2] - self._hma_history[-3]) if len(self._hma_history) >= 3 else 0.0

        self._log(
            f"price={close:.2f}  HMA({period})={hma_now:.2f}  slope={slope_now:+.2f}"
        )

        if slope_prev <= 0 and slope_now > 0 and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 HMA slope turned UP → BUY ATM CE", "signal")
                return atm

        elif slope_prev >= 0 and slope_now < 0 and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 HMA slope turned DOWN → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("period", 21))
        if len(self._prices) < period + 1:
            return {}
        hma = _hma(self._prices, period)
        slope = hma - (self._hma_history[-2] if len(self._hma_history) >= 2 else hma)
        return {
            f"HMA({period})": round(hma, 2),
            "slope": round(slope, 2),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 9. Price Channel Breakout (N-bar High/Low)
# ---------------------------------------------------------------------------


class PriceChannelBreakoutStrategy(_NiftyOptionBase):
    """Price breaks above N-bar channel high → buy CE; below channel low → buy PE."""

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        period: int = 20,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Price Channel Breakout NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={"period": period, "quantity": quantity},
        )

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period = int(self.params.get("period", 20))

        if self._warming_up(period + 1, close):
            return None

        # Channel based on previous period (exclude current bar)
        prev_window = self._prices[-period - 1:-1]
        chan_high = max(prev_window)
        chan_low  = min(prev_window)
        prev_close = self._prices[-2]

        self._log(
            f"price={close:.2f}  ChanHigh={chan_high:.2f}  ChanLow={chan_low:.2f}"
        )

        if prev_close <= chan_high and close > chan_high and self._position <= 0:
            atm = self._fetch_atm(close, "CE")
            if atm:
                self._position = 1
                self._log(f"🟢 Price breaks {period}-bar high {chan_high:.2f} → BUY ATM CE", "signal")
                return atm

        elif prev_close >= chan_low and close < chan_low and self._position >= 0:
            atm = self._fetch_atm(close, "PE")
            if atm:
                self._position = -1
                self._log(f"🔴 Price breaks {period}-bar low {chan_low:.2f} → BUY ATM PE", "signal")
                return atm

        return None

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("period", 20))
        if len(self._prices) < period + 1:
            return {}
        chan_high = max(self._prices[-period - 1:-1])
        chan_low  = min(self._prices[-period - 1:-1])
        return {
            f"Chan{period}H": round(chan_high, 2),
            f"Chan{period}L": round(chan_low, 2),
            "dist_to_high": round(self._prices[-1] - chan_high, 2),
            "dist_to_low":  round(self._prices[-1] - chan_low, 2),
            "price": round(self._prices[-1], 2),
        }


# ---------------------------------------------------------------------------
# 10. Volatility Squeeze (BB inside KC → breakout direction → CE/PE)
# ---------------------------------------------------------------------------


class VolatilitySqueezeStrategy(_NiftyOptionBase):
    """Detects BB-inside-KC squeeze then trades the breakout direction.

    Squeeze = BB bandwidth < KC bandwidth.
    On squeeze release: if price > mid → CE, if price < mid → PE.
    """

    def __init__(
        self,
        under_security_id: int = 13,
        under_exchange_segment: str = "IDX_I",
        period: int = 20,
        bb_mult: float = 2.0,
        kc_mult: float = 1.5,
        quantity: int = 1,
        product_type: str = "INTRADAY",
    ) -> None:
        super().__init__(
            name="Volatility Squeeze NIFTY",
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            quantity=quantity,
            product_type=product_type,
            params={
                "period": period,
                "bb_mult": bb_mult,
                "kc_mult": kc_mult,
                "quantity": quantity,
            },
        )
        self._in_squeeze: bool = False

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        close = float(data["close"].iloc[-1])
        self._prices.append(close)
        period  = int(self.params.get("period", 20))
        bb_mult = float(self.params.get("bb_mult", 2.0))
        kc_mult = float(self.params.get("kc_mult", 1.5))

        if self._warming_up(period + 1, close):
            return None

        mid     = _sma(self._prices, period)
        sd      = _std(self._prices, period)
        bb_up   = mid + bb_mult * sd
        bb_lo   = mid - bb_mult * sd
        bb_bw   = bb_up - bb_lo

        # Keltner Channel uses ATR proxy (std-dev of closes)
        atr_proxy = sd * 1.4142
        kc_up   = mid + kc_mult * atr_proxy
        kc_lo   = mid - kc_mult * atr_proxy
        kc_bw   = kc_up - kc_lo

        squeeze_now  = bb_bw < kc_bw
        squeeze_prev = self._in_squeeze

        self._log(
            f"price={close:.2f}  BB_BW={bb_bw:.2f}  KC_BW={kc_bw:.2f}"
            f"  squeeze={'YES' if squeeze_now else 'NO'}"
        )

        signal = None
        # Squeeze release (was in squeeze, now out)
        if squeeze_prev and not squeeze_now:
            if close > mid and self._position <= 0:
                atm = self._fetch_atm(close, "CE")
                if atm:
                    self._position = 1
                    self._log(f"🟢 Squeeze released BULLISH → BUY ATM CE", "signal")
                    signal = atm
            elif close < mid and self._position >= 0:
                atm = self._fetch_atm(close, "PE")
                if atm:
                    self._position = -1
                    self._log(f"🔴 Squeeze released BEARISH → BUY ATM PE", "signal")
                    signal = atm

        self._in_squeeze = squeeze_now
        return signal

    def get_live_indicators(self) -> dict:
        period = int(self.params.get("period", 20))
        if len(self._prices) < period:
            return {}
        mid   = _sma(self._prices, period)
        sd    = _std(self._prices, period)
        bb_bw = 4 * float(self.params.get("bb_mult", 2.0)) * sd
        kc_bw = 4 * float(self.params.get("kc_mult", 1.5)) * sd * 1.4142
        return {
            "BB_BW":    round(bb_bw, 2),
            "KC_BW":    round(kc_bw, 2),
            "Squeeze":  "YES" if bb_bw < kc_bw else "NO",
            "Midline":  round(mid, 2),
            "price":    round(self._prices[-1], 2),
        }
