"""Strategy registry with persistent enable/disable state and param management."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "strategies.json"
_TRADING_STATE_FILE = DATA_DIR / "trading_state.json"

# ---------------------------------------------------------------------------
# Strategy catalog – single source of truth for all available strategies
# ---------------------------------------------------------------------------
STRATEGY_CATALOG: dict[str, dict] = {
    "moving_average_cross": {
        "name": "Moving Average Crossover",
        "type": "Trend Following",
        "type_color": "blue",
        "description": (
            "Generates a BUY signal when the fast moving average crosses above "
            "the slow MA, and a SELL when it crosses below. Classic trend-following approach."
        ),
        "asset_type": "equity",
        "param_schema": [
            {"key": "symbol", "label": "Symbol", "type": "text", "default": "RELIANCE"},
            {"key": "security_id", "label": "Security ID (Dhan)", "type": "text", "default": "2885"},
            {
                "key": "fast_period",
                "label": "Fast Period",
                "type": "number",
                "default": 20,
                "min": 2,
                "max": 200,
                "step": 1,
            },
            {
                "key": "slow_period",
                "label": "Slow Period",
                "type": "number",
                "default": 50,
                "min": 5,
                "max": 500,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Quantity (shares)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10000,
                "step": 1,
            },
            {"key": "exchange_segment", "label": "Exchange Segment", "type": "text", "default": "NSE_EQ"},
        ],
    },
    "rsi": {
        "name": "RSI Mean Reversion",
        "type": "Mean Reversion",
        "type_color": "purple",
        "description": (
            "Buys when RSI drops below the oversold threshold and sells when it rises "
            "above the overbought threshold. Works best in ranging markets."
        ),
        "asset_type": "equity",
        "param_schema": [
            {"key": "symbol", "label": "Symbol", "type": "text", "default": "TCS"},
            {"key": "security_id", "label": "Security ID (Dhan)", "type": "text", "default": "11536"},
            {
                "key": "rsi_period",
                "label": "RSI Period",
                "type": "number",
                "default": 14,
                "min": 2,
                "max": 100,
                "step": 1,
            },
            {
                "key": "oversold",
                "label": "Oversold Level",
                "type": "number",
                "default": 30,
                "min": 5,
                "max": 49,
                "step": 1,
            },
            {
                "key": "overbought",
                "label": "Overbought Level",
                "type": "number",
                "default": 70,
                "min": 51,
                "max": 95,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Quantity (shares)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10000,
                "step": 1,
            },
            {"key": "exchange_segment", "label": "Exchange Segment", "type": "text", "default": "NSE_EQ"},
        ],
    },
    "bollinger_bands": {
        "name": "Bollinger Bands",
        "type": "Mean Reversion",
        "type_color": "purple",
        "description": (
            "Buys when price closes below the lower band (oversold) and sells when "
            "price closes above the upper band (overbought). Good for volatile stocks."
        ),
        "asset_type": "equity",
        "param_schema": [
            {"key": "symbol", "label": "Symbol", "type": "text", "default": "INFY"},
            {"key": "security_id", "label": "Security ID (Dhan)", "type": "text", "default": "10999"},
            {
                "key": "period",
                "label": "Period",
                "type": "number",
                "default": 20,
                "min": 5,
                "max": 200,
                "step": 1,
            },
            {
                "key": "num_std",
                "label": "Std Dev Multiplier",
                "type": "number",
                "default": 2.0,
                "min": 0.5,
                "max": 5.0,
                "step": 0.1,
            },
            {
                "key": "quantity",
                "label": "Quantity (shares)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10000,
                "step": 1,
            },
            {"key": "exchange_segment", "label": "Exchange Segment", "type": "text", "default": "NSE_EQ"},
        ],
    },
    "short_straddle": {
        "name": "Short Straddle",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Sells an ATM call and ATM put simultaneously to collect premium when "
            "IV exceeds the threshold. Profits from time decay and low volatility. "
            "Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {
                "key": "under_exchange_segment",
                "label": "Underlying Segment",
                "type": "text",
                "default": "IDX_I",
            },
            {
                "key": "min_iv_threshold",
                "label": "Min IV Threshold (%)",
                "type": "number",
                "default": 15.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    "pcr": {
        "name": "PCR Strategy",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Contrarian strategy based on Put-Call Ratio. Buys ATM calls when PCR "
            "is high (bearish crowd → likely bullish) and ATM puts when PCR is low "
            "(bullish crowd → likely bearish). Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {
                "key": "under_exchange_segment",
                "label": "Underlying Segment",
                "type": "text",
                "default": "IDX_I",
            },
            {
                "key": "bullish_pcr",
                "label": "Bullish PCR Threshold",
                "type": "number",
                "default": 1.5,
                "min": 0.5,
                "max": 5.0,
                "step": 0.1,
            },
            {
                "key": "bearish_pcr",
                "label": "Bearish PCR Threshold",
                "type": "number",
                "default": 0.5,
                "min": 0.1,
                "max": 2.0,
                "step": 0.1,
            },
            {
                "key": "quantity",
                "label": "Lots",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    # -------------------------------------------------------------------------
    # Long volatility strategies
    # -------------------------------------------------------------------------
    "long_straddle": {
        "name": "Long Straddle",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Buys the ATM call and ATM put simultaneously. Profits from a large "
            "move in either direction (earnings, events). Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    "long_strangle": {
        "name": "Long Strangle",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Buys an OTM call and OTM put simultaneously. Cheaper than a straddle but "
            "requires a larger move to profit. Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "otm_distance",
                "label": "OTM Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    # -------------------------------------------------------------------------
    # Debit spread strategies
    # -------------------------------------------------------------------------
    "bull_call_spread": {
        "name": "Bull Call Spread",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Buys an ATM call and sells an OTM call (debit spread). Profits when "
            "the underlying rises with capped risk and reward. Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "otm_distance",
                "label": "OTM Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    "bear_put_spread": {
        "name": "Bear Put Spread",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Buys an ATM put and sells an OTM put (debit spread). Profits when "
            "the underlying falls with capped risk and reward. Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "otm_distance",
                "label": "OTM Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    # -------------------------------------------------------------------------
    # Credit spread strategies
    # -------------------------------------------------------------------------
    "bull_put_spread": {
        "name": "Bull Put Spread",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Sells an ATM put and buys an OTM put (credit spread). Collects premium "
            "when the market is stable or bullish. Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "otm_distance",
                "label": "OTM Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    "bear_call_spread": {
        "name": "Bear Call Spread",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Sells an ATM call and buys an OTM call (credit spread). Collects premium "
            "when the market is stable or bearish. Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "otm_distance",
                "label": "OTM Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    # -------------------------------------------------------------------------
    # Multi-leg neutral strategies
    # -------------------------------------------------------------------------
    "iron_condor": {
        "name": "Iron Condor",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Sells an OTM CE + OTM PE and buys further OTM wings for protection. "
            "Profits when the market stays in a range. IV-gated entry. "
            "Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "short_otm_distance",
                "label": "Short Leg OTM Distance",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "long_otm_distance",
                "label": "Long Leg OTM Distance",
                "type": "number",
                "default": 2,
                "min": 2,
                "max": 15,
                "step": 1,
            },
            {
                "key": "min_iv_threshold",
                "label": "Min IV Threshold (%)",
                "type": "number",
                "default": 15.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    "iron_butterfly": {
        "name": "Iron Butterfly",
        "type": "Options",
        "type_color": "orange",
        "description": (
            "Sells ATM CE + ATM PE and buys OTM wing options for limited risk. "
            "A tighter, higher-premium variant of the iron condor. IV-gated entry. "
            "Requires live Dhan API credentials."
        ),
        "asset_type": "options",
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {
                "key": "wing_distance",
                "label": "Wing Distance (strikes)",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
            },
            {
                "key": "min_iv_threshold",
                "label": "Min IV Threshold (%)",
                "type": "number",
                "default": 15.0,
                "min": 1.0,
                "max": 100.0,
                "step": 0.5,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
    # -------------------------------------------------------------------------
    # EMA crossover on NIFTY50 futures
    # -------------------------------------------------------------------------
    "ema_crossover_nifty": {
        "name": "EMA Crossover NIFTY (9/21)",
        "type": "Trend Following",
        "type_color": "blue",
        "description": (
            "Uses EMA(9) vs EMA(21) on NIFTY50 Futures to detect trend crossovers, "
            "then trades the ATM Call (on golden cross) or ATM Put (on death cross) "
            "at the current LTP. Regime-controlled by default."
        ),
        "asset_type": "options",
        "regime_controlled": True,
        "param_schema": [
            {
                "key": "under_security_id",
                "label": "Underlying Security ID",
                "type": "number",
                "default": 13,
                "min": 1,
                "max": 99999,
                "step": 1,
            },
            {"key": "under_exchange_segment", "label": "Underlying Segment", "type": "text", "default": "IDX_I"},
            {"key": "futures_security_id", "label": "Futures Security ID", "type": "text", "default": "13"},
            {
                "key": "fast_period",
                "label": "Fast EMA Period",
                "type": "number",
                "default": 9,
                "min": 2,
                "max": 50,
                "step": 1,
            },
            {
                "key": "slow_period",
                "label": "Slow EMA Period",
                "type": "number",
                "default": 21,
                "min": 5,
                "max": 200,
                "step": 1,
            },
            {
                "key": "quantity",
                "label": "Lots per Leg",
                "type": "number",
                "default": 1,
                "min": 1,
                "max": 50,
                "step": 1,
            },
            {"key": "product_type", "label": "Product Type", "type": "text", "default": "INTRADAY"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_params(strategy_id: str) -> dict:
    schema = STRATEGY_CATALOG[strategy_id]["param_schema"]
    return {p["key"]: p["default"] for p in schema}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    state: dict = {}
    for sid in STRATEGY_CATALOG:
        state[sid] = {"enabled": True, "params": _default_params(sid)}
    return state


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


def _ensure_state_entry(state: dict, strategy_id: str) -> None:
    if strategy_id not in state:
        state[strategy_id] = {"enabled": True, "params": _default_params(strategy_id)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_all_strategies() -> list[dict]:
    """Return all strategies with current state merged in."""
    state = _load_state()
    result = []
    for sid, catalog in STRATEGY_CATALOG.items():
        _ensure_state_entry(state, sid)
        s_state = state[sid]
        result.append(
            {
                "id": sid,
                "name": catalog["name"],
                "type": catalog["type"],
                "type_color": catalog["type_color"],
                "description": catalog["description"],
                "asset_type": catalog["asset_type"],
                "regime_controlled": catalog.get("regime_controlled", False),
                "param_schema": catalog["param_schema"],
                "enabled": s_state.get("enabled", True),
                "params": s_state.get("params", _default_params(sid)),
                "regime_override": s_state.get("regime_override", None),
            }
        )
    return result


def get_strategy(strategy_id: str) -> dict | None:
    """Return a single strategy with state, or None if not found."""
    if strategy_id not in STRATEGY_CATALOG:
        return None
    for s in get_all_strategies():
        if s["id"] == strategy_id:
            return s
    return None


def toggle_strategy(strategy_id: str) -> dict | None:
    """Flip enabled/disabled for a strategy. Returns updated state or None."""
    if strategy_id not in STRATEGY_CATALOG:
        return None
    state = _load_state()
    _ensure_state_entry(state, strategy_id)
    state[strategy_id]["enabled"] = not state[strategy_id].get("enabled", True)
    _save_state(state)
    return {"id": strategy_id, "enabled": state[strategy_id]["enabled"]}


def update_params(strategy_id: str, params: dict) -> dict | None:
    """Update parameters for a strategy. Only known schema keys are accepted."""
    if strategy_id not in STRATEGY_CATALOG:
        return None
    state = _load_state()
    _ensure_state_entry(state, strategy_id)
    schema_keys = {p["key"] for p in STRATEGY_CATALOG[strategy_id]["param_schema"]}
    for key, val in params.items():
        if key in schema_keys:
            state[strategy_id]["params"][key] = val
    _save_state(state)
    return {"id": strategy_id, "params": state[strategy_id]["params"]}


def build_strategy_instance(strategy_id: str, params: dict) -> Any:
    """Instantiate and return a strategy object from its id and params dict.

    Returns ``None`` for unknown strategy IDs. Option strategies cannot be
    instantiated here (no broker) and also return ``None``.
    """
    if strategy_id == "moving_average_cross":
        from src.strategy.example_strategies import MovingAverageCrossStrategy

        return MovingAverageCrossStrategy(
            symbol=str(params.get("symbol", "RELIANCE")),
            security_id=str(params.get("security_id", "2885")),
            fast_period=int(params.get("fast_period", 20)),
            slow_period=int(params.get("slow_period", 50)),
            quantity=int(params.get("quantity", 1)),
            exchange_segment=str(params.get("exchange_segment", "NSE_EQ")),
        )
    if strategy_id == "rsi":
        from src.strategy.example_strategies import RSIStrategy

        return RSIStrategy(
            symbol=str(params.get("symbol", "TCS")),
            security_id=str(params.get("security_id", "11536")),
            rsi_period=int(params.get("rsi_period", 14)),
            oversold=float(params.get("oversold", 30)),
            overbought=float(params.get("overbought", 70)),
            quantity=int(params.get("quantity", 1)),
            exchange_segment=str(params.get("exchange_segment", "NSE_EQ")),
        )
    if strategy_id == "bollinger_bands":
        from src.strategy.example_strategies import BollingerBandsStrategy

        return BollingerBandsStrategy(
            symbol=str(params.get("symbol", "INFY")),
            security_id=str(params.get("security_id", "10999")),
            period=int(params.get("period", 20)),
            num_std=float(params.get("num_std", 2.0)),
            quantity=int(params.get("quantity", 1)),
            exchange_segment=str(params.get("exchange_segment", "NSE_EQ")),
        )
    if strategy_id == "long_straddle":
        from src.strategy.option_chain_strategy import LongStraddleStrategy

        return LongStraddleStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "long_strangle":
        from src.strategy.option_chain_strategy import LongStrangleStrategy

        return LongStrangleStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            otm_distance=int(params.get("otm_distance", 1)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "bull_call_spread":
        from src.strategy.option_chain_strategy import BullCallSpreadStrategy

        return BullCallSpreadStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            otm_distance=int(params.get("otm_distance", 1)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "bear_put_spread":
        from src.strategy.option_chain_strategy import BearPutSpreadStrategy

        return BearPutSpreadStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            otm_distance=int(params.get("otm_distance", 1)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "bull_put_spread":
        from src.strategy.option_chain_strategy import BullPutSpreadStrategy

        return BullPutSpreadStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            otm_distance=int(params.get("otm_distance", 1)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "bear_call_spread":
        from src.strategy.option_chain_strategy import BearCallSpreadStrategy

        return BearCallSpreadStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            otm_distance=int(params.get("otm_distance", 1)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "iron_condor":
        from src.strategy.option_chain_strategy import IronCondorStrategy

        return IronCondorStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            short_otm_distance=int(params.get("short_otm_distance", 1)),
            long_otm_distance=int(params.get("long_otm_distance", 2)),
            min_iv_threshold=float(params.get("min_iv_threshold", 15.0)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "iron_butterfly":
        from src.strategy.option_chain_strategy import IronButterflyStrategy

        return IronButterflyStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            wing_distance=int(params.get("wing_distance", 1)),
            min_iv_threshold=float(params.get("min_iv_threshold", 15.0)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    if strategy_id == "ema_crossover_nifty":
        from src.strategy.ema_crossover_strategy import EMACrossoverNiftyStrategy

        return EMACrossoverNiftyStrategy(
            under_security_id=int(params.get("under_security_id", 13)),
            under_exchange_segment=str(params.get("under_exchange_segment", "IDX_I")),
            futures_security_id=str(params.get("futures_security_id", "13")),
            fast_period=int(params.get("fast_period", 9)),
            slow_period=int(params.get("slow_period", 21)),
            quantity=int(params.get("quantity", 1)),
            product_type=str(params.get("product_type", "INTRADAY")),
        )
    return None


# ---------------------------------------------------------------------------
# Trading-active state
# ---------------------------------------------------------------------------


def get_trading_active() -> bool:
    """Return True when the master trading switch is ON."""
    try:
        if _TRADING_STATE_FILE.exists():
            with _TRADING_STATE_FILE.open() as f:
                return bool(json.load(f).get("active", False))
    except Exception:  # noqa: BLE001
        pass
    return False


def set_trading_active(active: bool) -> None:
    """Persist the master trading-active flag."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _TRADING_STATE_FILE.open("w") as f:
        json.dump({"active": active}, f)


# ---------------------------------------------------------------------------
# Per-strategy regime override
# ---------------------------------------------------------------------------


def get_strategy_regime_override(strategy_id: str) -> bool | None:
    """Return the regime override for a strategy.

    Returns
    -------
    True  – strategy runs regardless of regime
    False – strategy never runs (manually disabled by regime override)
    None  – follow the regime (default)
    """
    if strategy_id not in STRATEGY_CATALOG:
        return None
    state = _load_state()
    _ensure_state_entry(state, strategy_id)
    return state[strategy_id].get("regime_override", None)


def set_strategy_regime_override(strategy_id: str, override: bool | None) -> None:
    """Set the regime override for a strategy (True/False/None)."""
    if strategy_id not in STRATEGY_CATALOG:
        return
    state = _load_state()
    _ensure_state_entry(state, strategy_id)
    state[strategy_id]["regime_override"] = override
    _save_state(state)
