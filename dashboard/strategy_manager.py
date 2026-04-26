"""Strategy registry with persistent enable/disable state and param management."""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "strategies.json"

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
                "param_schema": catalog["param_schema"],
                "enabled": s_state.get("enabled", True),
                "params": s_state.get("params", _default_params(sid)),
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
    return None
