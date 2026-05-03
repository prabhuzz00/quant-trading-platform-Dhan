"""Risk management settings persistence and enforcement."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
_RISK_FILE = DATA_DIR / "risk_settings.json"

_DEFAULTS: dict = {
    "max_consecutive_trades_per_strategy": 5,
    "max_total_margin": 500000.0,
    "max_margin_per_strategy": 100000.0,
    "max_daily_loss": 10000.0,
    "max_daily_trades": 20,
    "max_open_positions": 10,
    "risk_per_trade_pct": 1.0,
    "min_trade_gap_seconds": 60,
}


def get_risk_settings() -> dict:
    """Return current risk settings, falling back to defaults."""
    try:
        if _RISK_FILE.exists():
            with _RISK_FILE.open() as f:
                saved = json.load(f)
            return {**_DEFAULTS, **saved}
    except Exception:  # noqa: BLE001
        pass
    return dict(_DEFAULTS)


def save_risk_settings(settings: dict) -> dict:
    """Persist risk settings.  Only known keys are accepted."""
    current = get_risk_settings()
    for key in _DEFAULTS:
        if key in settings:
            current[key] = settings[key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _RISK_FILE.open("w") as f:
        json.dump(current, f, indent=2)
    return current


def check_risk_limits(strategy_id: str = "", trade_value: float = 0.0) -> dict:
    """Check whether a new trade is allowed under current risk settings.

    Parameters
    ----------
    strategy_id:
        Identifier of the strategy requesting the trade.
    trade_value:
        Estimated margin / value of the trade in rupees.

    Returns
    -------
    dict
        ``{"allowed": bool, "reason": str}``
    """
    settings = get_risk_settings()

    # Margin per-trade check
    if trade_value > settings["max_margin_per_strategy"]:
        return {
            "allowed": False,
            "reason": (
                f"Trade value ₹{trade_value:,.0f} exceeds max margin per strategy "
                f"₹{settings['max_margin_per_strategy']:,.0f}"
            ),
        }

    return {"allowed": True, "reason": "OK"}
