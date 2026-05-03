"""Order management settings – stop-loss and target configuration."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
_ORDER_FILE = DATA_DIR / "order_settings.json"

_DEFAULTS: dict = {
    "default_sl_points": 50.0,
    "default_target_points": 100.0,
    "trailing_sl_enabled": False,
    "trailing_sl_points": 30.0,
    "sl_type": "FIXED",        # "FIXED" or "PERCENT"
    "sl_percent": 1.0,
    "target_type": "FIXED",    # "FIXED" or "PERCENT"
    "target_percent": 2.0,
    "order_type": "MARKET",    # "MARKET" or "LIMIT"
}


def get_order_settings() -> dict:
    """Return current order management settings, falling back to defaults."""
    try:
        if _ORDER_FILE.exists():
            with _ORDER_FILE.open() as f:
                saved = json.load(f)
            return {**_DEFAULTS, **saved}
    except Exception:  # noqa: BLE001
        pass
    return dict(_DEFAULTS)


def save_order_settings(settings: dict) -> dict:
    """Persist order management settings.  Only known keys are accepted."""
    current = get_order_settings()
    for key in _DEFAULTS:
        if key in settings:
            current[key] = settings[key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _ORDER_FILE.open("w") as f:
        json.dump(current, f, indent=2)
    return current


def get_sl_and_target(entry_price: float, action: str = "BUY") -> dict:
    """Compute stop-loss and target prices from current settings.

    Parameters
    ----------
    entry_price:
        The price at which the trade is entered.
    action:
        ``"BUY"`` (long) or ``"SELL"`` (short).

    Returns
    -------
    dict
        Keys: ``sl_price``, ``target_price``, ``sl_order_type``.
    """
    s = get_order_settings()
    is_long = str(action).upper() == "BUY"

    if s["sl_type"] == "PERCENT":
        sl_offset = entry_price * s["sl_percent"] / 100.0
    else:
        sl_offset = float(s["default_sl_points"])

    if s["target_type"] == "PERCENT":
        tgt_offset = entry_price * s["target_percent"] / 100.0
    else:
        tgt_offset = float(s["default_target_points"])

    if is_long:
        sl_price     = round(entry_price - sl_offset,  2)
        target_price = round(entry_price + tgt_offset, 2)
    else:
        sl_price     = round(entry_price + sl_offset,  2)
        target_price = round(entry_price - tgt_offset, 2)

    return {
        "sl_price":     sl_price,
        "target_price": target_price,
        "sl_order_type": s["order_type"],
    }
