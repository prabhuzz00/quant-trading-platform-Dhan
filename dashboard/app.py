"""Flask web dashboard for the Dhan Quant Trading Platform.

Run with:
    python run_dashboard.py          # convenience script at repo root
    python -m dashboard.app          # from repo root
    flask --app dashboard.app run    # via Flask CLI
"""

import json
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from dashboard.strategy_manager import (
    build_strategy_instance,
    get_all_strategies,
    get_strategy,
    toggle_strategy,
    update_params,
    get_trading_active,
    set_trading_active,
    get_strategy_regime_override,
    set_strategy_regime_override,
)
from dashboard.risk_manager import get_risk_settings, save_risk_settings, check_risk_limits
from dashboard.order_manager import get_order_settings, save_order_settings, get_sl_and_target
from dashboard.trade_journal import (
    record_trade_entry,
    record_trade_exit,
    get_open_trades,
    get_closed_trades,
    get_all_trades,
    get_strategy_pnl,
    get_all_strategy_stats,
)
from dashboard.regime_finder import (
    get_current_regime,
    update_regime,
    get_regime_details,
)
from src.backtesting.backtester import Backtester
from src.broker.dhan_broker import DhanBroker
from src.data.data_fetcher import DataFetcher
from src.utils.logger import load_config

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "config.yaml")
_CREDENTIALS_FILE = Path(__file__).parent / "data" / "credentials.json"

app = Flask(__name__, template_folder="templates")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Strategies API
# ---------------------------------------------------------------------------


@app.route("/api/strategies", methods=["GET"])
def list_strategies():
    return jsonify(get_all_strategies())


@app.route("/api/strategies/<strategy_id>/toggle", methods=["POST"])
def toggle(strategy_id: str):
    result = toggle_strategy(strategy_id)
    if result is None:
        return jsonify({"error": "Strategy not found"}), 404
    return jsonify(result)


@app.route("/api/strategies/<strategy_id>/params", methods=["PUT"])
def update_strategy_params(strategy_id: str):
    params = request.get_json(silent=True) or {}
    result = update_params(strategy_id, params)
    if result is None:
        return jsonify({"error": "Strategy not found"}), 404
    return jsonify(result)


# ---------------------------------------------------------------------------
# Backtest API
# ---------------------------------------------------------------------------


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    body = request.get_json(silent=True) or {}
    strategy_id = body.get("strategy_id")
    from_date = body.get("from_date", "2023-01-01")
    to_date = body.get("to_date", "2024-01-01")
    capital = float(body.get("capital", 100_000))
    commission = float(body.get("commission", 0.0003))
    slippage = float(body.get("slippage", 0.0001))

    strategy_info = get_strategy(strategy_id)
    if not strategy_info:
        return jsonify({"error": f"Strategy '{strategy_id}' not found"}), 404

    if strategy_info["asset_type"] == "options":
        return jsonify(
            {
                "error": (
                    "Options strategies require live Dhan API credentials and "
                    "cannot be backtested offline. Set DHAN_CLIENT_ID and "
                    "DHAN_ACCESS_TOKEN to use them in live/paper-trade mode."
                )
            }
        ), 400

    params = strategy_info["params"]
    strategy = build_strategy_instance(strategy_id, params)
    if strategy is None:
        return jsonify({"error": "Failed to build strategy instance"}), 500

    broker = DhanBroker(paper_trade=True)
    fetcher = DataFetcher(broker=broker)
    symbol = str(params.get("symbol", "RELIANCE"))
    security_id = str(params.get("security_id", ""))

    try:
        data = fetcher.get_historical_data(
            symbol=symbol,
            security_id=security_id,
            from_date=from_date,
            to_date=to_date,
        )
    except ImportError:
        return jsonify(
            {
                "error": (
                    "yfinance is not installed. Run: pip install yfinance "
                    "to enable historical data fetching."
                )
            }
        ), 400
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Failed to fetch historical data. Check the symbol and date range."}), 500

    if data.empty:
        return jsonify(
            {
                "error": (
                    f"No historical data returned for '{symbol}'. "
                    "Check the symbol name (use NSE ticker, e.g. RELIANCE, TCS) "
                    "or try a different date range."
                )
            }
        ), 400

    try:
        backtester = Backtester(
            strategy=strategy,
            data=data,
            initial_capital=capital,
            commission=commission,
            slippage=slippage,
        )
        results = backtester.run()
    except ValueError as exc:
        return jsonify({"error": f"Invalid backtest parameters: {exc}"}), 400
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Backtest failed due to an unexpected error. Check strategy parameters and data range."}), 500

    equity_df = backtester.equity_curve()
    trades_df = backtester.trade_history()

    # Downsample equity curve to ≤ 400 points for the chart
    step = max(1, len(equity_df) // 400)
    eq_data = [
        {
            "date": str(getattr(dt, "date", lambda: dt)()),
            "equity": round(float(row["equity"]), 2),
        }
        for dt, row in equity_df.iloc[::step].iterrows()
    ]

    trades_data = []
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            trades_data.append(
                {
                    "symbol": str(row["symbol"]),
                    "entry_price": round(float(row["entry_price"]), 2),
                    "exit_price": round(float(row["exit_price"]), 2),
                    "quantity": int(row["quantity"]),
                    "pnl": round(float(row["pnl"]), 2),
                }
            )

    return jsonify(
        {
            "strategy": strategy_info["name"],
            "symbol": symbol,
            "from_date": from_date,
            "to_date": to_date,
            "bars": len(data),
            "results": results,
            "equity_curve": eq_data,
            "trades": trades_data,
        }
    )


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------


@app.route("/api/config", methods=["GET"])
def get_config():
    try:
        config = load_config(_CONFIG_PATH)
        return jsonify(config)
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Failed to load configuration file."}), 500


# ---------------------------------------------------------------------------
# Credentials API
# ---------------------------------------------------------------------------


def _load_credentials() -> dict:
    try:
        if _CREDENTIALS_FILE.exists():
            with _CREDENTIALS_FILE.open() as f:
                return json.load(f)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("Could not read credentials file: %s", exc)
    return {}


def _save_credentials(client_id: str, access_token: str) -> None:
    _CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _CREDENTIALS_FILE.open("w") as f:
        json.dump({"client_id": client_id, "access_token": access_token}, f, indent=2)


@app.route("/api/credentials/status", methods=["GET"])
def credentials_status():
    """Return whether credentials are set and whether a Dhan connection can be made."""
    env_client = os.getenv("DHAN_CLIENT_ID", "")
    env_token = os.getenv("DHAN_ACCESS_TOKEN", "")
    saved = _load_credentials()
    client_id = env_client or saved.get("client_id", "")
    access_token = env_token or saved.get("access_token", "")
    connected = bool(client_id and access_token)
    return jsonify(
        {
            "client_id_set": bool(client_id),
            "access_token_set": bool(access_token),
            "connected": connected,
            "source": "env" if (env_client and env_token) else ("file" if connected else "none"),
        }
    )


@app.route("/api/credentials", methods=["POST"])
def save_credentials():
    """Persist client_id and access_token to the credentials file."""
    body = request.get_json(silent=True) or {}
    client_id = str(body.get("client_id", "")).strip()
    access_token = str(body.get("access_token", "")).strip()
    if not client_id or not access_token:
        return jsonify({"error": "Both client_id and access_token are required."}), 400
    try:
        _save_credentials(client_id, access_token)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).error("Failed to save credentials: %s", exc)
        return jsonify({"error": "Failed to save credentials to disk."}), 500
    return jsonify({"status": "saved", "client_id_set": True, "access_token_set": True, "connected": True})


# ---------------------------------------------------------------------------
# Risk Management API
# ---------------------------------------------------------------------------


@app.route("/api/risk", methods=["GET"])
def get_risk():
    return jsonify(get_risk_settings())


@app.route("/api/risk", methods=["PUT"])
def save_risk():
    body = request.get_json(silent=True) or {}
    return jsonify(save_risk_settings(body))


# ---------------------------------------------------------------------------
# Order Management API
# ---------------------------------------------------------------------------


@app.route("/api/order-management", methods=["GET"])
def get_order_mgmt():
    return jsonify(get_order_settings())


@app.route("/api/order-management", methods=["PUT"])
def save_order_mgmt():
    body = request.get_json(silent=True) or {}
    return jsonify(save_order_settings(body))


# ---------------------------------------------------------------------------
# Trade Journal API
# ---------------------------------------------------------------------------


@app.route("/api/trades", methods=["GET"])
def list_trades():
    strategy_id = request.args.get("strategy_id") or None
    status      = request.args.get("status", "all").lower()
    if status == "open":
        trades = get_open_trades(strategy_id)
    elif status == "closed":
        trades = get_closed_trades(strategy_id)
    else:
        trades = get_all_trades(strategy_id)
    return jsonify(trades)


@app.route("/api/trades/stats", methods=["GET"])
def trades_stats():
    return jsonify(get_all_strategy_stats())


@app.route("/api/trades/strategy/<strategy_id_param>", methods=["GET"])
def trades_strategy(strategy_id_param: str):
    return jsonify(get_strategy_pnl(strategy_id_param))


@app.route("/api/trades/<int:trade_id>", methods=["GET"])
def get_trade(trade_id: int):
    trades = get_all_trades()
    trade = next((t for t in trades if t["id"] == trade_id), None)
    if trade is None:
        return jsonify({"error": "Trade not found"}), 404
    return jsonify(trade)


@app.route("/api/trades/<int:trade_id>/close", methods=["POST"])
def close_trade(trade_id: int):
    body = request.get_json(silent=True) or {}
    exit_price = body.get("exit_price")
    if exit_price is None:
        return jsonify({"error": "exit_price is required"}), 400
    result = record_trade_exit(trade_id, float(exit_price))
    if not result:
        return jsonify({"error": "Trade not found"}), 404
    return jsonify(result)


# ---------------------------------------------------------------------------
# Regime API
# ---------------------------------------------------------------------------


@app.route("/api/regime", methods=["GET"])
def regime_status():
    return jsonify(get_regime_details())


@app.route("/api/regime/update", methods=["POST"])
def regime_update():
    body  = request.get_json(silent=True) or {}
    price = float(body.get("price", 0.0))
    high  = float(body.get("high",  0.0))
    low   = float(body.get("low",   0.0))
    if price <= 0:
        return jsonify({"error": "price is required"}), 400
    update_regime(price, high, low)
    return jsonify(get_regime_details())


# ---------------------------------------------------------------------------
# Trading state (master on/off)
# ---------------------------------------------------------------------------


@app.route("/api/trading/status", methods=["GET"])
def trading_status():
    return jsonify({"active": get_trading_active()})


@app.route("/api/trading/start", methods=["POST"])
def trading_start():
    set_trading_active(True)
    return jsonify({"active": True})


@app.route("/api/trading/stop", methods=["POST"])
def trading_stop():
    set_trading_active(False)
    return jsonify({"active": False})


# ---------------------------------------------------------------------------
# NIFTY50 snapshot
# ---------------------------------------------------------------------------


@app.route("/api/nifty/snapshot", methods=["GET"])
def nifty_snapshot():
    """Fetch NIFTY50 option chain snapshot via the Dhan API."""
    import datetime

    try:
        broker = DhanBroker(paper_trade=True)
        from src.data.option_chain import OptionChainFetcher

        fetcher = OptionChainFetcher(broker)
        expiries = fetcher.get_expiry_list(under_security_id=13, under_exchange_segment="IDX_I")
        if not expiries:
            return jsonify({"error": "No expiry data available. Check Dhan API credentials."}), 400

        expiry = expiries[0]
        chain  = fetcher.get_option_chain(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            expiry=expiry,
        )
        if chain.empty:
            return jsonify({"error": "Empty option chain returned."}), 400

        # Try to get spot price
        spot_price = fetcher.get_spot_price(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            expiry=expiry,
        )
        if spot_price <= 0 and not chain.empty:
            # Estimate from ATM strike
            spot_price = float(chain["strike_price"].median())

        atm       = fetcher.get_atm_options(chain, spot_price)
        pcr       = fetcher.calculate_pcr(chain)
        max_pain  = fetcher.get_max_pain(chain)
        near_atm  = fetcher.get_strikes_near_atm(chain, spot_price, n_strikes=5)

        strikes_data = []
        for _, row in near_atm.iterrows():
            strikes_data.append({
                "strike":     row["strike_price"],
                "call_sid":   row["call_security_id"],
                "call_ltp":   row["call_ltp"],
                "call_oi":    row["call_oi"],
                "call_iv":    row["call_iv"],
                "put_sid":    row["put_security_id"],
                "put_ltp":    row["put_ltp"],
                "put_oi":     row["put_oi"],
                "put_iv":     row["put_iv"],
            })

        return jsonify({
            "spot_price":    round(spot_price, 2),
            "expiry":        expiry,
            "atm_strike":    atm.get("strike_price", 0),
            "atm_call":      atm.get("call", {}),
            "atm_put":       atm.get("put", {}),
            "strikes_near_atm": strikes_data,
            "pcr":           round(pcr, 3),
            "max_pain":      round(max_pain, 2),
            "timestamp":     datetime.datetime.now().isoformat(timespec="seconds"),
        })

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to fetch NIFTY snapshot: {exc}"}), 500


# ---------------------------------------------------------------------------
# Per-strategy regime override
# ---------------------------------------------------------------------------


@app.route("/api/strategies/<strategy_id>/regime-override", methods=["POST"])
def strategy_regime_override(strategy_id: str):
    body = request.get_json(silent=True) or {}
    override = body.get("override")  # true / false / null
    if override is not None and not isinstance(override, bool):
        return jsonify({"error": "override must be true, false, or null"}), 400
    set_strategy_regime_override(strategy_id, override)
    return jsonify({"strategy_id": strategy_id, "regime_override": override})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
