"""Flask web dashboard for the Dhan Quant Trading Platform.

Run with:
    python run_dashboard.py          # convenience script at repo root
    python -m dashboard.app          # from repo root
    flask --app dashboard.app run    # via Flask CLI
"""

import os
import sys

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
)
from src.backtesting.backtester import Backtester
from src.broker.dhan_broker import DhanBroker
from src.data.data_fetcher import DataFetcher
from src.utils.logger import load_config

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "config.yaml")

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
    except ImportError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Data fetch error: {exc}"}), 500

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
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Backtest failed: {exc}"}), 500

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
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
