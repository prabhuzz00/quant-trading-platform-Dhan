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
    auto_refresh_regime,
    get_crude_regime_details,
    update_crude_regime,
    auto_refresh_crude_regime,
)
from src.backtesting.backtester import Backtester
from src.broker.dhan_broker import DhanBroker
from src.data.data_fetcher import DataFetcher
from src.utils.logger import load_config
from dashboard.trading_engine import get_engine

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "config.yaml")
_CREDENTIALS_FILE = Path(__file__).parent / "data" / "credentials.json"

app = Flask(__name__, template_folder="templates")


# ---------------------------------------------------------------------------
# Custom JSON encoder – makes Flask serialize numpy scalar types cleanly
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """Extend the default encoder to handle numpy int/float/bool scalars."""

    def default(self, obj: object) -> object:
        try:
            import numpy as np  # noqa: PLC0415
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


app.json_encoder = _NumpyEncoder  # type: ignore[attr-defined]


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
    # Drop cached instance so it is rebuilt with the new params on next tick
    get_engine().invalidate_instance(strategy_id)
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


@app.route("/api/regime/refresh", methods=["POST"])
def regime_refresh():
    """Fetch latest NIFTY50 1-min data and re-compute the regime."""
    try:
        broker = DhanBroker(paper_trade=True)
        details = auto_refresh_regime(broker)
        return jsonify(details)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Regime refresh failed: {exc}"}), 500


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


# ---- Crude Oil regime -------------------------------------------------------


@app.route("/api/regime/crude", methods=["GET"])
def crude_regime_status():
    return jsonify(get_crude_regime_details())


@app.route("/api/regime/crude/refresh", methods=["POST"])
def crude_regime_refresh():
    """Fetch latest Crude Oil Futures 1-min data and re-compute the regime."""
    body = request.get_json(silent=True) or {}
    # Prefer body param → scrip-master cache → strategy params → hard fallback
    security_id = str(body.get("security_id", "")).strip()
    if not security_id:
        security_id = str(_crude_security_id_cache.get("futures_security_id", "")).strip()
    if not security_id:
        # Last resort: read from ema_crossover_crude strategy params
        try:
            from dashboard.strategy_manager import get_strategy  # noqa: PLC0415
            s = get_strategy("ema_crossover_crude")
            if s:
                security_id = str(s.get("params", {}).get("futures_security_id", "")).strip()
        except Exception:  # noqa: BLE001
            pass
    if not security_id:
        security_id = "488290"  # known MCX Crude May-2026 – better than 16429
    exchange_segment = str(body.get("exchange_segment", "MCX_COMM"))
    try:
        broker = DhanBroker(paper_trade=True)
        details = auto_refresh_crude_regime(
            broker, security_id=security_id, exchange_segment=exchange_segment
        )
        return jsonify(details)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Crude regime refresh failed: {exc}"}), 500


# ---- Strategy signal log ---------------------------------------------------


@app.route("/api/strategies/<sid>/signal_log", methods=["GET"])
def strategy_signal_log(sid: str):
    """Return the in-memory signal log for a running strategy instance."""
    engine = get_engine()
    instance = engine.get_instance(sid)
    if instance is None:
        return jsonify([])
    return jsonify(list(instance._signal_log))


# ---------------------------------------------------------------------------
# Trading state (master on/off)
# ---------------------------------------------------------------------------


@app.route("/api/trading/status", methods=["GET"])
def trading_status():
    engine = get_engine()
    return jsonify({"active": get_trading_active(), "running": engine.is_running()})


@app.route("/api/trading/start", methods=["POST"])
def trading_start():
    set_trading_active(True)
    try:
        broker = DhanBroker(paper_trade=True)
        # Seed regime immediately so strategies see a valid regime from the first tick
        try:
            auto_refresh_regime(broker)
        except Exception as exc:  # noqa: BLE001
            pass  # non-fatal — regime will refresh on the first engine tick
        engine = get_engine()
        engine.invalidate_all()
        engine.start(broker, paper_trade=True)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"active": True, "running": False, "warning": str(exc)})
    return jsonify({"active": True, "running": engine.is_running()})


@app.route("/api/trading/stop", methods=["POST"])
def trading_stop():
    set_trading_active(False)
    get_engine().stop()
    return jsonify({"active": False, "running": False})


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
                "strike":     float(row["strike_price"]),
                "call_sid":   str(row["call_security_id"]),
                "call_ltp":   float(row["call_ltp"]),
                "call_oi":    int(row["call_oi"]),
                "call_iv":    float(row["call_iv"]),
                "put_sid":    str(row["put_security_id"]),
                "put_ltp":    float(row["put_ltp"]),
                "put_oi":     int(row["put_oi"]),
                "put_iv":     float(row["put_iv"]),
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
# Crude Oil instrument lookup
# ---------------------------------------------------------------------------

_DHAN_SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
_crude_security_id_cache: dict = {}   # {"security_id": str, "expiry": str, "fetched_at": float}


@app.route("/api/crude/security_id", methods=["GET"])
def crude_security_id():
    """Auto-detect MCX Crude Oil security IDs from Dhan's instrument master CSV.

    Returns:
      security_id          – option-chain underlying ID (from MCX_OPT OPTFUT rows, or futures ID as fallback)
      futures_security_id  – near-month FUTCOM contract ID (for LTP)
      opt_expiries         – list of upcoming option expiry dates from scrip master
      exchange_segment     – always MCX_COMM
    """
    import csv
    import io
    import time
    import urllib.request
    import datetime

    force = request.args.get("force", "0") == "1"
    cache_ttl = 3600

    if not force and _crude_security_id_cache.get("fetched_at", 0) + cache_ttl > time.time():
        return jsonify(_crude_security_id_cache)

    try:
        req = urllib.request.Request(
            _DHAN_SCRIP_MASTER_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to download instrument master: {exc}"}), 500

    try:
        today = datetime.date.today()
        reader = csv.DictReader(io.StringIO(raw))
        # Actual scrip master columns (confirmed from live CSV 2026-05):
        #   SEM_EXM_EXCH_ID     → exchange short code, e.g. "MCX"
        #   SEM_INSTRUMENT_NAME → "FUTCOM" | "OPTFUT"
        #   SM_SYMBOL_NAME      → base name, e.g. "CRUDEOIL"
        #   SEM_TRADING_SYMBOL  → full symbol, e.g. "CRUDEOIL-18May2026-FUT"
        #   SEM_CUSTOM_SYMBOL   → friendly name, e.g. "CRUDEOIL MAY FUT"
        #   SEM_SMST_SECURITY_ID → security ID
        #   SEM_EXPIRY_DATE     → "2026-05-18 23:30:00"  (take first 10 chars)

        best_futcom: dict | None = None
        opt_expiries: list[str] = []

        for row in reader:
            exc_id     = row.get("SEM_EXM_EXCH_ID", "").strip()
            instrument = row.get("SEM_INSTRUMENT_NAME", "").strip()
            sym_name   = row.get("SM_SYMBOL_NAME", "").strip().upper()
            trading    = row.get("SEM_TRADING_SYMBOL", "").strip().upper()

            if exc_id != "MCX":
                continue
            if "CRUDE" not in sym_name and "CRUDE" not in trading:
                continue

            exp_str = row.get("SEM_EXPIRY_DATE", "").strip()[:10]
            try:
                exp_date = datetime.date.fromisoformat(exp_str)
            except ValueError:
                continue
            if exp_date < today:
                continue

            if instrument == "FUTCOM":
                if best_futcom is None or exp_date < best_futcom["exp_date"]:
                    best_futcom = {
                        "security_id":  row.get("SEM_SMST_SECURITY_ID", "").strip(),
                        "display_name": row.get("SEM_CUSTOM_SYMBOL", trading).strip(),
                        "expiry":       exp_date.isoformat(),
                        "exp_date":     exp_date,
                    }
            elif instrument == "OPTFUT":
                iso = exp_date.isoformat()
                if iso not in opt_expiries:
                    opt_expiries.append(iso)

        if not best_futcom:
            return jsonify({"error": "No active MCX CRUDEOIL FUTCOM contract found in instrument master"}), 404

        opt_expiries.sort()
        futures_sid = best_futcom["security_id"]

        result = {
            "security_id":         futures_sid,   # FUTCOM id → used for expiry_list API
            "futures_security_id": futures_sid,
            "display_name":        best_futcom["display_name"],
            "expiry":              best_futcom["expiry"],
            "opt_expiries":        opt_expiries[:6],
            "exchange_segment":    "MCX_COMM",
            "instrument":          "FUTCOM",
            "fetched_at":          time.time(),
        }
        _crude_security_id_cache.clear()
        _crude_security_id_cache.update(result)
        return jsonify(result)

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to parse instrument master: {exc}"}), 500


# ---------------------------------------------------------------------------
# Crude Oil snapshot
# ---------------------------------------------------------------------------


@app.route("/api/crude/snapshot", methods=["GET"])
def crude_snapshot():
    """Fetch MCX Crude Oil option chain snapshot via the Dhan API.

    Query params:
      security_id          – option-chain underlying security ID (from /api/crude/security_id)
      futures_security_id  – FUTCOM contract ID for LTP (defaults to security_id)
      exchange_segment     – defaults to MCX_COMM
    """
    import datetime

    under_security_id      = int(request.args.get("security_id", 488290))
    futures_security_id    = str(request.args.get("futures_security_id", "")) or str(under_security_id)
    under_exchange_segment = str(request.args.get("exchange_segment", "MCX_COMM"))

    try:
        broker = DhanBroker(paper_trade=True)
        from src.data.option_chain import OptionChainFetcher

        fetcher = OptionChainFetcher(broker)
        expiries = fetcher.get_expiry_list(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
        )

        # Fallback: pull expiry dates from the scrip-master cache
        if not expiries:
            expiries = _crude_security_id_cache.get("opt_expiries", [])

        # Last resort: re-detect from scrip master synchronously
        if not expiries:
            try:
                import csv, io, urllib.request
                req = urllib.request.Request(
                    _DHAN_SCRIP_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                    raw_csv = resp.read().decode("utf-8", errors="replace")
                today = datetime.date.today()
                seen: set[str] = set()
                for row in csv.DictReader(io.StringIO(raw_csv)):
                    if (
                        row.get("SEM_EXM_EXCH_ID", "").strip() == "MCX"
                        and row.get("SEM_INSTRUMENT_NAME", "").strip() == "OPTFUT"
                        and "CRUDE" in row.get("SM_SYMBOL_NAME", "").upper()
                    ):
                        exp_str = row.get("SEM_EXPIRY_DATE", "").strip()[:10]
                        try:
                            ed = datetime.date.fromisoformat(exp_str)
                        except ValueError:
                            continue
                        if ed >= today:
                            iso = ed.isoformat()
                            if iso not in seen:
                                seen.add(iso)
                                expiries.append(iso)
                expiries.sort()
            except Exception:
                pass

        if not expiries:
            return jsonify({"error": (
                f"No expiry data found for MCX Crude Oil (security_id={under_security_id}). "
                "Use the 🔍 button to auto-detect the correct security_id, then refresh."
            )}), 400

        expiry = expiries[0]
        chain = fetcher.get_option_chain(
            under_security_id=under_security_id,
            under_exchange_segment=under_exchange_segment,
            expiry=expiry,
        )
        if chain.empty:
            return jsonify({"error": "Empty option chain returned."}), 400

        # Spot price from futures LTP (use futures_security_id, not the OC underlying ID)
        spot_price = broker.get_ltp(futures_security_id, under_exchange_segment)
        if spot_price <= 0:
            spot_price = fetcher.get_spot_price(
                under_security_id=under_security_id,
                under_exchange_segment=under_exchange_segment,
                expiry=expiry,
            )
        if spot_price <= 0 and not chain.empty:
            spot_price = float(chain["strike_price"].median())

        atm      = fetcher.get_atm_options(chain, spot_price)
        pcr      = fetcher.calculate_pcr(chain)
        max_pain = fetcher.get_max_pain(chain)
        near_atm = fetcher.get_strikes_near_atm(chain, spot_price, n_strikes=5)

        strikes_data = []
        for _, row in near_atm.iterrows():
            strikes_data.append({
                "strike":   float(row["strike_price"]),
                "call_sid": str(row["call_security_id"]),
                "call_ltp": float(row["call_ltp"]),
                "call_oi":  int(row["call_oi"]),
                "call_iv":  float(row["call_iv"]),
                "put_sid":  str(row["put_security_id"]),
                "put_ltp":  float(row["put_ltp"]),
                "put_oi":   int(row["put_oi"]),
                "put_iv":   float(row["put_iv"]),
            })

        return jsonify({
            "spot_price":       round(spot_price, 2),
            "expiry":           expiry,
            "security_id":      under_security_id,
            "exchange_segment": under_exchange_segment,
            "atm_strike":       atm.get("strike_price", 0),
            "atm_call":         atm.get("call", {}),
            "atm_put":          atm.get("put", {}),
            "strikes_near_atm": strikes_data,
            "pcr":              round(pcr, 3),
            "max_pain":         round(max_pain, 2),
            "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        })

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Failed to fetch Crude Oil snapshot: {exc}"}), 500


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
