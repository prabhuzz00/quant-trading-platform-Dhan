"""Convenience script to start the Dhan Quant Trading Platform dashboard.

Usage
-----
    python run_dashboard.py                    # runs on http://localhost:5000
    PORT=8080 python run_dashboard.py          # runs on a different port

The dashboard exposes:
    /                        Single-page trading dashboard
    /api/strategies          List / toggle / update strategies
    /api/backtest            Run a historical backtest
    /api/config              View platform configuration
"""

import os

from dotenv import load_dotenv

load_dotenv()

from dashboard.app import app  # noqa: E402  (import after env vars are loaded)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"  ⚡  Dhan Quant Trading Dashboard")
    print(f"  🌐  http://localhost:{port}")
    print(f"  ⏹   Press Ctrl+C to stop\n")
    app.run(debug=debug, host=host, port=port)
