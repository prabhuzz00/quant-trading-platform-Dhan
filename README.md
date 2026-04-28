# Dhan Quant Trading Platform

A modular, Python-based quantitative trading platform built on top of the **[Dhan](https://dhan.co/)** trading API. Supports equity backtesting, live option chain analysis, multiple option trading strategies, and a web-based dashboard — all out of the box.

## Features

### Core modules

| Module | Description |
|---|---|
| `DhanBroker` | Thin wrapper around the `dhanhq` SDK (v2.1+) with paper-trading support |
| `DataFetcher` | OHLCV data from Dhan Historical API (`intraday_minute_data` / `historical_daily_data`) or yfinance fallback |
| `BaseStrategy` | Abstract base class for custom equity strategies |
| `MovingAverageCrossStrategy` | Dual MA crossover (golden/death cross) |
| `RSIStrategy` | RSI mean-reversion |
| `BollingerBandsStrategy` | Bollinger Bands breakout |
| `PortfolioManager` | Position tracking, risk checks, P&L reporting |
| `Backtester` | Event-driven backtesting engine with commission & slippage |

### Option chain modules

| Module | Description |
|---|---|
| `OptionChainFetcher` | Live option chain data from Dhan API – expiry list, per-strike call/put OI, Greeks, IV |
| `OptionChainStrategy` | Abstract base for option strategies that use live chain data |
| `ShortStraddleStrategy` | Sell ATM call + put when IV exceeds a configurable threshold |
| `LongStraddleStrategy` | Buy ATM call + put to profit from high expected volatility |
| `PCRStrategy` | Contrarian directional strategy driven by the Put-Call Ratio (PCR) |
| `LongStrangleStrategy` | Buy OTM call + put for cheaper premium in a high-volatility play |
| `BullCallSpreadStrategy` | Buy lower-strike call, sell higher-strike call (debit spread, bullish) |
| `BearPutSpreadStrategy` | Buy higher-strike put, sell lower-strike put (debit spread, bearish) |
| `BullPutSpreadStrategy` | Sell higher-strike put, buy lower-strike put (credit spread, bullish) |
| `BearCallSpreadStrategy` | Sell lower-strike call, buy higher-strike call (credit spread, bearish) |
| `IronCondorStrategy` | Combine a bull put spread + bear call spread for range-bound markets |
| `IronButterflyStrategy` | Sell ATM straddle, buy OTM wings – maximum profit at expiry near ATM |

### Web dashboard

| Feature | Description |
|---|---|
| Single-page UI | Real-time strategy overview and controls at `http://localhost:5000` |
| Strategy toggle | Enable / disable individual strategies without restarting the server |
| Parameter editor | Edit strategy parameters live from the browser |
| Backtest runner | Run historical backtests and view equity-curve charts + trade history |
| Credentials tab | Save your Dhan `client_id` and `access_token` via the UI (stored in `dashboard/data/credentials.json`) |
| Config viewer | Inspect the full `config/config.yaml` from the browser |

---

## Setup

### Prerequisites

- Python **3.10+**
- A [Dhan](https://dhan.co/) trading account with API access (for live data & order execution)

### 1. Clone the repository

```bash
git clone https://github.com/prabhuzz00/quant-trading-platform-Dhan.git
cd quant-trading-platform-Dhan
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install yfinance   # optional – needed for data fallback when Dhan credentials are absent
```

> **Note:** `requirements.txt` pins `dhanhq>=2.1.0`. The platform uses the new `DhanContext` API introduced in that version.

### 4. Configure credentials

```bash
cp .env.example .env
# Open .env and fill in your DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN
```

Or export them as environment variables:

```bash
export DHAN_CLIENT_ID=your_client_id
export DHAN_ACCESS_TOKEN=your_access_token
```

Credentials are read automatically at startup via `python-dotenv`.

### 5. (Optional) Adjust `config/config.yaml`

Key settings you may want to change before running:

```yaml
trading:
  paper_trade: true          # set to false for live order execution

risk:
  max_position_size: 0.05    # max 5 % of equity per position
  stop_loss_pct: 0.02        # default 2 % stop-loss
  take_profit_pct: 0.04      # default 4 % take-profit

backtesting:
  initial_capital: 100000
  commission: 0.0003         # 0.03 % per trade
  slippage: 0.0001           # 0.01 % slippage

option_chain:
  default_underlying: nifty50
  short_straddle:
    min_iv_threshold: 15.0   # minimum IV (%) to enter a short straddle
  pcr_strategy:
    bullish_pcr: 1.5         # PCR above this → contrarian buy-call signal
    bearish_pcr: 0.5         # PCR below this → contrarian buy-put signal
```

### 6. Run the CLI entry point

```bash
python main.py
```

`main.py` will:
1. Run a **Moving Average Crossover** backtest on RELIANCE equity (using yfinance or Dhan data).
2. Run an **option chain demo** (requires valid Dhan credentials) that fetches the Nifty 50 chain, prints ATM prices / IV / PCR / Max Pain, and generates entry signals from both the PCR and Short Straddle strategies.

### 7. Run the web dashboard

```bash
python run_dashboard.py          # http://localhost:5000  (default)
PORT=8080 python run_dashboard.py   # custom port
```

The dashboard exposes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Single-page trading dashboard (SPA) |
| `/api/strategies` | GET | List all strategies with their state and parameters |
| `/api/strategies/<id>/toggle` | POST | Enable or disable a strategy |
| `/api/strategies/<id>/params` | PUT | Update strategy parameters |
| `/api/backtest` | POST | Run a historical backtest |
| `/api/config` | GET | View the full platform configuration |
| `/api/credentials` | POST | Save Dhan API credentials |
| `/api/credentials/status` | GET | Check if credentials are set and valid |

> **Tip:** Open the **Credentials** tab in the dashboard to enter your Dhan `client_id` and `access_token` — no need to edit files manually. Credentials are persisted in `dashboard/data/credentials.json` and take effect immediately.

### 8. Run tests

```bash
python -m pytest tests/ -v
```

---

## Configuration reference

Full configuration lives in `config/config.yaml`:

```yaml
broker:
  client_id: ""           # overridden by DHAN_CLIENT_ID env var
  access_token: ""        # overridden by DHAN_ACCESS_TOKEN env var

trading:
  paper_trade: true
  default_exchange: NSE
  default_product_type: INTRADAY
  default_order_type: LIMIT

risk:
  max_position_size: 0.05
  max_portfolio_risk: 0.20
  stop_loss_pct: 0.02
  take_profit_pct: 0.04

backtesting:
  initial_capital: 100000
  commission: 0.0003
  slippage: 0.0001

logging:
  level: INFO
  file: logs/trading.log

option_chain:
  nifty50_security_id: 13
  banknifty_security_id: 25
  finnifty_security_id: 27
  midcpnifty_security_id: 442
  default_underlying: nifty50
  default_exchange_segment: IDX_I   # IDX_I for indices, NSE_EQ for equities
  default_fno_segment: NSE_FNO
  short_straddle:
    min_iv_threshold: 15.0
    quantity: 1
    product_type: INTRADAY
  pcr_strategy:
    bullish_pcr: 1.5
    bearish_pcr: 0.5
    quantity: 1
    product_type: INTRADAY
```

---

## Writing a custom equity strategy

```python
from src.strategy.base_strategy import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy", params={"threshold": 0.5})

    def generate_signals(self, data: pd.DataFrame) -> dict | None:
        # data has at minimum a 'close' column
        close = data["close"].iloc[-1]
        if close > self.params["threshold"]:
            return {
                "symbol": "RELIANCE",
                "security_id": "2885",
                "exchange_segment": "NSE_EQ",
                "action": "BUY",
                "quantity": 1,
                "price": close,
                "order_type": "MARKET",
                "product_type": "INTRADAY",
            }
        return None
```

---

## Using the Option Chain module

```python
from src.broker.dhan_broker import DhanBroker
from src.data.option_chain import OptionChainFetcher

broker = DhanBroker()          # reads credentials from .env
fetcher = OptionChainFetcher(broker)

# List available expiry dates for Nifty 50
expiries = fetcher.get_expiry_list(under_security_id=13, under_exchange_segment="IDX_I")

# Fetch full option chain as a DataFrame
chain = fetcher.get_option_chain(
    under_security_id=13,
    under_exchange_segment="IDX_I",
    expiry=expiries[0],
)

spot = fetcher.get_spot_price(under_security_id=13, under_exchange_segment="IDX_I", expiry=expiries[0])

# ATM options, PCR, Max Pain
atm      = fetcher.get_atm_options(chain, spot_price=spot)
pcr      = fetcher.calculate_pcr(chain)
max_pain = fetcher.get_max_pain(chain)
near_atm = fetcher.get_strikes_near_atm(chain, spot_price=spot, n_strikes=5)
```

### Using option strategies

```python
from src.strategy.option_chain_strategy import PCRStrategy, ShortStraddleStrategy

# PCR-driven contrarian strategy
pcr_strategy = PCRStrategy(under_security_id=13, spot_price=spot)
pcr_strategy.attach_broker(broker)
signal = pcr_strategy.generate_signals(pd.DataFrame())   # returns a BUY signal or None

# Short straddle (sell ATM call + put when IV is high)
straddle = ShortStraddleStrategy(under_security_id=13, spot_price=spot, min_iv_threshold=15.0)
straddle.attach_broker(broker)
signal = straddle.generate_signals(pd.DataFrame())
```

---

## Project structure

```
quant-trading-platform-Dhan/
├── config/
│   └── config.yaml                    # Runtime configuration
├── dashboard/
│   ├── app.py                         # Flask REST API (web dashboard backend)
│   ├── strategy_manager.py            # Strategy registry – enable/disable, params
│   ├── data/
│   │   ├── credentials.json           # Saved Dhan API credentials (git-ignored)
│   │   └── strategies.json            # Persisted strategy enable/disable state
│   └── templates/
│       └── index.html                 # Single-page dashboard UI
├── src/
│   ├── broker/
│   │   └── dhan_broker.py             # Dhan API wrapper (dhanhq v2.1+)
│   ├── data/
│   │   ├── data_fetcher.py            # Historical & live OHLCV data
│   │   └── option_chain.py            # Option chain fetcher & analytics
│   ├── strategy/
│   │   ├── base_strategy.py           # Abstract strategy base class
│   │   ├── example_strategies.py      # MA Cross, RSI, Bollinger Bands
│   │   └── option_chain_strategy.py   # 10 option strategies (straddle, spreads, condor…)
│   ├── backtesting/
│   │   └── backtester.py              # Event-driven backtesting engine
│   ├── portfolio/
│   │   └── portfolio_manager.py       # Position & risk management
│   └── utils/
│       └── logger.py                  # Logging & config helpers
├── tests/
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   └── test_option_chain.py           # Option chain & strategy unit tests (123 tests)
├── main.py                            # Entry point (backtest + option chain demo)
├── run_dashboard.py                   # Convenience script to start the web dashboard
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Deploying on a VPS (Ubuntu 22.04)

This section walks you through hosting the dashboard as a persistent service on a fresh Ubuntu 22.04 VPS (e.g. DigitalOcean, AWS EC2, Hetzner).

### 1. Connect to your VPS and install system dependencies

```bash
ssh your_user@your_server_ip

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx
```

> Python 3.10+ is required. Ubuntu 22.04 ships Python 3.10 by default; `python3.11` is available from the `deadsnakes` PPA if you prefer 3.11.

### 2. Clone the repository

```bash
cd /opt
sudo git clone https://github.com/prabhuzz00/quant-trading-platform-Dhan.git
sudo chown -R $USER:$USER /opt/quant-trading-platform-Dhan
cd /opt/quant-trading-platform-Dhan
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install yfinance          # optional – needed for data fallback
deactivate
```

### 4. Configure credentials / environment variables

```bash
cp .env.example .env
nano .env   # fill in DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN
```

Alternatively, leave `.env` empty and use the **Credentials** tab in the dashboard after the service is running.

### 5. Create a systemd service

Create the unit file `/etc/systemd/system/dhan-dashboard.service`:

```bash
sudo nano /etc/systemd/system/dhan-dashboard.service
```

Paste the following (adjust `User` and paths if needed):

```ini
[Unit]
Description=Dhan Quant Trading Platform Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/quant-trading-platform-Dhan
EnvironmentFile=/opt/quant-trading-platform-Dhan/.env
ExecStart=/opt/quant-trading-platform-Dhan/.venv/bin/python run_dashboard.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable dhan-dashboard
sudo systemctl start dhan-dashboard
sudo systemctl status dhan-dashboard   # should show "active (running)"
```

Check logs at any time:

```bash
sudo journalctl -u dhan-dashboard -f
```

### 6. Configure Nginx as a reverse proxy

By default the dashboard runs on port 5000. Nginx forwards HTTP traffic on port 80 to it.

```bash
sudo nano /etc/nginx/sites-available/dhan-dashboard
```

```nginx
server {
    listen 80;
    server_name your_domain_or_ip;   # e.g. 203.0.113.10 or dashboard.example.com

    location / {
        proxy_pass         http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120;
    }
}
```

Enable the site and reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/dhan-dashboard /etc/nginx/sites-enabled/
sudo nginx -t          # verify config syntax
sudo systemctl reload nginx
```

The dashboard is now accessible at `http://your_domain_or_ip`.

### 7. (Recommended) Secure with HTTPS using Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

Follow the on-screen prompts. Certbot will update the Nginx config and set up automatic renewal.

### 8. Open the firewall

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

### 9. Update the application

```bash
cd /opt/quant-trading-platform-Dhan
git pull
source .venv/bin/activate
pip install -r requirements.txt
deactivate
sudo systemctl restart dhan-dashboard
```

---



This software is for educational and research purposes only. It is **not** financial advice. Trading involves significant risk of loss. Always test thoroughly in paper-trade mode before deploying real capital.