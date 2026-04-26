# Dhan Quant Trading Platform

A modular, Python-based quantitative trading platform built on top of the **[Dhan](https://dhan.co/)** trading API. Supports equity backtesting, live option chain analysis, and multiple option trading strategies out of the box.

## Features

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
| `OptionChainFetcher` | Live option chain data from Dhan API – expiry list, per-strike call/put OI, Greeks, IV |
| `OptionChainStrategy` | Abstract base for option strategies that use live chain data |
| `ShortStraddleStrategy` | Sell ATM call + put when IV exceeds a configurable threshold |
| `PCRStrategy` | Contrarian directional strategy driven by the Put-Call Ratio (PCR) |

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

### 6. Run

```bash
python main.py
```

`main.py` will:
1. Run a **Moving Average Crossover** backtest on RELIANCE equity (using yfinance or Dhan data).
2. Run an **option chain demo** (requires valid Dhan credentials) that fetches the Nifty 50 chain, prints ATM prices / IV / PCR / Max Pain, and generates entry signals from both the PCR and Short Straddle strategies.

### 7. Run tests

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
├── src/
│   ├── broker/
│   │   └── dhan_broker.py             # Dhan API wrapper (dhanhq v2.1+)
│   ├── data/
│   │   ├── data_fetcher.py            # Historical & live OHLCV data
│   │   └── option_chain.py            # Option chain fetcher & analytics
│   ├── strategy/
│   │   ├── base_strategy.py           # Abstract strategy base class
│   │   ├── example_strategies.py      # MA Cross, RSI, Bollinger Bands
│   │   └── option_chain_strategy.py   # OptionChainStrategy, ShortStraddle, PCRStrategy
│   ├── backtesting/
│   │   └── backtester.py              # Event-driven backtesting engine
│   ├── portfolio/
│   │   └── portfolio_manager.py       # Position & risk management
│   └── utils/
│       └── logger.py                  # Logging & config helpers
├── tests/
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   └── test_option_chain.py           # Option chain unit tests
├── main.py                            # Entry point (backtest + option chain demo)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Disclaimer

This software is for educational and research purposes only. It is **not** financial advice. Trading involves significant risk of loss. Always test thoroughly in paper-trade mode before deploying real capital.