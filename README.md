# Dhan Quant Trading Platform

A modular, Python-based quantitative trading platform built on top of the **[Dhan](https://dhan.co/)** trading API.

## Features

| Module | Description |
|---|---|
| `DhanBroker` | Thin wrapper around the `dhanhq` SDK with paper-trading support |
| `DataFetcher` | OHLCV data from Dhan Historical API or yfinance fallback |
| `BaseStrategy` | Abstract base class for custom strategies |
| `MovingAverageCrossStrategy` | Dual MA crossover (golden/death cross) |
| `RSIStrategy` | RSI mean-reversion |
| `BollingerBandsStrategy` | Bollinger Bands breakout |
| `PortfolioManager` | Position tracking, risk checks, P&L reporting |
| `Backtester` | Event-driven backtesting engine with commission & slippage |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/prabhuzz00/quant-trading-platform-Dhan.git
cd quant-trading-platform-Dhan
pip install -r requirements.txt
pip install yfinance   # optional – needed for data fallback
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN
```

Or export them directly:

```bash
export DHAN_CLIENT_ID=your_client_id
export DHAN_ACCESS_TOKEN=your_access_token
```

### 3. Run an example backtest

```bash
python main.py
```

This will run a Moving Average Crossover backtest on RELIANCE using Yahoo Finance data and print a performance summary.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Configuration

All defaults live in `config/config.yaml`:

```yaml
trading:
  paper_trade: true    # set to false for live orders

risk:
  max_position_size: 0.05   # 5 % of equity per position
  stop_loss_pct: 0.02       # 2 % stop-loss

backtesting:
  initial_capital: 100000
  commission: 0.0003
```

---

## Writing a custom strategy

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

## Project structure

```
quant-trading-platform-Dhan/
├── config/
│   └── config.yaml          # Runtime configuration
├── src/
│   ├── broker/
│   │   └── dhan_broker.py   # Dhan API wrapper
│   ├── data/
│   │   └── data_fetcher.py  # Historical & tick data
│   ├── strategy/
│   │   ├── base_strategy.py
│   │   └── example_strategies.py
│   ├── backtesting/
│   │   └── backtester.py
│   ├── portfolio/
│   │   └── portfolio_manager.py
│   └── utils/
│       └── logger.py
├── tests/
│   ├── test_strategy.py
│   └── test_portfolio.py
├── main.py
├── requirements.txt
└── .env.example
```

---

## Disclaimer

This software is for educational and research purposes only. It is **not** financial advice. Trading involves significant risk of loss. Always test thoroughly in paper-trade mode before deploying real capital.