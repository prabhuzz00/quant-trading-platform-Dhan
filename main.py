"""Quant Trading Platform for Dhan – main entry point.

Usage
-----
Run a paper-trade backtest::

    python main.py

Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables (or a .env
file) before running in live mode and set ``paper_trade: false`` in
config/config.yaml.
"""

import os

from dotenv import load_dotenv

from src.backtesting.backtester import Backtester
from src.broker.dhan_broker import DhanBroker
from src.data.data_fetcher import DataFetcher
from src.strategy.example_strategies import (
    BollingerBandsStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
)
from src.utils.logger import load_config, setup_logging

load_dotenv()


def main() -> None:
    config = load_config("config/config.yaml")

    log_cfg = config.get("logging", {})
    logger = setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    logger.info("=== Dhan Quant Trading Platform ===")

    trade_cfg = config.get("trading", {})
    paper_trade: bool = trade_cfg.get("paper_trade", True)

    broker = DhanBroker(paper_trade=paper_trade)
    fetcher = DataFetcher(broker=broker)

    bt_cfg = config.get("backtesting", {})
    risk_cfg = config.get("risk", {})

    # ------------------------------------------------------------------ #
    #  Example: backtest a Moving-Average Crossover on RELIANCE            #
    # ------------------------------------------------------------------ #
    symbol = "RELIANCE"
    security_id = "2885"  # Dhan security ID for RELIANCE NSE

    logger.info("Fetching historical data for %s …", symbol)
    data = fetcher.get_historical_data(
        symbol=symbol,
        security_id=security_id,
        from_date="2023-01-01",
        to_date="2024-01-01",
    )

    if data.empty:
        logger.error(
            "No data returned for %s. "
            "Ensure yfinance is installed (`pip install yfinance`) or "
            "provide valid Dhan credentials.",
            symbol,
        )
        return

    logger.info("Running MA Crossover backtest on %d bars …", len(data))
    strategy = MovingAverageCrossStrategy(
        symbol=symbol,
        security_id=security_id,
        fast_period=20,
        slow_period=50,
        quantity=1,
    )

    backtester = Backtester(
        strategy=strategy,
        data=data,
        initial_capital=bt_cfg.get("initial_capital", 100_000),
        commission=bt_cfg.get("commission", 0.0003),
        slippage=bt_cfg.get("slippage", 0.0001),
    )

    results = backtester.run()

    logger.info("=== Backtest Results ===")
    for key, value in results.items():
        logger.info("  %-25s %s", key + ":", value)

    trades = backtester.trade_history()
    if not trades.empty:
        logger.info("\nTrade history:\n%s", trades.to_string(index=False))


if __name__ == "__main__":
    main()
