"""Quant Trading Platform for Dhan – main entry point.

Usage
-----
Run a paper-trade backtest::

    python main.py

Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables (or a .env
file) before running in live mode and set ``paper_trade: false`` in
config/config.yaml.

Option chain demo
-----------------
When valid Dhan credentials are present the platform also demonstrates live
option chain fetching and the PCR-based option strategy.  Set
``paper_trade: true`` to simulate orders while still pulling live market
data from the Dhan API.
"""

import os

from dotenv import load_dotenv

from src.backtesting.backtester import Backtester
from src.broker.dhan_broker import DhanBroker
from src.data.data_fetcher import DataFetcher
from src.data.option_chain import OptionChainFetcher
from src.strategy.example_strategies import (
    BollingerBandsStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
)
from src.strategy.option_chain_strategy import PCRStrategy, ShortStraddleStrategy
from src.utils.logger import load_config, setup_logging

load_dotenv()


def run_equity_backtest(
    broker: DhanBroker,
    fetcher: DataFetcher,
    config: dict,
    logger,
) -> None:
    """Run a Moving-Average Crossover backtest on RELIANCE equity."""
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

    bt_cfg = config.get("backtesting", {})
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


def run_option_chain_demo(broker: DhanBroker, config: dict, logger) -> None:
    """Demonstrate live option chain fetching and strategy signals.

    Requires valid DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN credentials.
    """
    if broker._dhan is None:
        logger.info(
            "Skipping option chain demo – no Dhan API connection. "
            "Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to enable."
        )
        return

    oc_cfg = config.get("option_chain", {})
    nifty_id = oc_cfg.get("nifty50_security_id", OptionChainFetcher.NIFTY50_ID)
    idx_segment = oc_cfg.get("default_exchange_segment", "IDX_I")

    fetcher = OptionChainFetcher(broker)

    # ------------------------------------------------------------------ #
    #  1. Expiry list                                                      #
    # ------------------------------------------------------------------ #
    logger.info("--- Option Chain Demo: Nifty 50 ---")
    expiries = fetcher.get_expiry_list(
        under_security_id=nifty_id,
        under_exchange_segment=idx_segment,
    )
    if not expiries:
        logger.warning("No expiry dates returned from Dhan API.")
        return
    logger.info("Available expiries: %s", expiries[:5])

    nearest_expiry = expiries[0]

    # ------------------------------------------------------------------ #
    #  2. Option chain snapshot                                            #
    # ------------------------------------------------------------------ #
    chain = fetcher.get_option_chain(
        under_security_id=nifty_id,
        under_exchange_segment=idx_segment,
        expiry=nearest_expiry,
    )
    if chain.empty:
        logger.warning("Empty option chain for expiry=%s.", nearest_expiry)
        return

    spot = fetcher.get_spot_price(
        under_security_id=nifty_id,
        under_exchange_segment=idx_segment,
        expiry=nearest_expiry,
    )
    logger.info("Nifty 50 spot price: %.2f", spot)
    logger.info("Total strikes in chain: %d", len(chain))

    # ------------------------------------------------------------------ #
    #  3. ATM options                                                      #
    # ------------------------------------------------------------------ #
    atm = fetcher.get_atm_options(chain, spot)
    if atm:
        logger.info(
            "ATM strike: %.2f | CE LTP: %.2f (IV=%.1f%%) | PE LTP: %.2f (IV=%.1f%%)",
            atm["strike_price"],
            atm["call"]["ltp"],
            atm["call"]["iv"],
            atm["put"]["ltp"],
            atm["put"]["iv"],
        )

    # ------------------------------------------------------------------ #
    #  4. PCR and Max Pain                                                 #
    # ------------------------------------------------------------------ #
    pcr = fetcher.calculate_pcr(chain)
    max_pain = fetcher.get_max_pain(chain)
    logger.info("Put-Call Ratio (by OI): %.3f", pcr)
    logger.info("Max Pain strike: %.2f", max_pain)

    # ------------------------------------------------------------------ #
    #  5. PCR strategy signal                                              #
    # ------------------------------------------------------------------ #
    pcr_cfg = oc_cfg.get("pcr_strategy", {})
    pcr_strategy = PCRStrategy(
        under_security_id=nifty_id,
        under_exchange_segment=idx_segment,
        spot_price=spot,
        bullish_pcr=pcr_cfg.get("bullish_pcr", 1.5),
        bearish_pcr=pcr_cfg.get("bearish_pcr", 0.5),
        quantity=pcr_cfg.get("quantity", 1),
        product_type=pcr_cfg.get("product_type", "INTRADAY"),
    )
    pcr_strategy.attach_broker(broker)
    signal = pcr_strategy.generate_signals(pd.DataFrame())
    if signal:
        logger.info("PCR strategy signal: %s", signal)
    else:
        logger.info("PCR strategy: no signal (PCR=%.3f is within neutral zone).", pcr)

    # ------------------------------------------------------------------ #
    #  6. Short straddle strategy signal                                   #
    # ------------------------------------------------------------------ #
    ss_cfg = oc_cfg.get("short_straddle", {})
    straddle = ShortStraddleStrategy(
        under_security_id=nifty_id,
        under_exchange_segment=idx_segment,
        spot_price=spot,
        min_iv_threshold=ss_cfg.get("min_iv_threshold", 15.0),
        quantity=ss_cfg.get("quantity", 1),
        product_type=ss_cfg.get("product_type", "INTRADAY"),
    )
    straddle.attach_broker(broker)
    straddle_signal = straddle.generate_signals(pd.DataFrame())
    if straddle_signal:
        logger.info("Short straddle entry signal: %s", straddle_signal)
    else:
        logger.info("Short straddle: no entry signal (IV below threshold or no chain data).")


def main() -> None:
    import pandas as pd  # noqa: PLC0415 – local import to keep top-level clean

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

    # ------------------------------------------------------------------ #
    #  Equity backtest (MA Crossover on RELIANCE)                         #
    # ------------------------------------------------------------------ #
    run_equity_backtest(broker=broker, fetcher=fetcher, config=config, logger=logger)

    # ------------------------------------------------------------------ #
    #  Option chain demo (live data + strategy signals)                   #
    # ------------------------------------------------------------------ #
    run_option_chain_demo(broker=broker, config=config, logger=logger)


if __name__ == "__main__":
    main()

