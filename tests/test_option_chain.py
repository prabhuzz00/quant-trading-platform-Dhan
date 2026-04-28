"""Tests for OptionChainFetcher and option chain strategies."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.broker.dhan_broker import DhanBroker
from src.data.option_chain import OptionChainFetcher
from src.strategy.option_chain_strategy import (
    BearCallSpreadStrategy,
    BearPutSpreadStrategy,
    BullCallSpreadStrategy,
    BullPutSpreadStrategy,
    IronButterflyStrategy,
    IronCondorStrategy,
    LongStraddleStrategy,
    LongStrangleStrategy,
    PCRStrategy,
    ShortStraddleStrategy,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_broker(paper_trade: bool = True) -> DhanBroker:
    """Return a DhanBroker with no live API connection."""
    broker = DhanBroker.__new__(DhanBroker)
    broker.client_id = ""
    broker.access_token = ""
    broker.paper_trade = paper_trade
    broker._paper_orders = []
    broker._order_counter = 1
    broker._dhan = None
    return broker


def _mock_raw_chain(strikes: list[float], spot: float = 22000.0) -> dict:
    """Build a minimal raw option chain dict (as returned by Dhan API)."""
    oc_data = []
    for s in strikes:
        oc_data.append(
            {
                "strike_price": s,
                "call_options": {
                    "security_id": str(int(s * 10)),
                    "last_price": max(spot - s, 0) + 50.0,
                    "oi": 100_000,
                    "volume": 5_000,
                    "iv": 18.0,
                    "delta": 0.55,
                    "theta": -0.05,
                    "vega": 0.10,
                    "gamma": 0.001,
                },
                "put_options": {
                    "security_id": str(int(s * 10) + 1),
                    "last_price": max(s - spot, 0) + 30.0,
                    "oi": 90_000,
                    "volume": 4_000,
                    "iv": 17.5,
                    "delta": -0.45,
                    "theta": -0.04,
                    "vega": 0.09,
                    "gamma": 0.001,
                },
            }
        )
    return {"oc_data": oc_data, "last_price": spot}


STRIKES = [21800.0, 21900.0, 22000.0, 22100.0, 22200.0]
SPOT = 22000.0
RAW_CHAIN = _mock_raw_chain(STRIKES, SPOT)


# ---------------------------------------------------------------------------
# OptionChainFetcher tests
# ---------------------------------------------------------------------------


class TestOptionChainFetcher:
    def _fetcher(self) -> OptionChainFetcher:
        broker = _make_broker()
        # Inject a mocked _dhan so the broker reports it has a connection
        broker._dhan = MagicMock()
        broker._dhan.expiry_list.return_value = {
            "data": {"ExpiryDate": ["2024-11-28", "2024-12-26"]}
        }
        broker._dhan.option_chain.return_value = {"data": RAW_CHAIN}
        return OptionChainFetcher(broker)

    def test_get_expiry_list_returns_dates(self):
        fetcher = self._fetcher()
        expiries = fetcher.get_expiry_list(under_security_id=13)
        assert expiries == ["2024-11-28", "2024-12-26"]

    def test_get_expiry_list_no_connection(self):
        broker = _make_broker()  # _dhan is None
        fetcher = OptionChainFetcher(broker)
        assert fetcher.get_expiry_list(under_security_id=13) == []

    def test_get_option_chain_returns_dataframe(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            expiry="2024-11-28",
        )
        assert isinstance(chain, pd.DataFrame)
        assert len(chain) == len(STRIKES)
        assert set(chain.columns) >= {
            "strike_price",
            "call_ltp",
            "put_ltp",
            "call_oi",
            "put_oi",
            "call_iv",
            "put_iv",
        }

    def test_get_option_chain_sorted_by_strike(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        assert list(chain["strike_price"]) == sorted(STRIKES)

    def test_get_option_chain_empty_on_no_connection(self):
        broker = _make_broker()
        fetcher = OptionChainFetcher(broker)
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        assert chain.empty

    def test_get_spot_price(self):
        fetcher = self._fetcher()
        spot = fetcher.get_spot_price(13, "IDX_I", "2024-11-28")
        assert spot == SPOT

    def test_get_atm_options_returns_correct_strike(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        atm = fetcher.get_atm_options(chain, spot_price=22050.0)
        assert atm["strike_price"] == 22000.0  # closest to 22050

    def test_get_atm_options_empty_chain(self):
        fetcher = self._fetcher()
        result = fetcher.get_atm_options(pd.DataFrame(), spot_price=22000.0)
        assert result == {}

    def test_get_strikes_near_atm(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        near = fetcher.get_strikes_near_atm(chain, spot_price=22000.0, n_strikes=1)
        # ATM=22000 (index 2), n=1 → strikes at indices 1,2,3 = 21900, 22000, 22100
        assert len(near) == 3
        assert 22000.0 in near["strike_price"].values

    def test_calculate_pcr(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        pcr = fetcher.calculate_pcr(chain)
        expected = (90_000 * len(STRIKES)) / (100_000 * len(STRIKES))
        assert pcr == pytest.approx(expected)

    def test_calculate_pcr_empty_chain(self):
        fetcher = self._fetcher()
        assert fetcher.calculate_pcr(pd.DataFrame()) == 0.0

    def test_calculate_pcr_zero_call_oi(self):
        chain = pd.DataFrame({"call_oi": [0, 0], "put_oi": [1000, 2000]})
        fetcher = OptionChainFetcher(_make_broker())
        assert fetcher.calculate_pcr(chain) == 0.0

    def test_get_max_pain(self):
        fetcher = self._fetcher()
        chain = fetcher.get_option_chain(13, "IDX_I", "2024-11-28")
        max_pain = fetcher.get_max_pain(chain)
        # Max pain must be one of the strikes
        assert max_pain in STRIKES

    def test_get_max_pain_empty_chain(self):
        fetcher = OptionChainFetcher(_make_broker())
        assert fetcher.get_max_pain(pd.DataFrame()) == 0.0


# ---------------------------------------------------------------------------
# DhanBroker option chain delegation tests
# ---------------------------------------------------------------------------


class TestDhanBrokerOptionChain:
    def test_get_expiry_list_no_connection(self):
        broker = _make_broker()
        assert broker.get_expiry_list(13) == []

    def test_get_option_chain_no_connection(self):
        broker = _make_broker()
        assert broker.get_option_chain(13, "IDX_I", "2024-11-28") == {}

    def test_get_market_quote_no_connection(self):
        broker = _make_broker()
        assert broker.get_market_quote({"NSE_FNO": [52175]}) == {}

    def test_get_ltp_no_connection(self):
        broker = _make_broker()
        assert broker.get_ltp("52175", "NSE_FNO") == 0.0

    def test_get_expiry_list_with_mock_dhan(self):
        broker = _make_broker()
        broker._dhan = MagicMock()
        broker._dhan.expiry_list.return_value = {
            "data": {"ExpiryDate": ["2024-11-28"]}
        }
        result = broker.get_expiry_list(under_security_id=13)
        assert result == ["2024-11-28"]

    def test_get_option_chain_with_mock_dhan(self):
        broker = _make_broker()
        broker._dhan = MagicMock()
        broker._dhan.option_chain.return_value = {"data": RAW_CHAIN}
        result = broker.get_option_chain(13, "IDX_I", "2024-11-28")
        assert "oc_data" in result

    def test_place_option_order_paper_mode(self):
        broker = _make_broker(paper_trade=True)
        result = broker.place_option_order(
            security_id="52175",
            transaction_type="BUY",
            quantity=50,
            price=100.0,
        )
        assert result["status"] == "TRADED"
        assert result["order_id"].startswith("PAPER-")

    def test_get_market_quote_with_mock_dhan(self):
        broker = _make_broker()
        broker._dhan = MagicMock()
        broker._dhan.ticker_data.return_value = {"data": {"NSE_FNO": [{"last_price": 200.0}]}}
        result = broker.get_market_quote({"NSE_FNO": [52175]}, mode="ticker")
        assert "data" in result

    def test_get_ltp_with_mock_dhan(self):
        broker = _make_broker()
        broker._dhan = MagicMock()
        broker._dhan.ticker_data.return_value = {
            "data": {"NSE_FNO": [{"last_price": 150.5}]}
        }
        ltp = broker.get_ltp("52175", "NSE_FNO")
        assert ltp == pytest.approx(150.5)


# ---------------------------------------------------------------------------
# ShortStraddleStrategy tests
# ---------------------------------------------------------------------------


class TestShortStraddleStrategy:
    def _strategy_with_chain(self, call_iv: float = 18.0, put_iv: float = 17.5):
        """Return a ShortStraddleStrategy wired to a mock broker with live chain."""
        strategy = ShortStraddleStrategy(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            spot_price=SPOT,
            min_iv_threshold=15.0,
            quantity=1,
        )
        broker = _make_broker(paper_trade=True)
        broker._dhan = MagicMock()
        broker._dhan.expiry_list.return_value = {
            "data": {"ExpiryDate": ["2024-11-28"]}
        }
        # Build chain with custom IVs on ATM strike
        raw = _mock_raw_chain(STRIKES, SPOT)
        for rec in raw["oc_data"]:
            if rec["strike_price"] == 22000.0:
                rec["call_options"]["iv"] = call_iv
                rec["put_options"]["iv"] = put_iv
        broker._dhan.option_chain.return_value = {"data": raw}
        strategy.attach_broker(broker)
        return strategy

    def test_entry_signal_when_iv_above_threshold(self):
        strategy = self._strategy_with_chain(call_iv=20.0, put_iv=19.0)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "SELL"
        # Call leg returned; option_type CE embedded in symbol name
        assert "CE" in signal["symbol"]

    def test_no_signal_when_iv_below_threshold(self):
        strategy = self._strategy_with_chain(call_iv=10.0, put_iv=9.0)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is None

    def test_no_signal_when_position_already_open(self):
        strategy = self._strategy_with_chain(call_iv=20.0, put_iv=20.0)
        strategy.generate_signals(pd.DataFrame())  # open position
        signal = strategy.generate_signals(pd.DataFrame())  # should be None
        assert signal is None

    def test_no_signal_without_broker(self):
        strategy = ShortStraddleStrategy(spot_price=SPOT)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is None


# ---------------------------------------------------------------------------
# PCRStrategy tests
# ---------------------------------------------------------------------------


class TestPCRStrategy:
    def _strategy(self, put_oi: int = 90_000, call_oi: int = 100_000):
        """Return a PCRStrategy wired to a mock broker."""
        strategy = PCRStrategy(
            under_security_id=13,
            under_exchange_segment="IDX_I",
            spot_price=SPOT,
            bullish_pcr=1.5,
            bearish_pcr=0.5,
            quantity=1,
        )
        broker = _make_broker(paper_trade=True)
        broker._dhan = MagicMock()
        broker._dhan.expiry_list.return_value = {
            "data": {"ExpiryDate": ["2024-11-28"]}
        }
        raw = _mock_raw_chain(STRIKES, SPOT)
        # Override OI to control PCR
        for rec in raw["oc_data"]:
            rec["call_options"]["oi"] = call_oi
            rec["put_options"]["oi"] = put_oi
        broker._dhan.option_chain.return_value = {"data": raw}
        strategy.attach_broker(broker)
        return strategy

    def test_buy_call_when_pcr_high(self):
        # PCR = 200_000 / 100_000 = 2.0 > 1.5 → buy calls
        strategy = self._strategy(put_oi=200_000, call_oi=100_000)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "CE" in signal["symbol"]

    def test_buy_put_when_pcr_low(self):
        # PCR = 40_000 / 100_000 = 0.4 < 0.5 → buy puts
        strategy = self._strategy(put_oi=40_000, call_oi=100_000)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "PE" in signal["symbol"]

    def test_no_signal_in_neutral_zone(self):
        # PCR = 90_000 / 100_000 = 0.9 (within 0.5..1.5)
        strategy = self._strategy(put_oi=90_000, call_oi=100_000)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is None

    def test_no_duplicate_signal_in_same_direction(self):
        # PCR > 1.5 → first signal is bullish; second call should be None
        strategy = self._strategy(put_oi=200_000, call_oi=100_000)
        strategy.generate_signals(pd.DataFrame())  # position = +1
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is None

    def test_no_signal_without_broker(self):
        strategy = PCRStrategy(spot_price=SPOT)
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is None


# ---------------------------------------------------------------------------
# Helpers shared by new multi-leg strategy tests
# ---------------------------------------------------------------------------


def _make_broker_with_chain(call_iv: float = 18.0, put_iv: float = 17.5) -> DhanBroker:
    """Return a mock broker with a 5-strike option chain centred on SPOT."""
    broker = _make_broker(paper_trade=True)
    broker._dhan = MagicMock()
    broker._dhan.expiry_list.return_value = {"data": {"ExpiryDate": ["2024-11-28"]}}
    raw = _mock_raw_chain(STRIKES, SPOT)
    for rec in raw["oc_data"]:
        if rec["strike_price"] == 22000.0:
            rec["call_options"]["iv"] = call_iv
            rec["put_options"]["iv"] = put_iv
    broker._dhan.option_chain.return_value = {"data": raw}
    return broker


# ---------------------------------------------------------------------------
# LongStraddleStrategy
# ---------------------------------------------------------------------------


class TestLongStraddleStrategy:
    def test_entry_signal_generated(self):
        strategy = LongStraddleStrategy(spot_price=SPOT)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "CE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = LongStraddleStrategy(spot_price=SPOT)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = LongStraddleStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# LongStrangleStrategy
# ---------------------------------------------------------------------------


class TestLongStrangleStrategy:
    def test_entry_signal_generated(self):
        strategy = LongStrangleStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "CE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = LongStrangleStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = LongStrangleStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# BullCallSpreadStrategy
# ---------------------------------------------------------------------------


class TestBullCallSpreadStrategy:
    def test_entry_signal_generated(self):
        strategy = BullCallSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "CE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = BullCallSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = BullCallSpreadStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# BearPutSpreadStrategy
# ---------------------------------------------------------------------------


class TestBearPutSpreadStrategy:
    def test_entry_signal_generated(self):
        strategy = BearPutSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "BUY"
        assert "PE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = BearPutSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = BearPutSpreadStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# BullPutSpreadStrategy
# ---------------------------------------------------------------------------


class TestBullPutSpreadStrategy:
    def test_entry_signal_generated(self):
        strategy = BullPutSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "SELL"
        assert "PE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = BullPutSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = BullPutSpreadStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# BearCallSpreadStrategy
# ---------------------------------------------------------------------------


class TestBearCallSpreadStrategy:
    def test_entry_signal_generated(self):
        strategy = BearCallSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        signal = strategy.generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "SELL"
        assert "CE" in signal["symbol"]

    def test_no_duplicate_signal_when_position_open(self):
        strategy = BearCallSpreadStrategy(spot_price=SPOT, otm_distance=1)
        strategy.attach_broker(_make_broker_with_chain())
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = BearCallSpreadStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# IronCondorStrategy
# ---------------------------------------------------------------------------


class TestIronCondorStrategy:
    def _strategy(self, call_iv: float = 18.0, put_iv: float = 17.5) -> IronCondorStrategy:
        strategy = IronCondorStrategy(
            spot_price=SPOT,
            short_otm_distance=1,
            long_otm_distance=2,
            min_iv_threshold=15.0,
        )
        strategy.attach_broker(_make_broker_with_chain(call_iv=call_iv, put_iv=put_iv))
        return strategy

    def test_entry_signal_generated(self):
        signal = self._strategy(call_iv=20.0, put_iv=19.0).generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "SELL"
        assert "CE" in signal["symbol"]

    def test_no_signal_when_iv_below_threshold(self):
        signal = self._strategy(call_iv=10.0, put_iv=9.0).generate_signals(pd.DataFrame())
        assert signal is None

    def test_no_duplicate_signal_when_position_open(self):
        strategy = self._strategy(call_iv=20.0, put_iv=20.0)
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = IronCondorStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# IronButterflyStrategy
# ---------------------------------------------------------------------------


class TestIronButterflyStrategy:
    def _strategy(self, call_iv: float = 18.0, put_iv: float = 17.5) -> IronButterflyStrategy:
        strategy = IronButterflyStrategy(
            spot_price=SPOT,
            wing_distance=1,
            min_iv_threshold=15.0,
        )
        strategy.attach_broker(_make_broker_with_chain(call_iv=call_iv, put_iv=put_iv))
        return strategy

    def test_entry_signal_generated(self):
        signal = self._strategy(call_iv=20.0, put_iv=19.0).generate_signals(pd.DataFrame())
        assert signal is not None
        assert signal["action"] == "SELL"
        assert "CE" in signal["symbol"]

    def test_no_signal_when_iv_below_threshold(self):
        signal = self._strategy(call_iv=10.0, put_iv=9.0).generate_signals(pd.DataFrame())
        assert signal is None

    def test_no_duplicate_signal_when_position_open(self):
        strategy = self._strategy(call_iv=20.0, put_iv=20.0)
        strategy.generate_signals(pd.DataFrame())
        assert strategy.generate_signals(pd.DataFrame()) is None

    def test_no_signal_without_broker(self):
        strategy = IronButterflyStrategy(spot_price=SPOT)
        assert strategy.generate_signals(pd.DataFrame()) is None

