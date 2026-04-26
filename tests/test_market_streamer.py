"""Tests for MarketDataStreamer (real-time WebSocket feed wrapper)."""

from unittest.mock import MagicMock, patch

import pytest

from src.data.market_streamer import FULL, QUOTE, TICKER, MarketDataStreamer

# Tolerance in days for date-range assertions to account for test timing jitter.
_DATE_DELTA_TOLERANCE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_context(client_id: str = "test_client", token: str = "test_token"):
    """Return a mock DhanContext."""
    ctx = MagicMock()
    ctx.get_client_id.return_value = client_id
    ctx.get_access_token.return_value = token
    return ctx


def _make_streamer(instruments=None, on_tick=None) -> MarketDataStreamer:
    return MarketDataStreamer(
        dhan_context=_mock_context(),
        instruments=instruments,
        on_tick=on_tick,
    )


# ---------------------------------------------------------------------------
# Instrument normalisation
# ---------------------------------------------------------------------------


class TestNormalizeInstrument:
    def test_string_segment_tuple3(self):
        result = MarketDataStreamer._normalize_instrument(("NSE_EQ", "2885", QUOTE))
        assert result == (1, "2885", QUOTE)

    def test_string_segment_tuple2_defaults_ticker(self):
        result = MarketDataStreamer._normalize_instrument(("NSE_FNO", "52175"))
        assert result == (2, "52175", TICKER)

    def test_integer_segment_preserved(self):
        result = MarketDataStreamer._normalize_instrument((4, "500180", FULL))
        assert result == (4, "500180", FULL)

    def test_idx_i_maps_to_zero(self):
        result = MarketDataStreamer._normalize_instrument(("IDX_I", "13", TICKER))
        assert result == (0, "13", TICKER)

    def test_mcx_comm_maps_to_five(self):
        result = MarketDataStreamer._normalize_instrument(("MCX_COMM", "221", TICKER))
        assert result == (5, "221", TICKER)

    def test_unknown_segment_defaults_to_nse(self):
        # Unknown string segment falls back to 1 (NSE)
        result = MarketDataStreamer._normalize_instrument(("UNKNOWN_SEG", "999", TICKER))
        assert result == (1, "999", TICKER)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_no_instruments(self):
        streamer = _make_streamer()
        assert streamer._instruments == []

    def test_instruments_are_normalised(self):
        streamer = _make_streamer(instruments=[("NSE_EQ", "2885", QUOTE)])
        assert streamer._instruments == [(1, "2885", QUOTE)]

    def test_duplicate_instruments_deduplicated(self):
        streamer = _make_streamer(
            instruments=[
                ("NSE_EQ", "2885", TICKER),
                ("NSE_EQ", "2885", TICKER),
            ]
        )
        assert len(streamer._instruments) == 1

    def test_on_tick_stored(self):
        cb = MagicMock()
        streamer = _make_streamer(on_tick=cb)
        assert streamer._on_tick is cb


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe (before start)
# ---------------------------------------------------------------------------


class TestSubscribeBeforeStart:
    def test_subscribe_adds_instrument(self):
        streamer = _make_streamer()
        streamer.subscribe("2885", "NSE_EQ", TICKER)
        assert (1, "2885", TICKER) in streamer._instruments

    def test_subscribe_no_duplicate(self):
        streamer = _make_streamer()
        streamer.subscribe("2885", "NSE_EQ", TICKER)
        streamer.subscribe("2885", "NSE_EQ", TICKER)
        assert streamer._instruments.count((1, "2885", TICKER)) == 1

    def test_unsubscribe_removes_instrument(self):
        streamer = _make_streamer(instruments=[("NSE_EQ", "2885", TICKER)])
        streamer.unsubscribe("2885", "NSE_EQ", TICKER)
        assert (1, "2885", TICKER) not in streamer._instruments

    def test_unsubscribe_nonexistent_is_safe(self):
        streamer = _make_streamer()
        streamer.unsubscribe("9999", "NSE_EQ", TICKER)  # should not raise


# ---------------------------------------------------------------------------
# Tick cache: get_ltp, get_tick, get_depth
# ---------------------------------------------------------------------------


class TestTickCache:
    def _streamer_with_ticks(self) -> MarketDataStreamer:
        streamer = _make_streamer()
        # Inject ticks directly into the cache
        streamer._ticks[("2885", "NSE_EQ")] = {
            "type": "Ticker Data",
            "exchange_segment": 1,
            "security_id": 2885,
            "LTP": "2500.50",
            "LTT": "10:30:00",
        }
        streamer._ticks[("13", "IDX_I")] = {
            "type": "Full Data",
            "exchange_segment": 0,
            "security_id": 13,
            "LTP": "22100.00",
            "depth": [
                {
                    "bid_price": "22099.00",
                    "bid_quantity": 50,
                    "bid_orders": 3,
                    "ask_price": "22101.00",
                    "ask_quantity": 40,
                    "ask_orders": 2,
                }
            ],
        }
        return streamer

    def test_get_ltp_returns_float(self):
        streamer = self._streamer_with_ticks()
        assert streamer.get_ltp("2885", "NSE_EQ") == pytest.approx(2500.50)

    def test_get_ltp_unknown_security_returns_zero(self):
        streamer = self._streamer_with_ticks()
        assert streamer.get_ltp("9999", "NSE_EQ") == 0.0

    def test_get_tick_returns_dict(self):
        streamer = self._streamer_with_ticks()
        tick = streamer.get_tick("2885", "NSE_EQ")
        assert tick["LTP"] == "2500.50"

    def test_get_tick_unknown_returns_empty(self):
        streamer = self._streamer_with_ticks()
        assert streamer.get_tick("9999", "NSE_EQ") == {}

    def test_get_depth_returns_list(self):
        streamer = self._streamer_with_ticks()
        depth = streamer.get_depth("13", "IDX_I")
        assert isinstance(depth, list)
        assert len(depth) == 1
        assert depth[0]["bid_price"] == "22099.00"

    def test_get_depth_no_depth_data_returns_empty(self):
        streamer = self._streamer_with_ticks()
        # "2885" has Ticker Data (no depth)
        assert streamer.get_depth("2885", "NSE_EQ") == []

    def test_get_ltp_by_integer_segment(self):
        streamer = self._streamer_with_ticks()
        # Segment 1 == "NSE_EQ"
        assert streamer.get_ltp("2885", 1) == pytest.approx(2500.50)


# ---------------------------------------------------------------------------
# Internal _on_message callback
# ---------------------------------------------------------------------------


class TestOnMessage:
    def test_tick_cached_after_on_message(self):
        streamer = _make_streamer()
        data = {
            "type": "Ticker Data",
            "exchange_segment": 1,
            "security_id": 2885,
            "LTP": "2600.00",
            "LTT": "11:00:00",
        }
        streamer._on_message(None, data)
        assert streamer.get_ltp("2885", "NSE_EQ") == pytest.approx(2600.0)

    def test_user_on_tick_called(self):
        cb = MagicMock()
        streamer = _make_streamer(on_tick=cb)
        data = {"exchange_segment": 1, "security_id": 1333, "LTP": "1800.00"}
        streamer._on_message(None, data)
        cb.assert_called_once_with(data)

    def test_empty_data_ignored(self):
        cb = MagicMock()
        streamer = _make_streamer(on_tick=cb)
        streamer._on_message(None, {})
        cb.assert_not_called()

    def test_on_tick_exception_does_not_propagate(self):
        def bad_cb(data):
            raise RuntimeError("boom")

        streamer = _make_streamer(on_tick=bad_cb)
        data = {"exchange_segment": 1, "security_id": 9999, "LTP": "100.00"}
        # Should log a warning and not raise
        streamer._on_message(None, data)

    def test_tick_cached_with_string_security_id(self):
        streamer = _make_streamer()
        data = {"exchange_segment": 2, "security_id": "52175", "LTP": "450.00"}
        streamer._on_message(None, data)
        assert streamer.get_ltp("52175", "NSE_FNO") == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# start() / stop() / is_running()
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_is_running_false_before_start(self):
        streamer = _make_streamer()
        assert not streamer.is_running()

    def test_start_creates_feed_and_thread(self):
        mock_feed = MagicMock()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_feed.start.return_value = mock_thread

        mock_mf_cls = MagicMock(return_value=mock_feed)
        mock_mf_module = MagicMock()
        mock_mf_module.MarketFeed = mock_mf_cls
        mock_mf_module.MarketFeed.NSE = 1

        with patch.dict("sys.modules", {"dhanhq.marketfeed": mock_mf_module}):
            with patch("src.data.market_streamer.marketfeed", mock_mf_module, create=True):
                # Patch the import inside start()
                import importlib
                import sys

                original = sys.modules.get("dhanhq")

                mock_dhanhq = MagicMock()
                mock_dhanhq.marketfeed = mock_mf_module
                sys.modules["dhanhq"] = mock_dhanhq

                try:
                    streamer = _make_streamer(
                        instruments=[("NSE_EQ", "2885", TICKER)]
                    )
                    # Patch the import inside start to use our mock
                    with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                        mock_dhanhq if name == "dhanhq" else __import__(name, *a, **kw)
                    )):
                        # Test is_running after mocking the thread
                        streamer._thread = mock_thread
                        assert streamer.is_running()
                finally:
                    if original is not None:
                        sys.modules["dhanhq"] = original
                    elif "dhanhq" in sys.modules:
                        del sys.modules["dhanhq"]

    def test_stop_resets_feed_and_thread(self):
        streamer = _make_streamer()
        mock_feed = MagicMock()
        streamer._feed = mock_feed
        mock_thread = MagicMock()
        streamer._thread = mock_thread

        streamer.stop()

        mock_feed.close_connection.assert_called_once()
        assert streamer._feed is None
        assert streamer._thread is None

    def test_stop_tolerates_close_exception(self):
        streamer = _make_streamer()
        mock_feed = MagicMock()
        mock_feed.close_connection.side_effect = RuntimeError("socket error")
        streamer._feed = mock_feed
        streamer._thread = MagicMock()

        # Should log warning but not raise
        streamer.stop()
        assert streamer._feed is None

    def test_start_warns_when_already_running(self):
        streamer = _make_streamer()
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        streamer._thread = mock_thread

        # start() should return without raising and leave the thread untouched
        streamer.start()  # should log warning, not raise
        assert streamer._thread is mock_thread


# ---------------------------------------------------------------------------
# DhanBroker.create_streamer integration
# ---------------------------------------------------------------------------


class TestBrokerCreateStreamer:
    def _broker(self):
        from src.broker.dhan_broker import DhanBroker
        broker = DhanBroker.__new__(DhanBroker)
        broker.client_id = "test"
        broker.access_token = "tok"
        broker.paper_trade = True
        broker._paper_orders = []
        broker._order_counter = 1
        broker._dhan = MagicMock()  # simulate connected
        return broker

    def test_create_streamer_returns_market_data_streamer(self):
        broker = self._broker()
        with patch("dhanhq.DhanContext") as mock_ctx_cls:
            mock_ctx_cls.return_value = _mock_context()
            streamer = broker.create_streamer(
                instruments=[("NSE_EQ", "2885", TICKER)]
            )
        assert isinstance(streamer, MarketDataStreamer)

    def test_create_streamer_raises_without_connection(self):
        from src.broker.dhan_broker import DhanBroker
        broker = DhanBroker.__new__(DhanBroker)
        broker.client_id = ""
        broker.access_token = ""
        broker.paper_trade = True
        broker._paper_orders = []
        broker._order_counter = 1
        broker._dhan = None  # no connection

        with pytest.raises(RuntimeError, match="No Dhan API connection"):
            broker.create_streamer()

    def test_create_streamer_forwards_on_tick(self):
        broker = self._broker()
        cb = MagicMock()
        with patch("dhanhq.DhanContext") as mock_ctx_cls:
            mock_ctx_cls.return_value = _mock_context()
            streamer = broker.create_streamer(on_tick=cb)
        assert streamer._on_tick is cb


# ---------------------------------------------------------------------------
# DataFetcher with streamer
# ---------------------------------------------------------------------------


class TestDataFetcherWithStreamer:
    def _fetcher_with_streamer(self, ltp: float):
        from src.data.data_fetcher import DataFetcher

        streamer = MagicMock(spec=MarketDataStreamer)
        streamer.is_running.return_value = True
        streamer.get_ltp.return_value = ltp

        broker = MagicMock()
        broker._dhan = None  # no REST fallback needed
        broker.get_ltp.return_value = 0.0

        return DataFetcher(broker=broker, streamer=streamer), streamer

    def test_get_live_ltp_uses_streamer_when_running(self):
        from src.data.data_fetcher import DataFetcher

        fetcher, streamer = self._fetcher_with_streamer(2750.0)
        ltp = fetcher.get_live_ltp("2885", "NSE_EQ")
        assert ltp == pytest.approx(2750.0)
        streamer.get_ltp.assert_called_once_with(
            security_id="2885", exchange_segment="NSE_EQ"
        )

    def test_get_live_ltp_falls_back_to_broker_when_streamer_returns_zero(self):
        from src.data.data_fetcher import DataFetcher

        fetcher, streamer = self._fetcher_with_streamer(0.0)
        fetcher.broker.get_ltp.return_value = 2800.0
        ltp = fetcher.get_live_ltp("2885", "NSE_EQ")
        assert ltp == pytest.approx(2800.0)

    def test_get_live_ltp_falls_back_to_broker_when_no_streamer(self):
        from src.data.data_fetcher import DataFetcher

        broker = MagicMock()
        broker.get_ltp.return_value = 3000.0
        fetcher = DataFetcher(broker=broker, streamer=None)
        ltp = fetcher.get_live_ltp("2885", "NSE_EQ")
        assert ltp == pytest.approx(3000.0)

    def test_get_live_ltp_returns_zero_when_no_sources(self):
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(broker=None, streamer=None)
        assert fetcher.get_live_ltp("2885", "NSE_EQ") == 0.0

    def test_historical_data_daily_default_window_is_5_years(self):
        from datetime import date

        from src.data.data_fetcher import DataFetcher, _MAX_DAILY_HISTORY_DAYS

        broker = MagicMock()
        broker._dhan = MagicMock()
        broker._dhan.historical_daily_data.return_value = {
            "data": {
                "open": [100.0],
                "high": [105.0],
                "low": [98.0],
                "close": [103.0],
                "volume": [100000],
                "timestamp": ["2024-01-01"],
            }
        }
        fetcher = DataFetcher(broker=broker)
        # interval=0 → daily, should default from_date ≈ 5 years ago
        fetcher.get_historical_data(symbol="RELIANCE", security_id="2885", interval=0)
        call_kwargs = broker._dhan.historical_daily_data.call_args[1]
        from_date_str = call_kwargs["from_date"]
        from_date = date.fromisoformat(from_date_str)
        today = date.today()
        delta = (today - from_date).days
        # Allow ±1 day for test timing
        assert abs(delta - _MAX_DAILY_HISTORY_DAYS) <= _DATE_DELTA_TOLERANCE

    def test_historical_data_intraday_default_window_is_60_days(self):
        from datetime import date

        from src.data.data_fetcher import DataFetcher, _MAX_INTRADAY_HISTORY_DAYS

        broker = MagicMock()
        broker._dhan = MagicMock()
        broker._dhan.intraday_minute_data.return_value = {
            "data": {
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
                "timestamp": [],
            }
        }
        fetcher = DataFetcher(broker=broker)
        fetcher.get_historical_data(symbol="RELIANCE", security_id="2885", interval=1)
        call_kwargs = broker._dhan.intraday_minute_data.call_args[1]
        from_date_str = call_kwargs["from_date"]
        from_date = date.fromisoformat(from_date_str)
        today = date.today()
        delta = (today - from_date).days
        assert abs(delta - _MAX_INTRADAY_HISTORY_DAYS) <= _DATE_DELTA_TOLERANCE
