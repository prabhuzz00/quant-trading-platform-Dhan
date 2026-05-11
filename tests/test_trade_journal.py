"""Tests for dashboard/trade_journal.py using a temporary SQLite database."""

import importlib
import sys
from pathlib import Path

import pytest

import dashboard.trade_journal as tj


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Redirect the trade journal to a fresh temporary database for every test."""
    tmp_data = tmp_path / "data"
    tmp_data.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_data / "trades.db"

    monkeypatch.setattr(tj, "DATA_DIR", tmp_data)
    monkeypatch.setattr(tj, "DB_PATH", tmp_db)

    yield


class TestGetConn:
    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        """_get_conn must create DATA_DIR when it does not yet exist."""
        new_data = tmp_path / "brand_new" / "data"
        monkeypatch.setattr(tj, "DATA_DIR", new_data)
        monkeypatch.setattr(tj, "DB_PATH", new_data / "trades.db")

        conn = tj._get_conn()
        conn.close()

        assert new_data.exists()
        assert (new_data / "trades.db").exists()

    def test_creates_schema_on_first_open(self):
        conn = tj._get_conn()
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_schema_idempotent_on_second_open(self):
        """Opening the DB twice must not raise an error."""
        tj._get_conn().close()
        tj._get_conn().close()


class TestRecordTradeEntry:
    def test_returns_positive_integer(self):
        trade_id = tj.record_trade_entry(
            strategy_id="s1",
            strategy_name="Test Strategy",
            symbol="NIFTY",
            security_id="999",
            action="BUY",
            quantity=50,
            entry_price=100.0,
        )
        assert isinstance(trade_id, int)
        assert trade_id > 0

    def test_trade_is_open_after_entry(self):
        tj.record_trade_entry(
            strategy_id="s1",
            strategy_name="Test Strategy",
            symbol="NIFTY",
            security_id="999",
            action="BUY",
            quantity=50,
            entry_price=100.0,
        )
        open_trades = tj.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["status"] == "OPEN"

    def test_optional_fields_stored_correctly(self):
        tj.record_trade_entry(
            strategy_id="s2",
            strategy_name="Opt Strategy",
            symbol="BANKNIFTY",
            security_id="888",
            action="SELL",
            quantity=25,
            entry_price=200.0,
            option_type="CE",
            exchange_segment="NSE_FNO",
            sl_price=195.0,
            target_price=210.0,
            regime="BULLISH",
            notes="test note",
        )
        trades = tj.get_all_trades("s2")
        assert len(trades) == 1
        t = trades[0]
        assert t["option_type"] == "CE"
        assert t["sl_price"] == 195.0
        assert t["target_price"] == 210.0
        assert t["regime"] == "BULLISH"
        assert t["notes"] == "test note"

    def test_multiple_entries_increment_ids(self):
        id1 = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 1, 10.0)
        id2 = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 1, 10.0)
        assert id2 > id1


class TestRecordTradeExit:
    def test_closes_trade_and_computes_pnl_buy(self):
        trade_id = tj.record_trade_entry(
            strategy_id="s1",
            strategy_name="S",
            symbol="X",
            security_id="1",
            action="BUY",
            quantity=10,
            entry_price=100.0,
        )
        result = tj.record_trade_exit(trade_id, exit_price=110.0)
        assert result["status"] == "CLOSED"
        assert result["pnl"] == pytest.approx(100.0)

    def test_closes_trade_and_computes_pnl_sell(self):
        trade_id = tj.record_trade_entry(
            strategy_id="s1",
            strategy_name="S",
            symbol="X",
            security_id="1",
            action="SELL",
            quantity=10,
            entry_price=100.0,
        )
        result = tj.record_trade_exit(trade_id, exit_price=90.0)
        assert result["status"] == "CLOSED"
        assert result["pnl"] == pytest.approx(100.0)

    def test_trade_moves_from_open_to_closed(self):
        trade_id = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 5, 50.0)
        assert len(tj.get_open_trades()) == 1
        tj.record_trade_exit(trade_id, 60.0)
        assert len(tj.get_open_trades()) == 0
        assert len(tj.get_closed_trades()) == 1

    def test_unknown_trade_id_returns_empty_dict(self):
        result = tj.record_trade_exit(99999, exit_price=100.0)
        assert result == {}

    def test_custom_exit_time_stored(self):
        trade_id = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 1, 10.0)
        result = tj.record_trade_exit(trade_id, 12.0, exit_time="2024-01-01T10:00:00")
        assert result["exit_time"] == "2024-01-01T10:00:00"


class TestGetOpenTrades:
    def test_returns_only_open_trades(self):
        id1 = tj.record_trade_entry("s1", "S", "A", "1", "BUY", 1, 10.0)
        tj.record_trade_entry("s1", "S", "B", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 15.0)

        open_trades = tj.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["symbol"] == "B"

    def test_filter_by_strategy_id(self):
        tj.record_trade_entry("s1", "S1", "A", "1", "BUY", 1, 10.0)
        tj.record_trade_entry("s2", "S2", "B", "2", "BUY", 1, 20.0)

        trades_s1 = tj.get_open_trades("s1")
        assert all(t["strategy_id"] == "s1" for t in trades_s1)
        assert len(trades_s1) == 1

    def test_returns_empty_list_when_none(self):
        assert tj.get_open_trades() == []


class TestGetClosedTrades:
    def test_returns_only_closed_trades(self):
        id1 = tj.record_trade_entry("s1", "S", "A", "1", "BUY", 1, 10.0)
        tj.record_trade_entry("s1", "S", "B", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 15.0)

        closed = tj.get_closed_trades()
        assert len(closed) == 1
        assert closed[0]["symbol"] == "A"

    def test_filter_by_strategy_id(self):
        id1 = tj.record_trade_entry("s1", "S1", "A", "1", "BUY", 1, 10.0)
        id2 = tj.record_trade_entry("s2", "S2", "B", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 12.0)
        tj.record_trade_exit(id2, 22.0)

        closed_s1 = tj.get_closed_trades("s1")
        assert len(closed_s1) == 1
        assert closed_s1[0]["strategy_id"] == "s1"


class TestGetAllTrades:
    def test_returns_both_open_and_closed(self):
        id1 = tj.record_trade_entry("s1", "S", "A", "1", "BUY", 1, 10.0)
        tj.record_trade_entry("s1", "S", "B", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 15.0)

        all_trades = tj.get_all_trades()
        assert len(all_trades) == 2

    def test_filter_by_strategy_id(self):
        id1 = tj.record_trade_entry("s1", "S1", "A", "1", "BUY", 1, 10.0)
        tj.record_trade_entry("s2", "S2", "B", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 15.0)

        trades = tj.get_all_trades("s1")
        assert len(trades) == 1
        assert trades[0]["strategy_id"] == "s1"


class TestGetStrategyPnl:
    def test_summary_keys_present(self):
        summary = tj.get_strategy_pnl("s_missing")
        expected_keys = {
            "strategy_id", "open_trades", "closed_trades", "total_trades",
            "realized_pnl", "unrealized_pnl", "total_pnl",
            "winning_trades", "losing_trades", "win_rate",
        }
        assert expected_keys.issubset(summary.keys())

    def test_realized_pnl_correct(self):
        id1 = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 10, 100.0)
        id2 = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 5, 100.0)
        tj.record_trade_exit(id1, 110.0)  # PnL = +100
        tj.record_trade_exit(id2, 90.0)   # PnL = -50

        summary = tj.get_strategy_pnl("s1")
        assert summary["realized_pnl"] == pytest.approx(50.0)
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1

    def test_win_rate_zero_when_no_closed_trades(self):
        tj.record_trade_entry("s1", "S", "X", "1", "BUY", 1, 10.0)
        summary = tj.get_strategy_pnl("s1")
        assert summary["win_rate"] == 0.0

    def test_win_rate_100_when_all_winning(self):
        id1 = tj.record_trade_entry("s1", "S", "X", "1", "BUY", 1, 10.0)
        tj.record_trade_exit(id1, 20.0)
        summary = tj.get_strategy_pnl("s1")
        assert summary["win_rate"] == pytest.approx(100.0)


class TestGetAllStrategyStats:
    def test_returns_stats_for_each_strategy(self):
        id1 = tj.record_trade_entry("s1", "S1", "X", "1", "BUY", 1, 10.0)
        id2 = tj.record_trade_entry("s2", "S2", "Y", "2", "BUY", 1, 20.0)
        tj.record_trade_exit(id1, 12.0)

        stats = tj.get_all_strategy_stats()
        strategy_ids = {s["strategy_id"] for s in stats}
        assert {"s1", "s2"}.issubset(strategy_ids)

    def test_returns_empty_when_no_trades(self):
        stats = tj.get_all_strategy_stats()
        assert stats == []
