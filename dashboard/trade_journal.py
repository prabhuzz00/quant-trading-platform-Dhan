"""SQLite-backed trade journal for persistent trade history."""

import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "trades.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id       TEXT    NOT NULL DEFAULT '',
    strategy_name     TEXT    NOT NULL DEFAULT '',
    symbol            TEXT    NOT NULL DEFAULT '',
    security_id       TEXT    NOT NULL DEFAULT '',
    action            TEXT    NOT NULL DEFAULT '',
    quantity          INTEGER NOT NULL DEFAULT 0,
    entry_price       REAL    NOT NULL DEFAULT 0,
    exit_price        REAL             DEFAULT 0,
    entry_time        TEXT             DEFAULT '',
    exit_time         TEXT             DEFAULT '',
    pnl               REAL             DEFAULT 0,
    status            TEXT             DEFAULT 'OPEN',
    option_type       TEXT             DEFAULT '',
    exchange_segment  TEXT             DEFAULT '',
    sl_price          REAL             DEFAULT 0,
    target_price      REAL             DEFAULT 0,
    regime            TEXT             DEFAULT '',
    notes             TEXT             DEFAULT ''
);
"""


def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_trade_entry(
    strategy_id: str,
    strategy_name: str,
    symbol: str,
    security_id: str,
    action: str,
    quantity: int,
    entry_price: float,
    option_type: str = "",
    exchange_segment: str = "",
    sl_price: float = 0.0,
    target_price: float = 0.0,
    regime: str = "",
    notes: str = "",
) -> int:
    """Insert a new OPEN trade and return the new trade ID."""
    entry_time = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO trades
              (strategy_id, strategy_name, symbol, security_id, action, quantity,
               entry_price, entry_time, status, option_type, exchange_segment,
               sl_price, target_price, regime, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?)
            """,
            (strategy_id, strategy_name, symbol, security_id, action, quantity,
             entry_price, entry_time, option_type, exchange_segment,
             sl_price, target_price, regime, notes),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def record_trade_exit(
    trade_id: int,
    exit_price: float,
    exit_time: str | None = None,
) -> dict:
    """Mark a trade CLOSED and compute PnL.  Returns the updated trade dict."""
    exit_time = exit_time or datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        if row is None:
            return {}
        trade = _row_to_dict(row)
        action = trade["action"].upper()
        qty    = trade["quantity"]
        ep     = trade["entry_price"]
        pnl    = (exit_price - ep) * qty if action == "BUY" else (ep - exit_price) * qty
        conn.execute(
            """
            UPDATE trades
            SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED'
            WHERE id = ?
            """,
            (exit_price, exit_time, round(pnl, 2), trade_id),
        )
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        return _row_to_dict(updated) if updated else {}
    finally:
        conn.close()


def get_open_trades(strategy_id: str | None = None) -> list[dict]:
    """Return all OPEN trades, optionally filtered by strategy."""
    conn = _get_conn()
    try:
        if strategy_id:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' AND strategy_id = ? ORDER BY entry_time DESC",
                (strategy_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_closed_trades(strategy_id: str | None = None) -> list[dict]:
    """Return all CLOSED trades, optionally filtered by strategy."""
    conn = _get_conn()
    try:
        if strategy_id:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' AND strategy_id = ? ORDER BY exit_time DESC",
                (strategy_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_time DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_all_trades(strategy_id: str | None = None) -> list[dict]:
    """Return all trades, optionally filtered by strategy."""
    conn = _get_conn()
    try:
        if strategy_id:
            rows = conn.execute(
                "SELECT * FROM trades WHERE strategy_id = ? ORDER BY entry_time DESC",
                (strategy_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_strategy_pnl(strategy_id: str) -> dict:
    """Return PnL summary for a single strategy."""
    open_trades   = get_open_trades(strategy_id)
    closed_trades = get_closed_trades(strategy_id)
    realized_pnl   = sum(t["pnl"] for t in closed_trades)
    unrealized_pnl = sum(t["pnl"] for t in open_trades)
    winning = sum(1 for t in closed_trades if t["pnl"] > 0)
    losing  = sum(1 for t in closed_trades if t["pnl"] <= 0)
    win_rate = round(winning / len(closed_trades) * 100, 1) if closed_trades else 0.0
    return {
        "strategy_id":    strategy_id,
        "open_trades":    len(open_trades),
        "closed_trades":  len(closed_trades),
        "total_trades":   len(open_trades) + len(closed_trades),
        "realized_pnl":   round(realized_pnl,   2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "total_pnl":      round(realized_pnl + unrealized_pnl, 2),
        "winning_trades": winning,
        "losing_trades":  losing,
        "win_rate":       win_rate,
    }


def get_all_strategy_stats() -> list[dict]:
    """Return PnL summary for every strategy that has at least one trade."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT strategy_id FROM trades"
        ).fetchall()
        return [get_strategy_pnl(r["strategy_id"]) for r in rows]
    finally:
        conn.close()
