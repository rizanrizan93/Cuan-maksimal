
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

DEFAULT_STARTING_CAPITAL = 100_000_000.0
DEFAULT_ACCOUNT_ID = "default"
DEFAULT_CURRENCY = "IDR"
DEFAULT_STATE_FILENAME = "portfolio_state.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _env_db_url() -> str:
    for key in ("PORTFOLIO_DB_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _default_state_path() -> Path:
    override = os.environ.get("PORTFOLIO_STATE_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_project_root() / DEFAULT_STATE_FILENAME).resolve()


def get_state_path() -> Path:
    return _default_state_path()


def _backend_kind() -> str:
    db_url = _env_db_url().lower()
    if not db_url:
        return "sqlite"
    if db_url.startswith("sqlite"):
        return "sqlite"
    if "supabase" in db_url or db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
        return "postgres"
    if "postgres" in db_url:
        return "postgres"
    return "postgres"


def get_backend_label() -> str:
    kind = _backend_kind()
    if kind == "sqlite":
        return "local-sqlite"
    db_url = _env_db_url().lower()
    if "supabase" in db_url:
        return "supabase-postgres"
    return "remote-postgres"


def get_backend_warning() -> str:
    label = get_backend_label()
    if label == "local-sqlite":
        return (
            "Backend saat ini memakai SQLite lokal. Cocok untuk dev dan backup, "
            "tetapi tidak ideal sebagai satu-satunya storage untuk live money di Streamlit deploy."
        )
    return (
        "Backend eksternal aktif. Ini lebih cocok untuk live ledger, audit trail, "
        "dan reload-safe state." 
    )


def _postgres_dsn() -> str:
    db_url = _env_db_url().strip()
    if not db_url:
        raise RuntimeError("PORTFOLIO_DB_URL / SUPABASE_DB_URL / DATABASE_URL belum di-set")
    low = db_url.lower()
    if low.startswith("sqlite"):
        raise RuntimeError("DB URL menunjuk ke SQLite, bukan Postgres")
    if "sslmode=" not in low:
        sep = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{sep}sslmode=require"
    return db_url


@contextmanager
def _connect() -> Iterator[Any]:
    kind = _backend_kind()
    if kind == "postgres":
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except Exception as exc:  # pragma: no cover - dependency issue
            raise ImportError(
                "Backend Postgres aktif, tetapi psycopg2-binary belum terpasang. "
                "Tambahkan psycopg2-binary ke requirements.txt."
            ) from exc

        conn = psycopg2.connect(_postgres_dsn())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    path = get_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _normalize_sql(sql: str) -> str:
    return sql.replace("?", "%s") if _backend_kind() == "postgres" else sql


def _fetch_rows(sql: str, params: tuple[Any, ...] | list[Any] = ()) -> list[Any]:
    kind = _backend_kind()
    with _connect() as conn:
        if kind == "postgres":
            cur = conn.cursor()
            cur.execute(_normalize_sql(sql), params)
            rows = cur.fetchall()
            cur.close()
            return rows
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows


def _fetch_df(sql: str, params: tuple[Any, ...] | list[Any] = ()) -> pd.DataFrame:
    rows = _fetch_rows(sql, params)
    if not rows:
        return pd.DataFrame()
    first = rows[0]
    if isinstance(first, Mapping):
        return pd.DataFrame([dict(r) for r in rows])
    if hasattr(first, "keys"):
        return pd.DataFrame([dict(r) for r in rows])
    with _connect() as conn:
        cur = conn.cursor()
        try:
            cur.execute(_normalize_sql(sql), params)
            cols = [desc[0] for desc in cur.description] if cur.description else []
        finally:
            cur.close()
    return pd.DataFrame(rows, columns=cols)




def _row_to_dict(row: Any, columns: list[str] | None = None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return dict(row)
    if columns:
        return dict(zip(columns, row))
    return {}

def init_store() -> None:
    """Create required tables if they don't exist yet."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS accounts (
            account_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            currency TEXT NOT NULL,
            starting_capital REAL NOT NULL,
            cash REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            mark_price REAL,
            stop_loss REAL,
            target_1 REAL,
            target_2 REAL,
            source TEXT,
            strategy TEXT,
            signal_hash TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            order_type TEXT NOT NULL,
            limit_price REAL,
            stop_price REAL,
            status TEXT NOT NULL,
            source TEXT,
            signal_hash TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            filled_price REAL,
            filled_qty INTEGER,
            filled_at TEXT,
            broker_ref TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fills (
            fill_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            fill_price REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            fill_time TEXT NOT NULL,
            venue TEXT,
            notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trade_journal (
            journal_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            setup_stage TEXT NOT NULL,
            validity_ok INTEGER NOT NULL,
            validity_reason TEXT,
            next_action TEXT,
            decision TEXT,
            setup_kind TEXT,
            score REAL,
            ifs_score REAL,
            catalyst_score REAL,
            tradeability_score REAL,
            entry_price REAL,
            stop_price REAL,
            target_1 REAL,
            target_2 REAL,
            risk_reward_1 REAL,
            risk_reward_2 REAL,
            value_traded_20d REAL,
            spread_proxy_20d REAL,
            gap_proxy_20d REAL,
            notes TEXT,
            lifecycle_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    ]
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_orders_account_created ON orders(account_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_fills_account_time ON fills(account_id, fill_time DESC)",
        "CREATE INDEX IF NOT EXISTS idx_events_account_time ON events(account_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trade_journal_account_time ON trade_journal(account_id, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_trade_journal_symbol_time ON trade_journal(symbol, updated_at DESC)",
    ]
    with _connect() as conn:
        cur = conn.cursor()
        for stmt in statements + index_statements:
            cur.execute(_normalize_sql(stmt))
        cur.close()


def ensure_account(
    account_id: str = DEFAULT_ACCOUNT_ID,
    label: str = "Main",
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    currency: str = DEFAULT_CURRENCY,
) -> None:
    init_store()
    now = _utc_now()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(_normalize_sql("SELECT account_id FROM accounts WHERE account_id = ?"), (account_id,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                _normalize_sql(
                    """
                    INSERT INTO accounts (account_id, label, currency, starting_capital, cash, realized_pnl, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                    """
                ),
                (account_id, label, currency, float(starting_capital), float(starting_capital), now, now),
            )
        else:
            cur.execute(
                _normalize_sql(
                    """
                    UPDATE accounts
                    SET label = COALESCE(?, label),
                        currency = COALESCE(?, currency),
                        updated_at = ?
                    WHERE account_id = ?
                    """
                ),
                (label, currency, now, account_id),
            )
        cur.close()


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def record_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    account_id: str = DEFAULT_ACCOUNT_ID,
    symbol: str | None = None,
) -> str:
    init_store()
    payload = payload or {}
    event_id = f"evt_{_hash_payload({'event_type': event_type, 'payload': payload, 'symbol': symbol, 'time': _utc_now()})}"
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _normalize_sql(
                """
                INSERT INTO events (event_id, account_id, event_type, symbol, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id) DO UPDATE SET
                    account_id = excluded.account_id,
                    event_type = excluded.event_type,
                    symbol = excluded.symbol,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """
            ),
            (event_id, account_id, event_type, symbol, json.dumps(payload, ensure_ascii=False, default=str), _utc_now()),
        )
        cur.close()
    return event_id


def list_events(account_id: str = DEFAULT_ACCOUNT_ID, limit: int = 100) -> pd.DataFrame:
    init_store()
    return _fetch_df(
        """
        SELECT * FROM events
        WHERE account_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (account_id, int(limit)),
    )


def list_positions() -> pd.DataFrame:
    init_store()
    return _fetch_df("SELECT * FROM positions ORDER BY updated_at DESC, symbol ASC")


def list_orders(limit: int = 200, account_id: str = DEFAULT_ACCOUNT_ID) -> pd.DataFrame:
    init_store()
    return _fetch_df(
        """
        SELECT * FROM orders
        WHERE account_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (account_id, int(limit)),
    )


def list_fills(limit: int = 200, account_id: str = DEFAULT_ACCOUNT_ID) -> pd.DataFrame:
    init_store()
    return _fetch_df(
        """
        SELECT * FROM fills
        WHERE account_id = ?
        ORDER BY fill_time DESC
        LIMIT ?
        """,
        (account_id, int(limit)),
    )


def list_trade_journal(limit: int = 200, account_id: str = DEFAULT_ACCOUNT_ID) -> pd.DataFrame:
    init_store()
    return _fetch_df(
        """
        SELECT * FROM trade_journal
        WHERE account_id = ?
        ORDER BY updated_at DESC, symbol ASC
        LIMIT ?
        """,
        (account_id, int(limit)),
    )


def upsert_trade_journal(
    symbol: str,
    *,
    scan_date: str,
    setup_stage: str,
    validity_ok: bool,
    validity_reason: str = "",
    next_action: str = "",
    decision: str = "",
    setup_kind: str = "",
    score: float | None = None,
    ifs_score: float | None = None,
    catalyst_score: float | None = None,
    tradeability_score: float | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_1: float | None = None,
    target_2: float | None = None,
    risk_reward_1: float | None = None,
    risk_reward_2: float | None = None,
    value_traded_20d: float | None = None,
    spread_proxy_20d: float | None = None,
    gap_proxy_20d: float | None = None,
    notes: str = "",
    lifecycle_json: dict[str, Any] | None = None,
    account_id: str = DEFAULT_ACCOUNT_ID,
    source: str = "scanner",
) -> str:
    """Persist a trade journal snapshot for scanner review and post-trade analysis."""
    init_store()
    symbol = _normalize_symbol(symbol)
    scan_date = str(scan_date).strip() or _utc_now()[:10]
    setup_stage = str(setup_stage or "UNKNOWN").strip().upper()
    journal_id = f"jr_{_hash_payload({'symbol': symbol, 'scan_date': scan_date, 'source': source})}"
    now = _utc_now()
    payload = json.dumps(lifecycle_json or {}, ensure_ascii=False, default=str)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _normalize_sql(
                """
                INSERT INTO trade_journal
                (journal_id, account_id, symbol, scan_date, setup_stage, validity_ok, validity_reason, next_action, decision, setup_kind, score, ifs_score, catalyst_score, tradeability_score, entry_price, stop_price, target_1, target_2, risk_reward_1, risk_reward_2, value_traded_20d, spread_proxy_20d, gap_proxy_20d, notes, lifecycle_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (journal_id) DO UPDATE SET
                    account_id = excluded.account_id,
                    symbol = excluded.symbol,
                    scan_date = excluded.scan_date,
                    setup_stage = excluded.setup_stage,
                    validity_ok = excluded.validity_ok,
                    validity_reason = excluded.validity_reason,
                    next_action = excluded.next_action,
                    decision = excluded.decision,
                    setup_kind = excluded.setup_kind,
                    score = excluded.score,
                    ifs_score = excluded.ifs_score,
                    catalyst_score = excluded.catalyst_score,
                    tradeability_score = excluded.tradeability_score,
                    entry_price = excluded.entry_price,
                    stop_price = excluded.stop_price,
                    target_1 = excluded.target_1,
                    target_2 = excluded.target_2,
                    risk_reward_1 = excluded.risk_reward_1,
                    risk_reward_2 = excluded.risk_reward_2,
                    value_traded_20d = excluded.value_traded_20d,
                    spread_proxy_20d = excluded.spread_proxy_20d,
                    gap_proxy_20d = excluded.gap_proxy_20d,
                    notes = excluded.notes,
                    lifecycle_json = excluded.lifecycle_json,
                    updated_at = excluded.updated_at
                """
            ),
            (
                journal_id,
                account_id,
                symbol,
                scan_date,
                setup_stage,
                int(bool(validity_ok)),
                validity_reason,
                next_action,
                decision,
                setup_kind,
                score,
                ifs_score,
                catalyst_score,
                tradeability_score,
                entry_price,
                stop_price,
                target_1,
                target_2,
                risk_reward_1,
                risk_reward_2,
                value_traded_20d,
                spread_proxy_20d,
                gap_proxy_20d,
                notes,
                payload,
                now,
                now,
            ),
        )
        cur.close()
    record_event(
        "trade_journal_upserted",
        {"journal_id": journal_id, "symbol": symbol, "scan_date": scan_date, "setup_stage": setup_stage, "validity_ok": bool(validity_ok)},
        account_id=account_id,
        symbol=symbol,
    )
    return journal_id


def get_account_summary(account_id: str = DEFAULT_ACCOUNT_ID) -> dict[str, Any]:
    init_store()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(_normalize_sql("SELECT * FROM accounts WHERE account_id = ?"), (account_id,))
        account = cur.fetchone()
        if account is None:
            cur.close()
            ensure_account(account_id=account_id)
            with _connect() as conn2:
                cur2 = conn2.cursor()
                cur2.execute(_normalize_sql("SELECT * FROM accounts WHERE account_id = ?"), (account_id,))
                account = cur2.fetchone()
                cols = [d[0] for d in cur2.description] if cur2.description else None
                cur2.close()
        else:
            cols = [d[0] for d in cur.description] if cur.description else None
        positions = _fetch_df("SELECT * FROM positions ORDER BY symbol ASC")
        cur.close()

    acc = _row_to_dict(account, cols if 'cols' in locals() else None)

    cash = float(acc.get("cash", 0.0))
    starting_capital = float(acc.get("starting_capital", DEFAULT_STARTING_CAPITAL))
    realized_pnl = float(acc.get("realized_pnl", 0.0))

    gross_mark = 0.0
    gross_cost = 0.0
    open_positions = 0
    if not positions.empty:
        for _, row in positions.iterrows():
            qty = int(row.get("qty", 0) or 0)
            if qty <= 0:
                continue
            open_positions += 1
            avg_price = float(row.get("avg_price", 0) or 0)
            mark_price = row.get("mark_price", None)
            mp = float(mark_price) if mark_price is not None and pd.notna(mark_price) else avg_price
            gross_mark += qty * mp
            gross_cost += qty * avg_price

    equity = cash + gross_mark
    unrealized_pnl = gross_mark - gross_cost
    total_pnl = realized_pnl + unrealized_pnl
    return {
        "account_id": account_id,
        "label": acc.get("label", "Main"),
        "currency": acc.get("currency", DEFAULT_CURRENCY),
        "starting_capital": starting_capital,
        "cash": cash,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "open_positions": open_positions,
        "state_path": str(get_state_path()),
        "backend": get_backend_label(),
        "warning": get_backend_warning(),
        "updated_at": acc.get("updated_at"),
    }


def _normalize_side(side: str) -> str:
    s = str(side or "").strip().upper()
    if s in {"BUY", "LONG", "BULL"}:
        return "BUY"
    if s in {"SELL", "CLOSE", "EXIT"}:
        return "SELL"
    raise ValueError(f"Unsupported side: {side!r}")


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def upsert_position(
    symbol: str,
    side: str,
    qty: int,
    avg_price: float,
    *,
    mark_price: float | None = None,
    stop_loss: float | None = None,
    target_1: float | None = None,
    target_2: float | None = None,
    source: str = "manual",
    strategy: str = "",
    signal_hash: str = "",
    notes: str = "",
) -> None:
    init_store()
    symbol = _normalize_symbol(symbol)
    side = _normalize_side(side)
    qty = int(qty)
    if qty <= 0:
        raise ValueError("qty must be positive")
    now = _utc_now()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(_normalize_sql("SELECT * FROM positions WHERE symbol = ?"), (symbol,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                _normalize_sql(
                    """
                    INSERT INTO positions
                    (symbol, side, qty, avg_price, mark_price, stop_loss, target_1, target_2, source, strategy, signal_hash, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                ),
                (symbol, side, qty, float(avg_price), mark_price, stop_loss, target_1, target_2, source, strategy, signal_hash, notes, now),
            )
        else:
            cur.execute(
                _normalize_sql(
                    """
                    UPDATE positions
                    SET side = ?,
                        qty = ?,
                        avg_price = ?,
                        mark_price = COALESCE(?, mark_price),
                        stop_loss = COALESCE(?, stop_loss),
                        target_1 = COALESCE(?, target_1),
                        target_2 = COALESCE(?, target_2),
                        source = COALESCE(?, source),
                        strategy = COALESCE(?, strategy),
                        signal_hash = COALESCE(?, signal_hash),
                        notes = COALESCE(?, notes),
                        updated_at = ?
                    WHERE symbol = ?
                    """
                ),
                (side, qty, float(avg_price), mark_price, stop_loss, target_1, target_2, source, strategy, signal_hash, notes, now, symbol),
            )
        cur.close()


def close_position(symbol: str) -> None:
    init_store()
    symbol = _normalize_symbol(symbol)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(_normalize_sql("DELETE FROM positions WHERE symbol = ?"), (symbol,))
        cur.close()


def update_position_mark(symbol: str, mark_price: float) -> None:
    init_store()
    symbol = _normalize_symbol(symbol)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _normalize_sql("UPDATE positions SET mark_price = ?, updated_at = ? WHERE symbol = ?"),
            (float(mark_price), _utc_now(), symbol),
        )
        cur.close()


def create_order(
    symbol: str,
    side: str,
    qty: int,
    *,
    order_type: str = "LIMIT",
    limit_price: float | None = None,
    stop_price: float | None = None,
    status: str = "PLANNED",
    account_id: str = DEFAULT_ACCOUNT_ID,
    source: str = "scanner",
    signal_hash: str = "",
    notes: str = "",
    broker_ref: str = "",
) -> str:
    init_store()
    symbol = _normalize_symbol(symbol)
    side = _normalize_side(side)
    qty = int(qty)
    if qty <= 0:
        raise ValueError("qty must be positive")
    order_id = f"ord_{_hash_payload({'symbol': symbol, 'side': side, 'qty': qty, 'limit_price': limit_price, 'stop_price': stop_price, 'notes': notes, 'time': _utc_now()})}"
    now = _utc_now()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            _normalize_sql(
                """
                INSERT INTO orders
                (order_id, account_id, symbol, side, qty, order_type, limit_price, stop_price, status, source, signal_hash, notes, created_at, updated_at, filled_price, filled_qty, filled_at, broker_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """
            ),
            (order_id, account_id, symbol, side, qty, order_type, limit_price, stop_price, status, source, signal_hash, notes, now, now, broker_ref),
        )
        cur.close()
    record_event("order_created", {"order_id": order_id, "symbol": symbol, "side": side, "qty": qty, "status": status}, account_id=account_id, symbol=symbol)
    return order_id


def _position_after_fill(current: Mapping[str, Any] | None, side: str, qty: int, fill_price: float) -> tuple[str, int, float]:
    if current is None:
        return ("BUY", qty, fill_price)
    cur_qty = int(current.get("qty", 0) or 0)
    cur_avg = float(current.get("avg_price", 0) or 0)
    cur_side = str(current.get("side", "BUY") or "BUY").upper()
    if side == "BUY":
        new_qty = cur_qty + qty
        new_avg = ((cur_avg * cur_qty) + (fill_price * qty)) / max(new_qty, 1)
        return (cur_side or "BUY", new_qty, new_avg)
    new_qty = max(cur_qty - qty, 0)
    new_avg = cur_avg if new_qty > 0 else 0.0
    return (cur_side or "BUY", new_qty, new_avg)


def record_fill(
    order_id: str,
    fill_price: float,
    qty: int | None = None,
    *,
    fee: float = 0.0,
    venue: str = "sim",
    notes: str = "",
    account_id: str = DEFAULT_ACCOUNT_ID,
) -> dict[str, Any]:
    init_store()
    now = _utc_now()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(_normalize_sql("SELECT * FROM orders WHERE order_id = ? AND account_id = ?"), (order_id, account_id))
        order = cur.fetchone()
        if order is None:
            cur.close()
            raise ValueError(f"order_id not found: {order_id}")

        if isinstance(order, Mapping):
            order_map = dict(order)
        else:
            # sqlite row or tuple
            cols = [d[0] for d in cur.description] if cur.description else []
            order_map = dict(zip(cols, order))

        side = str(order_map.get("side", "BUY")).upper()
        symbol = _normalize_symbol(order_map.get("symbol", ""))
        fill_qty = int(qty or order_map.get("qty", 0) or 0)
        fill_price = float(fill_price)
        fee = float(fee)

        fill_id = f"fill_{_hash_payload({'order_id': order_id, 'symbol': symbol, 'qty': fill_qty, 'fill_price': fill_price, 'time': now})}"
        cur.execute(
            _normalize_sql(
                """
                INSERT INTO fills
                (fill_id, order_id, account_id, symbol, side, qty, fill_price, fee, fill_time, venue, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (fill_id) DO UPDATE SET
                    order_id = excluded.order_id,
                    account_id = excluded.account_id,
                    symbol = excluded.symbol,
                    side = excluded.side,
                    qty = excluded.qty,
                    fill_price = excluded.fill_price,
                    fee = excluded.fee,
                    fill_time = excluded.fill_time,
                    venue = excluded.venue,
                    notes = excluded.notes
                """
            ),
            (fill_id, order_id, account_id, symbol, side, fill_qty, fill_price, fee, now, venue, notes),
        )

        cur.execute(_normalize_sql("SELECT * FROM accounts WHERE account_id = ?"), (account_id,))
        account = cur.fetchone()
        if account is None:
            cur.close()
            raise ValueError(f"account_id not found: {account_id}")
        account_map = dict(account) if isinstance(account, Mapping) else {k[0]: v for k, v in zip([], [])}
        if not isinstance(account, Mapping):
            try:
                keys = [d[0] for d in cur.description] if cur.description else []
                account_map = dict(zip(keys, account))
            except Exception:
                account_map = {}

        cash = float(account_map.get("cash", 0.0))
        realized_pnl = float(account_map.get("realized_pnl", 0.0))

        cur.execute(_normalize_sql("SELECT * FROM positions WHERE symbol = ?"), (symbol,))
        pos = cur.fetchone()
        pos_map = dict(pos) if isinstance(pos, Mapping) else None
        if pos is not None and not isinstance(pos, Mapping):
            try:
                keys = [d[0] for d in cur.description] if cur.description else []
                pos_map = dict(zip(keys, pos))
            except Exception:
                pos_map = None

        if side == "BUY":
            cash -= (fill_qty * fill_price) + fee
            if pos_map is None:
                cur.execute(
                    _normalize_sql(
                        """
                        INSERT INTO positions
                        (symbol, side, qty, avg_price, mark_price, stop_loss, target_1, target_2, source, strategy, signal_hash, notes, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """
                    ),
                    (symbol, "BUY", fill_qty, fill_price, fill_price, None, None, None, "fill", "live", "", notes, now),
                )
            else:
                new_side, new_qty, new_avg = _position_after_fill(pos_map, side, fill_qty, fill_price)
                cur.execute(
                    _normalize_sql(
                        """
                        UPDATE positions
                        SET side = ?, qty = ?, avg_price = ?, mark_price = ?, updated_at = ?
                        WHERE symbol = ?
                        """
                    ),
                    (new_side, new_qty, new_avg, fill_price, now, symbol),
                )
        else:
            cash += (fill_qty * fill_price) - fee
            if pos_map is not None:
                cur_qty = int(pos_map.get("qty", 0) or 0)
                cur_avg = float(pos_map.get("avg_price", 0) or 0)
                sell_qty = min(fill_qty, cur_qty)
                realized_pnl += (fill_price - cur_avg) * sell_qty
                new_qty = max(cur_qty - sell_qty, 0)
                if new_qty <= 0:
                    cur.execute(_normalize_sql("DELETE FROM positions WHERE symbol = ?"), (symbol,))
                else:
                    cur.execute(
                        _normalize_sql("UPDATE positions SET qty = ?, mark_price = ?, updated_at = ? WHERE symbol = ?"),
                        (new_qty, fill_price, now, symbol),
                    )

        new_status = "FILLED" if side in {"BUY", "SELL"} else "ACK"
        cur.execute(
            _normalize_sql(
                """
                UPDATE orders
                SET status = ?, filled_price = ?, filled_qty = ?, filled_at = ?, updated_at = ?
                WHERE order_id = ?
                """
            ),
            (new_status, fill_price, fill_qty, now, now, order_id),
        )
        cur.execute(
            _normalize_sql("UPDATE accounts SET cash = ?, realized_pnl = ?, updated_at = ? WHERE account_id = ?"),
            (cash, realized_pnl, now, account_id),
        )
        cur.close()

    record_event(
        "fill_recorded",
        {"order_id": order_id, "symbol": symbol, "side": side, "qty": fill_qty, "fill_price": fill_price, "fee": fee, "venue": venue},
        account_id=account_id,
        symbol=symbol,
    )
    return {
        "fill_id": fill_id,
        "order_id": order_id,
        "symbol": symbol,
        "side": side,
        "qty": fill_qty,
        "fill_price": fill_price,
        "fee": fee,
        "status": new_status,
    }


def estimate_position_size(
    cash: float,
    entry_price: float,
    stop_price: float,
    *,
    risk_pct: float = 0.01,
    max_notional_pct: float = 0.20,
    lot_size: int = 100,
) -> dict[str, Any]:
    cash = float(cash)
    entry_price = float(entry_price)
    stop_price = float(stop_price)
    lot_size = max(1, int(lot_size))
    risk_budget = max(0.0, cash * float(risk_pct))
    notional_cap = max(0.0, cash * float(max_notional_pct))
    risk_per_share = abs(entry_price - stop_price)
    if not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return {
            "qty": 0,
            "risk_budget": risk_budget,
            "notional_cap": notional_cap,
            "risk_per_share": risk_per_share,
            "reason": "invalid_risk_per_share",
        }
    raw_qty = int(risk_budget // risk_per_share)
    capped_qty = int(min(raw_qty, notional_cap // max(entry_price, 1e-9)))
    qty = (capped_qty // lot_size) * lot_size
    return {
        "qty": max(qty, 0),
        "risk_budget": risk_budget,
        "notional_cap": notional_cap,
        "risk_per_share": risk_per_share,
        "reason": "ok" if qty > 0 else "too_small",
    }


def simulate_limit_execution(
    *,
    side: str,
    order_price: float,
    open_price: float | None = None,
    high_price: float | None = None,
    low_price: float | None = None,
    close_price: float | None = None,
    slippage_bps: float = 25.0,
    spread_bps: float = 12.0,
) -> dict[str, Any]:
    side = _normalize_side(side)
    order_price = float(order_price)
    open_price = float(open_price) if open_price is not None else order_price
    high_price = float(high_price) if high_price is not None else open_price
    low_price = float(low_price) if low_price is not None else open_price
    close_price = float(close_price) if close_price is not None else open_price
    slippage = abs(order_price) * float(slippage_bps) / 10_000.0
    spread = abs(order_price) * float(spread_bps) / 10_000.0

    if side == "BUY":
        touched = low_price <= order_price <= high_price
        if touched:
            fill = min(high_price, max(order_price, open_price) + slippage + spread / 2.0)
            return {"filled": True, "fill_price": float(fill), "reason": "limit_touched"}
        if open_price > order_price:
            return {"filled": False, "fill_price": None, "reason": "opened_above_limit"}
        return {"filled": False, "fill_price": None, "reason": "not_touched"}
    touched = low_price <= order_price <= high_price
    if touched:
        fill = max(low_price, min(order_price, open_price) - slippage - spread / 2.0)
        return {"filled": True, "fill_price": float(fill), "reason": "limit_touched"}
    if open_price < order_price:
        return {"filled": False, "fill_price": None, "reason": "opened_below_limit"}
    return {"filled": False, "fill_price": None, "reason": "not_touched"}


def export_state() -> dict[str, Any]:
    init_store()
    payload = {
        "exported_at": _utc_now(),
        "backend": get_backend_label(),
        "accounts": [],
        "positions": [],
        "orders": [],
        "fills": [],
        "events": [],
        "trade_journal": [],
        "meta": [],
    }
    for table in ["accounts", "positions", "orders", "fills", "events", "trade_journal", "meta"]:
        payload[table] = _fetch_df(f"SELECT * FROM {table}").to_dict("records")
    return payload


def import_state(payload: dict[str, Any], *, replace: bool = False) -> dict[str, int]:
    init_store()
    counts = {"accounts": 0, "positions": 0, "orders": 0, "fills": 0, "events": 0, "trade_journal": 0, "meta": 0}
    with _connect() as conn:
        cur = conn.cursor()
        if replace:
            for table in ["accounts", "positions", "orders", "fills", "events", "trade_journal", "meta"]:
                cur.execute(_normalize_sql(f"DELETE FROM {table}"))
        for table in counts.keys():
            rows = payload.get(table, []) or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                cols = list(row.keys())
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
                cur.execute(_normalize_sql(sql), [row[c] for c in cols])
                counts[table] += 1
        cur.close()
    return counts


def render_backend_notice() -> str:
    return get_backend_warning()
