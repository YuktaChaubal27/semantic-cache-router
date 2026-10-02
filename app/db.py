"""SQLite request log plus a tiny key-value table for the corpus version."""
import json
import sqlite3
import time
from contextlib import contextmanager

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id TEXT,
    mode TEXT,               -- 'full' or 'baseline'
    query TEXT NOT NULL,
    tenant TEXT,
    cache_hit INTEGER NOT NULL,
    route TEXT,              -- 'simple' / 'complex' / NULL on cache hit
    escalated INTEGER NOT NULL DEFAULT 0,
    model TEXT,              -- model that produced the final answer
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,   -- includes the wasted small call on escalation
    latency_ms REAL NOT NULL,
    quality REAL,            -- filled in by the benchmark script
    error TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(settings()["paths"]["db"])
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_request(**fields) -> int:
    fields.setdefault("ts", time.time())
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    with connect() as c:
        cur = c.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(fields.values()))
        return cur.lastrowid


def set_quality(request_id: int, quality: float) -> None:
    with connect() as c:
        c.execute("UPDATE requests SET quality = ? WHERE id = ?", (quality, request_id))


def get_corpus_version() -> int:
    with connect() as c:
        row = c.execute("SELECT value FROM meta WHERE key = 'corpus_version'").fetchone()
        return int(row["value"]) if row else 0


def bump_corpus_version() -> int:
    new = get_corpus_version() + 1
    with connect() as c:
        c.execute(
            "INSERT INTO meta (key, value) VALUES ('corpus_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(new),),
        )
    return new
