from __future__ import annotations

import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
  execution_id TEXT PRIMARY KEY,
  config_hash TEXT NOT NULL,
  label TEXT NOT NULL,
  split TEXT NOT NULL,
  task TEXT NOT NULL,
  difficulty TEXT NOT NULL,
  status TEXT NOT NULL,
  hostname TEXT,
  started_at TEXT,
  finished_at TEXT,
  exit_code INTEGER,
  artifact_dir TEXT,
  result_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS executions_label ON executions(label, split);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def connect(path: str | Path, attempts: int = 10) -> sqlite3.Connection:
    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        con = sqlite3.connect(db_file, timeout=30)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            return con
        except sqlite3.OperationalError as exc:
            con.close()
            if not _is_lock_error(exc) or attempt + 1 == attempts:
                raise
            time.sleep(0.05 + random.random() * 0.2 * (attempt + 1))
    raise AssertionError("unreachable")


def register(
    con: sqlite3.Connection, execution_id: str, digest: str, label: str, split: str, task: str, difficulty: str
) -> None:
    con.execute(
        "INSERT OR IGNORE INTO executions(execution_id, config_hash, label, split, task, difficulty, status) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
        (execution_id, digest, label, split, task, difficulty),
    )
    con.commit()


def mark_running(con: sqlite3.Connection, execution_id: str, hostname: str, artifact_dir: str) -> None:
    con.execute(
        "UPDATE executions SET status='running', hostname=?, started_at=?, finished_at=NULL, "
        "exit_code=NULL, artifact_dir=?, result_json=NULL, error=NULL WHERE execution_id=?",
        (hostname, now(), artifact_dir, execution_id),
    )
    con.commit()


def mark_finished(
    con: sqlite3.Connection,
    execution_id: str,
    status: str,
    exit_code: int | None,
    result: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    con.execute(
        "UPDATE executions SET status=?, finished_at=?, exit_code=?, result_json=?, error=? WHERE execution_id=?",
        (status, now(), exit_code, json.dumps(result) if result is not None else None, error, execution_id),
    )
    con.commit()


def mark_lost(con: sqlite3.Connection, execution_id: str, error: str, status: str = "infra_error") -> bool:
    cursor = con.execute(
        "UPDATE executions SET status=?, finished_at=?, error=? "
        "WHERE execution_id=? AND status IN ('pending', 'running')",
        (status, now(), error, execution_id),
    )
    con.commit()
    return cursor.rowcount > 0


def rows(con: sqlite3.Connection, label: str | None = None, split: str | None = None) -> list[sqlite3.Row]:
    if label is None:
        return list(con.execute("SELECT * FROM executions ORDER BY label, split, task"))
    return list(
        con.execute("SELECT * FROM executions WHERE label=? AND split=? ORDER BY task", (label, split))
    )
