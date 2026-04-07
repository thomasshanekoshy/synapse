"""
Core.memory.sqlite_manager — Persistent state, caching, token tracking,
A2A memory, and user-feedback storage.

Per SPEC §3.3 tables:
  - cache_exact
  - token_usage
  - a2a_memory
  - user_feedback
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("synapze.db")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cache_exact (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_hash     TEXT    NOT NULL UNIQUE,
    response        TEXT    NOT NULL,
    model_id        TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    model_id            TEXT    NOT NULL,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    latency_ms          REAL    NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS a2a_memory (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT    NOT NULL,
    agent_name              TEXT    NOT NULL,
    final_state_summary     TEXT    NOT NULL,
    created_at              TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS user_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    task_id         TEXT,
    user_rating     INTEGER,
    user_comment    TEXT,
    created_at      TEXT    NOT NULL
);
"""


class SQLiteManager:
    """Single entry-point for all SQLite persistence in Synapze."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Exact-match cache
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get_cache(self, prompt: str) -> str | None:
        """Return cached response for an exact prompt match, or None."""
        h = self._hash_prompt(prompt)
        row = (
            self._get_conn()
            .execute("SELECT response FROM cache_exact WHERE prompt_hash = ?", (h,))
            .fetchone()
        )
        if row:
            logger.debug("Cache HIT for prompt hash %s", h[:12])
            return row["response"]
        return None

    def set_cache(self, prompt: str, response: str, model_id: str) -> None:
        """Store a prompt→response pair in the exact cache."""
        h = self._hash_prompt(prompt)
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache_exact (prompt_hash, response, model_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (h, response, model_id, now),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Token usage tracking
    # ------------------------------------------------------------------

    def log_usage(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
        session_id: str | None = None,
    ) -> None:
        """Record a single inference event."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO token_usage "
            "(session_id, model_id, prompt_tokens, completion_tokens, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, model_id, prompt_tokens, completion_tokens, latency_ms, now),
        )
        conn.commit()

    def get_usage_summary(self, session_id: str | None = None) -> dict[str, Any]:
        """Aggregate token counts, optionally filtered by session."""
        conn = self._get_conn()
        if session_id:
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens),0) AS total_prompt, "
                "COALESCE(SUM(completion_tokens),0) AS total_completion, "
                "COUNT(*) AS call_count "
                "FROM token_usage WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens),0) AS total_prompt, "
                "COALESCE(SUM(completion_tokens),0) AS total_completion, "
                "COUNT(*) AS call_count "
                "FROM token_usage",
            ).fetchone()
        return {
            "total_prompt_tokens": row["total_prompt"],
            "total_completion_tokens": row["total_completion"],
            "call_count": row["call_count"],
        }

    # ------------------------------------------------------------------
    # A2A shared memory
    # ------------------------------------------------------------------

    def save_agent_state(
        self, session_id: str, agent_name: str, summary: str
    ) -> None:
        """Persist the final state summary of an agent for A2A handoff."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO a2a_memory (session_id, agent_name, final_state_summary, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, agent_name, summary, now),
        )
        conn.commit()

    def get_agent_states(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all agent state summaries for a session (ordered)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT agent_name, final_state_summary, created_at "
            "FROM a2a_memory WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # User feedback
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        user_rating: int,
        user_comment: str = "",
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Log a user-feedback entry."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO user_feedback "
            "(session_id, task_id, user_rating, user_comment, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, task_id, user_rating, user_comment, now),
        )
        conn.commit()

    def get_feedback(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return feedback entries, optionally filtered by session."""
        conn = self._get_conn()
        if session_id:
            rows = conn.execute(
                "SELECT * FROM user_feedback WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM user_feedback ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]
