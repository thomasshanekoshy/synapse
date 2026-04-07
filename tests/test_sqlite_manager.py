"""Tests for Core.memory.sqlite_manager — all 4 tables."""

import tempfile
from pathlib import Path

import pytest

from Core.memory.sqlite_manager import SQLiteManager


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    """Provide a fresh in-tempdir SQLite database per test."""
    return SQLiteManager(db_path=tmp_path / "test.db")


class TestExactCache:
    def test_cache_miss_returns_none(self, db: SQLiteManager):
        assert db.get_cache("never seen") is None

    def test_round_trip(self, db: SQLiteManager):
        db.set_cache("hello", "world", model_id="m1")
        assert db.get_cache("hello") == "world"

    def test_overwrite(self, db: SQLiteManager):
        db.set_cache("p", "r1", model_id="m1")
        db.set_cache("p", "r2", model_id="m1")
        assert db.get_cache("p") == "r2"


class TestTokenUsage:
    def test_log_and_summary(self, db: SQLiteManager):
        db.log_usage("m1", 10, 20, session_id="s1")
        db.log_usage("m1", 5, 15, session_id="s1")
        summary = db.get_usage_summary(session_id="s1")
        assert summary["total_prompt_tokens"] == 15
        assert summary["total_completion_tokens"] == 35
        assert summary["call_count"] == 2

    def test_global_summary(self, db: SQLiteManager):
        db.log_usage("m1", 10, 20, session_id="s1")
        db.log_usage("m2", 5, 15, session_id="s2")
        summary = db.get_usage_summary()
        assert summary["call_count"] == 2


class TestA2AMemory:
    def test_save_and_retrieve(self, db: SQLiteManager):
        db.save_agent_state("s1", "agent_a", "summary A")
        db.save_agent_state("s1", "agent_b", "summary B")
        states = db.get_agent_states("s1")
        assert len(states) == 2
        assert states[0]["agent_name"] == "agent_a"
        assert states[1]["agent_name"] == "agent_b"

    def test_isolation_by_session(self, db: SQLiteManager):
        db.save_agent_state("s1", "a", "x")
        db.save_agent_state("s2", "b", "y")
        assert len(db.get_agent_states("s1")) == 1
        assert len(db.get_agent_states("s2")) == 1


class TestUserFeedback:
    def test_record_and_get(self, db: SQLiteManager):
        db.record_feedback(4, "good", session_id="s1", task_id="t1")
        rows = db.get_feedback(session_id="s1")
        assert len(rows) == 1
        assert rows[0]["user_rating"] == 4
        assert rows[0]["user_comment"] == "good"
