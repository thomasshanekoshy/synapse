"""Tests for Core.evals — feedback_loop and benchmarks."""

from pathlib import Path

import pytest

from Core.evals.feedback_loop import FeedbackLoop
from Core.evals.benchmarks import BenchmarkRunner, EvalResult
from Core.memory.sqlite_manager import SQLiteManager
from pydantic import BaseModel


@pytest.fixture()
def db(tmp_path: Path) -> SQLiteManager:
    return SQLiteManager(db_path=tmp_path / "test.db")


# ── Feedback Loop ────────────────────────────────────────────────────


class TestFeedbackLoop:
    def test_record_and_avg(self, db: SQLiteManager):
        fb = FeedbackLoop(db)
        fb.record(5, session_id="s1")
        fb.record(3, session_id="s1")
        assert fb.average_rating(session_id="s1") == 4.0

    def test_invalid_rating_raises(self, db: SQLiteManager):
        fb = FeedbackLoop(db)
        with pytest.raises(ValueError):
            fb.record(0)
        with pytest.raises(ValueError):
            fb.record(6)

    def test_no_feedback_returns_none(self, db: SQLiteManager):
        fb = FeedbackLoop(db)
        assert fb.average_rating() is None


# ── Benchmark Runner ─────────────────────────────────────────────────


def _echo_generate(prompt: str) -> str:
    """Fake model that echoes the prompt."""
    return prompt


class TestBenchmarkRunner:
    def test_exact_match_pass(self):
        runner = BenchmarkRunner(generate_fn=_echo_generate)
        results = runner.eval_exact_match([{"prompt": "hello", "expected": "hello"}])
        assert results[0].passed is True
        assert results[0].score == 1.0

    def test_exact_match_fail(self):
        runner = BenchmarkRunner(generate_fn=_echo_generate)
        results = runner.eval_exact_match([{"prompt": "hello", "expected": "world"}])
        assert results[0].passed is False

    def test_fuzzy_match(self):
        runner = BenchmarkRunner(generate_fn=_echo_generate)
        results = runner.eval_fuzzy_match(
            [{"prompt": "hello world", "expected": "hello world!"}],
            threshold=0.8,
        )
        assert results[0].passed is True
        assert results[0].score > 0.8

    def test_schema_adherence_pass(self):
        class Out(BaseModel):
            value: int

        runner = BenchmarkRunner(generate_fn=lambda _: '{"value": 42}')
        results = runner.eval_schema_adherence(["test"], Out)
        assert results[0].passed is True

    def test_schema_adherence_fail(self):
        class Out(BaseModel):
            value: int

        runner = BenchmarkRunner(generate_fn=lambda _: "not json")
        results = runner.eval_schema_adherence(["test"], Out)
        assert results[0].passed is False

    def test_trajectory_eval(self):
        runner = BenchmarkRunner(generate_fn=_echo_generate)
        results = runner.eval_trajectory(
            [
                {
                    "steps": ["step1", "step2"],
                    "expected_final_state": "done",
                    "actual_final_state": "done",
                }
            ]
        )
        assert results[0].passed is True

    def test_summary(self):
        results = [
            EvalResult(case_id="a", passed=True, score=1.0),
            EvalResult(case_id="b", passed=False, score=0.3),
        ]
        s = BenchmarkRunner.summary(results)
        assert s["pass_rate"] == 0.5
        assert s["total"] == 2
