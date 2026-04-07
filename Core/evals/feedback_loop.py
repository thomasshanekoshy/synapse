"""
Core.evals.feedback_loop — User feedback logging and retrieval.

Wraps SQLiteManager's user_feedback table with a higher-level API
suitable for agentic eval pipelines.
"""

from __future__ import annotations

import logging
from typing import Any

from Core.memory.sqlite_manager import SQLiteManager

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Collects and surfaces user feedback for evaluation & fine-tuning loops.

    Parameters
    ----------
    db : SQLiteManager
        Shared database instance.
    """

    def __init__(self, db: SQLiteManager):
        self._db = db

    def record(
        self,
        rating: int,
        comment: str = "",
        session_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """Log a user's quality rating (1-5) and optional free-text comment."""
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be 1-5, got {rating}")
        self._db.record_feedback(
            user_rating=rating,
            user_comment=comment,
            session_id=session_id,
            task_id=task_id,
        )
        logger.info(
            "Feedback recorded: rating=%d session=%s task=%s",
            rating,
            session_id,
            task_id,
        )

    def average_rating(self, session_id: str | None = None) -> float | None:
        """Return the mean user rating, or None if no feedback exists."""
        entries = self._db.get_feedback(session_id=session_id)
        if not entries:
            return None
        return sum(e["user_rating"] for e in entries) / len(entries)

    def get_all(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return raw feedback entries."""
        return self._db.get_feedback(session_id=session_id)
