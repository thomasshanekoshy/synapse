"""
Core.rate_limiter — Concurrency control and backoff for the local Ollama
server.

Prevents overwhelming the single-GPU setup with parallel requests.
Uses a semaphore + exponential backoff on transient failures.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitExceeded(Exception):
    """Raised when max retries are exhausted."""


class RateLimiter:
    """Thread-safe concurrency gate with exponential backoff.

    Parameters
    ----------
    max_concurrent : int
        Maximum in-flight requests (default 1 for single-GPU).
    max_retries : int
        Number of retry attempts on transient failure.
    base_delay : float
        Initial backoff delay in seconds (doubles each retry).
    max_delay : float
        Cap on backoff delay.
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

        # Metrics
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_retries = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Acquire the semaphore, call *fn*, and retry with backoff on failure.

        Parameters
        ----------
        fn : Callable
            The function to execute (e.g., ``OllamaClient.generate``).

        Returns
        -------
        T
            Whatever *fn* returns.

        Raises
        ------
        RateLimitExceeded
            If all retries are exhausted.
        """
        self._semaphore.acquire()
        try:
            return self._call_with_backoff(fn, *args, **kwargs)
        finally:
            self._semaphore.release()

    @property
    def stats(self) -> dict[str, int]:
        """Return current call/retry counters."""
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "total_retries": self._total_retries,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_with_backoff(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        delay = self._base_delay
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 2):  # 1 initial + retries
            with self._lock:
                self._total_calls += 1
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt > self._max_retries:
                    break
                with self._lock:
                    self._total_retries += 1
                logger.warning(
                    "Attempt %d/%d failed (%s) — retrying in %.1fs",
                    attempt,
                    self._max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, self._max_delay)

        raise RateLimitExceeded(
            f"All {self._max_retries + 1} attempts failed"
        ) from last_exc
