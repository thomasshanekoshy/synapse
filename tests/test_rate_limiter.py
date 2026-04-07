"""Tests for Core.rate_limiter."""

import pytest

from Core.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiter:
    def test_successful_call(self):
        rl = RateLimiter()
        result = rl.execute(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        rl = RateLimiter(max_retries=3, base_delay=0.01)
        result = rl.execute(flaky)
        assert result == "ok"
        assert call_count == 3

    def test_exhausted_retries_raises(self):
        def always_fail():
            raise RuntimeError("permanent")

        rl = RateLimiter(max_retries=2, base_delay=0.01)
        with pytest.raises(RateLimitExceeded):
            rl.execute(always_fail)

    def test_stats_tracking(self):
        rl = RateLimiter(base_delay=0.01)
        rl.execute(lambda: 1)
        assert rl.stats["total_calls"] >= 1
