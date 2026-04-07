"""
Core.evals.benchmarks — Model & task-based evaluation harness.

Supports:
  - Ground-truth comparisons (exact match, fuzzy).
  - JSON-schema adherence scoring.
  - Agentic trajectory evaluation (did the agent reach the goal?).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Outcome of a single evaluation case."""

    case_id: str
    passed: bool
    score: float  # 0.0 – 1.0
    details: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run evaluation suites against model outputs.

    Parameters
    ----------
    generate_fn : Callable[[str], str]
        A function ``(prompt) -> raw_response`` used to query the model
        under test.
    """

    def __init__(self, generate_fn: Callable[[str], str]):
        self._generate = generate_fn

    # ------------------------------------------------------------------
    # Exact / fuzzy match
    # ------------------------------------------------------------------

    def eval_exact_match(
        self, cases: list[dict[str, str]]
    ) -> list[EvalResult]:
        """Evaluate model against ``[{"prompt": ..., "expected": ...}]``."""
        results: list[EvalResult] = []
        for i, case in enumerate(cases):
            response = self._generate(case["prompt"]).strip()
            passed = response == case["expected"].strip()
            score = 1.0 if passed else 0.0
            results.append(
                EvalResult(
                    case_id=case.get("id", f"case_{i}"),
                    passed=passed,
                    score=score,
                    details={"response": response, "expected": case["expected"]},
                )
            )
        return results

    def eval_fuzzy_match(
        self, cases: list[dict[str, str]], threshold: float = 0.8
    ) -> list[EvalResult]:
        """Score via SequenceMatcher ratio; pass if ≥ *threshold*."""
        results: list[EvalResult] = []
        for i, case in enumerate(cases):
            response = self._generate(case["prompt"]).strip()
            ratio = SequenceMatcher(None, response, case["expected"].strip()).ratio()
            results.append(
                EvalResult(
                    case_id=case.get("id", f"case_{i}"),
                    passed=ratio >= threshold,
                    score=ratio,
                    details={"response": response, "expected": case["expected"]},
                )
            )
        return results

    # ------------------------------------------------------------------
    # Schema adherence
    # ------------------------------------------------------------------

    def eval_schema_adherence(
        self,
        prompts: list[str],
        schema: type[BaseModel],
    ) -> list[EvalResult]:
        """Check whether the model output parses into *schema*."""
        results: list[EvalResult] = []
        for i, prompt in enumerate(prompts):
            raw = self._generate(prompt)
            try:
                data = json.loads(raw)
                schema.model_validate(data)
                results.append(
                    EvalResult(case_id=f"schema_{i}", passed=True, score=1.0)
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                results.append(
                    EvalResult(
                        case_id=f"schema_{i}",
                        passed=False,
                        score=0.0,
                        details={"error": str(exc), "raw": raw},
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Agentic trajectory eval
    # ------------------------------------------------------------------

    def eval_trajectory(
        self,
        trajectories: list[dict[str, Any]],
    ) -> list[EvalResult]:
        """Score agentic runs.

        Each trajectory dict must contain:
          - ``steps``: list[str] — ordered actions the agent took.
          - ``expected_final_state``: str — the desired end state.
          - ``actual_final_state``: str — what the agent actually achieved.
        """
        results: list[EvalResult] = []
        for i, traj in enumerate(trajectories):
            match_ratio = SequenceMatcher(
                None,
                traj["actual_final_state"],
                traj["expected_final_state"],
            ).ratio()
            results.append(
                EvalResult(
                    case_id=traj.get("id", f"traj_{i}"),
                    passed=match_ratio >= 0.9,
                    score=match_ratio,
                    details={
                        "steps": traj["steps"],
                        "expected": traj["expected_final_state"],
                        "actual": traj["actual_final_state"],
                    },
                )
            )
        return results

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summary(results: list[EvalResult]) -> dict[str, Any]:
        """Return pass-rate and average score across results."""
        if not results:
            return {"pass_rate": 0.0, "avg_score": 0.0, "total": 0}
        passed = sum(1 for r in results if r.passed)
        return {
            "pass_rate": passed / len(results),
            "avg_score": sum(r.score for r in results) / len(results),
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        }
