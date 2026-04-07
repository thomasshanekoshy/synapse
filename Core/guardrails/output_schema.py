"""
Core.guardrails.output_schema — Pydantic-based output validation with
one-step self-repair.

Per SPEC §3.4:
  - Attempts to parse raw LLM text into a Pydantic BaseModel.
  - On failure, constructs a repair prompt and re-queries the model.
  - Tool execution is out of scope — we only guarantee schema validity.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OutputValidationError(Exception):
    """Raised when the LLM output cannot be coerced into the target schema."""

    def __init__(self, raw: str, errors: list[dict[str, Any]]):
        self.raw = raw
        self.errors = errors
        super().__init__(f"Output validation failed: {errors}")


def _extract_json(text: str) -> str | None:
    """Best-effort extraction of a JSON object from freeform LLM text."""
    # Try to find a fenced code block first
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back to first { ... } pair
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


class OutputSchemaValidator:
    """Validates raw LLM output against a Pydantic model.

    Parameters
    ----------
    generate_fn : Callable[[str], str] | None
        A function that re-queries the model for a repair attempt.  Signature
        ``(repair_prompt: str) -> raw_response: str``.  If ``None``, repair
        is disabled and the validator raises immediately on failure.
    max_repair_attempts : int
        Number of repair retries before giving up (default 1).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str] | None = None,
        max_repair_attempts: int = 1,
    ):
        self._generate_fn = generate_fn
        self._max_repairs = max_repair_attempts

    def validate(
        self,
        raw_response: str,
        schema: Type[BaseModel],
    ) -> dict[str, Any]:
        """Parse *raw_response* into *schema* (with optional repair).

        Returns
        -------
        dict
            The validated data as a plain dict.

        Raises
        ------
        OutputValidationError
            If parsing fails after all repair attempts.
        """
        last_errors: list[dict[str, Any]] = []

        for attempt in range(1 + self._max_repairs):
            text = raw_response if attempt == 0 else self._repair(raw_response, schema, last_errors)
            if text is None:
                break

            json_str = _extract_json(text)
            if json_str is None:
                last_errors = [{"msg": "No JSON object found in response"}]
                raw_response = text
                continue

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as exc:
                last_errors = [{"msg": f"Invalid JSON: {exc}"}]
                raw_response = text
                continue

            try:
                validated = schema.model_validate(data)
                return validated.model_dump()
            except ValidationError as exc:
                last_errors = exc.errors()  # type: ignore[assignment]
                raw_response = text
                continue

        raise OutputValidationError(raw=raw_response, errors=last_errors)

    # ------------------------------------------------------------------
    # Repair logic
    # ------------------------------------------------------------------

    def _repair(
        self,
        original_response: str,
        schema: Type[BaseModel],
        errors: list[dict[str, Any]],
    ) -> str | None:
        """Construct a repair prompt and re-query the model."""
        if self._generate_fn is None:
            return None

        schema_json = schema.model_json_schema()
        repair_prompt = (
            "The previous response did not match the required JSON schema.\n\n"
            f"Required schema:\n```json\n{json.dumps(schema_json, indent=2)}\n```\n\n"
            f"Validation errors:\n{json.dumps(errors, indent=2, default=str)}\n\n"
            f"Original response:\n{original_response}\n\n"
            "Please output ONLY a corrected JSON object matching the schema above."
        )

        logger.info("Attempting output repair (schema=%s)", schema.__name__)
        return self._generate_fn(repair_prompt)
