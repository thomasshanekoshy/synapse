"""
Core.guardrails.input_filter — Lightweight input screening.

Per SPEC §3.4 & TDD (custom, lightweight guardrails):
  - Prompt-injection detection via pattern matching.
  - PII screening (email, phone, SSN patterns).
  - Configurable deny-list for forbidden terms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class InputGuardrailViolation(Exception):
    """Raised when an input prompt fails safety screening."""

    def __init__(self, reason: str, matched: str = ""):
        self.reason = reason
        self.matched = matched
        super().__init__(f"Input blocked — {reason}: {matched}")


# ---- Pattern banks ----

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?new\s+(ai|assistant|system)", re.I),
    re.compile(r"disregard\s+(all\s+)?(safety|system|prior)", re.I),
    re.compile(r"pretend\s+you\s+(are|have)\s+no\s+(rules|restrictions)", re.I),
    re.compile(r"\bDAN\b|\bDo\s+Anything\s+Now\b", re.I),
]

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_uk": re.compile(r"\b(?:\+44|0)\s?\d{4}\s?\d{6}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "nino": re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.I),
}


@dataclass
class InputFilterConfig:
    """Tuneable knobs for the input filter."""

    check_injection: bool = True
    check_pii: bool = True
    deny_list: list[str] = field(default_factory=list)
    max_prompt_length: int = 32_000


class InputFilter:
    """Stateless input guardrail that validates prompts before model calls."""

    def __init__(self, config: InputFilterConfig | None = None):
        self._cfg = config or InputFilterConfig()

    def validate(self, prompt: str) -> bool:
        """Return True if prompt passes all checks, else raise.

        Raises
        ------
        InputGuardrailViolation
            With a descriptive ``reason`` field.
        """
        # Length gate
        if len(prompt) > self._cfg.max_prompt_length:
            raise InputGuardrailViolation(
                "prompt_too_long",
                f"{len(prompt)} chars exceeds {self._cfg.max_prompt_length}",
            )

        # Injection detection
        if self._cfg.check_injection:
            for pat in _INJECTION_PATTERNS:
                m = pat.search(prompt)
                if m:
                    raise InputGuardrailViolation("prompt_injection", m.group())

        # PII detection
        if self._cfg.check_pii:
            for label, pat in _PII_PATTERNS.items():
                m = pat.search(prompt)
                if m:
                    raise InputGuardrailViolation(f"pii_detected_{label}", m.group())

        # Deny list
        lower = prompt.lower()
        for term in self._cfg.deny_list:
            if term.lower() in lower:
                raise InputGuardrailViolation("deny_list_match", term)

        return True
