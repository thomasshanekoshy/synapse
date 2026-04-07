"""
Core.routing.dynamic_router — Maps task categories to optimal local models.

Per SPEC §3.5 & TDD:
  Simple tasks → small, fast models (cost optimisation).
  Complex tasks → larger, capable models.
  Routing rules default to the models in models_tracker.csv but are
  fully configurable at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """A single task_category → model_id mapping."""

    task_category: str
    model_id: str
    description: str = ""


# Default rules derived from models_tracker.csv
_DEFAULT_RULES: list[RoutingRule] = [
    RoutingRule("simple_summary", "lfm2.5-thinking:latest", "Tiny, low-latency summariser"),
    RoutingRule("general", "granite4:latest", "Balanced agentic default"),
    RoutingRule("complex_reasoning", "deepseek-r1:14b", "Deep CoT reasoning"),
    RoutingRule("complex_code", "deepseek-coder-v2:latest", "Heavy code generation"),
    RoutingRule("vision", "qwen3-vl:latest", "Image / video understanding"),
    RoutingRule("ocr", "glm-ocr:latest", "Document OCR"),
    RoutingRule("multimodal", "gemma4:latest", "Audio + image + long context"),
    RoutingRule("embedding", "snowflake-arctic-embed2:568m", "Fast embedding generation"),
    RoutingRule("reranking", "xitao/bge-reranker-v2-m3:latest", "Re-ranking for RAG"),
]


class DynamicRouter:
    """Resolve a task category to the best-suited local model.

    Parameters
    ----------
    rules : list[RoutingRule] | None
        Override the built-in routing table.
    default_model : str
        Fallback model when no rule matches.
    """

    def __init__(
        self,
        rules: list[RoutingRule] | None = None,
        default_model: str = "granite4:latest",
    ):
        self._default = default_model
        self._rules: dict[str, RoutingRule] = {}
        for rule in rules or _DEFAULT_RULES:
            self._rules[rule.task_category] = rule

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, task_category: str) -> str:
        """Return the model_id for *task_category*, or the default fallback."""
        rule = self._rules.get(task_category)
        if rule:
            logger.info(
                "Routing '%s' → %s (%s)", task_category, rule.model_id, rule.description
            )
            return rule.model_id
        logger.info("No rule for '%s' — using default %s", task_category, self._default)
        return self._default

    def add_rule(self, rule: RoutingRule) -> None:
        """Register or overwrite a routing rule."""
        self._rules[rule.task_category] = rule

    def remove_rule(self, task_category: str) -> None:
        """Remove a routing rule (falls back to default)."""
        self._rules.pop(task_category, None)

    def list_rules(self) -> list[RoutingRule]:
        """Return all registered rules."""
        return list(self._rules.values())
