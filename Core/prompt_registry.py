"""
Core.prompt_registry — Version-controlled prompt storage.

Allows storing, versioning, and retrieving system prompts by name.
Designed to integrate with MCP context chunks — prompts can embed
placeholder slots that are filled at runtime.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """A single immutable snapshot of a prompt template."""

    version: int
    template: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PromptRegistry:
    """In-memory registry of named prompt templates with version history.

    Usage
    -----
    >>> reg = PromptRegistry()
    >>> reg.register("summarise", "Summarise the following: {text}")
    >>> prompt = reg.render("summarise", text="Hello world")
    """

    def __init__(self) -> None:
        self._store: dict[str, list[PromptVersion]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        template: str,
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Create a new version of prompt *name*.

        Parameters
        ----------
        name : str
            Unique identifier for the prompt.
        template : str
            The prompt text.  May contain ``{placeholder}`` slots.
        metadata : dict
            Arbitrary metadata (e.g., author, model target).

        Returns
        -------
        PromptVersion
            The newly created version object.
        """
        history = self._store.setdefault(name, [])
        version_num = len(history) + 1
        pv = PromptVersion(
            version=version_num,
            template=template,
            metadata=metadata or {},
        )
        history.append(pv)
        logger.info("Registered prompt '%s' v%d", name, version_num)
        return pv

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        """Return a specific version (default: latest) of prompt *name*.

        Raises
        ------
        KeyError
            If *name* is not registered.
        IndexError
            If *version* does not exist.
        """
        if name not in self._store or not self._store[name]:
            raise KeyError(f"Prompt '{name}' not registered")
        history = self._store[name]
        if version is None:
            return copy.deepcopy(history[-1])
        if version < 1 or version > len(history):
            raise IndexError(
                f"Prompt '{name}' has {len(history)} versions; requested v{version}"
            )
        return copy.deepcopy(history[version - 1])

    def render(self, name: str, version: int | None = None, **kwargs: Any) -> str:
        """Render the prompt template with ``str.format_map`` substitution.

        Parameters
        ----------
        name : str
            Registered prompt name.
        version : int | None
            Specific version to use (default latest).
        **kwargs
            Values for ``{placeholder}`` slots in the template.

        Returns
        -------
        str
            The fully rendered prompt string.
        """
        pv = self.get(name, version)
        return pv.template.format_map(kwargs)

    def list_prompts(self) -> dict[str, int]:
        """Return a mapping of prompt name → latest version number."""
        return {name: len(history) for name, history in self._store.items()}

    def history(self, name: str) -> list[PromptVersion]:
        """Return full version history for a prompt."""
        if name not in self._store:
            raise KeyError(f"Prompt '{name}' not registered")
        return list(self._store[name])
