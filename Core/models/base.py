"""
Core.models.base — Abstract base class for all model adapters.

Per SPEC §3.1: All model interactions go through a uniform adapter interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseModelAdapter(ABC):
    """Abstract contract that every model adapter must fulfil."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> dict:
        """Send a prompt and return Ollama-style response dict.

        Returns
        -------
        dict
            Must contain at minimum:
            - "response" (str): The generated text.
            - "prompt_eval_count" (int): Prompt token count.
            - "eval_count" (int): Completion token count.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is currently resident in VRAM."""

    @abstractmethod
    def unload(self) -> None:
        """Evict this model from VRAM."""
