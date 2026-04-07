"""
Core.models.ollama_client — Low-level Ollama API wrapper with strict
1-active-model VRAM management.

Per SPEC §3.1:
  - `_active_model_id`: tracks the currently loaded model.
  - `unload_model()`: drops a model via keep_alive=0.
  - `ensure_active()`: swaps models when needed.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_GENERATE_ENDPOINT = "/api/generate"


class OllamaClient:
    """Manages a single-model-at-a-time connection to Ollama."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: float = 300.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._active_model_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_model(self) -> str | None:
        """Return the model ID currently loaded in VRAM (or None)."""
        return self._active_model_id

    def unload_model(self, model_id: str) -> None:
        """Evict *model_id* from VRAM by setting keep_alive=0."""
        logger.info("Unloading model %s from VRAM", model_id)
        payload = {"model": model_id, "keep_alive": 0}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                client.post(f"{self._base_url}{_GENERATE_ENDPOINT}", json=payload)
        except httpx.HTTPError as exc:
            logger.warning("Failed to unload %s: %s", model_id, exc)
        if self._active_model_id == model_id:
            self._active_model_id = None

    def ensure_active(self, model_id: str) -> None:
        """Guarantee that *model_id* is the sole model in VRAM.

        If a different model is currently loaded, it is unloaded first.
        """
        if self._active_model_id == model_id:
            return
        if self._active_model_id is not None:
            self.unload_model(self._active_model_id)
        logger.info("Activating model %s", model_id)
        self._active_model_id = model_id

    def generate(self, model_id: str, prompt: str, **kwargs: Any) -> dict:
        """Send a generation request and return the full Ollama JSON response.

        Automatically calls `ensure_active` to enforce the 1-model policy.

        Parameters
        ----------
        model_id : str
            Ollama model tag (e.g. ``"deepseek-r1:14b"``).
        prompt : str
            The user/system prompt to send.
        **kwargs
            Extra keys forwarded to the Ollama ``/api/generate`` body
            (e.g. ``temperature``, ``num_predict``).

        Returns
        -------
        dict
            Full Ollama response including ``response``, ``prompt_eval_count``,
            ``eval_count``, ``total_duration``, etc.
        """
        self.ensure_active(model_id)

        payload: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}{_GENERATE_ENDPOINT}", json=payload
            )
            resp.raise_for_status()
            return resp.json()
