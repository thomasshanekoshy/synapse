"""
Core.memory.faiss_manager — Decoupled semantic caching layer.

This module is intentionally isolated so that FAISS can be swapped for any
other vector store without touching the rest of the codebase.

Usage:
    manager = FaissManager(dimension=384)
    manager.add("prompt text", "cached response")
    hit = manager.search("similar prompt text", threshold=0.85)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    prompt: str
    response: str
    vector: np.ndarray


class FaissManager:
    """Semantic cache backed by a FAISS flat-IP index.

    Parameters
    ----------
    dimension : int
        Embedding vector dimension (must match the embedding model output).
    embed_fn : Callable[[str], np.ndarray] | None
        A function that takes a string and returns a unit-normalised vector.
        If None, the manager is inert (``search`` always returns None).
    similarity_threshold : float
        Minimum cosine-similarity score for a cache hit (0.0–1.0).
    """

    def __init__(
        self,
        dimension: int = 384,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        similarity_threshold: float = 0.85,
    ):
        self._dimension = dimension
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._entries: list[_CacheEntry] = []
        self._index = None

        if embed_fn is not None:
            self._init_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _init_index(self) -> None:
        try:
            import faiss  # type: ignore[import-untyped]

            self._index = faiss.IndexFlatIP(self._dimension)
            logger.info("FAISS index initialised (dim=%d)", self._dimension)
        except ImportError:
            logger.warning("faiss-cpu not installed — semantic cache disabled")
            self._index = None

    @property
    def is_enabled(self) -> bool:
        return self._index is not None and self._embed_fn is not None

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, prompt: str, response: str) -> None:
        """Embed *prompt* and store alongside *response* in the index."""
        if not self.is_enabled:
            return
        vec = self._embed_fn(prompt).astype(np.float32).reshape(1, -1)  # type: ignore[union-attr]
        # Normalise to unit length so inner-product ≈ cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._index.add(vec)  # type: ignore[union-attr]
        self._entries.append(_CacheEntry(prompt=prompt, response=response, vector=vec.squeeze()))

    def search(self, prompt: str, k: int = 1) -> str | None:
        """Return the cached response for the nearest semantic match, or None.

        A result is returned only when the cosine similarity exceeds
        ``self._threshold``.
        """
        if not self.is_enabled or self.size == 0:
            return None

        vec = self._embed_fn(prompt).astype(np.float32).reshape(1, -1)  # type: ignore[union-attr]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        distances, indices = self._index.search(vec, k)  # type: ignore[union-attr]
        best_score = float(distances[0][0])
        best_idx = int(indices[0][0])

        if best_score >= self._threshold and best_idx < len(self._entries):
            logger.debug(
                "Semantic cache HIT (score=%.4f, idx=%d)", best_score, best_idx
            )
            return self._entries[best_idx].response

        logger.debug("Semantic cache MISS (best_score=%.4f)", best_score)
        return None

    def clear(self) -> None:
        """Drop all entries and reinitialise the index."""
        self._entries.clear()
        if self._index is not None:
            self._init_index()
