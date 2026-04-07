"""
Core.models.synapze_llm — The primary SDK class that agentic frameworks
(CrewAI, LangChain) instantiate.

Implements the full invoke() flow defined in SPEC §3.2:
  1. Input guardrails
  2. Exact cache check (SQLite)
  3. Semantic cache check (FAISS)
  4. Dynamic routing
  5. VRAM swap (OllamaClient.ensure_active)
  6. Generation
  7. Output schema validation (+ repair)
  8. Token logging
  9. Cache storage & A2A persistence
"""

from __future__ import annotations

import logging
import time
from typing import Any, Type

from pydantic import BaseModel

from Core.guardrails.input_filter import InputFilter
from Core.guardrails.output_schema import OutputSchemaValidator
from Core.memory.faiss_manager import FaissManager
from Core.memory.sqlite_manager import SQLiteManager
from Core.models.ollama_client import OllamaClient
from Core.rate_limiter import RateLimiter
from Core.routing.dynamic_router import DynamicRouter

logger = logging.getLogger(__name__)


class SynapzeLLM:
    """High-level adapter that CrewAI / LangChain agents call.

    Parameters
    ----------
    ollama : OllamaClient
        Low-level Ollama HTTP wrapper.
    router : DynamicRouter
        Task-to-model mapper.
    db : SQLiteManager
        Persistent storage backend.
    input_filter : InputFilter
        Input guardrail.
    faiss : FaissManager | None
        Optional semantic cache (decoupled).
    rate_limiter : RateLimiter | None
        Optional concurrency / backoff controller.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        router: DynamicRouter,
        db: SQLiteManager,
        input_filter: InputFilter,
        faiss: FaissManager | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self._ollama = ollama
        self._router = router
        self._db = db
        self._input_filter = input_filter
        self._faiss = faiss
        self._rate_limiter = rate_limiter

        # Wire up the output validator with a generate callback for repair
        self._output_validator = OutputSchemaValidator(
            generate_fn=self._raw_generate,
            max_repair_attempts=1,
        )

    # ------------------------------------------------------------------
    # Primary entry-point (SPEC §3.2)
    # ------------------------------------------------------------------

    def invoke(
        self,
        prompt: str,
        task_type: str = "general",
        expected_schema: Type[BaseModel] | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
        **kwargs: Any,
    ) -> str | dict:
        """Full orchestrated generation pipeline.

        Parameters
        ----------
        prompt : str
            User / agent prompt.
        task_type : str
            Category key used by the dynamic router.
        expected_schema : Type[BaseModel] | None
            If provided, the raw response is validated (and repaired)
            against this Pydantic model before returning.
        session_id : str | None
            Group calls for token-tracking and A2A memory.
        agent_name : str | None
            Identifier for A2A state persistence.
        **kwargs
            Forwarded to Ollama (e.g. ``temperature``, ``num_predict``).

        Returns
        -------
        str | dict
            Raw text when no schema is specified; validated dict otherwise.
        """
        # 1. Input guardrails
        self._input_filter.validate(prompt)

        # 2. Exact cache
        cached = self._db.get_cache(prompt)
        if cached is not None:
            logger.info("Returning exact-cache hit")
            return cached

        # 3. Semantic cache
        if self._faiss is not None:
            sem_hit = self._faiss.search(prompt)
            if sem_hit is not None:
                logger.info("Returning semantic-cache hit")
                return sem_hit

        # 4. Route to model
        model_id = self._router.get_model(task_type)

        # 5 & 6. Generate (via rate limiter if present)
        t0 = time.perf_counter()
        if self._rate_limiter:
            raw_resp = self._rate_limiter.execute(
                self._ollama.generate, model_id, prompt, **kwargs
            )
        else:
            raw_resp = self._ollama.generate(model_id, prompt, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1_000

        response_text: str = raw_resp.get("response", "")
        prompt_tokens: int = raw_resp.get("prompt_eval_count", 0)
        completion_tokens: int = raw_resp.get("eval_count", 0)

        # 7. Output validation
        result: str | dict
        if expected_schema is not None:
            result = self._output_validator.validate(response_text, expected_schema)
        else:
            result = response_text

        # 8. Token logging
        self._db.log_usage(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            session_id=session_id,
        )

        # 9. Persist to caches & A2A memory
        self._db.set_cache(prompt, response_text, model_id)
        if self._faiss is not None:
            self._faiss.add(prompt, response_text)
        if agent_name and session_id:
            self._db.save_agent_state(session_id, agent_name, response_text)

        return result

    # ------------------------------------------------------------------
    # Internal helper used by OutputSchemaValidator for repair prompts
    # ------------------------------------------------------------------

    def _raw_generate(self, repair_prompt: str) -> str:
        """Send a raw prompt (no caching / guardrails) for output repair."""
        model_id = self._ollama.active_model or self._router.get_model("general")
        resp = self._ollama.generate(model_id, repair_prompt)
        return resp.get("response", "")
