# Synapze Specifications (SPEC.md)

## 1. Overview
This document serves as the formal specification for the Synapze management layer. Following Specification-Driven Development (SDD), all code and tests must conform to the interfaces, data models, and component contracts defined here.

## 2. File and Directory Structure
```text
Synapze/
├── Core/
│   ├── __init__.py
│   ├── models/
│   │   ├── base.py                 # Abstract base class for models
│   │   ├── ollama_client.py        # Low-level VRAM management and Ollama API wrapper
│   │   └── synapze_llm.py          # The SDK adapter for CrewAI/Langchain
│   ├── memory/
│   │   ├── sqlite_manager.py       # Exact cache, token tracking, history, A2A state
│   │   └── faiss_manager.py        # Decoupled semantic caching
│   ├── routing/
│   │   └── dynamic_router.py       # Logic to map prompts/tasks to specific models
│   ├── guardrails/
│   │   ├── input_filter.py         # Prompt safety and injection screening
│   │   └── output_schema.py        # Pydantic validation and retry logic
│   └── evals/
│       ├── feedback_loop.py        # User feedback logging to SQLite
│       └── benchmarks.py           # Model and task-based eval logic
└── Orchestration/                  # Exposes interfaces for CrewAI usage
├── Tools/                          # Actionable code and FastAPI endpoints
├── Skills/                         # MD files, System prompts, Agent templates
└── Workflows/                      # Coordination schemas mapping Tools to Skills
```

## 3. Core Class Specifications

### 3.1 `OllamaClient` (VRAM Management)
Responsible for direct communication with the local Ollama API.
*   **State**: `_active_model_id` (str or None)
*   **Method `unload_model(model_id: str)`**: Issues a request to Ollama to drop the model from VRAM (e.g., via `/api/generate` with `keep_alive=0`).
*   **Method `ensure_active(model_id: str)`**: If `model_id != _active_model_id`, unloads `_active_model_id`, then updates the state to the new `model_id`.

### 3.2 `SynapzeLLM` (CrewAI/Agent Wrapper)
The public-facing class that agentic frameworks instantiate, functioning concurrently as the primary FastAPI endpoint provider.
*   **Dependencies**: Requires `OllamaClient`, `DynamicRouter`, `SQLiteManager`, `GuardrailsEngine`.
*   **Method `invoke(prompt: str, task_type: str = "general", expected_schema: BaseModel = None, session_id: str = None) -> Union[str, dict]`**:
    1.  Runs `GuardrailsEngine.validate_input(prompt)`.
    2.  Checks `SQLiteManager.get_cache(prompt)`. If hit, return.
    3.  If semantic caching is enabled, checks `FaissManager.search(prompt)`.
    4.  Calls `DynamicRouter.get_model_id(prompt, task_type)` to determine target model.
    5.  Calls `OllamaClient.ensure_active(target_model_id)`.
    6.  Sends generation request to Ollama.
    7.  If `expected_schema` exists, runs `GuardrailsEngine.validate_output()`. If failed, auto-corrects.
    8.  Logs tokens via `SQLiteManager.log_usage()`.
    9.  Returns response, saving to `SQLiteManager` for A2A and caching.

### 3.3 `SQLiteManager`
Manages all persistent state, telemetry, and exact caching.
*   **Database Schema**:
    *   `cache_exact`: (id, prompt_hash, response, timestamp)
    *   `token_usage`: (id, session_id, model_id, prompt_tokens, completion_tokens, timestamp)
    *   `a2a_memory`: (session_id, agent_name, final_state_summary, timestamp)
    *   `user_feedback`: (id, session_id, task_id, user_rating, user_comment, timestamp)

### 3.4 `GuardrailsEngine`
*   **Method `validate_input(prompt: str) -> bool`**: Returns false or raises exception if safety constraints are breached.
*   **Method `validate_output(raw_response: str, schema: BaseModel) -> dict`**: Attempts to parse the raw text output from Ollama into the Pydantic schema requested by the agent (tool signature). If JSON parsing fails, manages a 1-step repair prompt.

### 3.5 `DynamicRouter`
*   **Method `get_model(prompt: str, task_category: str) -> str`**:
    *   *Rules*: If `task_category == 'simple_summary'`, return `lfm2.5-thinking:latest`. If `task_category == 'complex_code'`, return `deepseek-r1:14b`. Else return default default fallback.

## 4. Workflows & Interactions

### MCP Integration Flow
When a Model Context Protocol payload is received, it is stored temporarily in `SQLiteManager.a2a_memory` or directly appended to the prompt context array before `GuardrailsEngine.validate_input(prompt)` executes.

### Tool Execution Flow (Out of Scope for Synapze)
1. CrewAI requests tool schema format.
2. `SynapzeLLM.invoke` passes schema requirements.
3. Ollama responds.
4. `GuardrailsEngine.validate_output()` guarantees valid JSON.
5. Returns JSON to CrewAI. CrewAI parses JSON and executes the physical tool.
