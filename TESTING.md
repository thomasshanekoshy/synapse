# Synapze Testing Strategy

This document outlines the testing methodologies (Unit, Integration, and End-to-End) required to maintain the stability and reliability of the Synapze framework and its various agentic modules.

## 1. Core Module (`/Core`)
Contains foundational logic, memory persistence, guardrails, and model LLM coordination APIs.

- **Unit Testing**:
  - Mock external providers (`synapze_llm`, `ollama_client`) using standard libraries (e.g., `unittest.mock`).
  - Pass synthetic text to `prompt_registry` and `rate_limiter` to assert expected deterministic logic (Token calculation limits, Jinja2 template resolutions).
  - Test SQLite transactions cleanly isolated using in-memory databases.

- **Integration Testing**:
  - Ensure SQLite outputs sync properly to LLM tool abstractions. Ensure that retrieved database contexts populate specific prompt templates correctly without escaping issues.

- **End-to-End (E2E) Testing**:
  - Spin up mock models or use a lightweight local model configuration (e.g. `gemma:2b`) and run full input cascades. Check that a system prompt goes through the router, hits rate limits (or not), generates text, and passes safely through output guardrails.

## 2. Orchestration Module (`/Orchestration`)
Contains higher-level multi-model coordination scripts and logical chains.

- **Unit Testing**:
  - Assert that fallback hierarchies and JSON chain configurations are structurally sound using schema validators like `Pydantic`.
  - Validate state transitions to ensure a tool failure leads to the defined fallback node.

- **Integration Testing**:
  - Load and coordinate two simulated agent endpoints through the router. Mock output from `Agent A` as input to `Agent B` to verify pipeline throughput formats.

- **End-to-End (E2E) Testing**:
  - Perform live execution paths using local models. Send a composite query requiring multi-steps (e.g., "Analyze the stock AAPL, and format it as an article fallback") and assert the holistic routing path triggered and succeeded.

## 3. Tools & Workflows (`/Tools`, `/Workflows`)
Concrete pipelines, YAML task sequences, and endpoint wrappers.

- **Unit Testing**:
  - Verify individual tool schemas and parameter bounds (e.g., parsing a specific date formatting regex).
  - Verify chunking algorithms (e.g., audio splitter length restrictions, pdf text extraction cleanly removing whitespace).

- **Integration Testing**:
  - Wire external scripts mockings (like `yt-dlp` output or `OpenBB SDK`) back to the ingestion queue and test indexation mappings (e.g. to a local FAISS vector database).

- **End-to-End (E2E) Testing**:
  - Run a complete declarative Markdown/YAML `Workflow` file from start to finish to ensure the runtime compiles and runs tool sequences predictably compared against known baseline responses.

## 4. Skills (`/Skills`)
These are configuration files specifying system prompt templates, schemas, and persona behaviors.

- **Unit / Integration Testing**:
  - Execute programmatic linters logic ensuring that prompt formats, required schema fields exist, and template variable strings map legally to known parameters. Run prompt injection security tests against generated prompts locally.

---

## Guidelines for UI & Desktop Automation E2E
Synapze is building toward frontend and desktop interactions via autonomous planners and UI visual overlays.

### Automated Testing (CI/CD)
While autonomous AI assistants (like Antigravity) are heavily capable of navigating dynamic UIs organically to perform live testing, reproducible automated build pipelines (CI/CD) for regressions in React webapps or GUIs **should strictly utilize Playwright for Python**. 

**Why Playwright?**
- Stable Python support out of the box (`pip install pytest-playwright`), keeping it unified with standard `pytest` logic.
- Highly reproducible, asynchronous interactions perfect for tracking deterministic, locked UI states to prevent regression errors.
- Extensive native recording and cross-browser coverage.

### Exploratory/Manual E2E (Agentic)
In unstructured evaluation and rapid-prototyping, utilize agentic tools for on-the-fly execution. Only enforce Playwright scripts when a feature module has "matured" into production to ensure reliable backward compatibility.
