# Synapze Technical Design Document (TDD)

## 1. Subsystem Purpose
Synapze is a foundational local model management and orchestration layer. It acts as a comprehensive proxy and intelligence router between local Ollama instances and higher-level agentic frameworks (such as CrewAI). The core focus is maintaining strict VRAM limits, reducing cost/latency via caching and dynamic routing, and enforcing privacy.

## 2. Scope & Out of Scope

### **In Scope**
*   **Agentic Framework SDK**: Base integration layer (exposing LLM classes conformable to CrewAI/LangChain standards).
*   **VRAM Management**: Strict 1-active-model policy. Logic to automatically unload models from Ollama memory when switching models.
*   **Storage & Caching**: 
    *   SQLite implementation for token tracking, state memory, and exact-match caching.
    *   Decoupled, modular FAISS implementation for semantic caching (easily swappable).
*   **Guardrails (Lightweight, Custom)**:
    *   *Input*: Lightweight screening for prompt injection/safety.
    *   *Output*: Enforced JSON schema validation checking prior to returning data to the agentic layer.
*   **Dynamic Routing**: Automated routing strategies (e.g., standardizing simple summarization tasks to `lfm2.5-thinking` and complex logical reasoning to `deepseek-r1:14b`).
*   **Prompt Registry Engine**: Embedded version control for system prompts and guardrail instructions.
*   **Concurrency & Rate Limiting**: Request buffering and exponential backoff to preventing Ollama server overload.
*   **Evaluations**: Model/Task-based benchmarking and structured **User Feedback loops** (recording user corrections in SQLite).

### **Out of Scope**
*   **Tool Execution & Calling Pipelines**: The execution, definition, and state management of external tools/APIs are a separate project. Synapze *only* guarantees that the LLM's output conforms to the requested tool's JSON schema constraints.
*   **Heavyweight External Guardrails**: Frameworks like LlamaGuard or NeMo-Guardrails are deferred in favor of fast, custom, lightweight checks.
*   **UI/Frontend**: No user interface components; pure SDK/Backend.

---

## 3. Integration Approaches & Modern Concepts

### 3.1 Agentic Frameworks (CrewAI)
Synapze will expose Python classes (e.g., `SynapzeModel`) that inherit from or mimic base interfaces expected by frameworks like CrewAI (typically Langchain's `BaseChatModel` or `BaseLLM`). 
*   **Workflow**: When an agent requests a completion, CrewAI calls `.invoke()` on `SynapzeModel`. Synapze intercepts this: checks SQLite cache -> if miss, checks FAISS semantic cache -> if miss, unloads current Ollama model -> loads target model -> queries Ollama -> runs lightweight output guardrails -> logs tokens/latency to SQLite -> returns formatted response to CrewAI.

### 3.2 Input/Output Schemas (Without Tool Execution)
Since tools are a separate project, Synapze handles I/O schemas purely as string-to-structured-data validation:
*   Instead of calling a web API, Synapze forces the Ollama model into returning structured JSON (via format/system prompt constraints).
*   If CrewAI passes up a tool schema signature, Synapze uses Pydantic to validate the LLM's raw text output against that schema. If invalid, Synapze internally prompts the model to correct itself. It then hands the valid JSON string back to CrewAI, which is responsible for the actual tool execution.

### 3.3 Model Context Protocol (MCP)
*   As the Model Context Protocol becomes standard for injecting context (e.g., local codebase, DB schemas) dynamically into prompts, Synapze's **Prompt Registry Engine** will be designed to naturally ingest MCP context chunks.
*   It will stitch MCP metadata cleanly into the prompt context window before applying Input Guardrails.

### 3.4 API Based Tools
*   Synapze remains completely agnostic to what the API tooling is doing.
*   Its only responsibility is maintaining state memory in SQLite so that if an agent receives a massive API payload back from a tool (executed by the separate project), Synapze can accurately track the token cost of processing that payload.

### 3.5 A2A (Agent-to-Agent) Communication
*   A2A requires robust shared memory. Because Synapze handles **State Memory** via SQLite, it acts as a centralized brain. 
*   When Agent A finishes a task in CrewAI, its final state and context trajectory are saved in Synapze's SQLite DB. When Agent B spins up, Synapze can hydrate Agent B's system prompt with Agent A's summarized outputs, facilitating seamless A2A handoffs without bloating the active memory limit.
