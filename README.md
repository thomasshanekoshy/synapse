# Synapze

Synapze is a repository for advanced agentic AI scripts, automated tasks, and local model orchestration operations.

## Local Models Tracker
We utilize various local reasoning engines, embedded agents, and vision-language models via Ollama. 

For an up-to-date breakdown of model quantizations, VRAM requirements, and standardized evaluation benchmarks (like LiveCodeBench, GPQA, and Arena Elo), please refer to the tracking documents:

- 📈 **[Models Tracker (CSV)](./models_tracker.csv)**

## Project Structure
- **`/Core`** - Foundational logic, FastAPI orchestration, and LLM utilities.
- **`/Orchestration`** - Multi-model coordination scripts and fallback definitions.
- **`/Tasks`** - Standard specialized automation targets.
- **`/Tools`** - Concrete Python execution pipelines, endpoint logic, and APIs.
- **`/Skills`** - Non-executable Markdown files specifying agent personas and templates.
- **`/Workflows`** - Declarative sequences combining specific Tools and Skills.

## Testing Strategy
To maintain reliability across the multi-agent pipelines and integrations, we maintain a strict testing methodology divided into Unit, Integration, and E2E testing for each module. 

For full details, reference the: [TESTING.md](./TESTING.md)
