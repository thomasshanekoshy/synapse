# Synapze TODO & Backlog

## Urgent Technical Explorations
- [ ] **React Exploration**: Learn how to utilize React and Tailwind to build custom frontend components (e.g., Chat UI, 4-Quadrant Planner view, Financial Sankey diagrams).
- [ ] **Off-The-Shelf UI**: Evaluate frameworks like LibreChat and Open-WebUI. Create a sub-list of criteria to see if custom React overhead can be completely avoided.
- [ ] **YouTube Testing**: Write a script to test `youtube-transcript-api` for basic scraping. If it fails, create a fallback script testing `yt-dlp` specifically to extract raw audio streams and extensive publisher/timestamp metadata.
- [ ] **Financial APIs**: Write basic data polling scripts for FMP (Financial Modeling Prep) API. Outline testing steps for `OpenBB Terminal SDK` for viability versus overhead. 

## Agent Implementations
### 1. Voice-to-Text
- [ ] Implement backend routing in `Core/models/ollama_client.py` to pipe byte-buffers of audio into Gemma-4 for single-pass multimodal inference without Whisper dependencies.
- [ ] Create API formatters in the `Tools` folder to ingest user audio and return JSON intent.

### 2. Planner
- [ ] Map FastAPI backends for the 4-Quadrant task planner GUI.
- [ ] Implement local LLM priority categorization endpoints.

### 3. Coder
- [ ] Setup `Aider` (CLI) workflows and IDE extensions (`Continue.dev`). 
- [ ] Investigate leveraging CrewAI or LangChain code-generation models by wrapping them in MCP (Model Context Protocol).

### 4. Stock Researcher & Financial Analyst
- [ ] Implement PDF parsing mechanisms for importing semi-structured banking and credit card statements.
- [ ] Pass the parsed metrics to the frontend to populate React visualizations.

### 5. PII Abstraction
- [ ] Validate Microsoft Presidio on sample local data to ensure high-accuracy deterministic masking (no LLM usage here).

### 6. Article Writer
- [ ] Establish Obsidian mapping. Test reading a structured graph containing templates and TODOs.

### 7. YouTube Summarizer & Tracker
- [ ] Pass `yt-dlp` extracted histories and metadata attributes into local Qwen embeddings (FAISS) for topic clustering.

## Testing Milestones
- [ ] Implement `unittest.mock` fixtures inside `/tests` targeting Core classes (`synapze_llm`, `ollama_client`, `sqlite_manager`).
- [ ] Configure `pytest` coverage constraints for GitHub CI runs once Workflows are defined.
- [ ] Script Integration mock tools to validate data-handshake between Multi-Agent chains in `Orchestration/`.
- [ ] Investigate Playwright for Python (`pytest-playwright`) to automate GUI & React testing to lock production E2E features, supplementing organic agent tests.
