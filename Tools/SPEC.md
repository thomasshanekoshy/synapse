# Synapze Tools Specification

## 1. Scope
The `Tools/` directory contains all actionable code and API scripts designed to parse real-world data and provide formatted state packages to the Synapze LLM Core. This is purely a backend operations layer; all graphical presentation logic resides in external **React** components.

## 2. Tool Workflows

### 2.1 Voice-to-Text (Ollama Multimodal)
- **Input**: Audio streams (mp3, wav).
- **Execution**: The audio bytes are routed via FastAPI (`Synapze/Core`) directly into a multimodal local model (e.g., `gemma4:latest`) via the Ollama client wrapper.
- **Output**: JSON payload containing accurate transcripts and speaker intent extracted in a single pass. 
- **Code Separation**: Model loading logic sits in `Core/models`. The data-ingestion handling and API endpoint formatting sits heavily in `Tools/`.

### 2.2 Financial Analyst (PDF Extractor)
- **Input**: Semi-structured PDF exports of bank statements or credit card histories.
- **Execution**: Parses tabular and semi-structured fields from PDFs, feeding the slice through the local model for semantic categorization (identifying runway, subscriptions, categories).
- **Output**: Returns structured JSON arrays to the **React dashboard** for immediate visual analysis (e.g., Sankey charts).

### 2.3 Stock Researcher
- **Input**: React Chat UI prompts.
- **Execution**: Polls `FMP API`/OpenBB for fundamentals. Performs web-search. 
- **Output**: Combines findings into markdown payloads streamed directly back to the React UI.

### 2.4 YouTube Extractor
- **Input**: YouTube URLs or User History datasets.
- **Execution**: Relies on `yt-dlp` to extract publisher context, detailed timestamps, and comprehensive metadata. Defaults to `youtube-transcript-api`; upon failure, `yt-dlp` downloads the raw audio track for native transcription.

### 2.5 Coder
- **Input**: Context parameters via Model Context Protocols (MCP).
- **Execution**: Integrates with robust off-the-shelf tools (`Aider` via CLI, `Continue.dev` / `Cline` via VS Code). Can be configured such that these tools trigger and leverage heavier `CrewAI` or `Langchain` orchestrations defined in our background processes.

### 2.6 PII Abstraction
- **Input**: Raw text blocks.
- **Execution**: Deploys Microsoft Presidio to deterministically scrub payloads prior to broader LLM exposure. No generative execution involved.
