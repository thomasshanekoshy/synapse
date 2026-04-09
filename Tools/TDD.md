# Synapze Tools TDD

## Architecture Blueprint
The scripts in `Tools/` operate as decoupled micro-services accessible via FastAPI orchestrators defined in `Core/`. 

### Stack Data Flow
`React GUI` $\rightarrow$ `FastAPI Core` $\rightarrow$ `Tools (Execution Scripts)` $\rightarrow$ `FastAPI Core (LLM Formatting)` $\rightarrow$ `React GUI`.

### Integration Matrix

- **Frontend Tech Constraint**: All visualization and interactive components are delegated to **React** and Tailwind, moving decisively away from Python-native GUI libraries like Streamlit. Alternatively, exploration of existing ecosystems (`LibreChat`, `Open-WebUI`) will be conducted prior to hard-coding React components.
- **Financial Architecture**: Relying on standard web-polling via the `FMP API` to ensure robust JSON data frames. `OpenBB SDK` will be considered strictly if complex analytical metrics outgrow FMP parameters. Parsing will necessitate standard PDF extraction implementations (`PyMuPDF` or equivalent) since CSV definitions are excluded for standard bank records.
- **IDE AI Workflows**: Utilizing `Aider` within terminal interfaces natively supports large-scale git repository editing. Future architectural enhancements aim to route native Code Assistant payloads directly into CrewAI-orchestrated workflows for massive multi-agent refactors.
