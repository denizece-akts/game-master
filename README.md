# GameMaster

GameMaster is a RAG-based chatbot for querying game data and reviews.

## Project Structure

```
.
├── main.py                 # Streamlit application entry point
├── config.json             # Main configuration
├── gamemaster/
│   ├── cli.py              # CLI entry point
│   ├── config/             # Configuration logic
│   ├── core/               # Core RAG engine logic
│   ├── services/           # External services (LLM, Embeddings)
│   ├── data/               # Data loading and processing
│   ├── evaluation/         # Evaluation scripts
│   ├── utils/              # Utility functions
│   └── resources/          # Static resources (Q&A Datasets)
```

## Setup

1. Install `uv` if you haven't already.
2. Run `uv sync` to install dependencies.
3. Configure `config.json` in the root directory if needed.

## Usage

### Run the Chatbot (Streamlit)

```bash
uv run streamlit run main.py
```

### Run the CLI

```bash
uv run python -m gamemaster.cli "Your question here"
```

### Run Evaluation

```bash
uv run python -m gamemaster.evaluation.run
```
