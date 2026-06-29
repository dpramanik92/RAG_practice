# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for querying India's Ancient History. The PDF source (`data/Indias_Ancient_history_full.pdf`) is chunked by chapter, embedded with a local HuggingFace model, stored in a FAISS vector index, and queried via a Streamlit chat UI backed by OpenAI.

## Environment Setup

The virtualenv lives at `./rag/` (Python 3.13). Activate it before running anything:

```bash
source rag/bin/activate
pip install -r requirements.txt
```

Required `.env` variables:
- `OPENAI_API_KEY` — used by `app.py` and notebooks for `gpt-4o-mini`
- `LANGSMITH_API_KEY` — optional; used for tracing in `rag_history_v2.ipynb`

## Commands

**Build the FAISS index** (must run before the app):
```bash
python build_index.py
```

**Run the Streamlit app:**
```bash
streamlit run app.py
```

**Run a notebook:**
```bash
jupyter notebook first_rag.ipynb
# or
jupyter notebook rag_history_v2.ipynb
```

## Architecture

### Index pipeline (`build_index.py`)
- Loads the PDF with PyMuPDF, skips the first 14 pages (front matter)
- Extracts chapter boundaries via regex, splitting the full text into 33 chapter segments
- Each segment becomes a `Document` with `{"chapter": <title>}` metadata
- Chunks with `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap)
- Embeds with `BAAI/bge-small-en-v1.5` (local HuggingFace model, no API key needed)
- Saves FAISS index to `data/faiss_index/` (`.faiss` and `.pkl` files are gitignored)

### Chat app (`app.py`)
- Loads the FAISS index at startup (cached via `@st.cache_resource`)
- On each query: similarity-searches top-5 chunks, builds a prompt with context + last 5 conversation turns, calls `gpt-4o-mini` directly via the OpenAI SDK
- Maintains chat history in `st.session_state.messages`

### Notebooks
- `first_rag.ipynb` — initial exploratory RAG implementation
- `rag_history_v2.ipynb` — extended version with richer chunk metadata (`chunk_label`, `chunk_level`), LangSmith tracing support, and RAG evaluation against a golden dataset (`data/golden_data_ancient.xlsx`)

### Data
- `data/Indias_Ancient_history_full.pdf` — source document (33 chapters)
- `data/faiss_index/` — persisted vector index (gitignored)
- `data/golden_data_ancient.xlsx` / `.csv` — evaluation Q&A pairs with ground-truth chunk references

## Git Hooks

A pre-push hook scans diffs and notebook cell outputs for API key patterns. Activate it once per clone:

```bash
git config core.hooksPath .githooks
```
