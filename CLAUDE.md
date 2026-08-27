# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for Python dependency management (Python >=3.13).

```bash
# Install dependencies
uv sync

# Run the app (quick start script — starts uvicorn from backend/)
chmod +x run.sh && ./run.sh

# Run the app manually, with auto-reload
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app needs a `.env` file in the repo root with `ANTHROPIC_API_KEY=...` (see `.env.example`). Once running: web UI at `http://localhost:8000`, Swagger docs at `http://localhost:8000/docs`.

There is currently no test suite, linter, or formatter configured in this repo.

## Architecture

Full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course transcripts. FastAPI backend + a static vanilla-JS frontend (no build step), ChromaDB for vector storage, and Anthropic's Claude for generation via tool-calling.

### Request flow (`POST /api/query`)

`frontend/script.js` → `backend/app.py` → `RAGSystem.query()` (`rag_system.py`) is the central orchestrator:

1. `app.py` creates a session via `SessionManager` if the client didn't send one.
2. `RAGSystem` pulls prior conversation history (last `MAX_HISTORY` exchanges) from `SessionManager`.
3. `AIGenerator.generate_response()` (`ai_generator.py`) calls the Anthropic Messages API with the `search_course_content` tool available (`tool_choice="auto"`).
4. If Claude's response has `stop_reason == "tool_use"`, `AIGenerator._handle_tool_execution()` runs each requested tool call through `ToolManager.execute_tool()`, appends the tool results as a follow-up message, and calls Claude **again without tools** to force a final answer. The system prompt caps this at one search per query.
5. `CourseSearchTool.execute()` (`search_tools.py`) calls `VectorStore.search()`, formats the results as `[Course - Lesson N]\n<chunk>` blocks, and separately caches plain-text source labels on `self.last_sources` for the UI's "Sources" panel.
6. Back in `RAGSystem`, sources are read via `ToolManager.get_last_sources()` and then reset (so they don't leak into the next query), and the exchange is saved to session history.

Note: `RAGSystem.query()` returns `(response, sources)` regardless of whether a tool was actually used — for general-knowledge questions, Claude answers directly and `sources` is empty (per the system prompt in `ai_generator.py`, only course-specific questions trigger a search).

### Search tool abstraction (`search_tools.py`)

`Tool` is an ABC (`get_tool_definition()` + `execute()`); `ToolManager` registers tools by name and exposes their Anthropic tool-schema definitions. `CourseSearchTool` is currently the only tool. New tools plug in by implementing `Tool` and calling `tool_manager.register_tool(...)` in `RAGSystem.__init__`.

### Vector store (`vector_store.py`)

Two ChromaDB collections, both using the `all-MiniLM-L6-v2` sentence-transformer embedding function:
- `course_catalog` — one entry per course (title as ID), used to resolve a fuzzy/partial course name (e.g. "MCP") to an exact title via semantic search.
- `course_content` — the actual chunk embeddings, queried with an optional `$and` filter on `course_title` / `lesson_number`.

`VectorStore.search()` is the single entry point: it resolves `course_name` → exact title first, builds the filter, then queries `course_content`.

### Document ingestion

On startup, `app.py` loads every file in `../docs` via `RAGSystem.add_course_folder()`, skipping any course whose title already exists in the vector store. `DocumentProcessor.process_course_document()` (`document_processor.py`) expects this exact format:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <lesson title>
...
```

Content is split into sentence-aware chunks (`chunk_text()`) with configurable size/overlap; the first chunk of each lesson is prefixed with `"Lesson N content: "` for retrieval context.

### Config (`backend/config.py`)

Single dataclass loaded from `.env`: `ANTHROPIC_MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE`/`CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, `CHROMA_PATH`. Change retrieval/chunking behavior here rather than hardcoding in `document_processor.py` or `vector_store.py`.

### Session management (`session_manager.py`)

Conversation history is kept **in-memory only** (a dict keyed by session ID) — it does not survive a server restart, and there's no persistence layer beyond ChromaDB's on-disk collections.
