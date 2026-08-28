import sys
import types
from unittest.mock import MagicMock

import pytest

from config import Config
from vector_store import VectorStore, SearchResults


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=VectorStore)
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    store.get_course_outline.return_value = None
    return store


@pytest.fixture
def populated_search_results():
    return SearchResults(
        documents=[
            "Lesson 1 content: Introduction to embeddings.",
            "Lesson 2 content: Vector similarity search.",
        ],
        metadata=[
            {"course_title": "Intro to RAG", "lesson_number": 1},
            {"course_title": "Intro to RAG", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    return SearchResults.empty("Search error: connection failed")


@pytest.fixture
def sample_course_outline_dict():
    return {
        "title": "Intro to RAG",
        "course_link": "https://example.com/intro-to-rag",
        "lessons": [
            {"lesson_number": 0, "lesson_title": "Welcome"},
            {"lesson_number": 1, "lesson_title": "Embeddings"},
        ],
    }


@pytest.fixture
def test_config(tmp_path):
    return Config(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_BASE_URL="",
        ANTHROPIC_MODEL="test-model",
        EMBEDDING_MODEL="test-embed-model",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        CHROMA_PATH=str(tmp_path / "chroma"),
    )


def _make_block(**kwargs):
    return types.SimpleNamespace(**kwargs)


@pytest.fixture
def make_text_response():
    def _make(text, stop_reason="end_turn"):
        return types.SimpleNamespace(
            stop_reason=stop_reason,
            content=[_make_block(type="text", text=text)],
        )

    return _make


@pytest.fixture
def make_tool_use_response():
    def _make(tool_calls):
        blocks = [
            _make_block(type="tool_use", id=call["id"], name=call["name"], input=call["input"])
            for call in tool_calls
        ]
        return types.SimpleNamespace(stop_reason="tool_use", content=blocks)

    return _make
