from unittest.mock import MagicMock

import pytest

from rag_system import RAGSystem


@pytest.fixture
def rag_system(mocker, test_config):
    mock_vector_store_cls = mocker.patch("rag_system.VectorStore")
    mock_ai_generator_cls = mocker.patch("rag_system.AIGenerator")
    mock_session_manager_cls = mocker.patch("rag_system.SessionManager")

    system = RAGSystem(test_config)

    system.mock_vector_store = mock_vector_store_cls.return_value
    system.mock_ai_generator = mock_ai_generator_cls.return_value
    system.mock_session_manager = mock_session_manager_cls.return_value
    return system


class TestRAGSystemQuery:
    def test_query_returns_response_and_sources(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        mocker.patch.object(
            rag_system.tool_manager, "get_last_sources", return_value=[{"text": "Course A", "link": None}]
        )
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        response, sources = rag_system.query("What is X?")

        assert response == "the answer"
        assert sources == [{"text": "Course A", "link": None}]

    def test_query_builds_expected_prompt_and_passes_tool_definitions(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("What is X?")

        call_kwargs = rag_system.mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["query"] == "Answer this question about course materials: What is X?"
        assert call_kwargs["tools"] == rag_system.tool_manager.get_tool_definitions()
        assert call_kwargs["tool_manager"] is rag_system.tool_manager

    def test_query_resets_sources_after_retrieval(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        get_sources = mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        reset_sources = mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("What is X?")

        get_sources.assert_called_once()
        reset_sources.assert_called_once()

    def test_query_with_session_id_fetches_and_passes_history(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        rag_system.mock_session_manager.get_conversation_history.return_value = "prior turns"
        mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("q", session_id="s1")

        rag_system.mock_session_manager.get_conversation_history.assert_called_once_with("s1")
        call_kwargs = rag_system.mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "prior turns"

    def test_query_without_session_id_skips_history_lookup(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("q")

        rag_system.mock_session_manager.get_conversation_history.assert_not_called()
        call_kwargs = rag_system.mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] is None

    def test_query_with_session_id_saves_exchange(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("q", session_id="s1")

        rag_system.mock_session_manager.add_exchange.assert_called_once_with("s1", "q", "the answer")

    def test_query_without_session_id_does_not_save_exchange(self, rag_system, mocker):
        rag_system.mock_ai_generator.generate_response.return_value = "the answer"
        mocker.patch.object(rag_system.tool_manager, "get_last_sources", return_value=[])
        mocker.patch.object(rag_system.tool_manager, "reset_sources")

        rag_system.query("q")

        rag_system.mock_session_manager.add_exchange.assert_not_called()
