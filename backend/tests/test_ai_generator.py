from unittest.mock import MagicMock

import pytest

import ai_generator
from ai_generator import AIGenerator


@pytest.fixture
def mock_anthropic_client(mocker):
    mock_client_cls = mocker.patch("ai_generator.anthropic.Anthropic")
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    return mock_client


@pytest.fixture
def generator(mock_anthropic_client):
    return AIGenerator(api_key="test-key", model="test-model")


class TestGenerateResponseWithoutTools:
    def test_no_tool_use_returns_text_directly(self, generator, mock_anthropic_client, make_text_response):
        mock_anthropic_client.messages.create.return_value = make_text_response("Paris is the capital of France.")

        result = generator.generate_response(query="What is the capital of France?")

        assert result == "Paris is the capital of France."
        assert mock_anthropic_client.messages.create.call_count == 1

    def test_includes_conversation_history_in_system_prompt(self, generator, mock_anthropic_client, make_text_response):
        mock_anthropic_client.messages.create.return_value = make_text_response("answer")

        generator.generate_response(query="follow up", conversation_history="User: foo\nAssistant: bar")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "User: foo\nAssistant: bar" in call_kwargs["system"]
        assert AIGenerator.SYSTEM_PROMPT in call_kwargs["system"]

    def test_without_tools_omits_tool_kwargs(self, generator, mock_anthropic_client, make_text_response):
        mock_anthropic_client.messages.create.return_value = make_text_response("answer")

        generator.generate_response(query="hello", tools=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_with_tools_passes_tool_choice_auto(self, generator, mock_anthropic_client, make_text_response):
        mock_anthropic_client.messages.create.return_value = make_text_response("answer")
        tools = [{"name": "search_course_content", "input_schema": {}}]

        generator.generate_response(query="hello", tools=tools)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}


class TestToolUseFlow:
    def test_tool_use_triggers_tool_execution(self, generator, mock_anthropic_client, make_tool_use_response, make_text_response):
        first_response = make_tool_use_response(
            [{"id": "tool_1", "name": "search_course_content", "input": {"query": "x"}}]
        )
        second_response = make_text_response("final answer")
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result text"

        result = generator.generate_response(
            query="search for x",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="x")
        assert result == "final answer"
        assert mock_anthropic_client.messages.create.call_count == 2

    def test_multi_tool_call_handling(self, generator, mock_anthropic_client, make_tool_use_response, make_text_response):
        first_response = make_tool_use_response(
            [
                {"id": "tool_1", "name": "search_course_content", "input": {"query": "x"}},
                {"id": "tool_2", "name": "get_course_outline", "input": {"course_name": "y"}},
            ]
        )
        second_response = make_text_response("final answer")
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = ["result 1", "result 2"]

        generator.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert tool_manager.execute_tool.call_args_list == [
            (("search_course_content",), {"query": "x"}),
            (("get_course_outline",), {"course_name": "y"}),
        ]

        second_call_messages = mock_anthropic_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_message = second_call_messages[-1]
        assert tool_result_message["role"] == "user"
        tool_use_ids = [block["tool_use_id"] for block in tool_result_message["content"]]
        assert tool_use_ids == ["tool_1", "tool_2"]
        contents = [block["content"] for block in tool_result_message["content"]]
        assert contents == ["result 1", "result 2"]

    def test_second_call_excludes_tools_kwarg(self, generator, mock_anthropic_client, make_tool_use_response, make_text_response):
        first_response = make_tool_use_response(
            [{"id": "tool_1", "name": "search_course_content", "input": {"query": "x"}}]
        )
        second_response = make_text_response("final answer")
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result"

        generator.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs

    def test_appends_assistant_and_user_messages_correctly(self, generator, mock_anthropic_client, make_tool_use_response, make_text_response):
        first_response = make_tool_use_response(
            [{"id": "tool_1", "name": "search_course_content", "input": {"query": "x"}}]
        )
        second_response = make_text_response("final answer")
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result"

        generator.generate_response(
            query="original query", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call_messages = mock_anthropic_client.messages.create.call_args_list[1].kwargs["messages"]
        assert second_call_messages[0] == {"role": "user", "content": "original query"}
        assert second_call_messages[1] == {"role": "assistant", "content": first_response.content}
        assert second_call_messages[2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "tool result"}],
        }


class TestThinkingDisabledRegression:
    def test_base_params_thinking_disabled(self, generator):
        assert generator.base_params["thinking"] == {"type": "disabled"}

    def test_thinking_disabled_in_both_calls_of_tool_use_flow(
        self, generator, mock_anthropic_client, make_tool_use_response, make_text_response
    ):
        first_response = make_tool_use_response(
            [{"id": "tool_1", "name": "search_course_content", "input": {"query": "x"}}]
        )
        second_response = make_text_response("final answer")
        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "tool result"

        generator.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        for call in mock_anthropic_client.messages.create.call_args_list:
            assert call.kwargs["thinking"] == {"type": "disabled"}


class TestExtractText:
    def test_extract_text_raises_on_no_text_block(self):
        response = type("Response", (), {})()
        response.content = [type("Block", (), {"type": "tool_use"})()]

        with pytest.raises(StopIteration):
            AIGenerator._extract_text(response)
