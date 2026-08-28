import pytest

from search_tools import CourseSearchTool, CourseOutlineTool, Tool, ToolManager


# ---------------------------------------------------------------------------
# CourseSearchTool
# ---------------------------------------------------------------------------

class TestCourseSearchTool:
    def test_execute_successful_search_formats_results(self, mock_vector_store, populated_search_results):
        mock_vector_store.search.return_value = populated_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="what is a vector")

        assert "[Intro to RAG - Lesson 1]" in result
        assert "Introduction to embeddings." in result
        assert "[Intro to RAG - Lesson 2]" in result
        assert "Vector similarity search." in result
        assert result.count("\n\n") == 1  # two blocks joined by a blank line

    def test_execute_search_without_lesson_number_in_metadata(self, mock_vector_store):
        from vector_store import SearchResults

        mock_vector_store.search.return_value = SearchResults(
            documents=["Course-level content."],
            metadata=[{"course_title": "Intro to RAG"}],
            distances=[0.1],
        )
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert result.startswith("[Intro to RAG]\nCourse-level content.")

    def test_execute_propagates_store_error(self, mock_vector_store, error_search_results):
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert result == "Search error: connection failed"
        assert tool.last_sources == []

    def test_execute_empty_results_no_filters(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_name_filter(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything", course_name="Intro to RAG")

        assert result == "No relevant content found in course 'Intro to RAG'."

    def test_execute_empty_results_with_course_and_lesson_filter(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything", course_name="Intro to RAG", lesson_number=2)

        assert result == "No relevant content found in course 'Intro to RAG' in lesson 2."

    def test_execute_populates_last_sources_with_links(self, mock_vector_store, populated_search_results):
        mock_vector_store.search.return_value = populated_search_results
        mock_vector_store.get_lesson_link.side_effect = (
            lambda course_title, lesson_number: f"https://example.com/lesson/{lesson_number}"
        )
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="what is a vector")

        assert tool.last_sources == [
            {"text": "Intro to RAG - Lesson 1", "link": "https://example.com/lesson/1"},
            {"text": "Intro to RAG - Lesson 2", "link": "https://example.com/lesson/2"},
        ]
        mock_vector_store.get_lesson_link.assert_any_call("Intro to RAG", 1)
        mock_vector_store.get_lesson_link.assert_any_call("Intro to RAG", 2)

    def test_execute_last_sources_when_link_lookup_returns_none(self, mock_vector_store, populated_search_results):
        mock_vector_store.search.return_value = populated_search_results
        mock_vector_store.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="what is a vector")

        assert all(source["link"] is None for source in tool.last_sources)

    def test_execute_passes_filters_through_to_store_search(self, mock_vector_store, empty_search_results):
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="q", course_name="C", lesson_number=3)

        mock_vector_store.search.assert_called_once_with(query="q", course_name="C", lesson_number=3)

    def test_get_tool_definition_shape(self, mock_vector_store):
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


# ---------------------------------------------------------------------------
# CourseOutlineTool
# ---------------------------------------------------------------------------

class TestCourseOutlineTool:
    def test_execute_course_found(self, mock_vector_store, sample_course_outline_dict):
        mock_vector_store.get_course_outline.return_value = sample_course_outline_dict
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Intro")

        assert "Course Title: Intro to RAG" in result
        assert "Course Link: https://example.com/intro-to-rag" in result
        assert "Lessons:" in result
        assert "0. Welcome" in result
        assert "1. Embeddings" in result

    def test_execute_course_not_found(self, mock_vector_store):
        mock_vector_store.get_course_outline.return_value = None
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_name="Nonexistent")

        assert result == "No course found matching 'Nonexistent'"

    def test_get_tool_definition_shape(self, mock_vector_store):
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class _FakeTool(Tool):
    def __init__(self, name, result="fake result"):
        self._name = name
        self._result = result
        self.executed_with = None

    def get_tool_definition(self):
        return {"name": self._name, "description": "fake", "input_schema": {}}

    def execute(self, **kwargs):
        self.executed_with = kwargs
        return self._result


class _NamelessTool(Tool):
    def get_tool_definition(self):
        return {"description": "no name here", "input_schema": {}}

    def execute(self, **kwargs):
        return "unused"


class TestToolManager:
    def test_register_tool_keys_by_name(self):
        manager = ToolManager()
        tool = _FakeTool("foo")

        manager.register_tool(tool)

        assert manager.execute_tool("foo") == "fake result"

    def test_register_tool_without_name_raises_value_error(self):
        manager = ToolManager()

        with pytest.raises(ValueError):
            manager.register_tool(_NamelessTool())

    def test_execute_tool_dispatches_to_correct_tool(self):
        manager = ToolManager()
        tool_a = _FakeTool("a", result="result-a")
        tool_b = _FakeTool("b", result="result-b")
        manager.register_tool(tool_a)
        manager.register_tool(tool_b)

        result = manager.execute_tool("b", query="hello")

        assert result == "result-b"
        assert tool_b.executed_with == {"query": "hello"}
        assert tool_a.executed_with is None

    def test_execute_tool_unknown_name_returns_message(self):
        manager = ToolManager()

        result = manager.execute_tool("nonexistent")

        assert result == "Tool 'nonexistent' not found"

    def test_get_tool_definitions_aggregates_all_registered(self):
        manager = ToolManager()
        manager.register_tool(_FakeTool("a"))
        manager.register_tool(_FakeTool("b"))

        definitions = manager.get_tool_definitions()

        assert {d["name"] for d in definitions} == {"a", "b"}

    def test_get_last_sources_aggregates_across_tools(self, mock_vector_store, populated_search_results):
        mock_vector_store.search.return_value = populated_search_results
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        search_tool.execute(query="anything")

        assert manager.get_last_sources() == search_tool.last_sources

    def test_reset_sources_clears_all_tools(self, mock_vector_store, populated_search_results):
        mock_vector_store.search.return_value = populated_search_results
        search_tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(search_tool)
        search_tool.execute(query="anything")
        assert search_tool.last_sources != []

        manager.reset_sources()

        assert search_tool.last_sources == []
        assert manager.get_last_sources() == []
