import pytest

from tau2.data_model.message import ToolCall
from tau2.utils.tools import (
    _evaluate_ast_node,
    extract_tool_calls_from_text,
    is_functional_tool_call,
    parse_functional_tool_call,
    to_functional_format,
)


class TestParseFunctionalToolCall:
    """Test cases for parse_functional_tool_call function."""

    def test_simple_function_call(self):
        """Test parsing a simple function call with string arguments."""
        result = parse_functional_tool_call(
            "search_flights(origin='NYC', destination='LAX')"
        )
        expected = ToolCall(
            name="search_flights",
            arguments={"origin": "NYC", "destination": "LAX"},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_numbers(self):
        """Test parsing function call with numeric arguments."""
        result = parse_functional_tool_call("book_ticket(flight_id=123, price=299.99)")
        expected = ToolCall(
            name="book_ticket",
            arguments={"flight_id": 123, "price": 299.99},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_booleans(self):
        """Test parsing function call with boolean arguments."""
        result = parse_functional_tool_call(
            "update_preferences(notifications=True, marketing=False)"
        )
        expected = ToolCall(
            name="update_preferences",
            arguments={"notifications": True, "marketing": False},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_none(self):
        """Test parsing function call with None value."""
        result = parse_functional_tool_call("set_value(key='test', value=None)")
        expected = ToolCall(
            name="set_value",
            arguments={"key": "test", "value": None},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_lists(self):
        """Test parsing function call with list arguments."""
        result = parse_functional_tool_call("process_items(items=['a', 'b', 'c'])")
        expected = ToolCall(
            name="process_items",
            arguments={"items": ["a", "b", "c"]},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_dicts(self):
        """Test parsing function call with dictionary arguments."""
        result = parse_functional_tool_call(
            "update_config(config={'timeout': 30, 'retries': 3})"
        )
        expected = ToolCall(
            name="update_config",
            arguments={"config": {"timeout": 30, "retries": 3}},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_tuples(self):
        """Test parsing function call with tuple arguments."""
        result = parse_functional_tool_call(
            "set_coordinates(coords=(40.7128, -74.0060))"
        )
        expected = ToolCall(
            name="set_coordinates",
            arguments={"coords": (40.7128, -74.0060)},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_nested_structures(self):
        """Test parsing function call with nested data structures."""
        result = parse_functional_tool_call(
            "complex_call(data={'users': [{'name': 'John', 'active': True}, {'name': 'Jane', 'active': False}]})"
        )
        expected = ToolCall(
            name="complex_call",
            arguments={
                "data": {
                    "users": [
                        {"name": "John", "active": True},
                        {"name": "Jane", "active": False},
                    ]
                }
            },
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_no_arguments(self):
        """Test parsing function call with no arguments."""
        result = parse_functional_tool_call("refresh()")
        expected = ToolCall(name="refresh", arguments={}, requestor="assistant")
        assert result == expected

    def test_function_call_with_spaces(self):
        """Test parsing function call with extra spaces."""
        result = parse_functional_tool_call(
            "  search_flights  (  origin = 'NYC' , destination = 'LAX'  )  "
        )
        expected = ToolCall(
            name="search_flights",
            arguments={"origin": "NYC", "destination": "LAX"},
            requestor="assistant",
        )
        assert result == expected

    def test_function_call_with_custom_requestor(self):
        """Test parsing function call with custom requestor."""
        result = parse_functional_tool_call(
            "search_flights(origin='NYC')", requestor="user"
        )
        expected = ToolCall(
            name="search_flights", arguments={"origin": "NYC"}, requestor="user"
        )
        assert result == expected

    def test_invalid_format_no_parentheses(self):
        """Test that invalid format without parentheses raises ValueError."""
        with pytest.raises(ValueError, match="Invalid functional call format"):
            parse_functional_tool_call("search_flights")

    def test_invalid_format_no_function_name(self):
        """Test that invalid format without function name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid functional call format"):
            parse_functional_tool_call("(origin='NYC')")

    def test_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Functional call cannot be empty"):
            parse_functional_tool_call("")

    def test_whitespace_only(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Functional call cannot be empty"):
            parse_functional_tool_call("   ")

    def test_invalid_arguments_syntax(self):
        """Test that invalid argument syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid arguments format"):
            parse_functional_tool_call("search_flights(origin=, destination='LAX')")


class TestEvaluateASTNode:
    """Test cases for _evaluate_ast_node function."""

    def test_constant_string(self):
        """Test evaluating string constant."""
        import ast

        node = ast.Constant(value="test")
        result = _evaluate_ast_node(node)
        assert result == "test"

    def test_constant_number(self):
        """Test evaluating numeric constant."""
        import ast

        node = ast.Constant(value=42)
        result = _evaluate_ast_node(node)
        assert result == 42

    def test_constant_float(self):
        """Test evaluating float constant."""
        import ast

        node = ast.Constant(value=3.14)
        result = _evaluate_ast_node(node)
        assert result == 3.14

    def test_constant_boolean(self):
        """Test evaluating boolean constant."""
        import ast

        node = ast.Constant(value=True)
        result = _evaluate_ast_node(node)
        assert result is True

    def test_constant_none(self):
        """Test evaluating None constant."""
        import ast

        node = ast.Constant(value=None)
        result = _evaluate_ast_node(node)
        assert result is None

    def test_list_node(self):
        """Test evaluating list node."""
        import ast

        node = ast.List(
            elts=[
                ast.Constant(value="a"),
                ast.Constant(value="b"),
                ast.Constant(value="c"),
            ]
        )
        result = _evaluate_ast_node(node)
        assert result == ["a", "b", "c"]

    def test_dict_node(self):
        """Test evaluating dict node."""
        import ast

        node = ast.Dict(
            keys=[ast.Constant(value="key1"), ast.Constant(value="key2")],
            values=[ast.Constant(value="value1"), ast.Constant(value="value2")],
        )
        result = _evaluate_ast_node(node)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_tuple_node(self):
        """Test evaluating tuple node."""
        import ast

        node = ast.Tuple(
            elts=[ast.Constant(value=1), ast.Constant(value=2), ast.Constant(value=3)]
        )
        result = _evaluate_ast_node(node)
        assert result == (1, 2, 3)

    def test_name_true(self):
        """Test evaluating name node with True."""
        import ast

        node = ast.Name(id="True")
        result = _evaluate_ast_node(node)
        assert result is True

    def test_name_false(self):
        """Test evaluating name node with False."""
        import ast

        node = ast.Name(id="False")
        result = _evaluate_ast_node(node)
        assert result is False

    def test_name_none(self):
        """Test evaluating name node with None."""
        import ast

        node = ast.Name(id="None")
        result = _evaluate_ast_node(node)
        assert result is None

    def test_name_other(self):
        """Test evaluating name node with other identifier."""
        import ast

        node = ast.Name(id="some_variable")
        result = _evaluate_ast_node(node)
        assert result == "some_variable"

    def test_unsupported_node_type(self):
        """Test that unsupported node type raises ValueError."""
        import ast

        node = ast.BinOp(
            left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2)
        )
        with pytest.raises(ValueError, match="Unsupported AST node type"):
            _evaluate_ast_node(node)

    def test_unary_negative(self):
        """Test evaluating unary negative operation."""
        import ast

        node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=42))
        result = _evaluate_ast_node(node)
        assert result == -42

    def test_unary_positive(self):
        """Test evaluating unary positive operation."""
        import ast

        node = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=42))
        result = _evaluate_ast_node(node)
        assert result == 42

    def test_unary_negative_float(self):
        """Test evaluating unary negative operation on float."""
        import ast

        node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=3.14))
        result = _evaluate_ast_node(node)
        assert result == -3.14


class TestIsFunctionalToolCall:
    """Test cases for is_functional_tool_call function."""

    def test_valid_function_call(self):
        """Test that valid function call returns True."""
        assert is_functional_tool_call("search_flights(origin='NYC')") is True

    def test_function_call_with_spaces(self):
        """Test that function call with spaces returns True."""
        assert is_functional_tool_call("  search_flights  (  origin='NYC'  )  ") is True

    def test_function_call_no_args(self):
        """Test that function call with no arguments returns True."""
        assert is_functional_tool_call("refresh()") is True

    def test_regular_text(self):
        """Test that regular text returns False."""
        assert is_functional_tool_call("Hello, how can I help you?") is False

    def test_empty_string(self):
        """Test that empty string returns False."""
        assert is_functional_tool_call("") is False

    def test_whitespace_only(self):
        """Test that whitespace-only string returns False."""
        assert is_functional_tool_call("   ") is False

    def test_no_parentheses(self):
        """Test that text without parentheses returns False."""
        assert is_functional_tool_call("search_flights") is False

    def test_no_function_name(self):
        """Test that text with only parentheses returns False."""
        assert is_functional_tool_call("(origin='NYC')") is False


class TestExtractToolCallsFromText:
    """Test cases for extract_tool_calls_from_text function."""

    def test_single_tool_call(self):
        """Test extracting a single tool call."""
        text = "search_flights(origin='NYC', destination='LAX')"
        results = extract_tool_calls_from_text(text)
        assert len(results) == 1
        assert results[0].name == "search_flights"
        assert results[0].arguments == {"origin": "NYC", "destination": "LAX"}

    def test_multiple_tool_calls_semicolon_separated(self):
        """Test extracting multiple tool calls separated by semicolons."""
        text = "search_flights(origin='NYC'); book_ticket(flight_id=123)"
        results = extract_tool_calls_from_text(text)
        assert len(results) == 2
        assert results[0].name == "search_flights"
        assert results[1].name == "book_ticket"

    def test_multiple_tool_calls_newline_separated(self):
        """Test extracting multiple tool calls separated by newlines."""
        text = "search_flights(origin='NYC')\nbook_ticket(flight_id=123)"
        results = extract_tool_calls_from_text(text)
        assert len(results) == 2
        assert results[0].name == "search_flights"
        assert results[1].name == "book_ticket"

    def test_mixed_content_with_tool_calls(self):
        """Test extracting tool calls from mixed content."""
        text = "Let me help you with that. search_flights(origin='NYC') and then book_ticket(flight_id=123)"
        results = extract_tool_calls_from_text(text)
        assert len(results) == 0  # No valid tool calls in this format

    def test_invalid_tool_calls_ignored(self):
        """Test that invalid tool calls are ignored."""
        text = "search_flights(origin=, destination='LAX'); valid_call(param='value')"
        results = extract_tool_calls_from_text(text)
        assert len(results) == 1
        assert results[0].name == "valid_call"

    def test_empty_text(self):
        """Test extracting from empty text."""
        results = extract_tool_calls_from_text("")
        assert results == []

    def test_whitespace_only(self):
        """Test extracting from whitespace-only text."""
        results = extract_tool_calls_from_text("   \n  \t  ")
        assert results == []

    def test_custom_requestor(self):
        """Test extracting with custom requestor."""
        text = "search_flights(origin='NYC')"
        results = extract_tool_calls_from_text(text, requestor="user")
        assert len(results) == 1
        assert results[0].requestor == "user"


class TestToFunctionalFormat:
    """Test cases for to_functional_format function."""

    def test_simple_string_arguments(self):
        """Test converting ToolCall with simple string arguments."""
        tool_call = ToolCall(
            name="search_flights", arguments={"origin": "NYC", "destination": "LAX"}
        )
        result = to_functional_format(tool_call)
        expected = "search_flights(destination='LAX', origin='NYC')"
        assert result == expected

    def test_numeric_arguments(self):
        """Test converting ToolCall with numeric arguments."""
        tool_call = ToolCall(
            name="book_ticket", arguments={"flight_id": 123, "price": 299.99}
        )
        result = to_functional_format(tool_call)
        expected = "book_ticket(flight_id=123, price=299.99)"
        assert result == expected

    def test_boolean_arguments(self):
        """Test converting ToolCall with boolean arguments."""
        tool_call = ToolCall(
            name="update_preferences",
            arguments={"notifications": True, "marketing": False},
        )
        result = to_functional_format(tool_call)
        expected = "update_preferences(marketing=False, notifications=True)"
        assert result == expected

    def test_none_argument(self):
        """Test converting ToolCall with None argument."""
        tool_call = ToolCall(name="set_value", arguments={"key": "test", "value": None})
        result = to_functional_format(tool_call)
        expected = "set_value(key='test', value=None)"
        assert result == expected

    def test_list_argument(self):
        """Test converting ToolCall with list argument."""
        tool_call = ToolCall(name="process_items", arguments={"items": ["a", "b", "c"]})
        result = to_functional_format(tool_call)
        expected = "process_items(items=['a', 'b', 'c'])"
        assert result == expected

    def test_dict_argument(self):
        """Test converting ToolCall with dict argument."""
        tool_call = ToolCall(
            name="update_config", arguments={"config": {"timeout": 30, "retries": 3}}
        )
        result = to_functional_format(tool_call)
        expected = "update_config(config={'timeout': 30, 'retries': 3})"
        assert result == expected

    def test_tuple_argument(self):
        """Test converting ToolCall with tuple argument."""
        tool_call = ToolCall(
            name="set_coordinates", arguments={"coords": (40.7128, -74.0060)}
        )
        result = to_functional_format(tool_call)
        expected = "set_coordinates(coords=(40.7128, -74.006))"
        assert result == expected

    def test_nested_structures(self):
        """Test converting ToolCall with nested data structures."""
        tool_call = ToolCall(
            name="complex_call",
            arguments={
                "data": {
                    "users": [
                        {"name": "John", "active": True},
                        {"name": "Jane", "active": False},
                    ]
                }
            },
        )
        result = to_functional_format(tool_call)
        expected = "complex_call(data={'users': [{'name': 'John', 'active': True}, {'name': 'Jane', 'active': False}]})"
        assert result == expected

    def test_no_arguments(self):
        """Test converting ToolCall with no arguments."""
        tool_call = ToolCall(name="refresh", arguments={})
        result = to_functional_format(tool_call)
        expected = "refresh()"
        assert result == expected

    def test_string_with_single_quotes(self):
        """Test converting ToolCall with strings containing single quotes."""
        tool_call = ToolCall(
            name="update_description", arguments={"description": "John's flight"}
        )
        result = to_functional_format(tool_call)
        expected = "update_description(description='John\\'s flight')"
        assert result == expected

    def test_negative_numbers(self):
        """Test converting ToolCall with negative numbers."""
        tool_call = ToolCall(
            name="set_coordinates", arguments={"lat": -40.7128, "lng": -74.0060}
        )
        result = to_functional_format(tool_call)
        expected = "set_coordinates(lat=-40.7128, lng=-74.006)"
        assert result == expected

    def test_mixed_types(self):
        """Test converting ToolCall with mixed argument types."""
        tool_call = ToolCall(
            name="mixed_call",
            arguments={
                "name": "John Doe",
                "age": 30,
                "active": True,
                "scores": [85, 92, 78],
                "metadata": {"department": "IT", "level": None},
            },
        )
        result = to_functional_format(tool_call)
        expected = "mixed_call(active=True, age=30, metadata={'department': 'IT', 'level': None}, name='John Doe', scores=[85, 92, 78])"
        assert result == expected

    def test_empty_string_argument(self):
        """Test converting ToolCall with empty string argument."""
        tool_call = ToolCall(name="set_name", arguments={"name": ""})
        result = to_functional_format(tool_call)
        expected = "set_name(name='')"
        assert result == expected

    def test_zero_values(self):
        """Test converting ToolCall with zero values."""
        tool_call = ToolCall(name="reset_counter", arguments={"count": 0, "total": 0.0})
        result = to_functional_format(tool_call)
        expected = "reset_counter(count=0, total=0.0)"
        assert result == expected

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        tool_call = ToolCall(name="", arguments={"param": "value"})
        with pytest.raises(ValueError, match="ToolCall must have a name"):
            to_functional_format(tool_call)

    def test_round_trip_parsing(self):
        """Test that to_functional_format and parse_functional_tool_call work together."""
        original_tool_call = ToolCall(
            name="search_flights",
            arguments={"origin": "NYC", "destination": "LAX", "date": "2024-01-15"},
        )

        # Convert to functional format
        functional_str = to_functional_format(original_tool_call)

        # Parse back to ToolCall
        parsed_tool_call = parse_functional_tool_call(functional_str)

        # Should be equivalent (ignoring id and requestor which aren't part of functional format)
        assert parsed_tool_call.name == original_tool_call.name
        assert parsed_tool_call.arguments == original_tool_call.arguments
