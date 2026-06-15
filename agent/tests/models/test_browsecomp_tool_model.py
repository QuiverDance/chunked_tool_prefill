from unittest.mock import MagicMock

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.browsecomp_tool_model import BrowseCompToolModel


def response_with_tool_calls(*tool_calls):
    message = MagicMock()
    message.tool_calls = list(tool_calls)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def tool_call(name: str, arguments: str, call_id: str = "call_1"):
    function = MagicMock()
    function.name = name
    function.arguments = arguments
    call = MagicMock()
    call.function = function
    call.id = call_id
    return call


def test_parse_search_tool_call():
    model = BrowseCompToolModel(model_name="hosted_vllm/test", cost_tracking="ignore_errors")

    actions = model._parse_actions(response_with_tool_calls(tool_call("search", '{"query": "who founded x"}')))

    assert actions == [
        {
            "tool_name": "search",
            "arguments": {"query": "who founded x"},
            "query": "who founded x",
            "tool_call_id": "call_1",
        }
    ]


def test_final_answer_without_tool_call_is_allowed():
    model = BrowseCompToolModel(model_name="hosted_vllm/test", cost_tracking="ignore_errors")

    assert model._parse_actions(response_with_tool_calls()) == []


def test_get_document_rejected_when_not_configured():
    model = BrowseCompToolModel(model_name="hosted_vllm/test", cost_tracking="ignore_errors")

    with pytest.raises(FormatError) as exc_info:
        model._parse_actions(response_with_tool_calls(tool_call("get_document", '{"docid": "doc-1"}')))

    assert "Unknown tool" in exc_info.value.messages[0]["content"]


def test_parse_get_document_when_configured():
    model = BrowseCompToolModel(
        model_name="hosted_vllm/test",
        cost_tracking="ignore_errors",
        include_get_document=True,
    )

    actions = model._parse_actions(response_with_tool_calls(tool_call("get_document", '{"docid": "doc-1"}')))

    assert actions == [
        {
            "tool_name": "get_document",
            "arguments": {"docid": "doc-1"},
            "docid": "doc-1",
            "tool_call_id": "call_1",
        }
    ]
