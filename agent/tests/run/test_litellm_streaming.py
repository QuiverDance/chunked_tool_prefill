from minisweagent.models.litellm_model import StreamingResponseBuilder, chunk_has_generated_payload
from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions


def test_streaming_response_builder_reconstructs_tool_call():
    builder = StreamingResponseBuilder()
    chunks = [
        {
            "id": "chatcmpl-test",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        },
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": ""},
                            }
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"command": "echo hi"'}}
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}}]},
        {
            "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
    ]

    for chunk in chunks:
        builder.add_chunk(chunk)

    response = builder.response()
    actions = parse_toolcall_actions(
        response.choices[0].message.tool_calls,
        format_error_template="{{ error }}",
    )

    assert actions == [{"command": "echo hi", "tool_call_id": "call_1"}]
    assert response.usage.prompt_tokens == 3


def test_chunk_has_generated_payload_skips_empty_role_chunk():
    assert not chunk_has_generated_payload({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
    assert chunk_has_generated_payload({"choices": [{"delta": {"reasoning": "x"}}]})
    assert chunk_has_generated_payload({"choices": [{"delta": {"reasoning_content": "x"}}]})
    assert chunk_has_generated_payload(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}}]}
    )
