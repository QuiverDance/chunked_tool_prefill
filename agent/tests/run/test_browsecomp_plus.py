import json

from minisweagent.run.benchmarks.browsecomp_plus import BrowseCompPlusTokenTimingAgent, result_items


class FakeProgress:
    def update_instance_status(self, instance_id, status):
        pass


class FakeModel:
    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return []

    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {"info": {"config": {"model": {}}}}


class FakeEnv:
    def get_template_vars(self, **kwargs):
        return kwargs

    def serialize(self):
        return {"info": {"config": {"environment": {}}}}


def test_browsecomp_agent_submits_final_answer_without_tool_call(tmp_path):
    agent = BrowseCompPlusTokenTimingAgent(
        FakeModel(),
        FakeEnv(),
        progress_manager=FakeProgress(),
        instance_id="q1",
        system_template="",
        instance_template="",
        tokenizer_path="",
        output_path=tmp_path / "q1.traj.json",
    )

    added = agent.execute_actions(
        {
            "role": "assistant",
            "content": "Explanation: because.\nExact Answer: answer\nConfidence: 80%",
            "extra": {"actions": []},
        }
    )

    assert added[0]["role"] == "exit"
    assert added[0]["extra"]["exit_status"] == "Submitted"
    assert "Exact Answer" in added[0]["extra"]["submission"]


def test_result_items_keeps_tool_calls_and_final_output():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "extra": {
                "actions": [
                    {
                        "tool_name": "search",
                        "arguments": {"query": "example"},
                    }
                ]
            },
        }
    ]
    records = [{"output": [{"docid": "d1", "snippet": "text"}]}]

    items = result_items(messages, records, "Exact Answer: final")

    assert items[0]["type"] == "tool_call"
    assert items[0]["tool_name"] == "search"
    assert json.loads(json.dumps(items[0]["output"]))[0]["docid"] == "d1"
    assert items[-1]["type"] == "output_text"
    assert items[-1]["output"] == "Exact Answer: final"
