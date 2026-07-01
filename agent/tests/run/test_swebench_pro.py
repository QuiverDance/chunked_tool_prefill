import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.swebench_pro import (
    build_task,
    build_swebench_pro_summary,
    instance_for_agent,
    load_instance_ids,
    main,
    preds_to_official_patches,
    select_instances,
    swebench_pro_image_name,
)


def make_instance(instance_id: str = "instance_fake") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "example/repo",
        "repo_language": "Python",
        "base_commit": "abc123",
        "dockerhub_tag": "example.repo-instance_fake",
        "problem_statement": "Fix the public issue.",
        "requirements": "Keep the public API stable.",
        "interface": {"function": "run"},
        "patch": "GOLD PATCH MUST NOT LEAK",
        "test_patch": "GOLD TEST PATCH MUST NOT LEAK",
        "fail_to_pass": ["tests/test_public.py::test_failure"],
        "pass_to_pass": ["tests/test_public.py::test_existing"],
        "selected_test_files_to_run": ["tests/test_public.py"],
        "before_repo_set_cmd": "echo hidden setup",
    }


def test_swebench_pro_image_name():
    assert swebench_pro_image_name(make_instance()) == "jefzda/sweap-images:example.repo-instance_fake"


def test_instance_for_agent_sets_task_and_image():
    instance = instance_for_agent(make_instance())

    assert instance["docker_image"] == "jefzda/sweap-images:example.repo-instance_fake"
    assert "Fix this SWE-bench Pro issue." in instance["problem_statement"]


def test_build_task_omits_gold_fields():
    task = build_task(make_instance())

    assert "Fix the public issue." in task
    assert "Keep the public API stable." in task
    assert "GOLD PATCH MUST NOT LEAK" not in task
    assert "GOLD TEST PATCH MUST NOT LEAK" not in task
    assert "tests/test_public.py::test_failure" not in task
    assert "echo hidden setup" not in task
    assert "selected_test_files_to_run" not in task


def test_load_instance_ids(tmp_path):
    instance_list = tmp_path / "instances.txt"
    instance_list.write_text("# comment\ninstance_one\n/path/to/instance_two\n")

    assert load_instance_ids("instance_single", None) == ["instance_single"]
    assert load_instance_ids(None, instance_list) == ["instance_one", "instance_two"]
    with pytest.raises(Exception):
        load_instance_ids("instance_single", instance_list)


def test_select_instances_filter_slice_shuffle():
    instances = [
        make_instance("case-c") | {"repo": "org/api", "repo_language": "Python"},
        make_instance("case-a") | {"repo": "org/web", "repo_language": "TypeScript"},
        make_instance("case-b") | {"repo": "org/api", "repo_language": "Python"},
        make_instance("other") | {"repo": "org/api", "repo_language": "Go"},
    ]

    selected = select_instances(
        instances,
        instance_ids=[],
        filter_spec=r"case-.*",
        repo_filter=r"org/api",
        language_filter=r"Python",
        slice_spec="0:2",
    )
    assert [instance["instance_id"] for instance in selected] == ["case-c", "case-b"]

    shuffled_once = select_instances(instances, instance_ids=[], shuffle=True)
    shuffled_twice = select_instances(instances, instance_ids=[], shuffle=True)
    assert [instance["instance_id"] for instance in shuffled_once] == [
        instance["instance_id"] for instance in shuffled_twice
    ]


def test_preds_to_official_patches(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "instance_b": {
                    "model_name_or_path": "model-b",
                    "instance_id": "instance_b",
                    "model_patch": "diff --git b\n",
                },
                "instance_a": {
                    "model_name_or_path": "model-a",
                    "instance_id": "instance_a",
                    "model_patch": "diff --git a\n",
                },
            }
        )
    )

    assert preds_to_official_patches(preds_path) == [
        {"instance_id": "instance_a", "patch": "diff --git a\n", "prefix": "model-a"},
        {"instance_id": "instance_b", "patch": "diff --git b\n", "prefix": "model-b"},
    ]


def test_summary_reads_official_eval_bool_map(tmp_path):
    (tmp_path / "swebench_pro_eval").mkdir()
    (tmp_path / "swebench_pro_eval" / "eval_results.json").write_text(
        json.dumps({"instance_a": True, "instance_b": False})
    )

    summary = build_swebench_pro_summary(tmp_path)

    assert summary["resolved"] == 1
    assert summary["pass_at_1"] == 0.5


class _SubmittingModelConfig:
    model_name = "submitting_model"


class _SubmittingModel:
    def __init__(self):
        self.cost = 0.0
        self.n_calls = 0
        self.config = _SubmittingModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        patch_text = "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
        raise Submitted(
            {"role": "exit", "content": patch_text, "extra": {"exit_status": "Submitted", "submission": patch_text}}
        )

    def format_message(self, **kwargs):
        return dict(**kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [self.format_message(role="user", content=str(output)) for output in outputs]

    def get_template_vars(self, **kwargs):
        return {"model_name": self.config.model_name, "n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self):
        return {"info": {"config": {"model": {"model_name": self.config.model_name}}}}


class _FakeEnv:
    container_id = None
    config = SimpleNamespace(executable="docker")

    def execute(self, action, cwd="", timeout=None):
        return {"output": "", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {"cwd": "/app", **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment": {"image": "jefzda/sweap-images:example.repo-instance_fake"}}}}

    def cleanup(self):
        pass


def test_swebench_pro_main_mocked(tmp_path):
    instance = make_instance()

    with (
        patch("minisweagent.run.benchmarks.swebench_pro.load_swebench_pro_instances", return_value=[instance]),
        patch("minisweagent.run.benchmarks.swebench_pro.get_environment", return_value=_FakeEnv()),
        patch("minisweagent.run.benchmarks.swebench_pro.get_model", return_value=_SubmittingModel()),
    ):
        main(
            instance_id="instance_fake",
            instance_list=None,
            slice_spec="",
            filter_spec="",
            repo_filter="",
            language_filter="",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            tokenizer_path=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench_pro.yaml")],
        )

    preds = json.loads((tmp_path / "preds.json").read_text())
    assert preds["instance_fake"]["model_name_or_path"] == "submitting_model"
    assert "diff --git" in preds["instance_fake"]["model_patch"]

    patches = json.loads((tmp_path / "swebench_pro_patches.json").read_text())
    assert patches == [
        {
            "instance_id": "instance_fake",
            "patch": preds["instance_fake"]["model_patch"],
            "prefix": "submitting_model",
        }
    ]

    raw_rows = (tmp_path / "swebench_pro_raw.jsonl").read_text().splitlines()
    assert len(raw_rows) == 1
    assert json.loads(raw_rows[0])["patch"] == "GOLD PATCH MUST NOT LEAK"

    trajectory = json.loads((tmp_path / "instance_fake" / "instance_fake.traj.json").read_text())
    assert trajectory["instance_id"] == "instance_fake"
    assert trajectory["info"]["swebench_pro"]["result"]["has_patch"] is True

    result_lines = (tmp_path / "swebench_pro_results.jsonl").read_text().splitlines()
    assert len(result_lines) == 1
    assert json.loads(result_lines[0])["submitted"] is True

    summary = json.loads((tmp_path / "summary.swebench_pro.json").read_text())
    assert summary["total"] == 1
    assert summary["generated_patches"] == 1
