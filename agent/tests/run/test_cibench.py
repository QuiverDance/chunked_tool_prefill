import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.cibench import (
    bugswarm_image_name,
    failed_repo_path,
    load_artifact_ids,
    main,
    select_artifacts,
)


def test_load_artifact_ids_single():
    assert load_artifact_ids("tananaev-traccar-64783123", None) == ["tananaev-traccar-64783123"]


def test_load_artifact_ids_file(tmp_path):
    artifact_list = tmp_path / "artifacts.txt"
    artifact_list.write_text(
        "\n".join(
            [
                "# comment",
                "tananaev-traccar-64783123",
                "",
                "mybatis-mybatis-3-117115624",
            ]
        )
    )

    assert load_artifact_ids(None, artifact_list) == [
        "tananaev-traccar-64783123",
        "mybatis-mybatis-3-117115624",
    ]


def test_load_artifact_ids_requires_one_source(tmp_path):
    with pytest.raises(Exception):
        load_artifact_ids(None, None)
    with pytest.raises(Exception):
        load_artifact_ids("a", tmp_path / "artifacts.txt")


def test_select_artifacts_filter_slice_shuffle():
    ids = ["repo-c-3", "repo-a-1", "repo-b-2", "other-4"]

    selected = select_artifacts(ids, filter_spec=r"repo-.*", slice_spec="1:3", shuffle=False)
    assert selected == [{"artifact_id": "repo-a-1"}, {"artifact_id": "repo-b-2"}]

    shuffled_once = select_artifacts(ids, shuffle=True)
    shuffled_twice = select_artifacts(ids, shuffle=True)
    assert shuffled_once == shuffled_twice
    assert shuffled_once != [{"artifact_id": artifact_id} for artifact_id in ids]


def test_bugswarm_image_name():
    assert bugswarm_image_name("tananaev-traccar-64783123") == "bugswarm/cached-images:tananaev-traccar-64783123"


def test_failed_repo_path_uses_metadata_repo_slug():
    metadata = {
        "ci_service": "travis",
        "repo": "traccar/traccar",
    }

    assert failed_repo_path(metadata) == "/home/travis/build/failed/traccar/traccar"


class _SubmittingModelConfig:
    model_name = "submitting_model"


class _SubmittingModel:
    def __init__(self):
        self.cost = 0.0
        self.n_calls = 0
        self.config = _SubmittingModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        patch = "diff --git a/src/Main.java b/src/Main.java\n--- a/src/Main.java\n+++ b/src/Main.java\n"
        raise Submitted({"role": "exit", "content": patch, "extra": {"exit_status": "Submitted", "submission": patch}})

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
        return {"cwd": "/home/travis/build/failed/traccar/traccar", **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment": {"image": "bugswarm/cached-images:fake-artifact"}}}}

    def cleanup(self):
        pass


def test_cibench_main_mocked(tmp_path, monkeypatch):
    metadata = {
        "image_tag": "fake-artifact",
        "repo": "traccar/traccar",
        "lang": "Java",
        "ci_service": "travis",
        "failed_job": {
            "job_id": 1,
            "trigger_sha": "abc123",
            "message": "broken",
            "failed_tests": "testBroken",
            "num_tests_failed": 1,
            "num_tests_run": 2,
            "config": {"language": "java"},
        },
        "passed_job": {"job_id": 2, "trigger_sha": "def456", "message": "fixed"},
    }
    validation = {
        "artifact_id": "fake-artifact",
        "patch_applies": True,
        "plausible": True,
        "run_failed_returncode": 0,
        "exit_status": "Plausible",
        "patch_path": str(tmp_path / "fake-artifact" / "generated.patch"),
        "validation_log_path": str(tmp_path / "fake-artifact" / "validation.log"),
    }

    monkeypatch.delenv("BUGSWARM_TOKEN", raising=False)
    with (
        patch("minisweagent.run.benchmarks.cibench.fetch_artifact_metadata", return_value=metadata),
        patch("minisweagent.run.benchmarks.cibench.fetch_failed_build_log", return_value="failed log"),
        patch("minisweagent.run.benchmarks.cibench.get_environment", return_value=_FakeEnv()),
        patch("minisweagent.run.benchmarks.cibench.get_model", return_value=_SubmittingModel()),
        patch("minisweagent.run.benchmarks.cibench.validate_generated_patch", return_value=validation),
    ):
        main(
            artifact_id="fake-artifact",
            artifact_list=None,
            slice_spec="",
            filter_spec="",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "cibench.yaml")],
            environment_class="docker",
            evaluate_sye=False,
        )

    preds = json.loads((tmp_path / "preds.json").read_text())
    assert preds["fake-artifact"]["model_name_or_path"] == "submitting_model"
    assert "diff --git" in preds["fake-artifact"]["model_patch"]

    trajectory = json.loads((tmp_path / "fake-artifact" / "fake-artifact.traj.json").read_text())
    assert trajectory["instance_id"] == "fake-artifact"
    assert trajectory["info"]["cibench"]["validation"]["plausible"] is True

    result_lines = (tmp_path / "cibench_results.jsonl").read_text().splitlines()
    assert len(result_lines) == 1
    assert json.loads(result_lines[0])["exit_status"] == "Plausible"

    summary = json.loads((tmp_path / "summary.cibench.json").read_text())
    assert summary["total"] == 1
    assert summary["plausible"] == 1
