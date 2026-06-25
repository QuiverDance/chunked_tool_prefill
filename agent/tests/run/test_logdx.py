import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.exceptions import Submitted
from minisweagent.run.benchmarks.logdx import (
    LogDxCase,
    build_task,
    case_id_from_list_entry,
    load_case_ids,
    main,
    diagnosis_for_scoring,
    normalize_diagnosis,
    parse_splits,
    parse_submission,
    safe_case_metadata,
    select_cases,
)


def make_case(case_id: str = "fake-case", split: str = "v2/dev") -> LogDxCase:
    return LogDxCase(
        case_id=case_id,
        split=split,
        raw_log="setup\nERROR: expected failure\n",
        case_metadata={
            "case_id": case_id,
            "repo": "example/repo",
            "source": "github_actions",
            "workflow_name": "CI",
            "job_name": "test",
            "framework": "pytest",
            "failure_category": "test_assertion",
        },
        ground_truth={"root_cause": {"category": "test_assertion"}},
    )


def test_parse_splits():
    assert parse_splits("all") == ["dev", "holdout", "stress", "v2/dev", "v2/holdout", "v2/stress"]
    assert parse_splits("dev,v2_dev") == ["dev", "v2/dev"]


def test_load_case_ids(tmp_path):
    case_list = tmp_path / "cases.txt"
    case_list.write_text("# comment\ncases/v2/dev/moby-buildx-bake-v2-001\nplain-case\n")

    assert load_case_ids("one-case", None) == ["one-case"]
    assert load_case_ids(None, case_list) == ["moby-buildx-bake-v2-001", "plain-case"]
    with pytest.raises(Exception):
        load_case_ids("one-case", case_list)


def test_case_id_from_list_entry():
    assert case_id_from_list_entry("cases/v2/dev/moby-buildx-bake-v2-001") == "moby-buildx-bake-v2-001"
    assert case_id_from_list_entry("/tmp/cases/dev/pytest-pandas-001/") == "pytest-pandas-001"


def test_select_cases_filter_slice_shuffle():
    cases = [make_case("case-c"), make_case("case-a"), make_case("other"), make_case("case-b")]

    selected = select_cases(cases, case_ids=[], filter_spec=r"case-.*", slice_spec="1:3")
    assert [case.case_id for case in selected] == ["case-a", "case-b"]

    shuffled_once = select_cases(cases, case_ids=[], shuffle=True)
    shuffled_twice = select_cases(cases, case_ids=[], shuffle=True)
    assert [case.case_id for case in shuffled_once] == [case.case_id for case in shuffled_twice]


def test_prompt_uses_safe_metadata_only(tmp_path):
    case = make_case()
    task = build_task(case, tmp_path)
    metadata = safe_case_metadata(case)

    assert metadata["repo"] == "example/repo"
    assert "failure_category" not in metadata
    assert "ground_truth" not in task
    assert "required_signals" not in task
    assert "failure_category" not in task
    assert "expected_diagnosis" not in task


def test_parse_submission_json_fence():
    parsed, error = parse_submission('```json\n{"summary": "ok"}\n```')
    assert error == ""
    assert parsed == {"summary": "ok"}


def test_normalize_diagnosis_requires_shape():
    diagnosis, error = normalize_diagnosis(
        {
            "summary": "summary",
            "root_cause_category": "test_assertion",
            "root_cause": "root cause",
            "confidence": "0.7",
            "evidence": [],
            "suggested_fix": "fix it",
        }
    )

    assert error == ""
    assert diagnosis["confidence"] == 0.7

    invalid, error = normalize_diagnosis({"summary": "missing"})
    assert invalid is None
    assert "missing fields" in error


def test_diagnosis_for_scoring_adds_official_wrapper_fields():
    case = make_case()
    diagnosis = {
        "summary": "summary",
        "root_cause_category": "test_assertion",
        "root_cause": "root cause",
        "confidence": 0.7,
        "evidence": [],
        "suggested_fix": "fix it",
    }

    wrapped = diagnosis_for_scoring(case, diagnosis)

    assert wrapped["case_id"] == "fake-case"
    assert wrapped["context_method"] == "mini-swe-agent-raw-log"
    assert wrapped["diagnoser"] == "mini-swe-agent-qwen"
    assert wrapped["mode"] == "root_cause_diagnosis"
    assert wrapped["metadata"]["provider_error"] is None


class _SubmittingModelConfig:
    model_name = "submitting_model"


class _SubmittingModel:
    def __init__(self):
        self.cost = 0.0
        self.n_calls = 0
        self.config = _SubmittingModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        diagnosis = {
            "summary": "The test failed.",
            "root_cause_category": "test_assertion",
            "root_cause": "An expected assertion failed.",
            "confidence": 0.8,
            "relevant_files": [],
            "relevant_tests": [],
            "evidence": [{"quote": "ERROR: expected failure", "reason": "shows the failure"}],
            "suggested_fix": "Update the failing assertion.",
        }
        text = json.dumps(diagnosis)
        raise Submitted({"role": "exit", "content": text, "extra": {"exit_status": "Submitted", "submission": text}})

    def format_message(self, **kwargs):
        return dict(**kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [self.format_message(role="user", content=str(output)) for output in outputs]

    def get_template_vars(self, **kwargs):
        return {"model_name": self.config.model_name, "n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self):
        return {"info": {"config": {"model": {"model_name": self.config.model_name}}}}


class _FakeEnv:
    config = SimpleNamespace()

    def execute(self, action, cwd="", timeout=None):
        return {"output": "", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {"cwd": "/tmp/fake", **kwargs}

    def serialize(self):
        return {"info": {"config": {"environment": {"cwd": "/tmp/fake"}}}}

    def cleanup(self):
        pass


def test_logdx_main_mocked(tmp_path):
    case = make_case()
    score = {
        "score": 0.75,
        "category_match": 1.0,
        "confident_error": False,
        "scoring_status": "ok",
    }

    with (
        patch("minisweagent.run.benchmarks.logdx.load_logdx_cases", return_value=[case]),
        patch("minisweagent.run.benchmarks.logdx.get_environment", return_value=_FakeEnv()),
        patch("minisweagent.run.benchmarks.logdx.get_model", return_value=_SubmittingModel()),
        patch("minisweagent.run.benchmarks.logdx.score_diagnosis", return_value=score),
    ):
        main(
            splits="v2/dev",
            case_id="fake-case",
            case_list=None,
            slice_spec="",
            filter_spec="",
            shuffle=False,
            output=str(tmp_path),
            workers=1,
            model=None,
            model_class=None,
            tokenizer_path=None,
            corpus_root=None,
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "logdx.yaml")],
        )

    preds = json.loads((tmp_path / "preds.json").read_text())
    assert preds["fake-case"]["model_name_or_path"] == "submitting_model"
    assert "model_diagnosis" in preds["fake-case"]

    trajectory = json.loads((tmp_path / "fake-case" / "fake-case.traj.json").read_text())
    assert trajectory["instance_id"] == "fake-case"
    assert trajectory["info"]["logdx"]["result"]["score"] == 0.75

    workspace = tmp_path / "fake-case" / "workspace"
    assert (workspace / "raw.log").exists()
    assert not (workspace / "ground_truth.json").exists()

    result_lines = (tmp_path / "logdx_results.jsonl").read_text().splitlines()
    assert len(result_lines) == 1
    assert json.loads(result_lines[0])["diagnosis_valid"] is True

    summary = json.loads((tmp_path / "summary.logdx.json").read_text())
    assert summary["total"] == 1
    assert summary["diagnosis_valid"] == 1
