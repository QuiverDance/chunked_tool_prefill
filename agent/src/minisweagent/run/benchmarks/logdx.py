#!/usr/bin/env python3

"""Run mini-SWE-agent on LogDx-CI root-cause diagnosis cases."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.live import Live

from minisweagent.agents import get_agent_class
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.run.benchmarks.utils.token_timing import count_tokens, load_tokenizer
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "logdx.yaml"
CANONICAL_SPLITS = ["dev", "holdout", "stress", "v2/dev", "v2/holdout", "v2/stress"]
SAFE_METADATA_FIELDS = ["case_id", "repo", "source", "workflow_name", "job_name", "framework"]
REQUIRED_DIAGNOSIS_FIELDS = [
    "summary",
    "root_cause_category",
    "root_cause",
    "confidence",
    "evidence",
    "suggested_fix",
]
_RESULTS_FILE_LOCK = threading.Lock()

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_HELP_TEXT = """Run mini-SWE-agent on LogDx-CI root-cause diagnosis cases."""
_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

Multiple configs are recursively merged from left to right.
"""


@dataclass(frozen=True)
class LogDxCase:
    case_id: str
    split: str
    raw_log: str
    case_metadata: dict[str, Any]
    ground_truth: dict[str, Any]


def parse_splits(splits: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[\s,]+", splits or "") if part.strip()]
    if not parts or any(part.lower() == "all" for part in parts):
        return list(CANONICAL_SPLITS)
    return [normalize_split(part) for part in parts]


def normalize_split(split: str) -> str:
    split = split.strip().replace("\\", "/")
    if split.startswith("v2_"):
        return split.replace("v2_", "v2/", 1)
    return split


def load_case_ids(case_id: str | None, case_list: Path | None) -> list[str]:
    if case_id and case_list:
        raise typer.BadParameter("Specify at most one of --case-id or --case-list.")
    if case_id:
        return [case_id.strip()]
    if case_list is None:
        return []

    ids = []
    for line in case_list.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(case_id_from_list_entry(value))
    return ids


def case_id_from_list_entry(value: str) -> str:
    value = value.strip().rstrip("/")
    if "/cases/" in value:
        value = value.split("/cases/", 1)[1]
    return value.split("/")[-1]


def select_cases(
    cases: list[LogDxCase],
    *,
    case_ids: list[str],
    filter_spec: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[LogDxCase]:
    selected = list(cases)
    if case_ids:
        wanted = set(case_ids)
        before = len(selected)
        selected = [case for case in selected if case.case_id in wanted]
        logger.info(f"Case id selection: {before} -> {len(selected)} cases")
    if filter_spec:
        before = len(selected)
        selected = [case for case in selected if re.match(filter_spec, case.case_id)]
        logger.info(f"Case filter: {before} -> {len(selected)} cases")
    if shuffle:
        selected = sorted(selected, key=lambda case: case.case_id)
        random.seed(42)
        random.shuffle(selected)
    if slice_spec:
        before = len(selected)
        parts = [int(part) if part else None for part in slice_spec.split(":")]
        selected = selected[slice(*parts)]
        logger.info(f"Case slice: {before} -> {len(selected)} cases")
    return selected


def load_logdx_cases(splits: list[str], corpus_root: Path | None = None) -> list[LogDxCase]:
    try:
        from logdx_ci.corpus import load_cases
    except ImportError as e:
        message = "logdx-ci is not installed. Install it in this environment with `pip install logdx-ci`."
        raise RuntimeError(message) from e

    root = str(corpus_root.expanduser()) if corpus_root else None
    raw_cases = load_cases(splits=splits, corpus_root=root)
    return [
        LogDxCase(
            case_id=case.case_id,
            split=case.split,
            raw_log=case.raw_log,
            case_metadata=dict(case.case_metadata or {}),
            ground_truth=dict(case.ground_truth or {}),
        )
        for case in raw_cases
    ]


def safe_case_metadata(case: LogDxCase) -> dict[str, Any]:
    metadata = dict(case.case_metadata)
    metadata.setdefault("case_id", case.case_id)
    return {field: metadata.get(field, "") for field in SAFE_METADATA_FIELDS}


def tail_text(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_task(case: LogDxCase, workspace: Path) -> str:
    metadata = safe_case_metadata(case)
    return f"""Diagnose this LogDx-CI failure.

Case: {case.case_id}
Split: {case.split}
Workspace: {workspace}

Safe case metadata:
```json
{json.dumps(metadata, indent=2, sort_keys=True)}
```

The full CI log is available at:

```text
raw.log
```

The safe metadata is also available at:

```text
safe_case_metadata.json
```

Do not use any file outside the workspace. The benchmark answer files are intentionally not present here.

Raw log tail preview:
```text
{tail_text(case.raw_log)}
```
"""


def prepare_workspace(case: LogDxCase, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "raw.log").write_text(case.raw_log, encoding="utf-8", errors="replace")
    (workspace / "safe_case_metadata.json").write_text(
        json.dumps(safe_case_metadata(case), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_submission(submission: str) -> tuple[dict[str, Any] | None, str]:
    if not submission.strip():
        return None, "empty submission"
    value = submission.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, ""
    return None, "submission does not contain a JSON object"


def normalize_diagnosis(diagnosis: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if diagnosis is None:
        return None, "missing diagnosis"

    missing = [field for field in REQUIRED_DIAGNOSIS_FIELDS if field not in diagnosis]
    if missing:
        return None, f"missing fields: {', '.join(missing)}"

    normalized = dict(diagnosis)
    try:
        normalized["confidence"] = float(normalized["confidence"])
    except (TypeError, ValueError):
        return None, "confidence must be a number"

    if not isinstance(normalized["evidence"], list):
        return None, "evidence must be a list"

    for field in ("summary", "root_cause_category", "root_cause", "suggested_fix"):
        normalized[field] = str(normalized[field])
    return normalized, ""


def score_diagnosis(case: LogDxCase, diagnosis: dict[str, Any] | None) -> dict[str, Any]:
    if diagnosis is None:
        return {
            "score": None,
            "category_match": None,
            "confident_error": False,
            "scoring_status": "skipped_invalid_diagnosis",
        }
    try:
        from logdx_ci.scoring import score_case
    except ImportError:
        return {
            "score": None,
            "category_match": None,
            "confident_error": False,
            "scoring_status": "skipped_missing_logdx_ci",
        }

    try:
        scored = score_case(
            diagnosis=diagnosis_for_scoring(case, diagnosis),
            ground_truth=case.ground_truth,
            reduced_context=case.raw_log,
        )
    except Exception as e:
        return {
            "score": None,
            "category_match": None,
            "confident_error": False,
            "scoring_status": "scorer_error",
            "scoring_error": str(e),
        }
    return {
        "score": scored.get("diagnosis_score_v1_1"),
        "category_match": scored.get("category_match_score_v1_1"),
        "confident_error": bool(scored.get("confident_error_v1_1")),
        "scoring_status": "ok",
    }


def diagnosis_for_scoring(case: LogDxCase, diagnosis: dict[str, Any]) -> dict[str, Any]:
    scoring_diagnosis = dict(diagnosis)
    scoring_diagnosis.setdefault("case_id", case.case_id)
    scoring_diagnosis.setdefault("context_method", "mini-swe-agent-raw-log")
    scoring_diagnosis.setdefault("diagnoser", "mini-swe-agent-qwen")
    scoring_diagnosis.setdefault("mode", "root_cause_diagnosis")
    scoring_diagnosis.setdefault("relevant_files", [])
    scoring_diagnosis.setdefault("relevant_tests", [])
    scoring_diagnosis.setdefault(
        "input",
        {
            "context_path": "raw.log",
            "context_tokens_estimate": 0,
        },
    )
    scoring_diagnosis.setdefault(
        "usage",
        {
            "processing_tokens_estimate": 0,
            "output_tokens_estimate": 0,
        },
    )
    scoring_diagnosis.setdefault(
        "metadata",
        {
            "provider": "mini-swe-agent",
            "prompt_sha256": "",
            "runtime_ms": 0,
            "provider_error": None,
        },
    )
    return scoring_diagnosis


def build_result_record(
    *,
    case: LogDxCase,
    exit_status: str | None,
    submission: str,
    prediction_path: Path,
    trajectory_path: Path,
    tokenizer,
) -> dict[str, Any]:
    parsed, parse_error = parse_submission(submission)
    diagnosis, validation_error = normalize_diagnosis(parsed)
    error = parse_error or validation_error
    scored = score_diagnosis(case, diagnosis)
    record = {
        "case_id": case.case_id,
        "split": case.split,
        "exit_status": exit_status,
        "diagnosis_valid": diagnosis is not None,
        "diagnosis_error": error,
        "score": scored.get("score"),
        "category_match": scored.get("category_match"),
        "confident_error": scored.get("confident_error"),
        "scoring_status": scored.get("scoring_status"),
        "raw_log_chars": len(case.raw_log),
        "raw_log_tokens": count_tokens(tokenizer, case.raw_log),
        "prediction_path": str(prediction_path),
        "trajectory_path": str(trajectory_path),
    }
    if diagnosis is not None:
        record["root_cause_category"] = diagnosis.get("root_cause_category")
        record["confidence"] = diagnosis.get("confidence")
    if scored.get("scoring_error"):
        record["scoring_error"] = scored["scoring_error"]
    return record


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with _RESULTS_FILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def update_preds_file(output_path: Path, case: LogDxCase, model_name: str, result: str) -> None:
    with _RESULTS_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[case.case_id] = {
            "model_name_or_path": model_name,
            "instance_id": case.case_id,
            "split": case.split,
            "model_diagnosis": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, case_id: str) -> None:
    if not output_path.exists():
        return
    with _RESULTS_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if case_id in output_data:
            del output_data[case_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def write_logdx_summary(output_dir: Path) -> dict[str, Any]:
    summary = build_logdx_summary(output_dir)
    (output_dir / "summary.logdx.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_logdx_summary(output_dir: Path) -> dict[str, Any]:
    records = read_jsonl(output_dir / "logdx_results.jsonl")
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "total": len(records),
        "submitted": sum(1 for record in records if record.get("exit_status") == "Submitted"),
        "diagnosis_valid": sum(1 for record in records if record.get("diagnosis_valid")),
        "score": numeric_summary(numbers_for(records, "score")),
        "category_match": numeric_summary(numbers_for(records, "category_match")),
        "confident_error_rate": bool_rate(records, "confident_error"),
        "raw_log_chars": numeric_summary(numbers_for(records, "raw_log_chars")),
        "raw_log_tokens": numeric_summary(numbers_for(records, "raw_log_tokens")),
        "exit_status_counts": statuses,
        "timing": timing_summary(output_dir),
    }


def timing_summary(output_dir: Path) -> dict[str, Any]:
    model_calls = []
    tool_calls = []
    problem_timings = []
    for trajectory in sorted(output_dir.glob("**/*.traj.json")):
        try:
            data = json.loads(trajectory.read_text())
        except Exception:
            continue
        problem = data.get("info", {}).get("token_timing", {}).get("problem", {})
        if problem:
            problem_timings.append(problem)
        for message in data.get("messages", []):
            if not isinstance(message, dict):
                continue
            timing = message.get("extra", {}).get("token_timing", {})
            if model_call := timing.get("model_call"):
                model_calls.append(model_call)
            tool_calls.extend(timing.get("tool_calls", []))

    return {
        "problem_e2e_s": numeric_summary(numbers_for(problem_timings, "e2e_s")),
        "ttft_s": numeric_summary(numbers_for(model_calls, "ttft_s")),
        "model_total_s": numeric_summary(numbers_for(model_calls, "model_total_s")),
        "decode_s": numeric_summary(numbers_for(model_calls, "decode_s")),
        "tool_duration_s": numeric_summary(numbers_for(tool_calls, "duration_s")),
        "raw_observation_tokens": numeric_summary(numbers_for(tool_calls, "raw_output_tokens")),
        "rendered_observation_tokens": numeric_summary(numbers_for(tool_calls, "rendered_observation_tokens")),
        "model_calls": len(model_calls),
        "tool_calls": len(tool_calls),
    }


def numbers_for(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0, "mean": None, "max": None}
    return {"count": len(values), "sum": sum(values), "mean": sum(values) / len(values), "max": max(values)}


def bool_rate(records: list[dict[str, Any]], field: str) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if record.get(field)) / len(records)


def load_metric_tokenizer(config: dict):
    tokenizer_path = config.get("agent", {}).get("tokenizer_path", "")
    if not tokenizer_path:
        return None
    local_files_only = bool(config.get("agent", {}).get("tokenizer_local_files_only", True))
    try:
        return load_tokenizer(tokenizer_path, local_files_only=local_files_only)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer for LogDx raw log metrics: {e}")
        return None


def process_instance(
    case: LogDxCase,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    tokenizer,
) -> None:
    instance_dir = output_dir / case.case_id
    workspace = instance_dir / "workspace"
    traj_path = instance_dir / f"{case.case_id}.traj.json"
    prediction_path = instance_dir / "diagnosis.json"

    remove_from_preds_file(output_dir / "preds.json", case.case_id)
    traj_path.unlink(missing_ok=True)

    progress_manager.on_instance_start(case.case_id)
    progress_manager.update_instance_status(case.case_id, "Preparing LogDx workspace")

    agent = None
    env = None
    model = None
    exit_status = None
    result = ""
    extra_info: dict[str, Any] = {}
    record: dict[str, Any] | None = None

    try:
        prepare_workspace(case, workspace)
        task = build_task(case, workspace)

        inst_config = copy.deepcopy(config)
        env_config = inst_config.setdefault("environment", {})
        env_config["cwd"] = str(workspace)

        model = get_model(config=inst_config.get("model", {}))
        env = get_environment(env_config, default_type="local")

        agent_config = dict(inst_config.get("agent", {}))
        agent_config["output_path"] = str(traj_path)
        agent_class_spec = agent_config.pop("agent_class", "")
        agent_class = get_agent_class(agent_class_spec) if agent_class_spec else ProgressTrackingAgent
        agent = agent_class(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=case.case_id,
            **agent_config,
        )
        agent.extra_template_vars = {"case": safe_case_metadata(case)}

        progress_manager.update_instance_status(case.case_id, "Running agent")
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission") or ""

        instance_dir.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(result, encoding="utf-8")
        record = build_result_record(
            case=case,
            exit_status=exit_status,
            submission=result,
            prediction_path=prediction_path,
            trajectory_path=traj_path,
            tokenizer=tokenizer,
        )
    except Exception as e:
        logger.error(f"Error processing LogDx case {case.case_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
        record = {
            "case_id": case.case_id,
            "split": case.split,
            "exit_status": "RunnerError",
            "diagnosis_valid": False,
            "diagnosis_error": str(e),
            "score": None,
            "category_match": None,
            "confident_error": False,
            "scoring_status": "runner_error",
            "raw_log_chars": len(case.raw_log),
            "raw_log_tokens": count_tokens(tokenizer, case.raw_log),
            "prediction_path": str(prediction_path),
            "trajectory_path": str(traj_path),
        }
    finally:
        if agent is not None:
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        "logdx": {
                            "result": record,
                            **extra_info,
                        },
                    },
                    "instance_id": case.case_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            cleanup()
        model_name = getattr(getattr(model, "config", None), "model_name", "")
        update_preds_file(output_dir / "preds.json", case, model_name, result)
        if record is not None:
            append_jsonl(output_dir / "logdx_results.jsonl", record)
            write_logdx_summary(output_dir)
        progress_manager.on_instance_end(case.case_id, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    splits: str = typer.Option("all", "--splits", help="Comma or space separated LogDx splits. Use 'all' for all 35 cases.", rich_help_panel="Data selection"),
    case_id: str | None = typer.Option(None, "--case-id", help="Single LogDx case id", rich_help_panel="Data selection"),
    case_list: Path | None = typer.Option(None, "--case-list", help="File containing one LogDx case id per line", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification, e.g. '0:5'", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter case ids by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle cases deterministically", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    tokenizer_path: str | None = typer.Option(None, "--tokenizer-path", help="Tokenizer path for raw log token metrics", rich_help_panel="Model"),
    corpus_root: Path | None = typer.Option(None, "--corpus-root", help="Optional local LogDx repo or corpus root", rich_help_panel="Data selection"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing cases", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
) -> None:
    # fmt: on
    output_path = Path(output) if output else Path(f"logdx_results_{int(time.time())}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    split_list = parse_splits(splits)
    cases = load_logdx_cases(split_list, corpus_root=corpus_root)
    cases = select_cases(
        cases,
        case_ids=load_case_ids(case_id, case_list),
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing = set(json.loads((output_path / "preds.json").read_text()).keys())
        if existing:
            logger.info(f"Skipping {len(existing)} existing cases")
            cases = [case for case in cases if case.case_id not in existing]
    logger.info(f"Running on {len(cases)} LogDx-CI cases")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "agent": {"tokenizer_path": tokenizer_path or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)
    tokenizer = load_metric_tokenizer(config)

    progress_manager = RunBatchProgressManager(len(cases), output_path / f"exit_statuses_{int(time.time())}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                cid = futures[future]
                logger.error(f"Error in future for LogDx case {cid}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(cid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, case, output_path, config, progress_manager, tokenizer): case.case_id
                for case in cases
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
