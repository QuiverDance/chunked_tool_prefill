#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench Pro public instances."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import typer
from rich.live import Live

from minisweagent.agents import get_agent_class
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    cleanup_environment,
    remove_docker_image_after_instance,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DATASET_SPLIT = "test"
DOCKERHUB_USERNAME = "jefzda"
DOCKER_IMAGE_REPO = "sweap-images"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_pro.yaml"
RESULTS_FILE = "swebench_pro_results.jsonl"
RAW_SAMPLE_FILE = "swebench_pro_raw.jsonl"
PATCH_FILE = "swebench_pro_patches.json"
GOLD_FIELDS = {
    "patch",
    "test_patch",
    "fail_to_pass",
    "pass_to_pass",
    "selected_test_files_to_run",
    "before_repo_set_cmd",
}
_RESULTS_FILE_LOCK = threading.Lock()

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_HELP_TEXT = """Run mini-SWE-agent on SWE-bench Pro public instances."""
_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

Multiple configs are recursively merged from left to right.
"""


def swebench_pro_image_name(instance: dict[str, Any]) -> str:
    tag = str(instance.get("dockerhub_tag") or "").strip()
    if not tag:
        raise ValueError(f"SWE-bench Pro instance has no dockerhub_tag: {instance.get('instance_id')!r}")
    return f"{DOCKERHUB_USERNAME}/{DOCKER_IMAGE_REPO}:{tag}"


def load_instance_ids(instance_id: str | None, instance_list: Path | None) -> list[str]:
    if instance_id and instance_list:
        raise typer.BadParameter("Specify at most one of --instance-id or --instance-list.")
    if instance_id:
        return [instance_id.strip()]
    if instance_list is None:
        return []

    ids = []
    for line in instance_list.read_text().splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.append(instance_id_from_list_entry(value))
    return ids


def instance_id_from_list_entry(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    if value.startswith("{"):
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return value
        return str(data.get("instance_id") or "")
    return value.split()[0].split("/")[-1]


def select_instances(
    instances: list[dict[str, Any]],
    *,
    instance_ids: list[str],
    filter_spec: str = "",
    repo_filter: str = "",
    language_filter: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict[str, Any]]:
    selected = [dict(instance) for instance in instances]
    if instance_ids:
        wanted = set(instance_ids)
        before = len(selected)
        selected = [instance for instance in selected if instance.get("instance_id") in wanted]
        logger.info(f"Instance id selection: {before} -> {len(selected)} instances")
    if filter_spec:
        before = len(selected)
        selected = [instance for instance in selected if re.match(filter_spec, str(instance.get("instance_id", "")))]
        logger.info(f"Instance filter: {before} -> {len(selected)} instances")
    if repo_filter:
        before = len(selected)
        selected = [instance for instance in selected if re.match(repo_filter, str(instance.get("repo", "")))]
        logger.info(f"Repo filter: {before} -> {len(selected)} instances")
    if language_filter:
        before = len(selected)
        selected = [instance for instance in selected if re.match(language_filter, str(instance.get("repo_language", "")))]
        logger.info(f"Language filter: {before} -> {len(selected)} instances")
    if shuffle:
        selected = sorted(selected, key=lambda instance: str(instance.get("instance_id", "")))
        random.seed(42)
        random.shuffle(selected)
    if slice_spec:
        before = len(selected)
        parts = [int(part) if part else None for part in slice_spec.split(":")]
        selected = selected[slice(*parts)]
        logger.info(f"Instance slice: {before} -> {len(selected)} instances")
    return selected


def build_task(instance: dict[str, Any]) -> str:
    sections = [
        "Fix this SWE-bench Pro issue.",
        "",
        f"Instance: {instance.get('instance_id', '')}",
        f"Repository: {instance.get('repo', '')}",
        f"Language: {instance.get('repo_language', '')}",
        "",
        "Problem statement:",
        "```text",
        str(instance.get("problem_statement") or "").strip(),
        "```",
    ]
    if requirements := formatted_field(instance.get("requirements")):
        sections.extend(["", "Requirements:", "```text", requirements, "```"])
    if interface := formatted_field(instance.get("interface")):
        sections.extend(["", "Interface notes:", "```text", interface, "```"])
    return "\n".join(sections).strip() + "\n"


def formatted_field(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2, sort_keys=True)


def instance_for_agent(instance: dict[str, Any]) -> dict[str, Any]:
    agent_instance = dict(instance)
    agent_instance["problem_statement"] = build_task(instance)
    agent_instance["docker_image"] = swebench_pro_image_name(instance)
    return agent_instance


def compact_instance_metadata(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": instance.get("instance_id"),
        "repo": instance.get("repo"),
        "repo_language": instance.get("repo_language"),
        "base_commit": instance.get("base_commit"),
        "dockerhub_tag": instance.get("dockerhub_tag"),
        "docker_image": swebench_pro_image_name(instance),
    }


def write_raw_samples(path: Path, instances: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for instance in instances:
            f.write(json.dumps(dict(instance), sort_keys=True) + "\n")


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


def write_official_patch_file(output_dir: Path) -> list[dict[str, Any]]:
    patches = preds_to_official_patches(output_dir / "preds.json")
    (output_dir / PATCH_FILE).write_text(json.dumps(patches, indent=2))
    return patches


def preds_to_official_patches(preds_path: Path) -> list[dict[str, Any]]:
    if not preds_path.exists():
        return []
    preds = json.loads(preds_path.read_text())
    patches = []
    for instance_id in sorted(preds):
        pred = preds[instance_id]
        patch = pred.get("model_patch") or pred.get("patch") or ""
        patches.append(
            {
                "instance_id": pred.get("instance_id") or instance_id,
                "patch": patch,
                "prefix": pred.get("model_name_or_path") or "mini-swe-agent",
            }
        )
    return patches


def build_result_record(
    *,
    instance: dict[str, Any],
    exit_status: str | None,
    submission: str,
    patch_path: Path,
    trajectory_path: Path,
) -> dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "repo": instance.get("repo"),
        "repo_language": instance.get("repo_language"),
        "base_commit": instance.get("base_commit"),
        "dockerhub_tag": instance.get("dockerhub_tag"),
        "docker_image": swebench_pro_image_name(instance),
        "exit_status": exit_status,
        "submitted": exit_status == "Submitted",
        "has_patch": has_patch(submission),
        "patch_chars": len(submission or ""),
        "patch_path": str(patch_path),
        "trajectory_path": str(trajectory_path),
    }


def has_patch(text: str) -> bool:
    return "diff --git " in (text or "")


def write_swebench_pro_summary(output_dir: Path) -> dict[str, Any]:
    summary = build_swebench_pro_summary(output_dir)
    (output_dir / "summary.swebench_pro.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_swebench_pro_summary(output_dir: Path) -> dict[str, Any]:
    records = read_jsonl(output_dir / RESULTS_FILE)
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1

    eval_summary = load_eval_summary(output_dir)
    return {
        "total": len(records),
        "submitted": sum(1 for record in records if record.get("submitted")),
        "generated_patches": sum(1 for record in records if record.get("has_patch")),
        "no_patch": sum(1 for record in records if not record.get("has_patch")),
        "resolved": eval_summary.get("resolved"),
        "pass_at_1": eval_summary.get("pass_at_1"),
        "eval_failures": eval_summary.get("eval_failures"),
        "exit_status_counts": statuses,
        "timing": timing_summary(output_dir),
        "official_eval": eval_summary,
    }


def load_eval_summary(output_dir: Path) -> dict[str, Any]:
    for path in (output_dir / "swebench_pro_eval" / "eval_results.json", output_dir / "eval_results.json"):
        if path.exists():
            return summarize_eval_results(json.loads(path.read_text()), path)
    return {"path": "", "total": 0, "resolved": None, "pass_at_1": None, "eval_failures": None}


def summarize_eval_results(data: Any, path: Path) -> dict[str, Any]:
    records = eval_records(data)
    total = len(records)
    resolved = sum(1 for record in records if eval_record_resolved(record))
    failures = sum(1 for record in records if eval_record_failed(record))
    return {
        "path": str(path),
        "total": total,
        "resolved": resolved,
        "pass_at_1": resolved / total if total else None,
        "eval_failures": failures,
    }


def eval_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("results", "instances", "eval_results"):
        value = data.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    if all(isinstance(value, bool) for value in data.values()):
        return [{"instance_id": key, "resolved": value} for key, value in data.items()]
    if all(isinstance(value, dict) for value in data.values()):
        return list(data.values())
    return [data]


def eval_record_resolved(record: dict[str, Any]) -> bool:
    for key in ("resolved", "success", "passed", "pass", "result"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"resolved", "pass", "passed", "success", "true"}:
            return True
    return False


def eval_record_failed(record: dict[str, Any]) -> bool:
    for key in ("error", "exception", "traceback", "failure"):
        if record.get(key):
            return True
    status = str(record.get("status") or "").lower()
    return status in {"error", "failed", "exception"}


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


def load_swebench_pro_instances() -> list[dict[str, Any]]:
    from datasets import load_dataset

    return [dict(instance) for instance in load_dataset(DATASET_NAME, split=DATASET_SPLIT)]


def process_instance(
    instance: dict[str, Any],
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    instance_id = str(instance["instance_id"])
    instance_dir = output_dir / instance_id
    traj_path = instance_dir / f"{instance_id}.traj.json"
    patch_path = instance_dir / "generated.patch"

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    traj_path.unlink(missing_ok=True)
    patch_path.unlink(missing_ok=True)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Starting SWE-bench Pro container")

    agent = None
    env = None
    model = None
    exit_status = None
    result = ""
    extra_info: dict[str, Any] = {}
    record: dict[str, Any] | None = None

    try:
        agent_instance = instance_for_agent(instance)
        inst_config = copy.deepcopy(config)
        env_config = inst_config.setdefault("environment", {})
        env_config["image"] = agent_instance["docker_image"]
        env_config.setdefault("cwd", "/app")

        model = get_model(config=inst_config.get("model", {}))
        env = get_environment(env_config, default_type="docker")
        env.execute({"command": f"git config --global --add safe.directory {env_config['cwd']} || true"})

        agent_config = dict(inst_config.get("agent", {}))
        agent_config["output_path"] = str(traj_path)
        agent_class_spec = agent_config.pop("agent_class", "")
        agent_class = get_agent_class(agent_class_spec) if agent_class_spec else ProgressTrackingAgent
        agent = agent_class(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **agent_config,
        )
        agent.extra_template_vars = {"swebench_pro": compact_instance_metadata(instance)}

        progress_manager.update_instance_status(instance_id, "Running agent")
        info = agent.run(agent_instance["problem_statement"])
        exit_status = info.get("exit_status")
        result = info.get("submission") or ""

        instance_dir.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(result, encoding="utf-8")
        record = build_result_record(
            instance=instance,
            exit_status=exit_status,
            submission=result,
            patch_path=patch_path,
            trajectory_path=traj_path,
        )
    except Exception as e:
        logger.error(f"Error processing SWE-bench Pro instance {instance_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
        record = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "repo_language": instance.get("repo_language"),
            "exit_status": "RunnerError",
            "submitted": False,
            "has_patch": False,
            "patch_chars": 0,
            "patch_path": str(patch_path),
            "trajectory_path": str(traj_path),
            "message": str(e),
        }
    finally:
        if agent is not None:
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        "swebench_pro": {
                            "result": record,
                            **extra_info,
                        },
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        if env is not None:
            cleanup_environment(env, instance_id)
            remove_docker_image_after_instance(config, env, instance_for_agent(instance))
        model_name = getattr(getattr(model, "config", None), "model_name", "")
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result)
        if record is not None:
            append_jsonl(output_dir / RESULTS_FILE, record)
            write_swebench_pro_summary(output_dir)
        progress_manager.on_instance_end(instance_id, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    instance_id: str | None = typer.Option(None, "--instance-id", help="Single SWE-bench Pro instance id", rich_help_panel="Data selection"),
    instance_list: Path | None = typer.Option(None, "--instance-list", help="File containing one instance id per line", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification, e.g. '0:5'", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance ids by regex", rich_help_panel="Data selection"),
    repo_filter: str = typer.Option("", "--repo-filter", help="Filter repositories by regex", rich_help_panel="Data selection"),
    language_filter: str = typer.Option("", "--language-filter", help="Filter repo_language by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances deterministically", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    tokenizer_path: str | None = typer.Option(None, "--tokenizer-path", help="Tokenizer path for token timing metrics", rich_help_panel="Model"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
) -> None:
    # fmt: on
    output_path = Path(output) if output else Path(f"swebench_pro_results_{int(time.time())}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    logger.info(f"Loading {DATASET_NAME} split {DATASET_SPLIT}")
    all_instances = load_swebench_pro_instances()
    selected_instances = select_instances(
        all_instances,
        instance_ids=load_instance_ids(instance_id, instance_list),
        filter_spec=filter_spec,
        repo_filter=repo_filter,
        language_filter=language_filter,
        slice_spec=slice_spec,
        shuffle=shuffle,
    )
    write_raw_samples(output_path / RAW_SAMPLE_FILE, selected_instances)

    instances = list(selected_instances)
    if not redo_existing and (output_path / "preds.json").exists():
        existing = set(json.loads((output_path / "preds.json").read_text()).keys())
        if existing:
            logger.info(f"Skipping {len(existing)} existing instances")
            instances = [instance for instance in instances if instance["instance_id"] not in existing]
    logger.info(f"Running on {len(instances)} SWE-bench Pro instances")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "agent": {"tokenizer_path": tokenizer_path or UNSET},
            "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
        }
    )
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{int(time.time())}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                iid = futures[future]
                logger.error(f"Error in future for SWE-bench Pro instance {iid}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(iid, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): str(instance["instance_id"])
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    write_official_patch_file(output_path)
    write_swebench_pro_summary(output_path)


if __name__ == "__main__":
    app()
