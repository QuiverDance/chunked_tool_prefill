#!/usr/bin/env python3

"""Run mini-SWE-agent on BugSwarm CI-Bench repair artifacts."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import re
import shlex
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import requests
import typer
from rich.live import Live

from minisweagent.agents import get_agent_class
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import cleanup_environment, remove_from_preds_file, update_preds_file
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

BUGSWARM_API_BASE = "http://www.api.bugswarm.org/v1"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "cibench.yaml"
_RESULTS_FILE_LOCK = threading.Lock()

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_HELP_TEXT = """Run mini-SWE-agent on BugSwarm CI-Bench repair artifacts."""
_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

Multiple configs are recursively merged from left to right.
"""


def bugswarm_image_name(artifact_id: str) -> str:
    return f"bugswarm/cached-images:{artifact_id}"


def load_artifact_ids(artifact_id: str | None, artifact_list: Path | None) -> list[str]:
    if bool(artifact_id) == bool(artifact_list):
        raise typer.BadParameter("Specify exactly one of --artifact-id or --artifact-list.")
    if artifact_id:
        return [artifact_id.strip()]
    assert artifact_list is not None
    ids = []
    for line in artifact_list.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def select_artifacts(
    artifact_ids: list[str],
    *,
    filter_spec: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict[str, str]]:
    ids = [artifact_id for artifact_id in artifact_ids if artifact_id]
    if shuffle:
        ids = sorted(ids)
        random.seed(42)
        random.shuffle(ids)
    if filter_spec:
        before = len(ids)
        ids = [artifact_id for artifact_id in ids if re.match(filter_spec, artifact_id)]
        logger.info(f"Artifact filter: {before} -> {len(ids)} artifacts")
    if slice_spec:
        before = len(ids)
        parts = [int(part) if part else None for part in slice_spec.split(":")]
        ids = ids[slice(*parts)]
        logger.info(f"Artifact slice: {before} -> {len(ids)} artifacts")
    return [{"artifact_id": artifact_id} for artifact_id in ids]


def request_auth(token: str | None):
    return requests.auth.HTTPBasicAuth(token, "") if token else None


def bugswarm_get_json(path: str, token: str | None) -> dict:
    url = f"{BUGSWARM_API_BASE}/{path.lstrip('/')}"
    response = requests.get(url, auth=request_auth(token), timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_artifact_metadata(artifact_id: str, token: str | None) -> dict[str, Any]:
    return bugswarm_get_json(f"artifacts/{artifact_id}", token)


def fetch_failed_build_log(metadata: dict[str, Any], token: str | None) -> str:
    job_id = metadata.get("failed_job", {}).get("job_id")
    if not job_id:
        return ""
    try:
        return str(bugswarm_get_json(f"logs/{job_id}", token).get("build_log", ""))
    except Exception as e:
        logger.warning(f"Failed to fetch build log for job {job_id}: {e}")
        return ""


def failed_repo_path(metadata: dict[str, Any]) -> str:
    ci_service = metadata.get("ci_service") or "travis"
    repo = metadata.get("repo") or ""
    if "/" not in repo:
        raise ValueError(f"BugSwarm metadata has no repo slug: {repo!r}")
    return f"/home/{ci_service}/build/failed/{repo}"


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    failed_job = metadata.get("failed_job", {})
    passed_job = metadata.get("passed_job", {})
    return {
        "image_tag": metadata.get("image_tag"),
        "repo": metadata.get("repo"),
        "lang": metadata.get("lang"),
        "ci_service": metadata.get("ci_service"),
        "failed_job": {
            "job_id": failed_job.get("job_id"),
            "trigger_sha": failed_job.get("trigger_sha"),
            "message": failed_job.get("message"),
            "failed_tests": failed_job.get("failed_tests"),
            "num_tests_failed": failed_job.get("num_tests_failed"),
            "num_tests_run": failed_job.get("num_tests_run"),
            "config": failed_job.get("config"),
        },
        "passed_job": {
            "job_id": passed_job.get("job_id"),
            "trigger_sha": passed_job.get("trigger_sha"),
            "message": passed_job.get("message"),
        },
    }


def tail_text(text: str, max_chars: int = 20000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_task(metadata: dict[str, Any], failed_log: str) -> str:
    data = compact_metadata(metadata)
    failed = data["failed_job"]
    passed = data["passed_job"]
    return f"""Fix this BugSwarm CI failure.

Artifact: {data["image_tag"]}
Repository: {data["repo"]}
Language: {data["lang"]}
CI service: {data["ci_service"]}
Working directory: {failed_repo_path(metadata)}

Failed job:
- job id: {failed["job_id"]}
- commit: {failed["trigger_sha"]}
- message: {failed["message"]}
- failed tests: {failed["failed_tests"]}
- tests run: {failed["num_tests_run"]}
- tests failed: {failed["num_tests_failed"]}
- config: {json.dumps(failed["config"], sort_keys=True)}

Passing job:
- job id: {passed["job_id"]}
- commit: {passed["trigger_sha"]}
- message: {passed["message"]}

Original failed build log tail:
```text
{tail_text(failed_log)}
```
"""


def docker_executable_from_env(env) -> str:
    return getattr(getattr(env, "config", None), "executable", None) or "docker"


def run_docker_exec(
    *,
    executable: str,
    container_id: str,
    user: str,
    cwd: str,
    command: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, "exec", "-u", user, "-w", cwd, container_id, "bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def write_validation_log(log_path: Path, title: str, result: subprocess.CompletedProcess[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n===== {title} =====\n")
        f.write(f"returncode={result.returncode}\n")
        if result.stdout:
            f.write("\n--- stdout ---\n")
            f.write(result.stdout)
        if result.stderr:
            f.write("\n--- stderr ---\n")
            f.write(result.stderr)


def empty_validation(
    artifact_id: str,
    patch_path: Path,
    log_path: Path,
    *,
    exit_status: str,
    message: str = "",
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "patch_applies": False,
        "plausible": False,
        "run_failed_returncode": "",
        "exit_status": exit_status,
        "patch_path": str(patch_path),
        "validation_log_path": str(log_path),
        "message": message,
    }


def validate_generated_patch(
    *,
    env,
    metadata: dict[str, Any],
    patch_path: Path,
    log_path: Path,
    evaluate_sye: bool,
    timeout: int = 1800,
) -> dict[str, Any]:
    artifact_id = metadata["image_tag"]
    if not patch_path.exists() or not patch_path.read_text(errors="replace").strip():
        return empty_validation(artifact_id, patch_path, log_path, exit_status="NoPatch")

    container_id = getattr(env, "container_id", None)
    if not container_id:
        return empty_validation(artifact_id, patch_path, log_path, exit_status="NoContainer")

    executable = docker_executable_from_env(env)
    ci_service = metadata.get("ci_service") or "travis"
    repo_path = failed_repo_path(metadata)
    failed_sha = metadata.get("failed_job", {}).get("trigger_sha") or "HEAD"
    container_patch = f"/tmp/cibench-{uuid.uuid4().hex}.patch"

    copy_result = subprocess.run(
        [executable, "cp", str(patch_path), f"{container_id}:{container_patch}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    write_validation_log(log_path, "docker cp patch", copy_result)
    if copy_result.returncode != 0:
        return empty_validation(artifact_id, patch_path, log_path, exit_status="PatchCopyFailed")

    reset_apply = " && ".join(
        [
            f"git config --global --add safe.directory {shlex.quote(repo_path)} || true",
            f"cd {shlex.quote(repo_path)}",
            "git clean -fdxq",
            f"git reset --hard {shlex.quote(str(failed_sha))}",
            f"cp {shlex.quote(container_patch)} model.patch",
            "git apply model.patch",
        ]
    )
    apply_result = run_docker_exec(
        executable=executable,
        container_id=container_id,
        user=ci_service,
        cwd=repo_path,
        command=reset_apply,
        timeout=300,
    )
    write_validation_log(log_path, "reset and apply patch", apply_result)
    if apply_result.returncode != 0:
        return empty_validation(artifact_id, patch_path, log_path, exit_status="PatchApplyFailed")

    run_result = run_docker_exec(
        executable=executable,
        container_id=container_id,
        user=ci_service,
        cwd=repo_path,
        command="/usr/local/bin/run_failed.sh",
        timeout=timeout,
    )
    write_validation_log(log_path, "run_failed.sh", run_result)
    combined_output = f"{run_result.stdout}\n{run_result.stderr}"
    plausible = run_result.returncode == 0 or "Done. Your build exited with 0." in combined_output
    record: dict[str, Any] = {
        "artifact_id": artifact_id,
        "patch_applies": True,
        "plausible": plausible,
        "run_failed_returncode": run_result.returncode,
        "exit_status": "Plausible" if plausible else "FailedBuild",
        "patch_path": str(patch_path),
        "validation_log_path": str(log_path),
    }
    record.update(evaluate_sye_if_requested(metadata, patch_path, evaluate_sye))
    return record


def evaluate_sye_if_requested(metadata: dict[str, Any], patch_path: Path, evaluate_sye: bool) -> dict[str, Any]:
    if not evaluate_sye:
        return {}
    if metadata.get("lang") != "Java":
        return {"sye_status": "skipped_non_java", "sye_output": ""}

    evaluate_sh = os.getenv("CIBENCH_EVALUATE_SH", "")
    if not evaluate_sh:
        return {"sye_status": "skipped_missing_cibench_evaluate_sh", "sye_output": ""}

    result = subprocess.run(
        [
            "bash",
            evaluate_sh,
            "--evaluation-metric",
            "SYE",
            "--artifact-id",
            str(metadata["image_tag"]),
            "--patch-file-path",
            str(patch_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return {
        "sye_status": "ok" if result.returncode == 0 else "failed",
        "sye_output": f"{result.stdout}\n{result.stderr}".strip(),
    }


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


def build_cibench_summary(results_path: Path) -> dict[str, Any]:
    records = read_jsonl(results_path)
    total = len(records)
    plausible = sum(1 for record in records if record.get("plausible"))
    patch_applies = sum(1 for record in records if record.get("patch_applies"))
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("exit_status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": total,
        "patch_applies": patch_applies,
        "plausible": plausible,
        "patch_apply_rate": patch_applies / total if total else 0,
        "plausibility_rate": plausible / total if total else 0,
        "exit_status_counts": statuses,
    }


def write_cibench_summary(output_dir: Path) -> dict[str, Any]:
    summary = build_cibench_summary(output_dir / "cibench_results.jsonl")
    (output_dir / "summary.cibench.json").write_text(json.dumps(summary, indent=2))
    return summary


def remove_bugswarm_image_after_instance(config: dict, env, artifact_id: str) -> None:
    if not config.get("run", {}).get("remove_docker_image_after_instance", False):
        return
    executable = docker_executable_from_env(env)
    image = bugswarm_image_name(artifact_id)
    result = subprocess.run(
        [executable, "rmi", image],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode == 0:
        logger.info(f"Removed Docker image after artifact {artifact_id}: {image}")
    else:
        logger.warning(f"Failed to remove Docker image after artifact {artifact_id}: {result.stderr or result.stdout}")


def process_instance(
    instance: dict[str, str],
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    *,
    token: str | None,
    evaluate_sye: bool,
) -> None:
    artifact_id = instance["artifact_id"]
    instance_dir = output_dir / artifact_id
    traj_path = instance_dir / f"{artifact_id}.traj.json"
    patch_path = instance_dir / "generated.patch"
    validation_log_path = instance_dir / "validation.log"

    remove_from_preds_file(output_dir / "preds.json", artifact_id)
    traj_path.unlink(missing_ok=True)

    progress_manager.on_instance_start(artifact_id)
    progress_manager.update_instance_status(artifact_id, "Fetching BugSwarm metadata")

    agent = None
    env = None
    model = None
    result = ""
    exit_status = None
    extra_info: dict[str, Any] = {}
    validation = empty_validation(artifact_id, patch_path, validation_log_path, exit_status="NotRun")

    try:
        metadata = fetch_artifact_metadata(artifact_id, token)
        failed_log = fetch_failed_build_log(metadata, token)
        task = build_task(metadata, failed_log)

        inst_config = copy.deepcopy(config)
        env_config = inst_config.setdefault("environment", {})
        env_config["image"] = bugswarm_image_name(artifact_id)
        env_config["cwd"] = failed_repo_path(metadata)

        model = get_model(config=inst_config.get("model", {}))
        progress_manager.update_instance_status(artifact_id, "Starting BugSwarm container")
        env = get_environment(env_config, default_type="docker")
        env.execute({"command": f"git config --global --add safe.directory {shlex.quote(env_config['cwd'])} || true"})

        agent_config = dict(inst_config.get("agent", {}))
        agent_config["output_path"] = str(traj_path)
        agent_class_spec = agent_config.pop("agent_class", "")
        agent_class = get_agent_class(agent_class_spec) if agent_class_spec else ProgressTrackingAgent
        agent = agent_class(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=artifact_id,
            **agent_config,
        )
        agent.extra_template_vars = {"artifact": compact_metadata(metadata)}

        progress_manager.update_instance_status(artifact_id, "Running agent")
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission") or ""

        instance_dir.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(result)
        progress_manager.update_instance_status(artifact_id, "Validating patch")
        validation = validate_generated_patch(
            env=env,
            metadata=metadata,
            patch_path=patch_path,
            log_path=validation_log_path,
            evaluate_sye=evaluate_sye,
        )
    except Exception as e:
        logger.error(f"Error processing CI-Bench artifact {artifact_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
        validation = {
            **validation,
            "exit_status": "RunnerError",
            "message": str(e),
        }
    finally:
        if agent is not None:
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "cibench": {
                            "validation": validation,
                            **extra_info,
                        },
                    },
                    "instance_id": artifact_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        if env is not None:
            cleanup_environment(env, artifact_id)
            remove_bugswarm_image_after_instance(config, env, artifact_id)
        model_name = getattr(getattr(model, "config", None), "model_name", "")
        update_preds_file(output_dir / "preds.json", artifact_id, model_name, result)
        append_jsonl(output_dir / "cibench_results.jsonl", validation)
        write_cibench_summary(output_dir)
        progress_manager.on_instance_end(artifact_id, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    artifact_id: str | None = typer.Option(None, "--artifact-id", help="Single BugSwarm artifact id", rich_help_panel="Data selection"),
    artifact_list: Path | None = typer.Option(None, "--artifact-list", help="File containing one BugSwarm artifact id per line", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification, e.g. '0:5'", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter artifact ids by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle artifacts deterministically", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing artifacts", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type, usually docker", rich_help_panel="Advanced"),
    evaluate_sye: bool = typer.Option(False, "--evaluate-sye", help="Run optional Java SYE evaluation when CIBENCH_EVALUATE_SH is set", rich_help_panel="Evaluation"),
) -> None:
    # fmt: on
    output_path = Path(output) if output else Path(f"cibench_results_{int(time.time())}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    token = os.getenv("BUGSWARM_TOKEN")
    if not token:
        logger.warning("BUGSWARM_TOKEN is not set. BugSwarm API calls may be rate-limited.")

    artifacts = select_artifacts(
        load_artifact_ids(artifact_id, artifact_list),
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing = set(json.loads((output_path / "preds.json").read_text()).keys())
        if existing:
            logger.info(f"Skipping {len(existing)} existing artifacts")
            artifacts = [artifact for artifact in artifacts if artifact["artifact_id"] not in existing]
    logger.info(f"Running on {len(artifacts)} CI-Bench artifacts")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(artifacts), output_path / f"exit_statuses_{int(time.time())}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                artifact = futures[future]
                logger.error(f"Error in future for CI-Bench artifact {artifact}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(artifact, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    artifact,
                    output_path,
                    config,
                    progress_manager,
                    token=token,
                    evaluate_sye=evaluate_sye,
                ): artifact["artifact_id"]
                for artifact in artifacts
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
