"""Harbor adapter for running local mini-swe-agent on Terminal-Bench 2 tasks."""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import traceback
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.mini_swe_agent import convert_and_save_trajectory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from minisweagent import __version__, package_dir
from minisweagent.agents import get_agent_class
from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import Submitted
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.utils.serialize import recursive_merge


class _Progress:
    def __init__(self, logger, instance_id: str):
        self.logger = logger
        self.instance_id = instance_id

    def update_instance_status(self, instance_id: str, status: str) -> None:
        self.logger.info("mini-swe-agent %s: %s", instance_id, status)


class HarborEnvironmentAdapter:
    """Synchronous mini-swe-agent environment backed by Harbor's async environment."""

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        *,
        cwd: str | None,
        env: dict[str, str],
    ):
        self.environment = environment
        self.loop = loop
        self.cwd = cwd
        self.env = env

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.environment.exec(
                    command,
                    cwd=cwd or self.cwd,
                    env=self.env,
                    timeout_sec=timeout,
                ),
                self.loop,
            )
            result = future.result()
            output = {
                "output": self._combined_output(result.stdout, result.stderr),
                "returncode": result.return_code,
                "exception_info": "",
            }
        except Exception as e:
            output = {
                "output": getattr(e, "stdout", "") or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    @staticmethod
    def _combined_output(stdout: str | None, stderr: str | None) -> str:
        if stdout and stderr:
            return stdout + stderr
        return stdout or stderr or ""

    def _check_finished(self, output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "cwd": self.cwd,
                        "env": self.env,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class MiniSweTokenTimingHarborAgent(BaseAgent):
    """Run local mini-swe-agent inside Harbor environments and keep token timing metrics."""

    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args,
        config_file: str | None = None,
        api_base: str = "http://127.0.0.1:8000/v1",
        tokenizer_path: str = "/home/pjw7200/models/Qwen3.6-27B",
        trajectories_dir: str | None = None,
        step_limit: str | int = 0,
        wall_time_limit_seconds: str | int = 10800,
        max_tokens: str | int | None = None,
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config_file = Path(config_file) if config_file else default_config_file()
        self.api_base = api_base
        self.tokenizer_path = tokenizer_path
        self.trajectories_dir = Path(trajectories_dir) if trajectories_dir else None
        self.step_limit = int(step_limit)
        self.wall_time_limit_seconds = int(wall_time_limit_seconds)
        self.max_tokens = int(max_tokens) if max_tokens not in (None, "") else None
        self.cwd = cwd
        self.extra_env = extra_env or {}

    @staticmethod
    def name() -> str:
        return "mini-swe-agent-token-timing"

    def version(self) -> str | None:
        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        loop = asyncio.get_running_loop()
        trajectory_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        await asyncio.to_thread(self._run_agent, instruction, environment, loop, trajectory_path)
        self._copy_trajectory(trajectory_path)
        self._populate_context(context, trajectory_path)
        self._write_atif_trajectory(trajectory_path)

    def _run_agent(
        self,
        instruction: str,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        trajectory_path: Path,
    ) -> None:
        instance_id = self.logs_dir.parent.name
        env = HarborEnvironmentAdapter(
            environment,
            loop,
            cwd=self.cwd,
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )
        config = self._agent_config(trajectory_path)
        model = get_model(config=config.get("model", {}))
        agent_config = config.get("agent", {}).copy()
        agent_class_spec = agent_config.pop("agent_class", "")
        agent_class = get_agent_class(agent_class_spec) if agent_class_spec else ProgressTrackingAgent
        agent = agent_class(
            model,
            env,
            progress_manager=_Progress(self.logger, instance_id),
            instance_id=instance_id,
            **agent_config,
        )
        try:
            agent.run(instruction)
        except Exception:
            agent.save(
                trajectory_path,
                {
                    "info": {
                        "exit_status": "Exception",
                        "traceback": traceback.format_exc(),
                    },
                    "instance_id": instance_id,
                },
            )
            raise
        else:
            agent.save(trajectory_path, {"instance_id": instance_id})

    def _agent_config(self, trajectory_path: Path) -> dict[str, Any]:
        config = get_config_from_spec(str(self.config_file))
        overrides = {
            "agent": {
                "step_limit": self.step_limit,
                "wall_time_limit_seconds": self.wall_time_limit_seconds,
                "output_path": trajectory_path,
                "tokenizer_path": self.tokenizer_path,
            },
            "model": {
                "model_name": self.model_name,
                "model_kwargs": {
                    "api_base": self.api_base,
                },
            },
        }
        if self.max_tokens is not None:
            overrides["model"]["model_kwargs"]["max_tokens"] = self.max_tokens
        return recursive_merge(config, overrides)

    def _copy_trajectory(self, trajectory_path: Path) -> None:
        if self.trajectories_dir is None or not trajectory_path.exists():
            return
        instance_id = self.logs_dir.parent.name
        target = self.trajectories_dir / instance_id / f"{instance_id}.traj.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trajectory_path, target)

    def _populate_context(self, context: AgentContext, trajectory_path: Path) -> None:
        if not trajectory_path.exists():
            return
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except Exception as e:
            context.metadata = {"mini_swe_agent_trajectory_error": str(e)}
            return

        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0
        for message in trajectory.get("messages") or []:
            usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            n_input_tokens += usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            n_output_tokens += usage.get("completion_tokens") or usage.get("output_tokens") or 0
            n_cache_tokens += details.get("cached_tokens") or 0

        context.n_input_tokens = n_input_tokens
        context.n_output_tokens = n_output_tokens
        context.n_cache_tokens = n_cache_tokens
        context.cost_usd = ((trajectory.get("info") or {}).get("model_stats") or {}).get("instance_cost") or 0.0
        context.metadata = {
            **(context.metadata or {}),
            "mini_swe_agent_trajectory": str(trajectory_path),
        }

    def _write_atif_trajectory(self, trajectory_path: Path) -> None:
        if not trajectory_path.exists():
            return
        try:
            convert_and_save_trajectory(
                mini_swe_agent_trajectory_path=trajectory_path,
                atif_trajectory_path=self.logs_dir / "trajectory.json",
                session_id=self.logs_dir.parent.name,
            )
        except Exception as e:
            self.logger.debug("Failed to convert mini-swe-agent trajectory to ATIF: %s", e)


def default_config_file() -> Path:
    return package_dir / "config" / "benchmarks" / "terminal_bench2_token_timing.yaml"
