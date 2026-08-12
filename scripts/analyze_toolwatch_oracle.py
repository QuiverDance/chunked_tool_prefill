"""Estimate the exact-repeat ToolWatch oracle from token-timing trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Record = dict[str, Any]
LEADING_WORKDIR = re.compile(r"^(?:cd\s+(?:/testbed|/workspace)\s*&&\s*)+")
EXPLICIT_MUTATION = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:
        apply_patch |
        patch |
        rm | mv | cp | touch | mkdir | rmdir | ln | chmod | chown |
        tee |
        apt(?:-get)?\s+(?:install|remove|purge|upgrade) |
        pip3?\s+(?:install|uninstall) |
        git\s+(?:add|apply|checkout|cherry-pick|clean|commit|merge|rebase|reset|restore|switch)
    )\b
    |
    \bsed\b[^\n;|]*\s-i(?:\s|$)
    |
    \bperl\b[^\n;|]*\s-i(?:\s|$)
    |
    \b(?:write_text|write_bytes|writeFile|writeFileSync)\s*\(
    |
    \bopen\s*\([^)]*,\s*["'][wax+]
    """,
    re.IGNORECASE | re.VERBOSE,
)
UNSAFE_QUERY = re.compile(
    r"""
    \b(?:curl|wget|ssh|scp|rsync|nc|netcat)\b |
    \b(?:sleep|watch)\b |
    \b(?:date|time)\b |
    /dev/random | /dev/urandom
    """,
    re.IGNORECASE | re.VERBOSE,
)
VIEW_PROGRAMS = {
    "awk",
    "basename",
    "cat",
    "cut",
    "dirname",
    "du",
    "echo",
    "file",
    "find",
    "free",
    "grep",
    "head",
    "ls",
    "nproc",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "tree",
    "uname",
    "uniq",
    "wc",
    "which",
}
SHELL_SEPARATORS = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
VERIFIER = re.compile(
    r"""
    \bpytest\b |
    \bpython3?\s+-m\s+(?:pytest|unittest)\b |
    \btox\b |
    \bgo\s+test\b |
    \bcargo\s+(?:test|check|clippy)\b |
    \b(?:npm|pnpm|yarn)\b[^\n;&|]*\b(?:test|lint|typecheck|check)\b |
    \b(?:mvn|mvnw)\b[^\n;&|]*\btest\b |
    \b(?:gradle|gradlew)\b[^\n;&|]*\btest\b |
    \b(?:ruff|mypy|eslint|flake8|pyright|tsc)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
BUILD = re.compile(
    r"""
    \b(?:make|ninja)\b |
    \bcmake\b[^\n;&|]*--build\b |
    \bcargo\s+build\b |
    \bgo\s+build\b |
    \bpython3?\s+setup\.py\s+(?:build|build_ext)\b |
    \b(?:mvn|mvnw)\b[^\n;&|]*\b(?:package|install)\b |
    \b(?:gradle|gradlew)\b[^\n;&|]*\bbuild\b |
    \b(?:npm|pnpm|yarn)\b[^\n;&|]*\bbuild\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class ToolCall:
    instance_id: str
    index: int
    command: str
    key: str
    duration_s: float
    start_s: float
    end_s: float
    output_hash: str
    returncode: int | None
    command_class: str
    mutates_state: bool
    write_paths: tuple[str, ...]
    unknown_write: bool
    generation: int
    last_invalidation_end_s: float | None


@dataclass(frozen=True)
class Dependency:
    path: str
    recursive: bool


@dataclass
class Opportunity:
    calls: int = 0
    duration_s: float = 0.0
    saved_s: float = 0.0
    cache_hits: int = 0
    materialized_hits: int = 0
    refresh_calls: int = 0
    refresh_ready: int = 0
    refresh_running: int = 0
    refresh_not_started: int = 0
    output_mismatches: int = 0

    def add_materialized_hit(self, call: ToolCall) -> None:
        self.calls += 1
        self.duration_s += call.duration_s
        self.saved_s += call.duration_s
        self.materialized_hits += 1

    def add_cache_hit(self, call: ToolCall) -> None:
        self.calls += 1
        self.duration_s += call.duration_s
        self.saved_s += call.duration_s
        self.cache_hits += 1

    def add_output_mismatch(self, call: ToolCall) -> None:
        self.calls += 1
        self.duration_s += call.duration_s
        self.output_mismatches += 1

    def add_refresh(self, call: ToolCall, slack_s: float) -> None:
        self.calls += 1
        self.duration_s += call.duration_s
        self.saved_s += min(call.duration_s, slack_s)
        self.refresh_calls += 1
        if slack_s >= call.duration_s:
            self.refresh_ready += 1
        elif slack_s > 0:
            self.refresh_running += 1
        else:
            self.refresh_not_started += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--instance-ids", type=Path)
    args = parser.parse_args()

    allowed_ids = load_instance_ids(args.instance_ids)
    trajectories = sorted(args.trace_dir.glob("**/*.traj.json"))
    if allowed_ids is not None:
        trajectories = [path for path in trajectories if path.stem.removesuffix(".traj") in allowed_ids]
    if not trajectories:
        raise SystemExit(f"No trajectories found below {args.trace_dir}")

    result = analyze(trajectories)
    print(json.dumps(result, indent=2, sort_keys=True))


def load_instance_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return {str(value) for value in payload}
    for field in ("completed_ids", "resolved_ids", "successful_ids"):
        values = payload.get(field)
        if isinstance(values, list):
            return {str(value) for value in values}
    raise ValueError(f"{path} does not contain a recognized instance-id list")


def analyze(trajectories: list[Path]) -> Record:
    calls: list[ToolCall] = []
    e2e_s = 0.0
    task_e2e_s: dict[str, float] = {}

    for path in trajectories:
        trajectory = json.loads(path.read_text())
        instance_id = str(trajectory.get("instance_id") or path.parent.name)
        problem = ((trajectory.get("info") or {}).get("token_timing") or {}).get("problem") or {}
        task_e2e = float(problem.get("e2e_s") or 0.0)
        e2e_s += task_e2e
        task_e2e_s[instance_id] = task_e2e
        calls.extend(extract_calls(trajectory, instance_id))

    total_tool_s = sum(call.duration_s for call in calls)
    policies = {
        "all_non_mutating": lambda call: not call.mutates_state and not UNSAFE_QUERY.search(call.command),
        "views": lambda call: call.command_class == "view",
        "verifiers": lambda call: call.command_class == "verifier",
        "views_and_verifiers": lambda call: call.command_class in {"view", "verifier"},
        "views_verifiers_builds": lambda call: call.command_class in {"view", "verifier", "build"},
    }

    policy_results: dict[str, Record] = {}
    for name, eligible in policies.items():
        opportunity, task_savings, class_savings = measure_opportunity(calls, eligible)
        policy_results[name] = summarize_opportunity(
            opportunity,
            e2e_s=e2e_s,
            total_tool_s=total_tool_s,
            task_savings=task_savings,
            task_e2e_s=task_e2e_s,
            class_savings=class_savings,
        )

    dependency_policies = {
        "dependency_oracle_all_non_mutating": lambda call: not call.mutates_state
        and not UNSAFE_QUERY.search(call.command),
        "dependency_oracle_views_verifiers_builds": lambda call: call.command_class
        in {"view", "verifier", "build"},
    }
    for name, eligible in dependency_policies.items():
        opportunity, task_savings, class_savings = measure_opportunity(
            calls,
            eligible,
            dependency_oracle=True,
        )
        policy_results[name] = summarize_opportunity(
            opportunity,
            e2e_s=e2e_s,
            total_tool_s=total_tool_s,
            task_savings=task_savings,
            task_e2e_s=task_e2e_s,
            class_savings=class_savings,
        )

    materialized_policies = {
        "materialized_views": {"view"},
        "materialized_views_repeat_verifiers": {"view", "verifier"},
        "materialized_views_repeat_verifiers_builds": {"view", "verifier", "build"},
    }
    for name, classes in materialized_policies.items():
        opportunity, task_savings, class_savings = measure_opportunity(
            calls,
            lambda call, selected=classes: call.command_class in selected,
            materialized_classes={"view"},
        )
        policy_results[name] = summarize_opportunity(
            opportunity,
            e2e_s=e2e_s,
            total_tool_s=total_tool_s,
            task_savings=task_savings,
            task_e2e_s=task_e2e_s,
            class_savings=class_savings,
        )

    opportunity, task_savings, class_savings = measure_opportunity(
        calls,
        lambda call: call.command_class in {"view", "verifier", "build"},
        materialized_classes={"view"},
        dependency_oracle=True,
    )
    policy_results["materialized_views_dependency_oracle_verifiers_builds"] = summarize_opportunity(
        opportunity,
        e2e_s=e2e_s,
        total_tool_s=total_tool_s,
        task_savings=task_savings,
        task_e2e_s=task_e2e_s,
        class_savings=class_savings,
    )

    dependency_policies = {
        "dependency_exact_views": {"view"},
        "dependency_exact_views_and_verifiers": {"view", "verifier"},
        "dependency_exact_views_verifiers_builds": {"view", "verifier", "build"},
    }
    for name, classes in dependency_policies.items():
        opportunity, task_savings, class_savings = measure_dependency_opportunity(calls, classes)
        policy_results[name] = summarize_opportunity(
            opportunity,
            e2e_s=e2e_s,
            total_tool_s=total_tool_s,
            task_savings=task_savings,
            task_e2e_s=task_e2e_s,
            class_savings=class_savings,
        )

    class_counts: dict[str, int] = defaultdict(int)
    class_duration_s: dict[str, float] = defaultdict(float)
    for call in calls:
        class_counts[call.command_class] += 1
        class_duration_s[call.command_class] += call.duration_s

    return {
        "trajectory_count": len(trajectories),
        "timed_tool_calls": len(calls),
        "e2e_s": e2e_s,
        "tool_s": total_tool_s,
        "tool_e2e_ratio": ratio(total_tool_s, e2e_s),
        "command_classes": {
            name: {
                "calls": class_counts[name],
                "duration_s": duration_s,
                "tool_time_ratio": ratio(duration_s, total_tool_s),
                "e2e_ratio": ratio(duration_s, e2e_s),
            }
            for name, duration_s in sorted(
                class_duration_s.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "policies": policy_results,
        "method": {
            "repeat_key": "Exact command after trimming and removing a leading cd /testbed|/workspace &&",
            "cache_hit": "No detected invalidation since the previous invocation and byte-identical output/returncode",
            "refresh": "Saved time is min(current duration, call start - last detected invalidation end)",
            "dependency_oracle": (
                "Byte-identical repeated output is treated as unaffected by intervening writes; "
                "changed output refresh starts at the earliest intervening mutation's tool start"
            ),
            "clock": "Tool result wall timestamp minus perf_counter duration",
            "important_limit": "No file-dependency trace; command-text mutation detection is an Experiment-A approximation",
            "dependency_adapter": "Recognized view paths are invalidated only by overlapping parsed write paths; verifier/build dependencies remain workspace-wide",
        },
    }


def extract_calls(trajectory: Record, instance_id: str) -> list[ToolCall]:
    actions: dict[str, str] = {}
    calls: list[ToolCall] = []
    generation = 0
    last_invalidation_end_s: float | None = None

    for message in trajectory.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for action in (message.get("extra") or {}).get("actions") or []:
                if isinstance(action, dict) and action.get("tool_call_id"):
                    actions[str(action["tool_call_id"])] = str(action.get("command") or "")
            continue
        if message.get("role") not in {"tool", "user"}:
            continue

        call_id = str(message.get("tool_call_id") or "")
        command = actions.get(call_id)
        if not command:
            continue
        extra = message.get("extra") or {}
        metric = matching_metric(extra, call_id)
        duration_s = float(metric.get("duration_s") or 0.0)
        end_s = float(extra.get("timestamp") or message.get("timestamp") or 0.0)
        if duration_s <= 0 or end_s <= 0:
            continue

        output = str(extra.get("raw_output") or "")
        returncode = integer_or_none(extra.get("returncode", metric.get("returncode")))
        mutates_state = bool(EXPLICIT_MUTATION.search(command) or has_output_redirection(command))
        write_paths, unknown_write = mutation_scope(command, mutates_state)
        normalized_command = normalize_command(command)
        call = ToolCall(
            instance_id=instance_id,
            index=len(calls),
            command=command,
            key=normalized_command,
            duration_s=duration_s,
            start_s=end_s - duration_s,
            end_s=end_s,
            output_hash=hashlib.sha256(output.encode()).hexdigest(),
            returncode=returncode,
            command_class=classify_command(normalized_command, mutates_state),
            mutates_state=mutates_state,
            write_paths=write_paths,
            unknown_write=unknown_write,
            generation=generation,
            last_invalidation_end_s=last_invalidation_end_s,
        )
        calls.append(call)

        if mutates_state:
            generation += 1
            last_invalidation_end_s = end_s

    return calls


def matching_metric(extra: Record, call_id: str) -> Record:
    metrics = ((extra.get("token_timing") or {}).get("tool_calls") or [])
    for metric in metrics:
        if isinstance(metric, dict) and str(metric.get("tool_call_id") or call_id) == call_id:
            return metric
    return {}


def normalize_command(command: str) -> str:
    normalized = command.replace("\r\n", "\n").strip()
    return LEADING_WORKDIR.sub("", normalized)


def has_output_redirection(command: str) -> bool:
    return bool(output_redirection_targets(command))


def output_redirection_targets(command: str) -> list[str]:
    targets = []
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if quote:
            if character == quote:
                quote = ""
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            index += 1
            continue
        if character != ">":
            index += 1
            continue

        target = command[index + 1 :].lstrip(">")
        target = target.lstrip()
        if target.startswith("&") or target.startswith("/dev/null"):
            index += 1
            continue
        try:
            words = shlex.split(target, comments=False, posix=True)
        except ValueError:
            words = target.split()
        if words:
            targets.append(words[0])
        index += 1
    return targets


def mutation_scope(command: str, mutates_state: bool) -> tuple[tuple[str, ...], bool]:
    if not mutates_state:
        return (), False

    raw_targets = output_redirection_targets(command)
    raw_targets.extend(write_api_targets(command))
    raw_targets.extend(shell_mutation_targets(command))
    workspace_paths = {
        normalized
        for target in raw_targets
        if (normalized := workspace_path(target)) is not None
    }
    return tuple(sorted(workspace_paths)), not raw_targets


def write_api_targets(command: str) -> list[str]:
    targets = []
    patterns = (
        r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wax+]",
        r"\bPath\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\.\s*write_(?:text|bytes)",
        r"\b(?:writeFile|writeFileSync)\s*\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        targets.extend(re.findall(pattern, command, flags=re.IGNORECASE))
    return targets


def shell_mutation_targets(command: str) -> list[str]:
    targets = []
    for part in SHELL_SEPARATORS.split(command):
        try:
            words = shlex.split(part, comments=False, posix=True)
        except ValueError:
            continue
        if not words:
            continue
        program = Path(words[0]).name
        arguments = [word for word in words[1:] if not word.startswith("-")]
        if program in {"rm", "touch", "mkdir", "rmdir", "chmod", "chown", "ln"}:
            targets.extend(arguments)
        elif program in {"cp", "mv"} and arguments:
            targets.append(arguments[-1])
        elif program == "sed" and any(word == "-i" or word.startswith("-i") for word in words[1:]):
            targets.extend(word for word in arguments if looks_like_path(word))
    return targets


def workspace_path(value: str) -> str | None:
    cleaned = value.strip().strip("'\"").rstrip("/:,.")
    if not cleaned or cleaned in {".", "/testbed", "/workspace"}:
        return ""
    if cleaned.startswith("/tmp/") or cleaned.startswith("/dev/") or cleaned.startswith("/proc/"):
        return None
    if cleaned.startswith("/testbed/"):
        return cleaned.removeprefix("/testbed/").rstrip("/")
    if cleaned.startswith("/workspace/"):
        return cleaned.removeprefix("/workspace/").rstrip("/")
    if cleaned.startswith("/"):
        return None
    return cleaned.removeprefix("./").rstrip("/")


def command_dependencies(call: ToolCall) -> tuple[Dependency, ...]:
    if call.command_class in {"verifier", "build"}:
        return (Dependency("", True),)
    if call.command_class != "view":
        return ()

    dependencies: set[Dependency] = set()
    for part in SHELL_SEPARATORS.split(call.key):
        try:
            words = shlex.split(part, comments=False, posix=True)
        except ValueError:
            return (Dependency("", True),)
        if not words:
            continue
        program = Path(words[0]).name
        if program == "git":
            return (Dependency("", True),)
        dependencies.update(dependencies_for_program(program, words[1:]))
    return tuple(sorted(dependencies, key=lambda dependency: (dependency.path, dependency.recursive)))


def dependencies_for_program(program: str, arguments: list[str]) -> set[Dependency]:
    positional = [argument for argument in arguments if not argument.startswith("-")]
    if program in {"grep", "rg"}:
        roots = positional[1:]
        return dependencies_from_paths(roots, recursive_by_default=True)
    if program == "find":
        roots = []
        for argument in positional:
            if argument.startswith(("!", "(", ")")):
                break
            roots.append(argument)
        return dependencies_from_paths(roots, recursive_by_default=True)
    if program in {"ls", "tree", "du"}:
        return dependencies_from_paths(positional, recursive_by_default=True)
    if program == "sed":
        return dependencies_from_paths(positional[1:], recursive_by_default=False)
    if program in VIEW_PROGRAMS:
        return dependencies_from_paths(positional, recursive_by_default=False)
    return {Dependency("", True)}


def dependencies_from_paths(values: list[str], *, recursive_by_default: bool) -> set[Dependency]:
    dependencies = set()
    for value in values:
        if not looks_like_path(value):
            continue
        path = workspace_path(value)
        if path is None:
            continue
        recursive = recursive_by_default and not Path(path).suffix
        dependencies.add(Dependency(path, recursive))
    return dependencies or {Dependency("", True)}


def looks_like_path(value: str) -> bool:
    if value in {".", "..", "/testbed", "/workspace"}:
        return True
    return (
        value.startswith(("/", "./", "../"))
        or "/" in value
        or bool(Path(value).suffix)
    )


def classify_command(command: str, mutates_state: bool) -> str:
    if mutates_state or UNSAFE_QUERY.search(command):
        return "unsafe"
    if VERIFIER.search(command):
        return "verifier"
    if BUILD.search(command):
        return "build"
    if is_view(command):
        return "view"
    return "other"


def is_view(command: str) -> bool:
    parts = [part.strip() for part in SHELL_SEPARATORS.split(command) if part.strip()]
    if not parts:
        return False
    for part in parts:
        words = re.findall(r"[A-Za-z0-9_./+-]+", part)
        if not words:
            return False
        program = Path(words[0]).name
        if program == "git":
            if len(words) < 2 or words[1] not in {"diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}:
                return False
            continue
        if program not in VIEW_PROGRAMS:
            return False
    return True


def measure_opportunity(
    calls: list[ToolCall],
    eligible: Any,
    *,
    materialized_classes: set[str] | None = None,
    dependency_oracle: bool = False,
) -> tuple[Opportunity, dict[str, float], dict[str, float]]:
    opportunity = Opportunity()
    previous_by_task_and_key: dict[tuple[str, str], ToolCall] = {}
    task_savings: dict[str, float] = defaultdict(float)
    class_savings: dict[str, float] = defaultdict(float)
    mutation_transitions = {
        (call.instance_id, call.generation + 1): call
        for call in calls
        if call.mutates_state
    }

    for call in calls:
        key = (call.instance_id, call.key)
        previous = previous_by_task_and_key.get(key)
        previous_by_task_and_key[key] = call
        if not eligible(call):
            continue

        if call.command_class in (materialized_classes or set()):
            saved_s = call.duration_s
            opportunity.add_materialized_hit(call)
        elif previous is None:
            continue
        elif dependency_oracle and (
            previous.output_hash == call.output_hash
            and previous.returncode == call.returncode
        ):
            saved_s = call.duration_s
            opportunity.add_cache_hit(call)
        elif previous.generation == call.generation:
            if (
                previous.output_hash == call.output_hash
                and previous.returncode == call.returncode
            ):
                saved_s = call.duration_s
                opportunity.add_cache_hit(call)
            else:
                saved_s = 0.0
                opportunity.add_output_mismatch(call)
        else:
            if dependency_oracle:
                first_invalidation = mutation_transitions.get(
                    (call.instance_id, previous.generation + 1)
                )
                refresh_start_s = first_invalidation.start_s if first_invalidation else call.start_s
            else:
                refresh_start_s = call.last_invalidation_end_s or call.start_s
            slack_s = max(0.0, call.start_s - refresh_start_s)
            saved_s = min(call.duration_s, slack_s)
            opportunity.add_refresh(call, slack_s)

        task_savings[call.instance_id] += saved_s
        class_savings[call.command_class] += saved_s

    return opportunity, task_savings, class_savings


def measure_dependency_opportunity(
    calls: list[ToolCall],
    eligible_classes: set[str],
) -> tuple[Opportunity, dict[str, float], dict[str, float]]:
    opportunity = Opportunity()
    previous_by_task_and_key: dict[tuple[str, str], tuple[ToolCall, int]] = {}
    mutations_by_task: dict[str, list[ToolCall]] = defaultdict(list)
    task_savings: dict[str, float] = defaultdict(float)
    class_savings: dict[str, float] = defaultdict(float)

    for call in calls:
        mutations = mutations_by_task[call.instance_id]
        key = (call.instance_id, call.key)
        previous_record = previous_by_task_and_key.get(key)
        previous_by_task_and_key[key] = (call, len(mutations))

        if previous_record is not None and call.command_class in eligible_classes:
            previous, mutation_index = previous_record
            dependencies = command_dependencies(call)
            relevant = [
                mutation
                for mutation in mutations[mutation_index:]
                if mutation_invalidates(dependencies, mutation)
            ]
            if not relevant:
                if previous.output_hash == call.output_hash and previous.returncode == call.returncode:
                    saved_s = call.duration_s
                    opportunity.add_cache_hit(call)
                else:
                    saved_s = 0.0
                    opportunity.add_output_mismatch(call)
            else:
                slack_s = max(0.0, call.start_s - relevant[-1].end_s)
                saved_s = min(call.duration_s, slack_s)
                opportunity.add_refresh(call, slack_s)

            task_savings[call.instance_id] += saved_s
            class_savings[call.command_class] += saved_s

        if call.mutates_state:
            mutations.append(call)

    return opportunity, task_savings, class_savings


def mutation_invalidates(dependencies: tuple[Dependency, ...], mutation: ToolCall) -> bool:
    if mutation.unknown_write or not dependencies:
        return True
    return any(
        dependency_overlaps_write(dependency, write_path)
        for dependency in dependencies
        for write_path in mutation.write_paths
    )


def dependency_overlaps_write(dependency: Dependency, write_path: str) -> bool:
    if not dependency.path or not write_path:
        return True
    if dependency.path == write_path:
        return True
    if dependency.recursive and write_path.startswith(f"{dependency.path}/"):
        return True
    return dependency.path.startswith(f"{write_path}/")


def summarize_opportunity(
    opportunity: Opportunity,
    *,
    e2e_s: float,
    total_tool_s: float,
    task_savings: dict[str, float],
    task_e2e_s: dict[str, float],
    class_savings: dict[str, float],
) -> Record:
    affected_ratios = [
        task_savings[instance_id] / task_e2e_s[instance_id]
        for instance_id in task_savings
        if task_e2e_s.get(instance_id, 0) > 0 and task_savings[instance_id] > 0
    ]
    return {
        "repeat_calls": opportunity.calls,
        "repeat_duration_s": opportunity.duration_s,
        "time_weighted_repeatability": ratio(opportunity.duration_s, total_tool_s),
        "oracle_saved_s": opportunity.saved_s,
        "oracle_tool_time_reduction": ratio(opportunity.saved_s, total_tool_s),
        "oracle_e2e_reduction": ratio(opportunity.saved_s, e2e_s),
        "cache_hits": opportunity.cache_hits,
        "materialized_hits": opportunity.materialized_hits,
        "refresh_calls": opportunity.refresh_calls,
        "refresh_ready": opportunity.refresh_ready,
        "refresh_running": opportunity.refresh_running,
        "refresh_not_started": opportunity.refresh_not_started,
        "refresh_ready_rate": ratio(opportunity.refresh_ready, opportunity.refresh_calls),
        "output_mismatches_without_invalidation": opportunity.output_mismatches,
        "affected_tasks": len(affected_ratios),
        "affected_task_e2e_reduction": distribution(affected_ratios),
        "saved_s_by_class": dict(sorted(class_savings.items(), key=lambda item: item[1], reverse=True)),
    }


def distribution(values: list[float]) -> Record:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": percentile(ordered, 0.9),
        "max": ordered[-1],
    }


def percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def integer_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


if __name__ == "__main__":
    main()
