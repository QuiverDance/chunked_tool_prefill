#!/usr/bin/env python3

"""Run mini-swe-agent on BrowseComp-Plus queries with local retrieval tools."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any

import typer
from rich.live import Live

from minisweagent.agents import get_agent_class
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.run.benchmarks.utils.token_timing import (
    TokenTimingProgressAgent,
    count_tokens,
)
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "browsecomp_plus_token_timing.yaml"

_HELP_TEXT = """Run mini-swe-agent on BrowseComp-Plus queries.

This runner uses the official BrowseComp-Plus fixed corpus and local retriever, but keeps the local
mini-swe-agent loop and token/tool timing trajectory format.
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

Multiple configs are recursively merged from left to right.
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class BrowseCompPlusEnvironment:
    """In-process BrowseComp-Plus retrieval environment."""

    def __init__(
        self,
        *,
        browsecomp_repo: Path,
        index_path: str,
        embedding_model: str,
        k: int,
        snippet_max_tokens: int,
        include_get_document: bool,
        normalize: bool = True,
        torch_dtype: str = "float16",
        retriever_attn_implementation: str = "sdpa",
        dataset_name: str = "Tevatron/browsecomp-plus-corpus",
    ):
        self.browsecomp_repo = browsecomp_repo.resolve()
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.k = k
        self.snippet_max_tokens = snippet_max_tokens
        self.include_get_document = include_get_document
        self.normalize = normalize
        self.torch_dtype = torch_dtype
        self.retriever_attn_implementation = retriever_attn_implementation
        self.dataset_name = dataset_name
        self.tool_call_counts: Counter[str] = Counter()
        self.retrieved_docids: list[str] = []
        self.records: list[dict[str, Any]] = []
        self.searcher = self._load_searcher()
        self.snippet_tokenizer = self._load_snippet_tokenizer()

    def reset(self) -> None:
        self.tool_call_counts = Counter()
        self.retrieved_docids = []
        self.records = []

    def _load_searcher(self):
        searcher_dir = self.browsecomp_repo / "searcher"
        searchers_dir = searcher_dir / "searchers"
        if not searcher_dir.is_dir():
            raise FileNotFoundError(f"BrowseComp-Plus searcher directory not found: {searcher_dir}")

        FaissSearcher = faiss_searcher_with_attention(
            load_faiss_searcher_class(searchers_dir),
            self.retriever_attn_implementation,
        )
        args = argparse.Namespace(
            index_path=self.index_path,
            model_name=self.embedding_model,
            normalize=self.normalize,
            pooling="eos",
            torch_dtype=self.torch_dtype,
            dataset_name=self.dataset_name,
            task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            max_length=8192,
            attn_implementation=self.retriever_attn_implementation,
        )
        return FaissSearcher(args)

    def _load_snippet_tokenizer(self):
        if self.snippet_max_tokens <= 0:
            return None
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        tool_name = action.get("tool_name")
        try:
            if tool_name == "search":
                payload = self.search(action.get("query", ""))
                returncode = 0
                exception_info = ""
            elif tool_name == "get_document" and self.include_get_document:
                payload = self.get_document(action.get("docid", ""))
                returncode = 0 if payload is not None else 1
                exception_info = "" if payload is not None else "Document not found"
            else:
                payload = {"error": f"Unknown tool: {tool_name}"}
                returncode = 1
                exception_info = f"Unknown tool: {tool_name}"
        except Exception as e:
            payload = {"error": str(e)}
            returncode = 1
            exception_info = f"An error occurred while executing {tool_name}: {e}"

        output = json.dumps(payload, ensure_ascii=False, indent=2)
        record = {
            "tool_name": tool_name,
            "arguments": action.get("arguments") or {},
            "output": payload,
            "returncode": returncode,
        }
        self.records.append(record)
        return {
            "output": output,
            "returncode": returncode,
            "exception_info": exception_info,
            "extra": {
                "tool_name": tool_name,
                "arguments": action.get("arguments") or {},
                "record_index": len(self.records) - 1,
            },
        }

    def search(self, query: str) -> list[dict[str, Any]]:
        self.tool_call_counts["search"] += 1
        candidates = self.searcher.search(query, self.k)
        results = []
        for candidate in candidates:
            docid = str(candidate.get("docid", ""))
            if docid and docid not in self.retrieved_docids:
                self.retrieved_docids.append(docid)
            snippet = self._snippet(candidate.get("text", ""))
            item = {"docid": docid, "snippet": snippet}
            if candidate.get("score") is not None:
                item["score"] = candidate.get("score")
            results.append(item)
        return results

    def get_document(self, docid: str) -> dict[str, Any] | None:
        self.tool_call_counts["get_document"] += 1
        document = self.searcher.get_document(docid)
        if document and docid not in self.retrieved_docids:
            self.retrieved_docids.append(docid)
        return document

    def _snippet(self, text: str) -> str:
        if self.snippet_tokenizer is None or self.snippet_max_tokens <= 0:
            return text
        tokens = self.snippet_tokenizer.encode(text or "", add_special_tokens=False)
        if len(tokens) <= self.snippet_max_tokens:
            return text or ""
        return self.snippet_tokenizer.decode(tokens[: self.snippet_max_tokens], skip_special_tokens=True)

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "k": self.k,
            "snippet_max_tokens": self.snippet_max_tokens,
            "include_get_document": self.include_get_document,
            "searcher_type": "faiss",
            "embedding_model": self.embedding_model,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "browsecomp_repo": str(self.browsecomp_repo),
                        "index_path": self.index_path,
                        "embedding_model": self.embedding_model,
                        "k": self.k,
                        "snippet_max_tokens": self.snippet_max_tokens,
                        "include_get_document": self.include_get_document,
                        "normalize": self.normalize,
                        "torch_dtype": self.torch_dtype,
                        "retriever_attn_implementation": self.retriever_attn_implementation,
                        "dataset_name": self.dataset_name,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class BrowseCompPlusTokenTimingAgent(TokenTimingProgressAgent):
    """Token timing agent for non-shell BrowseComp retrieval tools."""

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        if not actions:
            content = (message.get("content") or "").strip()
            return self.add_messages(
                self.model.format_message(
                    role="exit",
                    content=content,
                    extra={
                        "exit_status": "Submitted" if content else "NoAnswer",
                        "submission": content,
                    },
                )
            )

        outputs = [self.execute_timed_action(action) for action in actions]
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        return self.add_messages(*observations)

    def execute_timed_action(self, action: dict) -> dict:
        start_wall = time.time()
        start_perf = time.perf_counter()
        output = self.env.execute(action)
        end_perf = time.perf_counter()
        end_wall = time.time()

        metric = {
            "instance_id": self.instance_id,
            "tool_call_id": action.get("tool_call_id"),
            "sequence_index": 0,
            "sequence_separator": "tool",
            "command": json.dumps(action.get("arguments") or {}, ensure_ascii=False),
            "command_category": action.get("tool_name", "tool"),
            "start_ts": start_wall,
            "first_stdout_ts": end_wall,
            "last_stdout_ts": end_wall,
            "end_ts": end_wall,
            "duration_s": end_perf - start_perf,
            "time_to_first_stdout_s": end_perf - start_perf,
            "returncode": output.get("returncode"),
            "output_tokens": count_tokens(self.tokenizer, output.get("output", "")),
            "stdout_tokens": count_tokens(self.tokenizer, output.get("output", "")),
            "stderr_tokens": 0,
            "exception_info": output.get("exception_info", ""),
        }
        self.tool_metrics.append(metric)
        output.setdefault("extra", {})["token_timing"] = {"tool_calls": [metric]}
        return output


def load_queries(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"queries.tsv not found: {path}")
    queries = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"Malformed queries.tsv row: {row!r}")
            queries.append({"query_id": row[0].strip(), "query": row[1].strip()})
    return queries


def load_faiss_searcher_class(searchers_dir: Path):
    """Load only the FAISS searcher without importing the BM25/JDK-dependent package init."""

    package_name = "_browsecomp_plus_searchers"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(searchers_dir)]
        sys.modules[package_name] = package

    for module_name in ("base", "faiss_searcher"):
        full_name = f"{package_name}.{module_name}"
        if full_name in sys.modules:
            continue
        module_path = searchers_dir / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load BrowseComp-Plus module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.faiss_searcher"].FaissSearcher


def faiss_searcher_with_attention(base_class, attn_implementation: str):
    class MiniAgentFaissSearcher(base_class):
        def _load_model(self) -> None:
            import torch
            from tevatron.retriever.arguments import ModelArguments
            from tevatron.retriever.driver.encode import DenseModel
            from transformers import AutoTokenizer

            model_args = ModelArguments(
                model_name_or_path=self.args.model_name,
                normalize=self.args.normalize,
                pooling=self.args.pooling,
                cache_dir=os.getenv("HF_HOME") or None,
                attn_implementation=attn_implementation or None,
            )

            dtype_by_name = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_by_name.get(self.args.torch_dtype, torch.float16)

            self.model = DenseModel.load(
                model_args.model_name_or_path,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                lora_name_or_path=model_args.lora_name_or_path,
                cache_dir=model_args.cache_dir,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name or model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                padding_side="left",
            )

    return MiniAgentFaissSearcher


def select_queries(
    queries: list[dict[str, str]],
    *,
    limit: int,
    offset: int,
    query_ids: str,
) -> list[dict[str, str]]:
    if query_ids.strip():
        wanted = [item.strip() for item in query_ids.split(",") if item.strip()]
        wanted_set = set(wanted)
        ordered = {query["query_id"]: query for query in queries if query["query_id"] in wanted_set}
        return [ordered[query_id] for query_id in wanted if query_id in ordered]
    selected = queries[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


def process_query(
    query: dict[str, str],
    *,
    output_dir: Path,
    result_dir: Path,
    config: dict[str, Any],
    env_config: dict[str, Any],
    progress_manager: RunBatchProgressManager,
    redo_existing: bool,
    model: Any | None = None,
    env: BrowseCompPlusEnvironment | None = None,
) -> None:
    query_id = query["query_id"]
    query_dir = output_dir / "trajectories" / query_id
    trajectory_path = query_dir / f"{query_id}.traj.json"
    result_path = result_dir / f"{query_id}.json"
    if not redo_existing and result_path.exists() and trajectory_path.exists():
        progress_manager.on_instance_end(query_id, "Skipped")
        return

    result_path.unlink(missing_ok=True)
    trajectory_path.unlink(missing_ok=True)

    model = model or get_model(config=config.get("model", {}))
    env = env or BrowseCompPlusEnvironment(**env_config)
    env.reset()
    agent_config = config.get("agent", {}).copy()
    agent_config["output_path"] = trajectory_path
    agent_class_spec = agent_config.pop("agent_class", "")
    agent_class = get_agent_class(agent_class_spec) if agent_class_spec else ProgressTrackingAgent

    progress_manager.on_instance_start(query_id)
    progress_manager.update_instance_status(query_id, "Running")
    agent = agent_class(
        model,
        env,
        progress_manager=progress_manager,
        instance_id=query_id,
        **agent_config,
    )

    exit_status = "Exception"
    submission = ""
    extra_info: dict[str, Any] = {}
    try:
        info = agent.run(query["query"], query_id=query_id)
        exit_status = info.get("exit_status", "")
        submission = info.get("submission", "")
    except Exception as e:
        logger.error(f"Error processing BrowseComp query {query_id}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        agent.save(
            trajectory_path,
            {
                "instance_id": query_id,
                "query_id": query_id,
                "info": {
                    "exit_status": exit_status,
                    "submission": submission,
                    **extra_info,
                },
            },
        )
        write_result_json(result_path, query_id, query["query"], env, agent, exit_status, submission, config)
        progress_manager.on_instance_end(query_id, exit_status)


def write_result_json(
    path: Path,
    query_id: str,
    query: str,
    env: BrowseCompPlusEnvironment,
    agent: BrowseCompPlusTokenTimingAgent,
    exit_status: str,
    submission: str,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "metadata": {
            "model": config.get("model", {}).get("model_name"),
            "api_base": config.get("model", {}).get("model_kwargs", {}).get("api_base"),
            "query": query,
        },
        "query_id": query_id,
        "tool_call_counts": dict(env.tool_call_counts),
        "status": "completed" if exit_status == "Submitted" and submission.strip() else exit_status or "incomplete",
        "retrieved_docids": env.retrieved_docids,
        "result": result_items(agent.messages, env.records, submission),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def result_items(messages: list[dict], records: list[dict[str, Any]], submission: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    record_index = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if isinstance(reasoning, str) and reasoning.strip():
            items.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": [reasoning.strip()],
                }
            )
        for action in message.get("extra", {}).get("actions", []):
            record = records[record_index] if record_index < len(records) else {}
            record_index += 1
            items.append(
                {
                    "type": "tool_call",
                    "tool_name": action.get("tool_name"),
                    "arguments": action.get("arguments") or {},
                    "output": record.get("output"),
                }
            )

    final_text = submission.strip()
    if final_text:
        items.append(
            {
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": final_text,
            }
        )
    return items


def build_config(
    config_spec: list[str],
    *,
    model: str | None,
    api_base: str | None,
    tokenizer_path: str | None,
    max_tokens: int | None,
    step_limit: int | None,
    include_get_document: bool,
) -> dict[str, Any]:
    configs = [get_config_from_spec(spec) for spec in config_spec]
    overrides: dict[str, Any] = {
        "model": {
            "model_name": model or UNSET,
            "include_get_document": include_get_document,
            "model_kwargs": {
                "api_base": api_base or UNSET,
                "max_tokens": max_tokens if max_tokens is not None else UNSET,
            },
        },
        "agent": {
            "tokenizer_path": tokenizer_path or UNSET,
            "step_limit": step_limit if step_limit is not None else UNSET,
        },
    }
    return recursive_merge(*configs, overrides)


def default_browsecomp_repo() -> Path:
    return Path.cwd() / "external" / "BrowseComp-Plus"


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    browsecomp_repo: Path = typer.Option(default_browsecomp_repo(), "--browsecomp-repo", help="BrowseComp-Plus repo directory.", rich_help_panel="Data"),
    queries: Path | None = typer.Option(None, "--queries", help="Path to topics-qrels/queries.tsv.", rich_help_panel="Data"),
    index_path: str | None = typer.Option(None, "--index-path", help="Glob path for qwen3-embedding-8b corpus shard pickle files.", rich_help_panel="Data"),
    embedding_model: str = typer.Option("Qwen/Qwen3-Embedding-8B", "--embedding-model", help="Dense retriever model name.", rich_help_panel="Data"),
    output: Path = typer.Option(Path("runs/browsecomp_plus_qwen36_smoke"), "-o", "--output", help="Output run directory.", rich_help_panel="Basic"),
    limit: int = typer.Option(3, "--limit", help="Number of queries to run. 0 means all selected queries.", rich_help_panel="Data selection"),
    offset: int = typer.Option(0, "--offset", help="Query offset when not using --query-ids.", rich_help_panel="Data selection"),
    query_ids: str = typer.Option("", "--query-ids", help="Comma-separated query ids to run.", rich_help_panel="Data selection"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads. Keep 1 for a single local retriever.", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing result/trajectory files.", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name.", rich_help_panel="Model"),
    api_base: str | None = typer.Option(None, "--api-base", help="OpenAI-compatible vLLM API base.", rich_help_panel="Model"),
    tokenizer_path: str | None = typer.Option(None, "--tokenizer-path", help="Tokenizer path for timing token counts.", rich_help_panel="Model"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum generation tokens.", rich_help_panel="Model"),
    step_limit: int | None = typer.Option(None, "--step-limit", help="Maximum mini-agent steps.", rich_help_panel="Model"),
    k: int = typer.Option(5, "--k", help="Retriever top-k.", rich_help_panel="Retriever"),
    snippet_max_tokens: int = typer.Option(512, "--snippet-max-tokens", help="Search snippet max tokens.", rich_help_panel="Retriever"),
    include_get_document: bool = typer.Option(False, "--include-get-document", help="Expose get_document tool in addition to search.", rich_help_panel="Retriever"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Normalize dense vectors.", rich_help_panel="Retriever"),
    torch_dtype: str = typer.Option("float16", "--torch-dtype", help="Retriever torch dtype.", rich_help_panel="Retriever"),
    retriever_attn_implementation: str = typer.Option("sdpa", "--retriever-attn-implementation", help="Retriever attention implementation.", rich_help_panel="Retriever"),
) -> None:
    # fmt: on
    browsecomp_repo = browsecomp_repo.resolve()
    query_path = (queries or browsecomp_repo / "topics-qrels" / "queries.tsv").resolve()
    resolved_index_path = index_path or str(browsecomp_repo / "indexes" / "qwen3-embedding-8b" / "corpus.shard*.pkl")
    if not glob.glob(resolved_index_path):
        raise FileNotFoundError(f"No FAISS index shards matched: {resolved_index_path}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    result_dir = output / "browsecomp_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "minisweagent.log")

    all_queries = load_queries(query_path)
    selected = select_queries(all_queries, limit=limit, offset=offset, query_ids=query_ids)
    logger.info(f"Loaded {len(all_queries)} queries from {query_path}; running {len(selected)}")

    config = build_config(
        config_spec,
        model=model,
        api_base=api_base,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        step_limit=step_limit,
        include_get_document=include_get_document,
    )
    env_config = {
        "browsecomp_repo": browsecomp_repo,
        "index_path": resolved_index_path,
        "embedding_model": embedding_model,
        "k": k,
        "snippet_max_tokens": snippet_max_tokens,
        "include_get_document": include_get_document,
        "normalize": normalize,
        "torch_dtype": torch_dtype,
        "retriever_attn_implementation": retriever_attn_implementation,
    }
    progress_manager = RunBatchProgressManager(len(selected), output / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                query_id = futures[future]
                logger.error(f"Error in future for BrowseComp query {query_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(query_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers == 1:
            shared_model = get_model(config=config.get("model", {}))
            shared_env = BrowseCompPlusEnvironment(**env_config)
            for query in selected:
                process_query(
                    query,
                    output_dir=output,
                    result_dir=result_dir,
                    config=config,
                    env_config=env_config,
                    progress_manager=progress_manager,
                    redo_existing=redo_existing,
                    model=shared_model,
                    env=shared_env,
                )
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_query,
                    query,
                    output_dir=output,
                    result_dir=result_dir,
                    config=config,
                    env_config=env_config,
                    progress_manager=progress_manager,
                    redo_existing=redo_existing,
                ): query["query_id"]
                for query in selected
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
