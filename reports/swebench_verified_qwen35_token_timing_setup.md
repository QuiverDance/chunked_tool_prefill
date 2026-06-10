# SWE-bench Verified Qwen 27B Token Timing Setup

Last updated: 2026-06-09T16:21:56Z

This document summarizes the current experiment environment for running SWE-bench Verified with the local Qwen3.5-27B model through vLLM and mini SWE Agent.

## Goal

Run the full SWE-bench Verified `test` split with Qwen3.5-27B, split across GPU 0 and GPU 1.

The experiment records:

- model input tokens from the API usage field
- model output tokens from the API usage field
- raw tool output size tokenized with the Qwen tokenizer
- tool execution duration, measured around `env.execute(...)`
- command categories such as `ls`, `grep`, `pytest`, not full command strings with options

## Repository Layout

- Experiment repo: `/home/pjw7200/chunked_tool_prefill`
- Clean mini SWE Agent checkout: `/home/pjw7200/chunked_tool_prefill/agent`
- Reports: `/home/pjw7200/chunked_tool_prefill/reports`
- Run logs and trajectories: `/home/pjw7200/chunked_tool_prefill/runs`
- vLLM scripts: `/home/pjw7200/chunked_tool_prefill/scripts`
- Local model: `/home/pjw7200/models/Qwen3.5-27B`

`runs/` is ignored by git. The local vLLM conda environment `.conda/` is also ignored.

## Versions

- mini SWE Agent: `2.3.0`
- Current repo commit: `2395c9d`
- vLLM env: `/home/pjw7200/chunked_tool_prefill/.conda/vllm-py312`
- Python in vLLM env: `3.12.13`
- vLLM: `0.22.1`
- PyTorch: `2.11.0+cu130`
- CUDA visible to the vLLM env: available, 8 devices detected

The active vLLM servers run from the local conda Python 3.12 environment above. The start script points at:

```text
/home/pjw7200/chunked_tool_prefill/.conda/vllm-py312/bin/vllm
```

and launches vLLM with `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `PATH`, and `PYTHONNOUSERSITE=1` set for that environment. This avoids the earlier DeepGEMM/Python ABI mismatch seen in the Python 3.11 environment.

## vLLM Server Setup

Start script:

```bash
./scripts/start_qwen35_vllm.sh
```

Stop script:

```bash
./scripts/stop_qwen35_vllm.sh
```

The start script launches two independent single-GPU vLLM servers:

| GPU | Port | Served model name | Model path |
| --- | ---: | --- | --- |
| 0 | 8000 | `qwen35-27b` | `/home/pjw7200/models/Qwen3.5-27B` |
| 1 | 8001 | `qwen35-27b` | `/home/pjw7200/models/Qwen3.5-27B` |

Explicit vLLM CLI options:

```bash
vllm serve /home/pjw7200/models/Qwen3.5-27B \
  --host 127.0.0.1 \
  --port 8000-or-8001 \
  --served-model-name qwen35-27b \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --language-model-only
```

All other vLLM server options are left at vLLM defaults.

Observed server state:

- `/version` reports vLLM `0.22.1` on both ports
- `/health` returns OK on both ports
- `/v1/models` reports model id `qwen35-27b`
- `/v1/models` reports `max_model_len: 262144`
- engine dtype from vLLM logs: `torch.bfloat16`
- quantization from vLLM logs: `None`
- quantization config from vLLM logs: `None`
- tensor parallel size: `1`
- pipeline parallel size: `1`
- data parallel size: `1`
- prefix caching: `False`
- chunked prefill: `True`
- GPU memory utilization: `0.9200`
- KV cache size: `1,766,757` tokens per server
- seed: `0`

## Sampling Settings

mini SWE Agent does not send `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, or `frequency_penalty` in the current config.

vLLM therefore applies the model `generation_config.json` for the sampling fields that are present there:

```json
{
  "do_sample": true,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95
}
```

For the remaining sampling parameters, vLLM uses its own defaults:

```text
min_p=0.0
presence_penalty=0.0
repetition_penalty=1.0
frequency_penalty=0.0
```

vLLM logs confirm:

```text
Default vLLM sampling parameters have been overridden by the model's generation_config.json:
{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
```

The request-side generation cap is set by mini SWE Agent:

```yaml
model:
  model_kwargs:
    max_tokens: 32768
```

This was checked with a local proxy: the actual `/v1/chat/completions` request includes `max_tokens: 32768`.

## mini SWE Agent Model Config

Main config:

```yaml
model:
  model_name: "hosted_vllm/qwen35-27b"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:8000/v1"
    drop_params: true
    max_tokens: 32768
    parallel_tool_calls: true
```

For GPU 1 runs, the launcher overrides only `api_base`:

```yaml
model.model_kwargs.api_base: "http://127.0.0.1:8001/v1"
```

## Token Timing Agent

Token timing override:

```yaml
agent:
  agent_class: "minisweagent.run.benchmarks.utils.token_timing.TokenTimingProgressAgent"
  tokenizer_path: "/home/pjw7200/models/Qwen3.5-27B"
  tokenizer_local_files_only: true
```

Implementation file:

```text
agent/src/minisweagent/run/benchmarks/utils/token_timing.py
```

What it records in each trajectory:

- `extra.token_timing.model_call.prompt_tokens`
- `extra.token_timing.model_call.completion_tokens`
- `extra.token_timing.model_call.total_tokens`
- `extra.token_timing.tool_call.duration_seconds`
- `extra.token_timing.tool_call.output_tokens`
- `extra.token_timing.tool_call.output_chars`
- `extra.token_timing.tool_call.command_category`
- `extra.token_timing.tool_call.command_categories`

Tool output token counting uses:

```python
tokenizer.encode(raw_output, add_special_tokens=False)
```

Tool duration uses `time.perf_counter()` around the actual environment execution. It includes Docker exec overhead and command runtime, but not the later local tokenization pass.

Final submission actions that raise mini SWE Agent flow exceptions are also recorded before the exception is re-raised.

## Command Categorization

The metric keeps both:

- `command_category`: primary category used for aggregation
- `command_categories`: all command-like entries found in the shell string

The primary category skips setup-only shell commands when a real command follows.

Setup categories:

```text
cd, export, source, ., alias, unalias, set, unset
```

Examples:

| Command | command_categories | command_category |
| --- | --- | --- |
| `ls -al` | `["ls"]` | `ls` |
| `cd /testbed && pytest -q` | `["cd", "pytest"]` | `pytest` |
| `grep -R foo . | head` | `["grep", "head"]` | `grep` |
| `VAR=1 timeout 30 python -m pytest` | `["python"]` | `python` |

This is intentionally category-level, not option-level.

## SWE-bench Verified Launcher

Launcher:

```bash
./scripts/run_verified_token_timing.sh
```

The launcher uses the local mini SWE Agent CLI environment by default:

```text
/home/pjw7200/chunked_tool_prefill/.conda/miniswe-py311/bin/mini-extra
```

You can override it with `MINI_EXTRA=/path/to/mini-extra`. If the local executable is missing, the script falls back to `mini-extra` on `PATH`.

It runs:

```bash
mini-extra swebench \
  --subset verified \
  --split test \
  --workers 1 \
  --config swebench.yaml \
  --config swebench_token_timing.yaml \
  --config run.remove_docker_image_after_instance=true
```

The Docker backend starts each SWE-bench environment with `--rm`, so containers are removed after they stop. The additional `run.remove_docker_image_after_instance=true` setting makes the runner remove the corresponding `swebench/sweb.eval...` Docker image after each instance finishes. This keeps the full 500-instance run from retaining hundreds of large eval images on disk.

Split:

| Worker | Endpoint | Slice | Output |
| --- | --- | --- | --- |
| GPU 0 worker | `http://127.0.0.1:8000/v1` | `0:250` | `runs/$RUN_NAME/gpu0` |
| GPU 1 worker | `http://127.0.0.1:8001/v1` | `250:` | `runs/$RUN_NAME/gpu1` |

The second slice is open-ended on purpose. If the Verified split is exactly 500 instances, this is 250 and 250. If the dataset length changes, the second worker still consumes the remainder.

## Outputs

Each instance writes a trajectory:

```text
runs/$RUN_NAME/gpu0/<instance_id>/<instance_id>.traj.json
runs/$RUN_NAME/gpu1/<instance_id>/<instance_id>.traj.json
```

The launcher runs the summary script after both workers finish:

```bash
python scripts/summarize_token_timing.py "$RUN_DIR" --output-dir "$REPORT_DIR"
```

Summary outputs:

```text
reports/$RUN_NAME/summary.json
reports/$RUN_NAME/model_calls.csv
reports/$RUN_NAME/tool_calls.csv
```

`summary.json` contains global distributions and command-category distributions. The CSV files keep per-call rows for later plotting or deeper analysis.

## Quick Run Checklist

1. Start vLLM:

   ```bash
   ./scripts/start_qwen35_vllm.sh
   ```

2. Check health:

   ```bash
   curl -fsS http://127.0.0.1:8000/health
   curl -fsS http://127.0.0.1:8001/health
   ```

3. Run the experiment:

   ```bash
   ./scripts/run_verified_token_timing.sh
   ```

4. Read the report:

   ```bash
   less reports/<run_name>/summary.json
   ```
