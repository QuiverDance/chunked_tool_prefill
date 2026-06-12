# SWE-bench Verified Qwen3.6-27B Token Timing Setup

Last updated: 2026-06-11T14:56:00Z

## Model

- Hugging Face repo: `Qwen/Qwen3.6-27B`
- Local path: `/home/pjw7200/models/Qwen3.6-27B`
- Local size: about `52G`
- Checkpoint shards: `15`
- vLLM dtype: `bfloat16`
- vLLM quantization: `None`
- Served model name: `qwen36-27b`

The local download is the standard safetensors checkpoint, not the FP8 variant.

## vLLM Scripts

Start Qwen3.6 on GPU 0 and GPU 1:

```bash
./scripts/start_qwen36_vllm.sh
```

Stop Qwen3.6:

```bash
./scripts/stop_qwen36_vllm.sh
```

The start script uses the same local vLLM Python 3.12 environment as Qwen3.5:

```text
/home/pjw7200/chunked_tool_prefill/.conda/vllm-py312/bin/vllm
```

It launches:

| GPU | Port | Served model name | Model path |
| --- | ---: | --- | --- |
| 0 | 8000 | `qwen36-27b` | `/home/pjw7200/models/Qwen3.6-27B` |
| 1 | 8001 | `qwen36-27b` | `/home/pjw7200/models/Qwen3.6-27B` |

The script refuses to start if port `8000` or `8001` is already occupied, so stop Qwen3.5 first:

```bash
./scripts/stop_qwen35_vllm.sh
./scripts/start_qwen36_vllm.sh
```

## Experiment Launcher

Run the same SWE-bench Verified token timing experiment with Qwen3.6:

```bash
./scripts/run_verified_token_timing_qwen36.sh
```

This wrapper calls `run_verified_token_timing.sh` with:

```bash
RUN_NAME_PREFIX=swebench_verified_qwen36_token_timing
SERVED_MODEL_NAME=qwen36-27b
TOKENIZER_PATH=/home/pjw7200/models/Qwen3.6-27B
```

The shared launcher now verifies that `/v1/models` on each port contains the expected served model id before starting mini SWE Agent. This prevents accidentally running a Qwen3.6 experiment against still-running Qwen3.5 servers.

## Smoke Test

Qwen3.6 was smoke-tested on free GPU 2 with port `8010`, without touching the active Qwen3.5 servers on GPU 0 and GPU 1.

Observed vLLM state:

```text
vLLM version: 0.22.1
resolved architecture: Qwen3_5ForConditionalGeneration
dtype: torch.bfloat16
quantization: None
max model len: 262144
model loading memory: 50.22 GiB
GPU KV cache size: 1,767,533 tokens
```

The smoke server passed `/health`, `/v1/models`, and a short `/v1/chat/completions` request, then was stopped.
