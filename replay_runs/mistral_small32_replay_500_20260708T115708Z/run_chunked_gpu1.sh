#!/usr/bin/env sh
set -eu

cd /home/pjw7200/chunked_tool_prefill/agent

PYTHONPATH=src /home/pjw7200/chunked_tool_prefill/.conda/miniswe-py311/bin/python -m minisweagent.run.replay \
  /home/pjw7200/chunked_tool_prefill/traces/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z \
  -o /home/pjw7200/chunked_tool_prefill/replay_runs/mistral_small32_replay_500_20260708T115708Z/chunked_gpu1 \
  --algorithm chunked \
  --limit 500 \
  -c swebench_replay_output_first \
  -c /home/pjw7200/chunked_tool_prefill/replay_runs/mistral_small32_replay_500_20260708T115708Z/mistral_gpu1.yaml
