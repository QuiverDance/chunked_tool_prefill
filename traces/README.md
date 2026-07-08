# Traces

This directory stores curated trace artifacts that should be tracked in git.

Local execution outputs remain ignored elsewhere:

- `runs/`: local run outputs and scratch runs
- `reports/`: generated summaries and CSV reports
- `logs/`: detached launcher logs and pid files

Keep only canonical traces here. Reports should be regenerated from these traces
with `scripts/summarize_token_timing.py` when needed.

## Current Traces

- `swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z`
  - SWE-bench Verified test split, 500 trajectories
  - model: `hosted_vllm/qwen36-27b`
  - source run: `runs/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z`
  - evaluation summary copied under `evaluation/`
