# TraceLab Codex token-native replay

This experiment compares baseline and incremental tool-result prefill with the
public TraceLab Codex metadata. The public release removes prompt text, tool
arguments, and tool-result text, so this is a synthetic-token serving workload,
not a semantic conversation replay.

## Trace format and loader gap

The release file is gzip-compressed JSONL with one model round per row. Codex
rows are contiguous by `session_id` and ordered by `round_index`. A usable pair
joins a multi-tool round to its immediately following round:

- the current row's `tools` contain call IDs, names, emitted/result timestamps,
  and result character counts;
- the following row's `timing_events` contain the serialized tool-result order;
- `prefix_tokens`, `newly_append_tokens`, `input_tokens_total`, and
  `output_tokens` provide aggregate token counts;
- prompt text, tool arguments, result text, and token IDs are absent.

The existing incremental replay loader consumes saved mini/SWE-chat
trajectories with actual messages and tool output. It can tokenize those
messages, construct partial prompts, and reuse the shared asynchronous prefill
worker and HTTP backend. It cannot infer those message payloads from TraceLab's
count-only rows, nor can it directly seed TraceLab's reported cached-prefix
length.

The TraceLab integration therefore adds a narrow prepared-workload interface
instead of weakening the trajectory loader:

1. `tracelab_workload.py` validates adjacent rows, derives the count-only prompt
   layout, builds cumulative completion checkpoints, samples deterministically,
   and writes a fixed manifest.
2. `tracelab_replay.py` materializes deterministic valid token IDs from that
   manifest and reuses the existing prefill worker, backend, cancellation, and
   metrics helpers.
3. `run_tracelab_codex_replay.sh` performs health/reset preflight and a
   two-phase GPU crossover with a per-trial measurement barrier.
4. the Mistral vLLM launch configuration enables prompt-token details so the
   final request supplies authoritative cache-hit readback.

The release scan found 357,161 rows, 216,823 Codex rounds, 215,234 adjacent
Codex pairs, and 19,585 eligible pairs. The dominant exclusions were
non-multi-tool rounds (167,815), a next-round result-ID mismatch (14,536),
context capacity (12,469), a negative inferred result-token budget (443), and
concurrent user input (373).

## Measurement contract

One trial represents a multi-tool Codex round and the immediately following LLM
round. The prepared manifest fixes:

- the final prompt and completion token counts;
- the reported cached-prefix length;
- a static suffix available before tool results;
- result-token allocations and serialization order;
- tool completion offsets and cumulative prefill checkpoints.

Both algorithms receive the same final token IDs, completion count, cache salt
within a trial, trial order, and prepared manifest.

The timed E2E region starts only after:

1. the compressed TraceLab source has been scanned and hashed;
2. eligibility filtering and deterministic sampling have completed;
3. all token IDs and checkpoint slices for the trial have been materialized;
4. the server prefix cache has been reset;
5. the reported common prefix has been seeded and acknowledged.

E2E then contains the scaled tool phase, incremental request scheduling and
cancellation when enabled, the final completion request, TTFT, and decode.
`replay_ttft_s` starts immediately before the final HTTP completion request.
Cache reset and seed durations are retained as `setup_*` diagnostics but are
excluded from E2E.

Warm-up trials are discarded. Trace preparation, warm-up, health checks, result
serialization, summary generation, and server-version readback are outside
every timed region.

The paired GPU processes load the fixed manifest and finish warm-up before
either begins measurement. To bound memory for full-trace runs, they materialize
one trial at a time after the preceding measurement barrier and before cache
setup. A filesystem barrier then keeps every trial in lockstep:

1. both runners reset and seed their own GPU;
2. both signal that setup is complete;
3. both start the timed region;
4. both signal that measurement is complete before either starts the next
   trial's setup.

The barrier's file reads and writes occur before or after the E2E timestamps.
Each completed record is also flushed to a partial JSONL checkpoint after the
post-measurement barrier. The next measurement cannot begin until both runners
have subsequently materialized and seeded their next trial. This prevents one
runner's token construction, cache setup, checkpoint write, or final result
serialization from overlapping the peer's measured region.

If either final response fails completion-count or cache-readback validation,
both runners discard that attempt and repeat the same trial from cache reset.
Retries use new cache salts and paired attempt-specific barrier markers. Three
failed paired attempts remain fatal rather than admitting an unauthoritative
measurement.

## Authority checks

A trial is valid only when:

- the next round consumes exactly the current round's tool-call IDs;
- all result timestamps and character counts are present and consistent;
- the next round has no concurrent user input;
- prefix, append, prompt, and completion counts are internally consistent;
- the final prompt plus completion fits the configured context limit;
- the live server produces exactly the requested completion-token count;
- the live final response reports at least the prepared cached-prefix tokens.

The vLLM servers must use `--enable-prefix-caching` and
`--enable-prompt-tokens-details`. The latter provides the per-request
`prompt_tokens_details.cached_tokens` readback used by the final two checks.

`final_cached_prompt_suffix_tokens` and `final_cached_tool_result_tokens` come
from this authoritative final-response readback. Worker completion counters are
also recorded, but an aborted prefill can leave reusable cache blocks even when
the prefill HTTP request did not finish before tool end.

vLLM omits `prompt_tokens_details` when a request has zero cache hits. A missing
readback is therefore interpreted as zero only when the prepared prefix is zero
and the runner issued neither a seed nor an incremental prefill request.
`cached_token_readback_inferred_zero=true` records that case. Missing readback
remains an error whenever any cache reuse is expected or possible.

## Synthetic token policy

The next prompt uses deterministic valid vocabulary IDs derived from the trial
ID. Its length is exactly TraceLab's `input_tokens_total`.

The prepared static suffix is:

```text
max(0, current_input_tokens + current_output_tokens - next_prefix_tokens)
```

The remaining `newly_append_tokens` are assigned to tool results
proportionally to `result_chars`, with at least one token per result. This
assumption is stored in the manifest. Result chunks use the next round's
`timing_events` order, while availability uses each tool's `result_at`.
Incremental prefill advances only through the longest available prefix of that
serialized order.

## Reproducible execution

Prepare once:

```bash
PYTHONPATH=agent/src \
  /data/pjw7200/src/mini-swe-agent/.venv/bin/python \
  -m minisweagent.run.extra.tracelab_replay prepare \
  /home/pjw7200/traces/tracelab/raw/release-v0.0.1/syfi_coding_trace.jsonl.gz \
  --output replay_runs/tracelab_codex_v001_prepared_100.json \
  --limit 100 \
  --sample-seed tracelab-codex-v1 \
  --max-context-tokens 131072 \
  --cache-block-tokens 16
```

Use `--all` instead of `--limit 100` to prepare every eligible pair. The
crossover script accepts the equivalent `LIMIT=all`.

An interrupted run can append after its validated partial checkpoints:

```bash
RESUME=1 RESUME_LABEL=resume_6097 CROSSOVER=0 \
  OUTPUT_DIR="$PWD/replay_runs/tracelab_codex_full_20260730_r2" \
  MANIFEST="$PWD/replay_runs/tracelab_codex_v001_prepared_all_v2.json" \
  scripts/run_tracelab_codex_replay.sh
```

Run the fixed manifest concurrently and swap GPU roles:

```bash
MANIFEST="$PWD/replay_runs/tracelab_codex_v001_prepared_100.json" \
  scripts/run_tracelab_codex_replay.sh
```

The script runs:

- phase A: incremental on GPU 0, baseline on GPU 1;
- phase B: baseline on GPU 0, incremental on GPU 1.

`TIME_SCALE=1.0` is required for trace-timing results. Any other value is
recorded as `measurement_valid_for_trace_timing=false`.

Each summary records the TraceLab source SHA-256, prepared-manifest SHA-256,
runner and workload-source SHA-256 values, endpoint, model, and live vLLM
version. The current TraceLab source digest is
`9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`.

## 100-pair pilot

The measurement-safe crossover pilot is stored under
`replay_runs/tracelab_codex_pilot100_20260730`. It used vLLM 0.19.1,
`TIME_SCALE=1`, one discarded warm-up, and manifest SHA-256
`6efb84e62993c9e16b265b6e2c3f094c2abbcbd69db91749278bba7c5eebb78f`.

All four runs produced 100 valid and zero invalid trials. Both phases had 200
paired setup markers, 200 paired measurement markers, and no failure marker.
Every final response returned exactly the requested completion count and an
authoritative cached-token count within the prepared prefix/prompt bounds.

| Metric | Baseline, 200 observations | Incremental, 200 observations | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 0.3768 s | 0.3032 s | -19.55% |
| Median TTFT | 0.2259 s | 0.1546 s | -31.56% |
| p95 TTFT | 1.0885 s | 1.0062 s | -7.56% |
| Mean E2E | 5.5311 s | 5.4999 s | -0.57% |
| Median E2E | 3.0955 s | 3.0795 s | -0.52% |
| p95 E2E | 16.6921 s | 16.6719 s | -0.12% |

Incremental TTFT was lower in 185/200 paired observations; E2E was lower in
118/200. The final server readback attributed an average of 1,764.8 prompt
suffix tokens, including 980.4 inferred tool-result tokens, to cache hits in
incremental runs; both baseline runs reported zero suffix cache hits.

Cache reset and common-prefix seeding averaged about 3.72 seconds per trial and
are recorded only as `setup_total_s_excluded_from_e2e`. They are not included
in either TTFT or E2E. These results characterize serving behavior for the
count- and timing-preserving synthetic token workload; they do not claim
semantic equivalence to the redacted Codex prompts.
