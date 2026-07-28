# SWE-chat workload distributions

The distributions use only turns that were valid in the full replay runs.
Percentiles follow the TTFT report convention: arithmetic mean, median p50,
and nearest-rank p90/p95/p99.

Prefill elapsed time is omitted because the replay records only the prefill
completion timestamp, not its start timestamp or elapsed duration.

## OpenCode

| metric | n | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| tool call duration (ms/call) | 15,661 | 897.83 | 9 | 624 | 791 | 4,491 |
| tool phase critical path (ms/tool turn) | 7,737 | 1,461.35 | 21 | 742 | 1,576 | 12,708 |
| raw tool output (tokens/call) | 15,661 | 2,743.36 | 866 | 8,202 | 13,351 | 20,064 |
| raw tool output sum (tokens/tool turn) | 7,737 | 5,553.03 | 1,198 | 14,958 | 24,511 | 52,215 |
| replay LLM input (tokens/request) | 8,490 | 30,741.08 | 24,522 | 65,652 | 84,790 | 120,583 |
| requested LLM output (tokens/request) | 8,490 | 440.87 | 184 | 1,127 | 1,886 | 3,776 |

## Codex

| metric | n | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| tool call duration (ms/call) | 7,282 | 1,516.68 | 429 | 1,555 | 5,004 | 30,156 |
| tool phase critical path (ms/tool turn) | 4,965 | 1,697.05 | 678 | 1,938 | 5,361 | 30,184 |
| raw tool output (tokens/call) | 7,282 | 1,601.88 | 492 | 3,548 | 6,171 | 16,094 |
| raw tool output sum (tokens/tool turn) | 4,965 | 2,349.42 | 621 | 5,834 | 10,745 | 27,441 |
| replay LLM input (tokens/request) | 5,206 | 54,331.76 | 50,753.5 | 103,034 | 116,347 | 127,236 |
| requested LLM output (tokens/request) | 5,206 | 464.37 | 248.5 | 1,038 | 1,562 | 3,453 |

## Definitions

- **Tool call duration**: one duration per individual tool call.
- **Tool phase critical path**: the time from the start of a tool group until
  its last result is observed. A parallel multi-tool group is one sample here.
- **Raw tool output per call**: the Mistral tokenizer count of the raw output
  emitted by one tool, excluding the observation-template wrapper.
- **Raw tool output sum per turn**: the sum of the individual output token
  counts in one assistant tool-call turn.
- **Replay LLM input**: the reconstructed prompt tokens actually sent by the
  replay, not the source trace's provider-reported prompt count.
- **Requested LLM output**: the source trace completion-token count used as the
  replay request's forced output length.

Codex tool durations use `function_call_output` event timestamps. Parallel
results that were emitted together therefore include synchronization or
batch-flush delay and are not always individual physical tool runtimes.

## Active prefill cancellation latency

| trace | n | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| OpenCode | 1,102 | 3.71 ms | 3.36 ms | 5.30 ms | 5.90 ms | 9.49 ms |
| Codex | 192 | 4.72 ms | 4.18 ms | 6.55 ms | 7.48 ms | 17.67 ms |

All cancellation requests completed without a recorded error. Amortized over
all valid turns, the recorded cancellation latency was 0.482 ms/turn for
OpenCode and 0.174 ms/turn for Codex.

These full runs predate the synchronous abort-and-drain change. The values
therefore measure the old abort RPC latency and do not guarantee that the
original prefill coroutine had fully exited when the request returned.
