# Session-boundary-corrected TTFT

The original replay outputs remain unchanged. This report replaces only the two
Codex session-first measurements that followed the special boundary below:

- the previous turn cancelled an active incremental prefill;
- all remaining turns in that session were skipped for context capacity;
- replay then moved directly to the next session.

OpenCode contained no such boundary. Codex contained two. Each affected Codex
request was rerun five times after resetting the prefix cache on both servers,
using the original GPU assignment (incremental on GPU 0, baseline on GPU 1).
The median of the five isolated measurements replaces the original value.

## TTFT (ms) — SWE-chat OpenCode

| metric | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| baseline | 195.18 | 116.47 | 441.57 | 641.54 | 1101.48 |
| incremental tool prefill | 186.20 (-4.60%) | 111.85 (-3.96%) | 414.63 (-6.10%) | 598.30 (-6.74%) | 1063.86 (-3.42%) |

OpenCode: 8,490 paired valid turns; no measurements replaced.

## TTFT (ms) — SWE-chat Codex

| metric | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| baseline | 208.98 | 133.10 | 351.15 | 500.19 | 1262.46 |
| incremental tool prefill | 203.81 (-2.48%) | 128.82 (-3.22%) | 337.40 (-3.92%) | 482.91 (-3.45%) | 1250.42 (-0.95%) |

Codex: 5,206 paired valid turns; two session-first measurements replaced for
each algorithm.

## Corrected cases

| next session | prompt tokens | algorithm | original TTFT | isolated median | isolated range |
|---|---:|---|---:|---:|---:|
| `019d6314-b36c-7102-9028-f996571c253e` | 21,080 | baseline | 879.82 ms | 871.60 ms | 865.84–877.57 ms |
| `019d6314-b36c-7102-9028-f996571c253e` | 21,080 | incremental | 1265.13 ms | 877.94 ms | 871.55–882.44 ms |
| `019d69f9-9203-7891-8104-44787e8db422` | 9,882 | baseline | 388.33 ms | 373.79 ms | 372.73–377.44 ms |
| `019d69f9-9203-7891-8104-44787e8db422` | 9,882 | incremental | 391.20 ms | 383.30 ms | 378.80–384.14 ms |

Percentiles use the original report convention: arithmetic mean, median p50,
and nearest-rank p90/p95/p99.
