# SWE-chat tool duration by command

Durations are in milliseconds and include only tool calls belonging to valid
replay turns. Percentiles use median p50 and nearest-rank p90/p95/p99.

OpenCode durations come directly from each tool's
`state.time.end - state.time.start`. Codex durations span from the call event to
the corresponding output event, so synchronization and batch-flush delay may
be included.

`exec_command` can yield while its child process is still running and return a
session ID. Later `write_stdin` calls collect the remaining output. Therefore
an `exec_command` duration is the latency of that tool response, not
necessarily the end-to-end lifetime of the spawned process. The shell
executable table below describes tool-result availability and is not a
normalized OS process-runtime comparison.

## OpenCode tool APIs

| tool | n | share | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `read` | 8,911 | 56.90% | 390.43 | 5 | 13 | 18 | 36 |
| `bash` | 3,117 | 19.90% | 1,688.79 | 586 | 2,351 | 3,435 | 19,503 |
| `grep` | 1,646 | 10.51% | 205.83 | 16 | 29 | 38 | 243 |
| `glob` | 1,270 | 8.11% | 129.85 | 17 | 28 | 34 | 67 |
| `apply_patch` | 243 | 1.55% | 689.05 | 27 | 243 | 721 | 3,036 |
| `todowrite` | 145 | 0.93% | 1.70 | 1 | 3 | 4 | 5 |
| `edit` | 98 | 0.63% | 517.73 | 18 | 201 | 3,003 | 3,005 |
| `webfetch` | 83 | 0.53% | 289.46 | 202 | 623 | 813 | 1,175 |
| `write` | 45 | 0.29% | 67.47 | 16 | 195 | 597 | 860 |
| `websearch` | 22 | 0.14% | 1,670.59 | 964.5 | 3,511 | 7,428 | 11,016 |
| `task` | 21 | 0.13% | 128,173.67 | 89,801 | 252,244 | 255,261 | 274,943 |
| `skill` | 20 | 0.13% | 16.90 | 17.5 | 28 | 28 | 34 |

These rows cover 99.74% of the 15,661 OpenCode tool calls.

## OpenCode commands inside `bash`

| first executable | n | share of `bash` | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `git` | 2,065 | 66.25% | 509.90 | 595 | 806 | 895 | 1,698 |
| `pnpm` | 231 | 7.41% | 3,997.23 | 2,722 | 4,265 | 6,325 | 43,605 |
| `npm` | 185 | 5.94% | 2,550.34 | 271 | 4,982 | 6,518 | 32,566 |
| `openspec` | 67 | 2.15% | 346.76 | 341 | 406 | 485 | 525 |
| `ls` | 65 | 2.09% | 29.51 | 11 | 27 | 48 | 362 |
| `gh` | 47 | 1.51% | 903.98 | 397 | 2,214 | 2,564 | 2,862 |
| `go` | 46 | 1.48% | 5,749.74 | 2,716.5 | 7,129 | 8,260 | 120,121 |
| `node` | 43 | 1.38% | 541.23 | 233 | 980 | 1,332 | 2,492 |
| `mkdir` | 40 | 1.28% | 141.50 | 7 | 17 | 35 | 5,261 |
| `python3` | 34 | 1.09% | 4,589.21 | 948.5 | 13,266 | 13,381 | 16,257 |
| `cat` | 33 | 1.06% | 1,345.30 | 11 | 2,699 | 5,744 | 25,042 |
| `docker` | 29 | 0.93% | 18,890.45 | 1,542 | 31,096 | 52,151 | 258,214 |
| `wc` | 19 | 0.61% | 480.95 | 549 | 683 | 701 | 742 |
| `python` | 18 | 0.58% | 632.00 | 99.5 | 1,356 | 1,837 | 5,003 |
| `sed` | 18 | 0.58% | 1.44 | 0 | 1 | 1 | 22 |

These rows cover 94.32% of the 3,117 OpenCode `bash` calls.

## Codex tool APIs

| tool | n | share | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exec_command` | 3,034 | 41.66% | 1,054.32 | 797 | 1,173 | 1,478 | 10,643 |
| `shell_command` | 2,471 | 33.93% | 1,890.20 | 201 | 2,502 | 7,947 | 45,218 |
| `apply_patch` | 349 | 4.79% | 164.23 | 88 | 321 | 388 | 1,273 |
| `exec` | 264 | 3.63% | 1,290.91 | 360.5 | 3,784 | 10,004 | 10,005 |
| `write_stdin` | 234 | 3.21% | 2,998.03 | 83.5 | 6,892 | 9,976 | 30,014 |
| `update_plan` | 223 | 3.06% | 88.05 | 21 | 157 | 232 | 1,979 |
| `wait` | 137 | 1.88% | 2,173.34 | 317 | 5,215 | 10,357 | 20,002 |
| `mcp__sequential_thinking__sequentialthinking` | 129 | 1.77% | 428.78 | 21 | 200 | 2,155 | 10,234 |
| `mcp__filesystem__read_text_file` | 128 | 1.76% | 176.97 | 153 | 268 | 315 | 2,466 |
| `mcp__zls__diagnostics` | 55 | 0.76% | 7,529.11 | 6,003 | 13,118 | 15,938 | 20,877 |
| `mcp__filesystem__read_multiple_files` | 31 | 0.43% | 1,282.90 | 164 | 319 | 433 | 19,862 |
| `mcp__omx_code_intel__lsp_diagnostics` | 30 | 0.41% | 600.50 | 477.5 | 1,184 | 1,378 | 1,609 |

These rows cover 97.30% of the 7,282 Codex tool calls.

## Codex commands inside `exec_command` and `shell_command`

| first executable | n | share of shell calls | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sed` | 1,541 | 27.99% | 567.79 | 706 | 932 | 1,017 | 1,192 |
| `git` | 1,200 | 21.80% | 948.88 | 731.5 | 1,043 | 1,511 | 9,670 |
| `rg` | 750 | 13.62% | 785.38 | 704 | 1,101 | 1,290 | 5,589 |
| `nl` | 382 | 6.94% | 500.22 | 228 | 806 | 876 | 1,014 |
| `Get-Content` | 210 | 3.81% | 689.26 | 475.5 | 1,299 | 1,478 | 1,621 |
| `pnpm` | 177 | 3.22% | 6,482.62 | 1,458 | 15,422 | 20,139 | 30,198 |
| `zig` | 165 | 3.00% | 11,436.69 | 1,674 | 45,218 | 48,567 | 51,113 |
| `printf` | 140 | 2.54% | 930.22 | 179 | 1,173 | 2,860 | 14,303 |
| `grep` | 121 | 2.20% | 813.79 | 797 | 1,112 | 1,168 | 1,180 |
| `omx` | 120 | 2.18% | 4,054.67 | 2,336.5 | 6,958 | 20,008 | 45,005 |
| `cat` | 111 | 2.02% | 784.98 | 747 | 1,305 | 2,306 | 3,573 |
| `Select-String` | 78 | 1.42% | 574.99 | 402 | 1,102 | 1,258 | 1,735 |
| shell script | 64 | 1.16% | 491.56 | 398 | 907 | 944 | 1,303 |
| `find` | 63 | 1.14% | 1,016.49 | 202 | 1,026 | 8,637 | 9,430 |
| `python3` | 61 | 1.11% | 1,272.84 | 244 | 2,656 | 2,781 | 9,308 |

These rows cover 94.15% of the 5,505 Codex shell calls.

## Interpretation

- 8,911 OpenCode calls are direct `read` operations with a 5 ms median.
- OpenCode's direct `grep` and `glob` tools have 16 ms and 17 ms medians.
- Only 3,117 OpenCode calls are `bash`; their median is 586 ms.
- 56.13% of all OpenCode calls finish within 10 ms and 83.53% within
  50 ms. The corresponding Codex shares are 1.43% and 9.87%.
- OpenCode timing has millisecond resolution, so sub-millisecond calls can
  appear as 0 ms.
- OpenCode means are distorted by a few extreme lifecycle outliers. For
  example, `read` has a 5 ms median and 36 ms p99 but a 954,779 ms maximum.
  Median and upper percentiles are more representative than the mean.
- The trace systems are not timing-equivalent: OpenCode measures its direct
  tool implementation, while Codex measures call-to-output event latency.
