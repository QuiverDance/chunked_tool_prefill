# BranchFill Prefix Opportunity

- Trajectories: 500
- Tool calls: 26435

## Findings

- The causal any-prior oracle reuses 9.90% of model-visible payload tokens (trajectory-bootstrap 95% CI 9.38%–10.40%).
- Restricting candidates to the same command category retains 65.45% of those oracle-reusable tokens.
- Exact-argument history exists for 1755 calls (6.64% of all calls). Within those eligible calls it reuses 43.22% of payload tokens, or 2.76% globally.
- Among exact-argument-eligible calls, 1020 (58.12%) reach at least 32 exact prefix tokens.
- The best four-branch policy is `command_similarity` at 8.48% payload reuse, capturing 85.70% of the any-prior oracle. The combined ranker reaches 8.43%.
- The largest combined-k4 category contribution is `sed` at 21.21% of reusable tokens.

## Oracle reuse

| View | Candidate pool | Output tokens | Reusable tokens | Reuse ratio |
|---|---:|---:|---:|---:|
| model_visible | any_prior | 8051727 | 797173 | 9.90% |
| model_visible | recorded_category | 8051727 | 604348 | 7.51% |
| model_visible | same_category | 8051727 | 521711 | 6.48% |
| model_visible | same_signature | 8051727 | 465906 | 5.79% |
| model_visible | exact_args | 8051727 | 222239 | 2.76% |
| raw | any_prior | 10073811 | 846051 | 8.40% |
| raw | recorded_category | 10073811 | 617636 | 6.13% |
| raw | same_category | 10073811 | 535645 | 5.32% |
| raw | same_signature | 10073811 | 479038 | 4.76% |
| raw | exact_args | 10073811 | 233196 | 2.31% |

## Model-visible LCP coverage

| Candidate pool | Eligible calls | Positive LCP calls | Median LCP | P90 LCP | P99 LCP |
|---|---:|---:|---:|---:|---:|
| any_prior | 25635 | 12767 | 0 | 67 | 521.0 |
| recorded_category | 20158 | 9261 | 0 | 47 | 444.7 |
| same_category | 18238 | 8953 | 0 | 35 | 404 |
| same_signature | 14169 | 7562 | 0 | 22 | 382 |
| exact_args | 1755 | 1373 | 0 | 0 | 220 |

## Per-trajectory reuse ratio

| Candidate pool | P25 | Median | P75 | P90 |
|---|---:|---:|---:|---:|
| any_prior | 5.25% | 8.46% | 12.17% | 17.14% |
| recorded_category | 3.60% | 5.86% | 9.36% | 13.70% |
| same_category | 2.40% | 4.56% | 8.40% | 12.22% |
| same_signature | 1.91% | 3.94% | 7.47% | 11.68% |
| exact_args | 0.21% | 1.27% | 3.41% | 6.58% |

## Causal policy frontier

| Policy | k | Reuse ratio | 95% CI | Trajectory median (P25–P75) | Any-prior capture | Same-signature capture | Eligible calls | LCP ≥32 calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_args_recent | 1 | 2.67% | 2.37%–2.98% | 1.18% (0.21%–3.35%) | 26.97% | 46.15% | 1758 | 995 |
| exact_args_recent | 2 | 2.75% | 2.44%–3.05% | 1.25% (0.21%–3.41%) | 27.73% | 47.45% | 1758 | 1017 |
| exact_args_recent | 4 | 2.76% | 2.46%–3.07% | 1.27% (0.21%–3.41%) | 27.88% | 47.70% | 1758 | 1020 |
| exact_args_recent | 8 | 2.76% | 2.46%–3.07% | 1.27% (0.21%–3.41%) | 27.88% | 47.70% | 1758 | 1020 |
| signature_recent | 1 | 3.16% | 2.89%–3.43% | 2.19% (1.01%–4.09%) | 31.91% | 54.60% | 14425 | 1618 |
| signature_recent | 2 | 4.08% | 3.77%–4.42% | 2.91% (1.34%–5.49%) | 41.25% | 70.58% | 14425 | 1923 |
| signature_recent | 4 | 4.97% | 4.57%–5.36% | 3.41% (1.66%–6.58%) | 50.19% | 85.88% | 14425 | 2161 |
| signature_recent | 8 | 5.50% | 5.10%–5.90% | 3.80% (1.86%–7.15%) | 55.56% | 95.07% | 14425 | 2304 |
| resource_aware_recent | 1 | 4.24% | 3.90%–4.59% | 2.77% (1.29%–5.36%) | 42.80% | 73.23% | 14425 | 1833 |
| resource_aware_recent | 2 | 4.92% | 4.54%–5.31% | 3.36% (1.57%–6.48%) | 49.74% | 85.10% | 14425 | 2090 |
| resource_aware_recent | 4 | 5.43% | 5.02%–5.83% | 3.74% (1.77%–7.07%) | 54.81% | 93.79% | 14425 | 2270 |
| resource_aware_recent | 8 | 5.71% | 5.30%–6.10% | 3.92% (1.89%–7.36%) | 57.68% | 98.69% | 14425 | 2359 |
| command_similarity | 1 | 6.66% | 6.25%–7.08% | 5.22% (3.13%–8.45%) | 67.25% | 76.49% | 25935 | 2829 |
| command_similarity | 2 | 7.72% | 7.27%–8.17% | 6.24% (3.83%–9.68%) | 77.92% | 87.64% | 25935 | 3310 |
| command_similarity | 4 | 8.48% | 7.98%–8.95% | 6.97% (4.30%–10.78%) | 85.70% | 94.95% | 25935 | 3596 |
| command_similarity | 8 | 9.11% | 8.61%–9.59% | 7.76% (4.83%–11.30%) | 91.99% | 98.58% | 25935 | 3800 |
| combined | 1 | 6.54% | 6.10%–6.98% | 5.06% (2.98%–8.41%) | 66.05% | 79.19% | 25935 | 2748 |
| combined | 2 | 7.54% | 7.07%–8.00% | 6.16% (3.70%–9.49%) | 76.15% | 88.67% | 25935 | 3222 |
| combined | 4 | 8.43% | 7.94%–8.89% | 6.93% (4.22%–10.44%) | 85.12% | 95.52% | 25935 | 3570 |
| combined | 8 | 9.09% | 8.61%–9.57% | 7.61% (4.86%–11.31%) | 91.83% | 98.80% | 25935 | 3793 |

## Output-length breakdown

| Output tokens | Calls | Output tokens | Any-prior reusable tokens | Reuse ratio |
|---|---:|---:|---:|---:|
| 0 | 2191 | 0 | 0 | n/a |
| 1-31 | 3725 | 52387 | 6420 | 12.25% |
| 32-127 | 6238 | 461560 | 72111 | 15.62% |
| 128-511 | 9721 | 2608540 | 408786 | 15.67% |
| 512-2047 | 4025 | 3526574 | 287311 | 8.15% |
| 2048+ | 535 | 1402666 | 22545 | 1.61% |

## Rendering

- Full outputs: 26033 calls / 6983072 payload tokens
- Truncated outputs: 402 calls / 1068655 visible head-and-tail payload tokens
- Truncated-output reuse is conservatively capped at the visible 5,000-character head.

## Largest effective command categories

| Category | Calls | Output tokens | Any-prior oracle | Same-signature oracle | Combined k1 | Combined k4 |
|---|---:|---:|---:|---:|---:|---:|
| `sed` | 4821 | 1629667 | 10.08% | 9.75% | 5.85% | 8.83% |
| `cat` | 1787 | 1327895 | 10.36% | 5.73% | 5.50% | 6.31% |
| `python` | 3662 | 762671 | 16.74% | 11.04% | 10.90% | 15.15% |
| `python|tail` | 1147 | 518197 | 3.52% | 1.65% | 2.21% | 3.07% |
| `compound` | 2248 | 423408 | 7.08% | 4.85% | 4.55% | 5.97% |
| `python|head` | 921 | 377631 | 8.23% | 2.86% | 5.22% | 7.92% |
| `ls` | 452 | 372260 | 0.21% | 0.16% | 0.17% | 0.20% |
| `git` | 1258 | 340908 | 13.78% | 9.88% | 11.87% | 13.16% |
| `grep|head` | 1521 | 327349 | 2.29% | 0.33% | 1.43% | 1.85% |
| `cat|head` | 202 | 215630 | 7.10% | 1.16% | 5.08% | 6.20% |
| `grep` | 2070 | 209915 | 4.88% | 3.01% | 3.39% | 4.18% |
| `cat|sed` | 294 | 151224 | 11.59% | 9.87% | 7.37% | 10.63% |
| `find|head` | 452 | 98812 | 1.25% | 0.59% | 1.12% | 1.23% |
| `git&&cat` | 273 | 98736 | 88.06% | 1.13% | 81.14% | 83.62% |
| `python3` | 531 | 80925 | 11.09% | 7.78% | 5.91% | 9.07% |
| `head` | 177 | 77404 | 6.03% | 0.28% | 3.33% | 3.80% |
| `git|head` | 146 | 74914 | 9.33% | 0.26% | 5.86% | 9.06% |
| `pip&&cd&&python|head` | 67 | 72896 | 3.82% | 0.50% | 0.89% | 3.63% |
| `python|grep` | 210 | 62494 | 4.26% | 3.04% | 3.50% | 4.03% |
| `pip&&python` | 158 | 58001 | 5.07% | 2.98% | 3.88% | 5.02% |

## Method

For each tool call, candidates are restricted to completed earlier calls in the same trajectory. The oracle chooses the candidate with the longest exact target-tokenizer prefix. No text normalization or future/cross-trajectory output is used. Candidate pools are all prior calls, the effective command category after leading setup commands, the normalized command signature, and exact tool arguments. Calls without a candidate remain in the denominator.

Policy ranking uses only the current command and causal history. `exact_args_recent` and `signature_recent` use recency; `resource_aware_recent` prioritizes shared resource keys within a signature; `command_similarity` ranks command-token Jaccard similarity; and `combined` ranks exact arguments, signature, resource overlap, category, similarity, then recency. Identical historical outputs are deduplicated before selecting k branches. These deterministic policies have no trained or dataset-tuned weights.

Oracle capture is computed per call as the policy LCP capped by that call's oracle LCP, then summed over the oracle tokens. It therefore measures how much of the named oracle opportunity the policy covers and cannot exceed 100%.

The model-visible metric requires the candidate and current output to use the same full/truncated rendering form. This preserves the exact formatter boundary and token positions needed for KV reuse. The raw metric compares every causal candidate regardless of rendering form.

`per_call.jsonl.gz` contains every comparison result. `policy_per_call.jsonl.gz` and `policy_frontier.json` isolate the causal policy results. `top_matches.json` contains the 50 largest model-visible any-prior matches for manual inspection.