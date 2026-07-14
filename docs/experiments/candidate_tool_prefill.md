# Candidate Tool Prefill replay

Candidate Tool Prefill uses tool execution time to prefill a small set of historical tool-output candidates. A candidate is never accepted as the tool result. Once real output becomes visible, only an exact token prefix can contribute reusable KV; everything after the first mismatch is discarded or recomputed.

This implementation adds the algorithm to trace replay so that its serving cost and latency opportunity can be measured before changing the live agent.

## Replay flow

For each tool call that has a later assistant turn, replay does the following:

1. Search tool calls already completed in the same trajectory.
2. Rank their commands by token-set Jaccard similarity, using recency as the tie-breaker.
3. Keep up to `top_k` distinct raw outputs.
4. Render each candidate with the output-first streaming observation format. Return code and exception fields receive blank placeholders rather than historical values, so they are not predicted even if a custom streaming template references them.
5. Tokenize the candidate prompts, remove duplicate token sequences, and skip prompts beyond the configured context limit.
6. Build a prefix tree and submit every branch in depth-first subtree order with the same cache salt. vLLM's normal automatic prefix cache shares the history trunk and already materialized candidate trunks.
7. As actual output becomes visible, remove every branch whose raw text no longer starts with that exact text.
8. If there is no eligible candidate, or every branch misses, continue with actual-output chunked prefill. These streaming prompts also use blank return-code and exception placeholders until the tool completes. A completed candidate's verified prefix remains the starting point, so fallback does not recalculate matching blocks.
9. At the tool deadline, snapshot and cancel outstanding work before final-prompt tokenization. Only requests completed no later than that deadline are considered.
10. Compare those completed prefill requests with the final prompt token by token. Only the block-aligned longest common prefix is reported as reusable.

The current replay MVP speculates only for the first action in a multi-action assistant turn. Final verification still uses the complete, actual observation sequence, so this limitation can reduce opportunity but cannot admit incorrect KV.

## Prefix-tree scheduling

`CandidatePrefillPlan` receives the cached prompt and the tokenized top-k candidates. It records the block-aligned prefix already materialized for each branch, then orders branches so that a high-ranked candidate's shared subtree is finished before moving to a different subtree.

All requests remain full prompt prefixes. The replay does not splice KV or modify the inference engine. Shared work is recovered by using one cache salt, one sequential worker, and vLLM automatic prefix caching.

## Configuration and command

The SWE-bench replay formatter puts output first and limits streaming output to 5,000 characters:

```yaml
replay:
  stream_output_char_limit: 5000
  candidate_prefill:
    top_k: 4
```

Run a Candidate Tool Prefill replay with:

```bash
ALGORITHM=candidate CANDIDATE_TOP_K=4 scripts/run_verified_replay_mistral_small32.sh
```

The script accepts the same trace, model, endpoint, context, and cache-block environment variables as the baseline and chunked replay modes.

## Metrics

- `candidate_selected_count`: distinct, in-context token branches in the final plan.
- `candidate_skipped_capacity_count`: retrieved candidates skipped because their prompt exceeds the context limit.
- `candidate_submitted_count` and `candidate_completed_count`: candidate requests sent to and completed by the prefill worker.
- `candidate_shared_prefix_tokens`: block-aligned speculative suffix shared by all selected branches beyond the post-assistant history seed. With top-1, this is that candidate's full speculative suffix.
- `candidate_verified_prefix_tokens`: longest block-aligned full-prompt prefix shared by the final actual prompt and any completed candidate.
- `candidate_verified_tool_output_tokens`: verified tokens beyond the post-assistant history seed, including the streaming observation wrapper.
- `candidate_pruned_count` and `candidate_surviving_count`: branches rejected, and branches still viable at the last visible-output checkpoint.
- `candidate_fallback_to_chunked`: whether every candidate missed and actual-output chunking took over.
- `candidate_cancelled_count`: candidate requests removed or aborted after verification.

The general prefill metrics still describe the final usable work. In particular, `prefill_completed_prompt_tokens` is derived from exact comparison with the actual final prompt, not from candidate length.

## Deliberate MVP limits

- Candidates come only from earlier calls in the same trajectory.
- Candidate search is a linear scan over history and retokenizes command strings.
- Candidate work is sequential, but replay does not yet model a production low-priority, preemptible GPU scheduler.
- This is replay-only; the live `TokenTimingProgressAgent` is unchanged.
- The implementation measures serving behavior but does not claim a performance gain until candidate replay is run against the target vLLM deployment.

The next design task is an efficient runtime lookup for similar historical commands: decide the key representation, index structure, update policy, and environment/repository-snapshot boundaries instead of keeping the current `O(history)` scan.
