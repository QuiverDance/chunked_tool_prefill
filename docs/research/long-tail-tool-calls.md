# Long-tail tool calls와 agent E2E latency

조사일: 2026-07-29

## 결론

Long tail에는 분명한 특징이 있다. 다만 하나의 workload class가 아니라 다음이 섞인 혼합물이다.

- package install, build, test처럼 실제 계산과 I/O가 오래 걸리는 명령
- timeout에 걸린 실패, poll, 명시적인 `sleep`
- subagent와 orchestration 대기
- 사용자 입력과 승인 대기
- tool runner 바깥의 scheduling, startup, output propagation
- session 중단·재개 또는 기록 방식 때문에 실행시간처럼 보이는 timestamp gap

따라서 **p99 이상인 호출을 하나의 정책으로 최적화하면 안 된다.** 도구 이름만 보는 것도 부족하다. 명령의 의미, timeout, 최근 실행시간, output 도착 모양, return code, 사람·subagent 의존성을 함께 보는 semantic/runtime classifier가 필요하다.

직접 tool runtime을 줄이는 최적화는 E2E에 바로 반영될 수 있다. 반면 Candidate/Chunked Tool Prefill은 tool 시간을 줄이지 않고 그 안에 다음 LLM prefill 일부를 숨긴다. 둘의 기회 크기는 같지 않다.

- SWE-bench Qwen trace에서는 tool 전체가 E2E의 8.98%다. 상위 1% tool을 완전히 없애도 E2E ceiling은 2.67%, 2배 가속이면 1.33%다.
- AnalysisBench GLM trace에서는 tool이 E2E의 84.98%다. 그러나 p99가 60초 timeout에 쌓여 있다. 상위 1% 2배 가속 ceiling은 3.66%, 상위 5%는 18.09%다.
- SWE-bench의 상위 1%는 output이 늦게 몰리고 과거 output과의 exact-prefix 재사용률도 낮다. **tail-only Candidate Tool Prefill의 hideable next-TTFT 합은 E2E의 0.035%**에 불과했다. 같은 계산을 모든 tool call에 적용하면 2.79%였다.
- SWE-chat incremental replay에서도 tail turn 하나가 실제로 prefill 가능한 경우에는 다음 TTFT가 수십 ms 줄었지만, 그 조건을 만족하는 tail turn 자체가 드물었고 full-run E2E 감소는 관측되지 않았다.

현재 증거가 지지하는 우선순위는 다음과 같다.

1. timeout·poll·install/build/test의 실제 critical path를 먼저 줄인다.
2. duration만이 아니라 semantics와 runtime signal로 tail을 예측한다.
3. Chunked Tool Prefill은 tail-only 정책이 아니라, **충분한 overlap window와 일찍 확정되는 재사용 가능 token을 함께 가진 call**에 적용한다.
4. single long call에는 actual-output chunking만으로 부족하므로 output streaming 개선 또는 Candidate Tool Prefill이 필요하다. 다만 현재 command-similarity candidate는 tail에서 특히 약하다.

## 질문을 둘로 나눠야 한다

한 tool step의 단순한 critical path는 다음처럼 쓸 수 있다.

```text
현재 LLM decode → tool critical path → 다음 prompt prefill → 다음 LLM decode
```

### 직접 tool 최적화

tool 시간을 `ΔT`만큼 줄이면, 그 call이 실제 critical path에 있는 한 E2E도 최대 `ΔT`만큼 줄어든다. 병렬 tool group에서는 개별 duration의 합이 아니라 마지막 result가 도착하는 시각이 기준이다.

### Tool-time prefill

prefill overlap의 이득은 tool duration 자체가 아니라 다음 식으로 제한된다.

```text
benefit ≤ min(사용 가능한 tool slack, 미리 계산 가능한 다음 prompt prefill)
```

vLLM의 Automatic Prefix Caching도 prefill만 줄이며 decode token 생성시간은 줄이지 않는다. 따라서 long tool call이 60초여도 다음 prompt의 실제 uncached prefill이 50ms라면 prefill 계열 최적화의 이득은 50ms를 넘지 못한다. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

직접 tool 가속과 prefill overlap의 이득도 단순 합산되지 않는다. tool을 빠르게 만들수록 prefill을 숨길 window가 짧아진다.

## 측정 사실 1: 제어된 local benchmark traces

아래 값은 외부 timestamp가 아니라 runner가 `perf_counter`로 계측한 local benchmark trace에서 계산했다. Tool timing에는 duration, time-to-first-output, return code, raw output, cumulative output events가 있다. 계측 경로와 schema는 [token timing instrumentation](../experiments/token_timing_instrumentation.md)에, workload metadata는 [SWE-bench manifest](../../traces/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z/manifest.json)와 [AnalysisBench manifest](../../traces/analysisbench_minisweagent_toolcall_full_20260709T131115Z/manifest.json)에 있다.

### E2E ceiling은 benchmark마다 크게 다르다

| trace | task | timed tool | tool / E2E | top 1% perfect elimination | top 1% 2× | p99로 clamp | top 5% perfect elimination | top 5% 2× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-bench Verified, Qwen3.6-27B | 500 | 18,448 | 8.98% | 2.67% | 1.33% | 1.87% | — | — |
| AnalysisBench, GLM-5.2 | 35 | 2,496 | 84.98% | 7.33% | 3.66% | — | 36.18% | 18.09% |

여기서 perfect elimination은 해당 bucket의 duration을 0으로 만드는 비현실적인 상한이다. `2×`는 그 duration을 절반으로 줄이는 counterfactual이고, clamp는 p99 초과분만 p99로 낮춘 값이다. 모델의 행동, tool 결과, retry 수가 그대로라는 가정이므로 causal speedup 측정이 아니다.

두 trace가 다른 답을 주는 이유도 중요하다.

- SWE-bench에서는 model 시간이 대부분이라 top 1% tool을 크게 줄여도 E2E ceiling이 작다.
- AnalysisBench의 p99는 60초 timeout에 pile-up되어 있다. Top 1%가 전부 timeout 또는 nonzero return이고, 의미상 poll/install/build가 중심이다. 여기서는 timeout 처리와 progress detection이 곧 E2E 최적화다.
- 따라서 “long tail만 최적화하면 몇 % 줄어드는가”에는 workload-independent 답이 없다. 먼저 `tool / E2E`와 tail이 실제 성공 작업인지 timeout인지 확인해야 한다.

### SWE-bench tail의 semantic signature

SWE-bench 상위 1% timed tool call의 semantic mix는 다음과 같다.

| semantic class | tail call 비중 | tail time 비중 |
| --- | ---: | ---: |
| install / dependency setup | 54.6% | 61.1% |
| test | 19.5% | — |
| inspect / build-like | 25.4% | — |

Exact command repeat는 6.49%뿐이었다. 즉 tail은 install/build/test 쪽으로 강하게 치우치지만, 완전히 같은 명령의 memoization만으로 잡을 수 있는 비중은 작다.

### Tail output은 prefill에 불리하다

SWE-bench 상위 1% tool call의 67.0%가 end-loaded output이었다. 첫 output도 대체로 실행 종료에 가깝게 도착했다. 이는 actual-output chunking의 핵심 제약이다. 시간이 길어도 확정된 output token이 일찍 오지 않으면 prefill할 payload가 없다.

과거 명령을 token-set Jaccard로 ranking하고 top-4 candidate를 택했을 때, 실제 output과 exact prefix로 재사용 가능한 token 비율은 tail 2.20%, body 6.34%였다. 분석 정의와 causal-history 제약은 [branchfill prefix opportunity analyzer](../../agent/src/minisweagent/run/extra/branchfill_prefix_opportunity.py)에 있다.

이 결과를 next-turn TTFT와 결합한 opportunity 계산에서는:

- tail-only hideable next-TTFT 합: 전체 E2E의 0.035%
- 모든 tool call 대상: 전체 E2E의 2.79%

따라서 **현재 command-similarity top-4 정책에서 long duration은 candidate quality의 proxy가 아니다.** 오히려 tail은 install log, build/test result, timeout output처럼 환경 상태에 민감해 과거 output prefix가 덜 반복된다.

## 측정 사실 2: SWE-chat에서 보이는 tail의 성격

SWE-chat은 실제 사용자가 공개 repository에서 opt-in으로 남긴 session을 Entire checkpoint에서 수집한 데이터다. 논문은 dataset이 약 6,000 sessions, 355,000 tool calls를 포함하고, 한 turn에 여러 tool을 자주 호출하며 bash가 전체 tool의 약 1/3이라고 설명한다. [SWE-chat paper](https://arxiv.org/html/2604.20779), [official dataset card](https://huggingface.co/datasets/SALT-NLP/SWE-chat)

이번 local snapshot과 parser provenance는 [SWE-chat README](../../../traces/swe-chat/README.md)와 [provenance note](../../../traces/swe-chat/analysis/provenance-research.md)에 있다. Chunked replay가 사용한 subset은 OpenCode-format 600 sessions와 Codex-format 168 sessions다.

### Tail은 “큰 output”이 아니라 “오래 기다리는 operation”에 가깝다

Replay-valid individual calls에서 각 agent별 p99를 tail로 정의했다.

| subset | calls | p99 duration | top 1%의 tool-time 비중 | tail output 중앙값 | body output 중앙값 | duration-output Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCode | 15,661 | 4.574s | 84.0% | 1,059 chars | 2,661 chars | 0.081 |
| Codex | 7,282 | 30.161s | 32.2% | 74 chars | 1,634 chars | 0.216 |

Tail output이 오히려 작고 duration-output 상관도 약하다. “오래 걸리니 많은 output을 미리 prefill할 수 있을 것”이라는 가정은 이 trace에서 성립하지 않는다.

Tail의 API/command mix도 단일하지 않다.

- OpenCode tail 157 calls: `bash` 100, `task` 21, `question` 11, `read` 8이 대부분이다. `bash` 안에는 npm/pnpm, Python, Go, Docker, build/test가 많다. 모든 `task` call이 이 p99 bucket에 들어갔다.
- Codex tail 73 calls: `shell_command` 53, `exec_command` 10, `request_user_input` 5, `wait_agent` 2가 중심이다. Shell 내부에는 Zig build/test, pnpm test, 명시적인 `sleep`, orchestration status polling이 많았다.

이것은 네 가지 서로 다른 개선 대상을 한 tail bucket이 섞고 있음을 보여준다.

1. 실제 build/test/install 가속
2. timeout과 polling 제거
3. subagent orchestration의 비동기화
4. human wait와 lifecycle gap의 별도 처리

### SWE-chat duration은 agent별로 같은 물리량이 아니다

OpenCode는 각 tool part의 `state.time.start → end`를 사용한다. Codex는 call event에서 output event까지를 사용하므로 batch flush, synchronization, client bookkeeping이 섞일 수 있다. `exec_command`가 session ID를 반환하면 그 duration은 child process 전체 수명이 아니라 그 tool response가 available해질 때까지의 시간이다. 자세한 정의는 [local command-duration report](../../replay_runs/swe_chat_tool_duration_by_command_20260726.md)와 [replay source adapter](../../agent/src/minisweagent/run/replay_sources.py)에 있다.

실제 사용자 trace에는 `question`, `request_user_input`, 며칠 뒤 session 재개, 의도적인 wait가 섞인다. 이런 구간을 그대로 “tool implementation이 느리다”고 해석하면 E2E 기회를 과대평가한다. 사람 대기와 장시간 gap을 제거한 sensitivity analysis도 방향성 확인에는 유용하지만, 이 local benchmark의 직접 speedup으로 옮길 수는 없다.

## 측정 사실 3: SWE-chat incremental prefill replay

Local replay는 Mistral-Small-3.2-24B, 131,072-token context, 16-token cache block, `time_scale=1.0`으로 baseline과 incremental을 실행했다. 결과는 [OpenCode baseline](../../replay_runs/swe_chat_opencode_full_20260724T104006Z/baseline_gpu1/summary.json), [OpenCode incremental](../../replay_runs/swe_chat_opencode_full_20260724T104006Z/incremental_gpu0/summary.json), [Codex baseline](../../replay_runs/swe_chat_codex_full_20260725T122549Z/baseline_gpu1/summary.json), [Codex incremental](../../replay_runs/swe_chat_codex_full_20260725T122549Z/incremental_gpu0/summary.json)에 있다.

이 replay는 일반적인 streaming-output chunking과 다르다. SWE-chat raw에는 tool result의 최종 timestamp만 있으므로 source adapter가 call마다 `output_events=[{t: duration, output_chars: final_length}]` 하나를 만든다. Incremental runner는 multi-tool group에서 마지막 tool보다 먼저 끝난 sibling result만 prefill한다. 따라서:

- single long call은 prefill할 수 없다.
- 여러 result의 timestamp가 같으면 prefill할 수 없다.
- staggered multi-tool turn에서 먼저 확정된 result만 사용할 수 있다.

이 동작은 [incremental replay implementation](../../agent/src/minisweagent/run/extra/incremental_replay.py)의 `completed_at_s < tool_duration_s` 조건에 명시되어 있다.

### Tail에 시간이 많아도 eligibility가 작다

Tool-turn critical path의 p99를 기준으로 보면:

| subset | tool turns | p99 critical path | tail turns | multi-tool tail | prefill completed | 다음 valid request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenCode | 7,737 | 12.708s | 78 | 21 | 20 | 18 |
| Codex | 4,965 | 30.185s | 50 | 18 | 10 | 10 |

즉 tail turn 중 실제 incremental prefill이 완료된 비율은 OpenCode 25.6%, Codex 20.0%다. 나머지는 대체로 single-tool이거나 completion timestamp가 같은 group이었다.

Prefill이 완료되고 다음 replay request도 valid한 tail turn만 step `i → i+1`로 연결하면:

| subset | next requests | prefilled tool-output tokens 중앙값 | next TTFT 절감 중앙값 | next model-total 절감 중앙값 |
| --- | ---: | ---: | ---: | ---: |
| OpenCode | 18 | 725.5 | 38.1ms | 34.6ms |
| Codex | 10 | 722.5 | 32.7ms | 25.4ms |

조건을 만족한 call에서는 exact cached prefix가 실제로 다음 TTFT를 줄였다. 그러나 대상이 각각 18개와 10개뿐이어서 전체 E2E를 움직이기에는 작았다.

Full-run session E2E 중앙값은 오히려 다음처럼 소폭 증가했다.

| subset | baseline p50 E2E | incremental p50 E2E |
| --- | ---: | ---: |
| OpenCode | 54.557s | 54.694s |
| Codex | 138.337s | 138.546s |

이 차이를 prefill overhead의 causal effect로 해석할 수는 없다. 두 조건은 서로 다른 GPU에서 동시에 한 번 실행됐고 반복 실험이 없으며, 이 run은 synchronous abort-and-drain 변경 전이다. 정확한 결론은 **이 full replay가 material E2E speedup을 입증하지 못했다**는 정도다.

## 1차 문헌이 말하는 것

이 절은 local 측정이 아니라 각 source 저자의 주장이다.

### Tail dominance는 다른 real-world trace에서도 반복된다

TraceLab은 43명의 Claude Code/Codex 사용에서 432,510 tool calls를 분석했다. 저자들은 Claude에서 1분 이상 call 4.9%가 tool time의 92%, Codex에서는 3.1%가 61%를 차지한다고 보고한다. 동시에 같은 tool type 안에서도 latency 분산이 크므로, tool name뿐 아니라 operation semantics와 최근 latency history를 써야 한다고 제안한다. [TraceLab §6](https://arxiv.org/html/2606.30560)

TraceLab은 Codex의 internal execution time과 call-to-result span을 분리할 수 있는 253,391 calls에서 후자가 418.1시간, 전자가 341.5시간이었다고 보고한다. 양의 residual은 77.8시간, 18.6%이며 p99 residual은 10초다. 이는 tail 일부가 tool implementation이 아니라 approval, runtime scheduling, shell startup, output propagation 같은 바깥 overhead일 가능성을 지지한다. 저자들도 이를 가설로 명시한다.

### Prefill overlap은 가능하지만 보통 E2E 이득은 tool 비중보다 작다

Sutradhara는 tool-independent prompt slice를 tool 실행과 겹치고, tool call JSON이 완성되는 즉시 dispatch하며, KV cache에 semantic priority를 주는 orchestrator-engine co-design이다. 저자들의 ablation에서는 prompt splitting alone이 median E2E를 3.5%, streaming dispatch를 더하면 누적 9.0%, KV policy까지 더하면 10.8% 줄였다. 저자들은 decode가 E2E를 지배하고 마지막 decode는 overlap할 tool이 없어서 FTR보다 E2E gain이 작다고 설명한다. [Sutradhara design and ablation](https://arxiv.org/html/2601.12967)

이 수치는 현재 local replay의 예상 speedup으로 사용할 수 없다. Sutradhara는 production-derived synthetic trace, A100, 다른 model/load, 수정된 vLLM scheduler를 사용하며 prompt splitting과 streaming dispatch까지 함께 평가한다.

### Streaming은 scheduling과 cache pressure까지 함께 풀어야 한다

Stream2LLM은 retrieval chunk가 도착할 때마다 prefill을 겹치면 low-load TTFT가 크게 줄 수 있지만, concurrency와 memory pressure에서는 scheduling과 preemption policy가 중요하다고 보고한다. 이 결과는 tool output streaming에도 구조적으로 유사하지만, 논문의 workload는 web crawling과 ANN retrieval이지 coding-agent shell output이 아니다. [Stream2LLM](https://arxiv.org/html/2604.16395)

Sarathi-Serve는 작은 prefill chunk가 decode stall을 줄이지만, chunk가 너무 작으면 kernel launch와 반복 KV read overhead가 생긴다고 보고한다. 해당 Yi-34B 실험에서는 512-token chunk가 prefill runtime에 최대 약 25% overhead를 보였고 2,048-token budget에서는 거의 사라졌다. [Sarathi-Serve §4.3 and §5.4](https://arxiv.org/html/2403.02310)

vLLM도 `max_num_batched_tokens`가 작으면 ITL에, 크면 TTFT에 유리하다고 문서화한다. 따라서 production Tool Prefill은 idle GPU를 전제로 해서는 안 되고, foreground decode보다 낮은 priority로 preemptible하게 schedule해야 한다. [vLLM Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/)

## 가설과 설계 함의

이 절은 위 측정과 문헌으로부터 도출한 가설이며 아직 검증된 사실이 아니다.

### 가설 1: duration classifier보다 opportunity classifier가 낫다

Prefill 실행 여부는 단순히 `P(tool is p99)`로 결정할 문제가 아니다. 적어도 다음 세 항을 함께 예측해야 한다.

```text
expected value
≈ P(tool slack > prefill cost)
 × expected exact reusable tokens
 × next-TTFT value
 - GPU opportunity cost
```

필요한 feature는 tool type보다 세밀해야 한다.

- normalized command signature와 semantic class
- package manager, test/build target, repository size
- timeout과 retry/poll 여부
- 같은 command/resource의 최근 latency
- time-to-first-output와 output rate
- output stability 또는 candidate-prefix hit history
- return code와 human/subagent dependency
- 현재 GPU queue, KV pressure, 다음 prompt의 uncached suffix

### 가설 2: tail-only Candidate Tool Prefill은 현재 retrieval policy로는 우선순위가 낮다

[Candidate Tool Prefill design](../experiments/candidate_tool_prefill.md)은 historical output candidates를 prefill하고 실제 output과 exact token prefix만 재사용한다. 이 방식은 single long call의 빈 시간을 쓸 수 있다는 장점이 있다.

하지만 SWE-bench tail의 command-sim top-4 reusable-token rate가 2.20%였고 exact repeat도 6.49%뿐이다. Tail은 시간이 길지만 install/build/test의 상태 의존 output이 많다. 따라서 다음 중 하나가 먼저 필요할 가능성이 높다.

- command similarity 대신 repository snapshot, lockfile, target, environment를 포함한 retrieval key
- stdout 전체가 아니라 안정적인 prologue/header만 candidate로 생성
- build/test framework별 structured candidate
- 과거 prefix hit rate를 online calibration하여 낮은-confidence branch를 제출하지 않기

현재 design 문서도 replay MVP가 production low-priority scheduler를 모델링하지 않으며 target vLLM에서 run하기 전에는 performance gain을 주장하지 않는다고 명시한다.

### 가설 3: end-loaded tool은 prefill보다 tool/runtime 개선이 먼저다

첫 output이 종료 직전에 오는 install/build/test는 actual-output chunking의 payload가 늦다. 이 경우 다음 최적화가 더 직접적일 수 있다.

- package cache, build cache, test selection
- progress-aware timeout과 stuck detection
- long process를 background job으로 넘기고 event로 완료 통지
- 고정 sleep polling을 event-driven wait로 교체
- shell startup, approval, output propagation overhead 축소

단, backgrounding은 agent가 stale result를 사용하거나 불완전한 테스트를 통과로 오인하지 않도록 dependency semantics를 보존해야 한다.

### 가설 4: prefill과 direct tool optimization은 함께 측정해야 한다

Tool을 2배 빠르게 만든 실험과 Tool Prefill을 켠 실험을 따로 측정한 뒤 합산하면 중복 계산할 수 있다. 동일 trace에 대해 최소 네 조건이 필요하다.

1. baseline
2. direct tool optimization only
3. Tool Prefill only
4. both

`both`의 효과가 2와 3의 합보다 작은 것이 정상일 수 있다. 줄어든 tool duration이 prefill overlap window도 줄이기 때문이다.

## 다음 실험

### 1. Tail taxonomy를 trace schema에 고정한다

각 call에 다음 clock을 분리해서 기록한다.

- orchestrator dispatch
- runner start/end
- child process start/end
- first byte/first token/last byte
- result serialization과 agent-visible timestamp
- approval/human wait
- subagent wait
- timeout reason

이렇게 해야 “실제 실행”, “framework overhead”, “사람 대기”, “resume gap”을 서로 다른 optimization target으로 분리할 수 있다.

### 2. Tail policy를 p99가 아니라 semantic/runtime bucket으로 평가한다

최소한 다음 bucket을 별도로 본다.

- install
- build
- test
- read/search
- network fetch
- poll/wait/sleep
- subagent
- human input/approval
- timeout/failure

각 bucket마다 call count보다 다음을 보고해야 한다.

- critical-path time share
- E2E perfect-elimination ceiling
- 2× speedup ceiling
- first-output distribution
- output tokens와 exact-prefix reuse
- next-turn uncached prefill과 TTFT

### 3. Tail-only와 value-based prefill을 같은 GPU에서 반복 비교한다

정책은 세 개면 충분하다.

- all eligible calls
- p99-duration calls
- opportunity score 상위 calls

동일 GPU, 동일 trace order, 여러 seed/run에서 paired next-TTFT와 session E2E를 측정한다. 동시에 foreground decode ITL, prefill cancellation, wasted candidate tokens, KV eviction을 기록한다.

성공 기준은 “prefill token 수”가 아니라 다음이어야 한다.

- E2E 감소
- next-turn TTFT 감소
- foreground request의 tail ITL 비악화
- GPU-token당 hideable latency
- timeout/failure 및 task success 비악화

## Causal limitations

- SWE-chat은 public repository에 checkpoint logging을 opt-in한 early adopters의 trace다. 일반적인 enterprise workload를 대표하지 않는다.
- SWE-chat의 OpenCode와 Codex duration은 서로 다른 timing semantics를 갖는다.
- SWE-chat raw에는 human wait, resume gap, orchestration wait가 섞인다.
- Local SWE-chat baseline/incremental은 서로 다른 GPU에서 동시에 한 번 실행됐고 반복 trial이 없다.
- SWE-chat replay는 final-only output event를 합성하므로 실제 stdout streaming opportunity를 재현하지 못한다.
- SWE-bench와 AnalysisBench counterfactual은 tool duration을 줄여도 model 행동, retry, output, task quality가 변하지 않는다고 가정한다.
- SWE-bench timing 분석은 500 trajectory 전체의 18,448 timed tools를 사용했지만, manifest의 평가는 completed 77, resolved 53, error 421이다. 따라서 timing workload 분석에는 쓸 수 있어도 성공한 task의 대표적 latency나 quality-preserving speedup으로 일반화할 수 없다. 반면 AnalysisBench manifest는 35/35 instances가 successful이다.
- AnalysisBench의 timeout pile-up은 최적화 기회이면서 동시에 censoring이다. 실제 완료시간 분포를 알려주지 않는다.
- Candidate reusable-token 계산은 exact token prefix와 causal history만 허용하지만, 실제 GPU scheduling cost와 multi-tenant KV eviction을 포함하지 않는다.
- Prefix caching/prefill은 decode 시간을 없애지 않는다. Tool-time share를 prefill speedup으로 해석하면 안 된다.

## 최종 판단

**Long-tail tool optimization은 가치가 있다.** 그러나 효과는 “tail이 tool time의 몇 %인가”가 아니라 “그 tail이 전체 E2E critical path의 몇 %이고, 실제로 줄일 수 있는 runtime인가”로 판단해야 한다.

- SWE-bench형 workload에서는 tail tool을 직접 2배 빠르게 해도 E2E 약 1%대가 현실적인 1차 ceiling이다.
- timeout과 polling이 지배하는 AnalysisBench형 workload에서는 top 5%를 다루는 것이 두 자릿수 E2E 기회가 될 수 있다.
- 현재 Candidate/Chunked Tool Prefill은 tail-only로 좁힐 이유가 없다. Tail은 output이 늦고 candidate prefix가 덜 반복되어, 시간은 많지만 prefill material이 부족하다.
- 가장 유망한 정책은 **semantic/runtime tail reduction + value-based, low-priority Tool Prefill**의 조합이다.
