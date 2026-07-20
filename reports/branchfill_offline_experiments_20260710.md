# BranchFill Offline Prefix Opportunity 실험 보고서

- 작성일: 2026-07-10
- 데이터: SWE-bench Verified 500개 trajectory
- 대상 모델·토크나이저: `hosted_vllm/qwen36-27b`, `/home/pjw7200/models/Qwen3.6-27B`
- 범위: 기존 trace를 이용한 offline 분석
- 제외 범위: replay, speculative prefill GPU 비용, latency 절감, scheduler 간섭

## 요약

이 실험은 tool 실행 중 과거 tool output을 미리 prefill해 두고, 실제 output이 도착하면 정확히 일치하는 prefix의 KV만 재사용하는 BranchFill의 기회가 SWE-bench Verified agent trace에 얼마나 존재하는지 측정했다.

가장 넓은 causal opportunity인 `any_prior` oracle에서는 전체 model-visible tool output 8,051,727 token 중 797,173 token, 즉 **9.90%**를 재사용할 수 있었다. 이 수치는 각 tool call 시점에 같은 trajectory에서 이미 완료된 모든 과거 output을 후보로 놓고, 실제 output과 exact token prefix가 가장 긴 후보를 사후에 선택한 상한이다.

실제 output을 보지 않고 현재 command와 과거 command의 token Jaccard similarity만으로 후보를 선택해도 다음 결과를 얻었다.

| 미리 prefill하는 후보 수 | 전체 output token 재사용률 | Any-prior oracle 포착률 |
|---:|---:|---:|
| 1 | 6.66% | 67.25% |
| 2 | 7.72% | 77.92% |
| 4 | **8.48%** | **85.70%** |
| 8 | 9.11% | 91.99% |

따라서 비용을 고려하지 않는 offline 관점에서는 command similarity 기반 후보 4개만으로 관측된 최대 prefix 기회의 85.7%를 포착할 수 있다. 복합 heuristic은 k=4에서 8.43%로 단순 command similarity보다 좋아지지 않았다.

이 결과는 BranchFill의 prefix predictability 가설을 지지하지만, 8.48%가 곧 latency 개선율이나 GPU 절감률을 뜻하지는 않는다. 이번 실험은 후보 prefill 비용, tool 대기 시간 안에 완료할 수 있는 양, GPU 포화와 선점, KV 메모리 비용을 측정하지 않았다.

## 1. 연구 질문

이번 offline 분석은 다음 질문에 답하기 위해 두 단계로 진행했다.

1. 실제 tool output의 앞부분이 과거 output과 exact token 단위로 얼마나 반복되는가?
2. 실제 output을 보지 않고 command history만 사용해 그 과거 output을 후보로 선택할 수 있는가?
3. 후보 수 k를 1, 2, 4, 8로 늘릴 때 발견하는 기회가 얼마나 증가하는가?
4. 단순 exact-command 재실행 이외에도 signature, resource, command similarity가 의미 있는 신호인가?

첫 번째 실험은 가능한 prefix 재사용량의 상한을 측정했고, 두 번째 실험은 runtime에서 구현할 수 있는 causal 후보 선택 정책이 그 상한을 얼마나 회수하는지 측정했다.

## 2. BranchFill의 정확성 조건

현재 tool output을 \(O_i\), 미리 prefill한 과거 output 후보를 \(O_c\)라고 하면 BranchFill이 재사용하는 것은 다음 구간뿐이다.

\[
K_{reuse}=K(H\Vert\mathrm{LCP}(O_i,O_c))
\]

여기서 LCP는 두 output을 대상 모델 토크나이저로 변환했을 때 처음부터 연속해서 정확히 일치하는 token 수다. 실제 output이 도착한 뒤 후보와 token 단위로 검증하며, 첫 불일치 이후의 candidate KV는 모두 폐기한다.

이 실험에서는 다음 조건을 지켰다.

- 현재 call보다 먼저 완료된 output만 후보로 사용했다.
- 같은 trajectory 안의 history만 사용했다.
- text normalization, fuzzy matching, semantic matching을 하지 않았다.
- 실제로 일치하는 exact token prefix만 재사용 가능 token으로 집계했다.
- full output과 truncated output처럼 formatter 형태가 다른 후보의 KV는 호환되지 않는 것으로 처리했다.
- 후보가 없거나 첫 token부터 다르면 재사용량을 0으로 처리했다.
- 후보가 없는 call도 전체 재사용률의 분모에 포함했다.

따라서 잘못 예측한 token이 모델 입력이나 decode에 사용되는 경우는 분석상 존재하지 않는다.

## 3. 데이터와 분석 단위

입력은 Qwen3.6-27B 기반 mini-swe-agent로 수집한 SWE-bench Verified 500개 trajectory다.

| 항목 | 값 |
|---|---:|
| Trajectory | 500 |
| Tool call | 26,435 |
| Model-visible output token | 8,051,727 |
| Raw output token | 10,073,811 |
| Full-render call | 26,033 |
| Truncated-render call | 402 |

한 trajectory는 하나의 SWE-bench instance를 해결하는 agent 실행이며, 각 tool call의 command, raw output, 모델에 표시된 output 형태, return code, command category를 trace에서 읽었다.

10,000자보다 긴 output은 agent observation formatter에서 앞 5,000자와 뒤 5,000자로 잘린다. Prefix 재사용은 output의 시작부터만 가능하므로 truncated call에서는 보이는 head만 candidate LCP로 인정했다. 분모에는 모델에 표시된 head와 tail token을 모두 포함했다. 이는 긴 output의 opportunity를 보수적으로 측정한다.

### 3.1 Tool output token 분포

다음은 26,435개 tool call 각각의 output 길이 분포다. Percentile은 정렬된 call-level 값에 선형 보간을 적용했다.

| 기준 | Mean | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| Model-visible payload token | 304.6 | 152 | 744 | 1,133.3 | 2,398 |
| Raw output token | 381.1 | 152 | 744 | 1,133.3 | 3,563.3 |

Model-visible payload는 BranchFill 분석의 분모로 사용한 값이다. `<returncode>`와 `<output>` 같은 고정 wrapper는 제외하고, 실제 모델에 표시되는 output payload만 센다. Truncated call에서는 head와 tail payload를 포함하지만 exact-prefix LCP는 head에서만 인정한다. Raw output은 truncation 전 전체 command output이다. 2,191개 call은 빈 output이었으며 call-level 분포에는 포함했고 token-weighted 재사용률에는 0 token으로 반영했다.

### 3.2 LLM input·output token 분포

다음은 26,842개 LLM request의 API usage에 기록된 token 분포다. 모든 model call에 usage가 존재했다.

| 기준 | Mean | P50 | P90 | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| Input (`prompt_tokens`) | 23,147.7 | 18,140 | 45,679.8 | 58,236 | 97,305.8 |
| Output (`completion_tokens`) | 290.4 | 111 | 719 | 1,031.0 | 2,180.8 |

LLM input은 각 request가 받은 누적 conversation prompt 전체다. 따라서 모든 request의 `prompt_tokens`를 더한 621,331,727 token은 고유 text 양이나 실제 KV prefill 연산량과 같지 않다. Prefix/KV cache가 이미 처리한 history도 API usage의 prompt token에는 반복해서 집계될 수 있다. LLM output에는 visible response뿐 아니라 provider usage가 completion으로 센 reasoning token이 포함된다.

## 4. 메트릭 정의

### 4.1 Token 재사용률

보고서의 핵심 `reuse ratio`는 겹치는 command의 비율이 아니라, 전체 tool output token 중 재사용 가능한 exact prefix token의 비율이다.

\[
\mathrm{ReuseRatio}=
\frac{\sum_i \max_{c\in C_i}\mathrm{LCP}(O_i,O_c)}
{\sum_i |O_i|}
\]

- \(C_i\): 현재 call에서 정책이 선택한 과거 output 후보 집합
- 분자: call별 후보 중 실제 output과 가장 긴 exact LCP token의 합
- 분모: 모든 tool call의 model-visible output token 합

예를 들어 output 길이가 각각 100, 300, 600 token이고 재사용 가능한 prefix가 20, 0, 100 token이면 재사용률은 command 기준 2/3이 아니라 `(20 + 0 + 100) / 1,000 = 12%`다. 따라서 긴 output과 긴 LCP의 영향이 더 큰 token-weighted metric이다.

### 4.2 Call 단위 보조 메트릭

- `eligible calls`: 후보 선택이 가능한 call 수
- `positive LCP calls`: 한 token 이상 정확히 겹친 call 수
- `LCP ≥ N calls`: N token 이상의 prefix를 재사용할 수 있는 call 수
- `trajectory reuse ratio`: trajectory마다 먼저 재사용률을 계산한 분포
- `95% CI`: trajectory 단위 bootstrap으로 계산한 전체 재사용률 신뢰구간

### 4.3 Oracle 포착률

`Any-prior oracle capture`는 정책이 재사용한 token을 any-prior oracle의 재사용 가능 token과 비교한 값이다. Any-prior oracle은 같은 causal history에서 가능한 가장 긴 LCP를 선택하므로 모든 정책의 상한이다.

Same-signature capture처럼 정책의 검색 범위가 oracle pool보다 넓을 수 있는 경우에는 call별로 `min(policy LCP, oracle LCP)`를 합산한다. 따라서 named oracle capture는 100%를 넘지 않는다.

## 5. 실험 1: Prefix opportunity oracle

### 5.1 후보 pool

- `any_prior`: 같은 trajectory에서 완료된 모든 과거 output
- `recorded_category`: trace에 기록된 기존 command category가 같은 과거 output
- `same_category`: 선행 `cd /testbed &&` 같은 setup command를 제거한 effective command category가 같은 output
- `same_signature`: executable, subcommand, module, flag를 정규화한 signature가 같은 output
- `exact_args`: tool 이름과 전체 argument가 정확히 같은 output

`recorded_category`는 첫 실험에서 사용한 기존 category 기준이다. 두 번째 실험에서 setup command를 제거해 command feature를 정제하면서 기존 수치를 보존하기 위해 이 이름으로 분리했다.

### 5.2 Oracle 결과

| 후보 pool | Eligible call | Positive LCP call | 재사용 token | 전체 token 재사용률 |
|---|---:|---:|---:|---:|
| Any prior | 25,635 | 12,767 | 797,173 | **9.90%** |
| Recorded category | 20,158 | 9,261 | 604,348 | 7.51% |
| Effective category | 18,238 | 8,953 | 521,711 | 6.48% |
| Same signature | 14,169 | 7,562 | 465,906 | 5.79% |
| Exact arguments | 1,755 | 1,373 | 222,239 | 2.76% |

Any-prior oracle의 95% trajectory-bootstrap CI는 9.38–10.40%다. Trajectory별 재사용률 중앙값은 8.46%, P25는 5.25%, P75는 12.17%, P90은 17.14%였다. 소수 trajectory만으로 전체 결과가 만들어진 것은 아니지만 instance별 편차는 크다.

Exact-argument history가 존재한 call은 전체의 6.64%뿐이다. 해당 call 내부에서는 output token의 43.22%를 재사용할 수 있었지만, 전체 기준으로는 2.76%다. 따라서 exact command cache만으로는 opportunity 대부분을 놓친다.

### 5.3 LCP 길이

| 후보 pool | LCP ≥32 call | LCP ≥64 call |
|---|---:|---:|
| Any prior | 4,013 | 2,726 |
| Recorded category | 3,056 | 2,135 |
| Effective category | 2,772 | 1,860 |
| Same signature | 2,391 | 1,640 |
| Exact arguments | 1,020 | 738 |

Any-prior 후보가 있는 call 중 절반가량은 LCP가 0이다. Opportunity는 모든 command에 얕게 퍼져 있다기보다 일부 call에서 수십에서 수백 token의 긴 prefix가 반복되는 형태다.

### 5.4 Output 길이에 따른 opportunity

| Output 길이 | Call | Output token | Any-prior 재사용률 |
|---|---:|---:|---:|
| 1–31 | 3,725 | 52,387 | 12.25% |
| 32–127 | 6,238 | 461,560 | 15.62% |
| 128–511 | 9,721 | 2,608,540 | **15.67%** |
| 512–2,047 | 4,025 | 3,526,574 | 8.15% |
| 2,048+ | 535 | 1,402,666 | 1.61% |

가장 큰 opportunity는 32–511 token 구간에서 관찰됐다. 매우 긴 output은 전체 token 비중은 크지만 앞부분이 정확히 반복되는 비율은 낮았다. 무조건 긴 output을 우선하는 정책보다 command predictability와 output 길이를 함께 고려할 필요가 있다.

### 5.5 Model-visible과 raw output 비교

Raw output 전체를 formatter 호환성과 무관하게 비교한 보조 결과는 any-prior 8.40%, recorded category 6.13%, effective category 5.32%, same signature 4.76%, exact arguments 2.31%였다. Model-visible any-prior의 9.90%보다 낮은 이유는 raw output에 매우 긴 tail이 포함되어 분모가 10,073,811 token으로 커지기 때문이다.

KV 재사용 가능성을 판단하는 주 메트릭은 model-visible 결과다. 실제 모델 prompt에 포함되는 formatter 형태와 token position이 같아야 candidate KV가 호환되기 때문이다. Raw 결과는 output 자체의 반복성을 이해하기 위한 보조 통계로만 사용했다.

## 6. 실험 2: Causal top-k 후보 정책

Oracle은 실제 output을 보고 가장 잘 맞는 과거 output을 고르므로 runtime 후보 선택 정책이 아니다. 두 번째 실험에서는 현재 command와 완료된 history만으로 후보를 먼저 순위화하고, 상위 k개의 서로 다른 과거 raw output을 선택했다. 실제 output은 후보 선택 이후 exact LCP 검증에만 사용했다.

### 6.1 평가한 정책

- `exact_args_recent`: argument가 완전히 같은 과거 call을 최신순으로 선택
- `signature_recent`: 정규화한 command signature가 같은 call을 최신순으로 선택
- `resource_aware_recent`: 같은 signature 안에서 공통 파일·resource가 많은 call을 우선
- `command_similarity`: 현재와 과거 command token set의 Jaccard similarity, 이후 recency 순
- `combined`: exact arguments, signature, resource overlap, category, command similarity, recency 순으로 lexicographic ranking

Command similarity는 다음과 같이 계산했다.

\[
\mathrm{Jaccard}(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

각 정책은 k=1, 2, 4, 8을 평가했다. 동일한 raw output이 history에 여러 번 등장하면 branch를 낭비하지 않도록 중복 제거했다.

### 6.2 전체 policy frontier

| 정책 | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| Exact arguments recent | 2.67% | 2.75% | 2.76% | 2.76% |
| Signature recent | 3.16% | 4.08% | 4.97% | 5.50% |
| Resource-aware recent | 4.24% | 4.92% | 5.43% | 5.71% |
| **Command similarity** | **6.66%** | **7.72%** | **8.48%** | **9.11%** |
| Combined | 6.54% | 7.54% | 8.43% | 9.09% |

Command similarity가 모든 k에서 가장 좋은 결과를 보였다. Combined policy는 더 많은 feature를 사용하지만 단순 Jaccard ranking을 넘지 못했다. 현재 데이터에서는 복잡한 heuristic보다 command 전체의 유사도가 더 안정적인 retrieval signal이다.

### 6.3 Command-similarity 상세 결과

| k | 재사용 token | 재사용률 | 95% CI | Trajectory 중앙값 | Any-prior 포착률 | Positive LCP call | LCP ≥64 call |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 536,060 | 6.66% | 6.25–7.08% | 5.22% | 67.25% | 7,421 | 1,876 |
| 2 | 621,192 | 7.72% | 7.27–8.17% | 6.24% | 77.92% | 9,166 | 2,188 |
| 4 | 683,145 | **8.48%** | 7.98–8.95% | 6.97% | **85.70%** | 10,570 | 2,375 |
| 8 | 733,334 | 9.11% | 8.61–9.59% | 7.76% | 91.99% | 11,631 | 2,544 |

k=1에서 k=4로 늘리면 147,085 token의 추가 opportunity를 발견한다. k=4에서 k=8로 늘리면 추가량은 50,189 token이다. 비용을 제외한 opportunity는 k와 함께 계속 증가하지만 한계효용은 감소한다. 이 결과만으로 k=4가 runtime 최적이라고 단정할 수는 없지만, 이후 비용 실험의 우선 baseline으로 삼기에는 적절하다.

k=4 command similarity는 전체 26,435 call 중 10,570 call에서 한 token 이상의 LCP를 찾았고 3,596 call에서 32 token 이상, 2,375 call에서 64 token 이상을 찾았다. 다시 말해 8.48%는 “전체 command의 8.48%가 맞았다”는 뜻이 아니라, 전체 output token 중 8.48%가 exact prefix로 재사용 가능했다는 뜻이다.

### 6.4 Command category별 기여

Combined k=4를 기준으로 재사용 token 기여가 큰 category는 다음과 같다. Command similarity와 combined의 전체 결과가 유사하므로 opportunity가 어디서 발생하는지 보는 보조 분석으로 사용했다.

| Category | 재사용 token | 해당 category 재사용률 | 전체 재사용 token 기여 |
|---|---:|---:|---:|
| `sed` | 143,934 | 8.83% | 21.21% |
| `python` | 115,525 | 15.15% | 17.03% |
| `cat` | 83,736 | 6.31% | 12.34% |
| `git&&cat` | 82,559 | 83.62% | 12.17% |
| `git` | 44,863 | 13.16% | 6.61% |

상위 다섯 category가 재사용 token의 69.36%를 차지한다. 다만 `sed`, `python`, `cat`은 원래 output token 양도 큰 category이며, 한 category만으로 전체 결과가 만들어진 것은 아니다. 수동으로 상위 match를 확인했을 때 반복적인 파일 출력, 동일 파일의 `cat`, 반복 `git diff`, compiler·test diagnostic 같은 실제 exact-prefix 사례가 포함됐다.

## 7. 해석

### 7.1 BranchFill opportunity는 존재한다

전체 output token의 9.90%가 같은 trajectory의 causal history와 exact prefix로 겹쳤다. Semantic similarity나 normalization을 전혀 사용하지 않은 결과이므로, 모델 입력을 바꾸지 않고 검증된 KV만 재사용한다는 BranchFill의 안전 조건과 직접 연결된다.

### 7.2 Exact cache보다 유사 command retrieval이 중요하다

Exact arguments oracle은 2.76%에 그쳤지만 command-similarity k=1만으로 6.66%를 얻었다. 동일한 command가 그대로 반복되는 경우뿐 아니라 파일이나 옵션 일부가 달라져도 formatter, header, diagnostic, 파일 내용의 앞부분이 반복되는 경우가 상당하다.

### 7.3 후보 여러 개의 가치가 확인된다

Top-k는 과거 output을 실제 output과 비교한 뒤 가장 긴 것을 고른다는 의미가 아니다. 실제 output이 오기 전에 command history만으로 상위 k개 output branch를 모두 prefill하고, output 도착 후 각 branch의 exact LCP를 검증한다는 의미다. k가 필요한 이유는 retrieval 순위 1위가 항상 실제 prefix와 가장 잘 맞지는 않기 때문이다.

k=1에서 6.66%, k=4에서 8.48%로 증가한 결과는 여러 branch를 준비하는 것이 opportunity를 실제로 확장함을 보여준다. 반면 k=8의 추가 이득은 작아져 이후에는 branch 비용과 함께 판단해야 한다.

### 7.4 복잡한 heuristic의 이득은 아직 없다

Exact arguments, signature, resource, category를 조합한 combined ranker가 단순 command similarity를 넘지 못했다. 다음 단계에서도 학습 기반 retrieval을 바로 도입하기보다 단순 policy를 강한 baseline으로 유지하는 편이 타당하다.

## 8. 한계와 해석 시 주의점

이번 결과는 opportunity 분석이며 실제 serving 성능 결과가 아니다.

- Tool 대기 시간 안에 후보 prefill이 얼마나 완료되는지 측정하지 않았다.
- 후보 하나당 GPU 연산량과 KV 메모리 사용량을 계산하지 않았다.
- GPU 포화 상태에서 low-priority speculative work가 다른 요청에 주는 영향을 측정하지 않았다.
- 후보가 틀렸을 때 폐기되는 연산량을 측정하지 않았다.
- History KV 공유와 candidate별 delta KV 저장의 구현 비용을 반영하지 않았다.
- 현재 trace의 command·repository 분포에 특화된 결과일 수 있다.
- 후보는 같은 trajectory에만 제한했다. Cross-trajectory exact-prefix cache를 사용하면 기회가 늘 수도 있지만 environment mismatch 위험도 함께 커진다.
- 같은 trajectory라도 command 사이에 repository 상태가 변할 수 있다. 현재 policy는 snapshot hash나 file state를 feature로 사용하지 않는다.
- Truncated output의 tail은 실제로 모델에 보이지만 prefix KV로 재사용할 수 없으므로 opportunity에서 제외했다.
- 실제 output은 후보 선택에는 사용하지 않았지만, 여러 branch 중 재사용 가능한 LCP를 측정하는 검증 단계에는 사용했다. 이는 runtime의 exact verify 동작을 offline으로 재현한 것이다.

따라서 8.48%를 latency 8.48% 감소, GPU 비용 8.48% 감소 또는 end-to-end 처리량 8.48% 증가로 해석하면 안 된다.

## 9. 결론과 권고

Offline 결과만 놓고 보면 BranchFill을 바로 포기할 이유는 없다. 오히려 단순한 command similarity 후보 4개가 any-prior 상한의 85.70%를 포착했다는 점은 다음 단계의 비용·latency 검증을 정당화한다.

현재 근거에 기반한 권고안은 다음과 같다.

1. 이후 구현의 retrieval baseline은 `command_similarity`, k=1/2/4로 둔다.
2. k=4를 opportunity 중심 기본점으로 사용하되 k=1과 k=2를 반드시 함께 비교한다.
3. 다음 평가에서는 tool latency window 안에 실제로 끝난 speculative token만 이득으로 인정한다.
4. Candidate prefill은 low-priority·preemptible로 실행하고 foreground 요청 간섭을 별도 측정한다.
5. Net benefit은 `절약된 post-tool prefill - speculative prefill - 폐기 work - scheduler interference`로 계산한다.
6. Repository snapshot, return-code state, command category별로 결과를 분해해 잘못된 candidate에 쓰는 비용을 줄일 수 있는지 확인한다.

Replay와 GPU 비용 분석은 이번 보고서의 실험 범위에 포함하지 않았으며, 위 결론은 그 결과를 미리 가정하지 않는다.

## 10. 재현 자료

### 분석 코드

- [BranchFill 분석기](../agent/src/minisweagent/run/extra/branchfill_prefix_opportunity.py)
- [분석기 테스트](../agent/tests/run/test_branchfill_prefix_opportunity.py)

### 통합 결과 artifact

초기 opportunity oracle 결과는 아래 policy 실험의 `summary.json`, `per_call.jsonl.gz`, `top_matches.json`에 함께 들어 있다. Policy 실험에서 command feature와 candidate pool을 확장하면서 기존 recorded-category 수치도 보존했으므로, 아래 artifact가 두 단계 실험을 모두 재현하는 최종 결과다.

- [실험 결과 보고서](branchfill_policy_frontier_swebench_verified_qwen36_20260710/report.md)
- [전체 요약](branchfill_policy_frontier_swebench_verified_qwen36_20260710/summary.json)
- [Policy frontier](branchfill_policy_frontier_swebench_verified_qwen36_20260710/policy_frontier.json)
- [Category 분석](branchfill_policy_frontier_swebench_verified_qwen36_20260710/policy_categories.json)
- [Call별 oracle 결과](branchfill_policy_frontier_swebench_verified_qwen36_20260710/per_call.jsonl.gz)
- [Call별 policy 결과](branchfill_policy_frontier_swebench_verified_qwen36_20260710/policy_per_call.jsonl.gz)
- [상위 oracle match](branchfill_policy_frontier_swebench_verified_qwen36_20260710/top_matches.json)
- [상위 policy match](branchfill_policy_frontier_swebench_verified_qwen36_20260710/top_policy_matches.json)

관련 코드 커밋은 다음과 같다.

- `3aff52f`: 초기 BranchFill prefix opportunity 분석
- `1f60dcd`, `ac0795d`: 분석기 구조와 프로젝트 기준 정리
- `1c5fcc2`: offline causal candidate policy frontier
- `4b3b5f4`: oracle capture와 uncertainty 보고 정정

분석기 테스트 8개, 환경 독립 프로젝트 테스트 505개가 통과했고 Ruff 및 Standards/Spec 리뷰도 통과했다. 전체 raw test collection은 설치되지 않은 선택 의존성 `contree_sdk`, `swerex`, `portkey_ai` 때문에 실행 대상에서 제외했다.
