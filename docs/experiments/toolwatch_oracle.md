# ToolWatch exact-repeat oracle

측정일: 2026-07-29

## 결론

현재 두 token-timing trace에서는 ToolWatch를 그대로 구현할 근거가 부족하다.

- SWE-bench Qwen에서 안전하지 않은 명령을 제외한 exact-repeat oracle은 E2E의 **0.205%**다.
- 관련 없는 write는 cache를 무효화하지 않고, 결과가 달라진 query만 실제 dependency write 때 refresh한다고 보는 낙관적 dependency oracle은 **0.238%**다.
- `cat`, `grep`, `sed`, `find`, `git diff` 같은 view를 첫 호출 전부터 완벽하게 유지하고, 반복 verifier와 build까지 dependency-aware refresh하는 가장 낙관적인 oracle도 E2E의 **1.655%**다.
- AnalysisBench에서는 같은 dependency-aware 두 수치가 각각 **0.448%**, **0.403%**다. 첫 값은 임의의 non-mutating repeat까지 포함하고, 두 번째는 ToolWatch의 view/verifier/build 범위만 포함한다.
- 첨부 아이디어의 go/no-go 기준인 5%에도 미치지 못한다. SWE-bench Qwen 전체 trace는 tool time 자체가 E2E의 8.98%라서, tool wait만 줄여 10%를 달성하는 것은 수학적으로 불가능하다.
- Qwen에서 첫 호출까지 포함해 모든 non-mutating command를 비용 0으로 만드는 비현실적인 상한도 E2E 약 **4.97%**다.

Qwen에서는 edit 이후 slack이 부족한 것이 주된 문제가 아니었다. Exact repeat 자체가 적고 반복되는 verifier의 누적 시간이 작았다. AnalysisBench는 tool-bound이지만 install, timeout, one-off build가 지배하여 continuous query로 재사용할 대상이 거의 없었다.

## 2026-07-29 dependency-oracle 보정

최초 계산은 Experiment A의 간단한 근사로 모든 감지된 write가 모든 과거 query를 무효화한다고 보았다. 이는 ToolWatch의 dependency-level versioning을 완전히 반영하지 못했다.

보정된 oracle은 trace에서 직접 dependency를 알 수 없으므로 다음과 같이 더 낙관적으로 계산한다.

- intervening write가 있어도 반복 output과 return code가 같으면 관련 dependency는 변하지 않았다고 보고 이전 결과 전체를 재사용한다.
- output이 달라졌으면 이전 호출 이후 최초 mutation이 시작된 시점부터 refresh했다고 본다.
- mutation tool 안에서 실제 write가 언제 발생했는지 알 수 없으므로 tool 시작부터 refresh한 것으로 처리한다. 이는 같은 tool 안의 남은 subcommand 시간까지 전부 주는 상한이다.

이 보정은 미래 output을 이용하므로 실제 구현 가능한 정책이 아니라 perfect-dependency upper bound다. Qwen의 가장 낙관적인 E2E 상한은 1.627%에서 1.655%로 0.028%p 증가했다. 따라서 global invalidation 근사가 결론을 좌우하지는 않았다.

## Oracle 정의

한 task 안에서 앞서 실행된 것과 같은 command가 다시 등장할 때만 repeat opportunity로 본다. Command key는 앞뒤 whitespace를 제거하고 선행 `cd /testbed &&` 또는 `cd /workspace &&`를 제거한 문자열이다.

Tool 종료시각은 trace의 wall timestamp를 사용하고 시작시각은 다음처럼 복원한다.

```text
tool start = tool result timestamp - perf_counter duration
```

### 수정이 없는 repeat

직전 동일 command 이후 감지된 mutation이 없고 stdout, stderr, exit code가 byte-equivalent이면 현재 duration 전체를 절감한다.

```text
saved = current tool duration
```

Mutation이 없는데 결과가 달라진 호출은 cache miss로 처리했다.

### 수정 이후 repeat

마지막 mutation이 끝난 즉시 현재 workspace version을 위한 refresh가 시작되었다고 가정한다.

```text
slack = current call start - last invalidation end
saved = min(current tool duration, slack)
```

이는 refresh duration을 미래의 실제 duration과 같다고 아는 oracle이며, CPU contention과 잘못 시작한 refresh 비용을 무시한다.

### Full materialization

첨부 아이디어의 A 모드를 평가하기 위해 read-only view는 첫 호출을 포함해 전부 즉시 반환된다고 가정하는 별도 상한을 계산했다. Index 생성과 mutation 반영 비용은 0이다.

## 결과

### 전체 요약

| Trace | Task | Tool / E2E | Exact-repeat call | Time-weighted repeatability | Global-invalidation repeat oracle | Dependency repeat oracle | Materialized view + dependency verifier/build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SWE-bench Qwen, 전체 | 500 | 8.98% | 746 / 18,448 (4.04%) | 3.78% of tool time | 0.205% | **0.238%** | **1.655%** |
| SWE-bench Qwen, 평가 completed | 77 | 11.41% | 111 / 2,657 (4.18%) | 1.99% of tool time | 0.179% | **0.182%** | **1.748%** |
| AnalysisBench GLM | 35 | 84.98% | 37 / 2,496 (1.48%) | 1.77% of tool time | 0.128% | **0.448%** | **0.403%** |

`Exact-repeat E2E oracle`은 mutation command, network, 명시적인 wait 같은 안전하지 않은 query를 제외했지만, 그 외 임의 command까지 포함한 넓은 상한이다.

`Materialized view + dependency verifier/build`은 모든 view call을 0초로 만들고, verifier/build의 exact repeat에는 perfect-dependency refresh oracle을 적용한다.

### SWE-bench Qwen

전체 500 task의 E2E 합은 111,110.2초, timed tool 합은 9,973.9초다.

| 정책 | 대상 tool time | 절감 tool time | 절감 E2E | 영향받은 task의 E2E 절감 p50 / p90 / max |
| --- | ---: | ---: | ---: | ---: |
| 모든 non-mutating exact repeat | 3.78% | 2.28% | **0.205%** | 0.217% / 0.798% / 2.046% |
| View exact repeat | 0.82% | 0.76% | 0.069% | 0.132% / 0.341% / 1.238% |
| Verifier exact repeat | 0.52% | 0.42% | 0.038% | 0.447% / 1.041% / 1.374% |
| View + verifier exact repeat | 1.35% | 1.19% | 0.106% | 0.146% / 0.452% / 2.046% |
| 모든 view 완전 materialization | 17.58% | 17.58% | 1.578% | 1.576% / 3.372% / 8.037% |
| 위 정책 + global-invalidation repeat verifier/build | 19.24% | 18.12% | 1.627% | 1.613% / 3.492% / 8.037% |
| 위 정책 + dependency repeat verifier/build | 19.24% | 18.44% | **1.655%** | 1.620% / 3.569% / 9.235% |

Verifier repeat는 52건뿐이었고 42.0초를 숨길 수 있었다. 그중 mutation 이후 refresh 대상은 46건이며 37건, 80.4%는 실제 call 전에 refresh가 끝날 만큼 slack이 길었다.

View + verifier repeat에서는 refresh 437건 중 428건, 97.9%가 call 전에 완료 가능했다. 즉 이 workload에서 실패 원인은 refresh window가 아니라 다음 두 가지다.

1. 동일 command가 다시 등장하는 비율이 작다.
2. 반복되는 command의 duration이 짧다.

평가가 completed인 77 task만 분리해도 full materialization oracle은 1.75%로 결론이 달라지지 않는다. 다만 전체 Qwen manifest는 500개 중 completed 77, resolved 53, error 421이므로 성공 trajectory의 대표성에는 한계가 있다.

### AnalysisBench GLM

전체 E2E 합은 20,478.7초이고 tool 합은 17,403.6초다.

| 정책 | 대상 tool time | 절감 tool time | 절감 E2E |
| --- | ---: | ---: | ---: |
| 모든 non-mutating exact repeat | 1.77% | 0.15% | 0.128% |
| Dependency-aware non-mutating exact repeat | 1.77% | 0.53% | **0.448%** |
| View + verifier exact repeat | 0.006% | 0.004% | 0.003% |
| 모든 view 완전 materialization | 0.38% | 0.38% | 0.325% |
| 위 정책 + global-invalidation repeat verifier/build | 1.07% | 0.41% | 0.345% |
| 위 정책 + dependency repeat verifier/build | 1.07% | 0.47% | **0.403%** |

AnalysisBench는 tool 시간이 많지만 그중 78.4%가 install, explicit mutation, network, wait처럼 현재 안전 정책에서 continuous query로 취급할 수 없는 명령이다. 반복 build refresh 5건은 모두 실제 call 시점에도 실행 중일 만큼 slack이 짧았다.

## CPU-only dependency-aware 후속 실험

GPU, model server, vLLM을 사용하지 않고 기존 trace만 다시 분석했다. `cat`, `sed`, `grep`, `rg`, `find`, `ls` 등은 command argument에서 dependency path를 추출하고, redirection, `sed -i`, `rm`, `cp`, `mv`, write API 등에서 추출한 write path와 겹칠 때만 invalidation했다. Verifier와 build는 import dependency가 없으므로 workspace-wide dependency를 유지했다.

Qwen 전체 trace에서 global invalidation과 dependency adapter의 결과는 다음과 같다.

| Exact-repeat 정책 | Global invalidation | Dependency adapter |
| --- | ---: | ---: |
| View cache hit | 38 calls | 89 calls |
| View refresh | 391 calls | 337 calls |
| View output mismatch | 33 calls | 36 calls |
| View saved time | 76.171s | 75.719s |
| View + verifier/build E2E oracle | 0.11704% | **0.11714%** |

관련 없는 write를 무시하면서 cache hit은 51건 늘었지만, recorded output이 달라 cache할 수 없는 호출도 늘었다. 전체 E2E 순증가는 약 **0.00010 percentage point**, 0.108초뿐이다. 평가 completed 77개에서는 E2E oracle이 두 방식 모두 0.1043%였고, AnalysisBench도 차이가 없었다.

즉 command-level dependency adapter로도 결론은 바뀌지 않는다. Exact repeat가 차지하는 시간 자체가 너무 작다.

원래 계획한 live `strace` replay도 확인했지만 현재 머신에는 Docker daemon, 원본 sandbox snapshot, `strace`가 없다. Docker CLI만 있고 socket이나 다른 container runtime도 없어 recorded command를 원래 workspace에서 재실행할 수 없었다. 이 확인 과정과 위 adapter 분석은 모두 CPU-only였으며 GPU를 사용하지 않았다.

## Go/no-go 판단

첨부 아이디어가 제안한 기준에 대입하면 두 trace 모두 **no-go**다.

| 기준 | SWE-bench Qwen | AnalysisBench |
| --- | ---: | ---: |
| Cache 또는 refresh 대상 tool time ≥25% | 19.24% | 1.07% |
| Whole-task E2E oracle ≥10% | 1.66% | 0.40% |
| 제한적 진행 기준 E2E ≥5% | 미달 | 미달 |

이 결과는 “continuous materialization이 구현하기 어렵다”는 판단보다 강하다. Maintenance cost, wasted refresh, CPU interference를 모두 0으로 둔 상한도 5%에 못 미친다.

## 남은 가능성

정적 dependency adapter까지 검증했으므로 남은 범위는 syscall-level dependency와 non-identical query containment이다. 예를 들어 서로 다른 `grep`이 같은 repository index를 공유하거나, 실제 import graph상 수정된 파일과 무관한 test cache를 유지하는 효과는 exact-repeat oracle에 포함되지 않는다.

다만 그 가능성을 살리려면 새 trace에서 다음을 직접 기록해야 한다.

- filesystem read/write dependency
- normalized query와 shared index key
- mutation마다 발생하는 실제 refresh CPU
- refresh가 재사용되기 전 다시 invalidated된 wasted work
- foreground model/tool과의 CPU, I/O interference

이를 더 확인하려면 Docker daemon과 SWE-bench image를 복구한 뒤 recorded command만 `strace`로 재실행하면 된다. 모델을 다시 호출할 필요가 없으므로 이 경우에도 GPU는 필요 없다. 다만 이미 모든 view를 공짜로 만드는 상한이 1.58%이므로, live replay를 위한 환경 구축보다 이 방향을 종료하는 판단이 현재 증거에는 더 부합한다.

## 재현

분석기는 [analyze_toolwatch_oracle.py](../../scripts/analyze_toolwatch_oracle.py)다.

```bash
python scripts/analyze_toolwatch_oracle.py \
  traces/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z

python scripts/analyze_toolwatch_oracle.py \
  traces/analysisbench_minisweagent_toolcall_full_20260709T131115Z
```

입력 trace의 계측 정의는 [token timing instrumentation](token_timing_instrumentation.md)에 있고 workload metadata는 [SWE-bench manifest](../../traces/swebench_verified_qwen36_trace_token_timing_full_20260706T113200Z/manifest.json), [AnalysisBench manifest](../../traces/analysisbench_minisweagent_toolcall_full_20260709T131115Z/manifest.json)에 있다.

## 한계

- File dependency가 없으므로 mutation은 command text로 감지했다.
- `python`이나 build tool 내부의 숨은 write를 완전히 찾지 못한다.
- Global-invalidation 결과는 dependency와 무관한 write도 invalidation한다. 정적 adapter는 경로가 겹치지 않는 write를 제외하지만 숨은 dependency를 놓칠 수 있다.
- Full materialization oracle은 index 유지비용과 첫 model call 전 준비비용을 무시한다.
- Direct tool-time 절감이 model 행동, retry 수, task success를 바꾸지 않는다고 가정한다.
- Background work의 CPU, memory, disk contention과 wasted refresh를 무시한다.
- AnalysisBench의 60초 timeout은 censoring되어 실제 완료시간을 알 수 없다.
