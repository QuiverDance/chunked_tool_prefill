# Token Timing Instrumentation Report

이 문서는 현재 token timing 실험에서 trace가 어떻게 만들어지는지, runtime critical path에는 무엇이 남고 무엇이 offline으로 빠졌는지, 그리고 생성된 trace를 어떻게 후처리하는지 정리한다.

## 목적

이 계측의 목적은 일반적인 agent 실행을 가능한 한 덜 건드리면서 다음 정보를 남기는 것이다.

- model call의 TTFT, 전체 응답 시간, decode 시간
- tool call의 전체 실행 시간, 첫 output 도착 시간, returncode
- tool 실행 중 output이 언제 관측됐는지 나타내는 raw event timeline
- 후처리 단계에서 재구성 가능한 raw output token count와 stream sample 통계
- 문제 단위 end-to-end 시간과 model/tool 시간이 차지하는 비율

중요한 설계 원칙은 runtime에서 비싼 tokenization과 50ms sample row 생성을 하지 않는 것이다. Runtime trace에는 raw output과 cumulative output event만 저장하고, 25ms, 50ms, 100ms 같은 sample interval은 나중에 `scripts/summarize_token_timing.py`에서 재구성한다.

## 실행 경로

대표 실행 스크립트는 다음이다.

```bash
scripts/run_verified_token_timing_qwen36.sh
```

이 스크립트는 `scripts/run_verified_token_timing.sh`에 환경값을 넘긴다.

- `RUN_NAME_PREFIX`: 기본값 `swebench_verified_qwen36_token_timing`
- `SERVED_MODEL_NAME`: 기본값 `qwen36-27b`
- `TOKENIZER_PATH`: 기본값 `/home/pjw7200/models/Qwen3.6-27B`

실제 runner는 다음 일을 한다.

1. `RUN_DIR=$ROOT_DIR/runs/$RUN_NAME`와 `REPORT_DIR=$ROOT_DIR/reports/$RUN_NAME`를 만든다.
2. Docker daemon과 vLLM server 두 개를 확인한다.
   - port 8000
   - port 8001
3. SWE-bench Verified test split을 둘로 나눈다.
   - `gpu0`: slice `0:250`
   - `gpu1`: slice `250:`
4. 각 slice를 `mini-extra swebench`로 실행한다.
5. 두 slice가 끝나면 `scripts/summarize_token_timing.py`로 CSV와 summary JSON을 만든다.

핵심 command shape는 아래와 같다.

```bash
mini-extra swebench \
  --subset verified \
  --split test \
  --slice "$slice_spec" \
  --workers 1 \
  --output "$RUN_DIR/$label" \
  --config agent/src/minisweagent/config/benchmarks/swebench.yaml \
  --config agent/src/minisweagent/config/benchmarks/swebench_token_timing.yaml \
  --config "model.model_name=hosted_vllm/${SERVED_MODEL_NAME}" \
  --config "model.model_kwargs.api_base=http://127.0.0.1:${port}/v1" \
  --config "agent.tokenizer_path=${TOKENIZER_PATH}"
```

`swebench_token_timing.yaml`이 계측을 켠다.

```yaml
agent:
  agent_class: "minisweagent.run.benchmarks.utils.token_timing.TokenTimingProgressAgent"
  tokenizer_path: "/home/pjw7200/models/Qwen3.5-27B"
  tokenizer_local_files_only: true

model:
  model_kwargs:
    stream: true
    stream_options:
      include_usage: true
```

여기서 중요한 값은 `agent.agent_class`와 `model.model_kwargs.stream`이다. Agent class를 `TokenTimingProgressAgent`로 바꾸면 tool timing과 problem timing이 trace에 붙고, streaming model call을 사용하면 TTFT를 측정할 수 있다.

## Trajectory 파일 생성

SWE-bench runner는 instance마다 agent를 만들고 `agent.run(task)`를 호출한다. 실행이 끝나면 다음 위치에 trajectory를 저장한다.

```text
runs/<RUN_NAME>/<gpu label>/<instance_id>/<instance_id>.traj.json
```

예시는 다음과 같다.

```text
runs/swebench_verified_qwen36_token_timing_20260706T000000Z/gpu0/astropy__astropy-7671/astropy__astropy-7671.traj.json
```

Trajectory의 기본 구조는 mini-swe-agent의 일반 trajectory와 같다.

```json
{
  "info": {
    "model_stats": {
      "instance_cost": 0.0,
      "api_calls": 0
    },
    "config": {
      "agent": {},
      "model": {},
      "environment": {}
    },
    "token_timing": {
      "problem": {}
    }
  },
  "messages": [],
  "trajectory_format": "mini-swe-agent-1.1",
  "instance_id": "..."
}
```

계측값은 크게 세 곳에 붙는다.

- `info.token_timing.problem`: 문제 단위 wall/perf timing
- assistant message의 `extra.token_timing.model_call`: model call timing
- tool observation message의 `extra.token_timing.tool_calls`: tool call timing

## Problem Timing

`TokenTimingProgressAgent.run()`은 문제 풀이 시작과 끝을 감싼다.

1. 시작 시점에 `time.time()`과 `time.perf_counter()`를 기록한다.
2. 일반 agent 실행을 그대로 호출한다.
3. `finally`에서 종료 wall time과 elapsed perf time을 기록한다.
4. `info.token_timing.problem`에 저장한다.

현재 저장되는 problem timing은 다음 형태다.

```json
{
  "start_wall_s": 1780000000.0,
  "end_wall_s": 1780000123.4,
  "e2e_s": 123.4
}
```

`e2e_s`는 `time.perf_counter()` 기준이라 wall clock 조정의 영향을 덜 받는다. 후처리에서는 이 값과 model/tool duration 합을 비교해서 agent overhead를 계산한다.

## Model Call Timing

Model timing은 `LitellmModel._query_streaming()`에서 만들어진다. BrowseComp tool model도 같은 구조를 사용한다.

흐름은 다음과 같다.

1. `start = time.perf_counter()`를 찍는다.
2. `litellm.completion(..., stream=True, stream_options={"include_usage": True})`를 호출한다.
3. streaming chunk를 순회한다.
4. 첫 chunk가 도착하면 `first_chunk`를 기록한다.
5. 실제 생성 payload가 처음 들어온 chunk를 만나면 `first_token`을 기록한다.
6. stream이 끝나면 `end = time.perf_counter()`를 찍는다.
7. chunk들을 `StreamingResponseBuilder`로 합쳐 일반 completion response 형태로 복원한다.
8. response 객체에 `_mswea_model_timing`을 붙인다.

TTFT에서 말하는 "first token"은 단순히 첫 HTTP chunk가 아니다. `chunk_has_generated_payload()`가 true인 chunk다. 즉 content, reasoning content, function call, tool call id/name/arguments처럼 model이 실제로 생성한 payload가 들어와야 첫 token으로 본다.

Response에는 내부적으로 아래 값이 붙는다.

```json
{
  "request_start_s": 12345.0,
  "first_chunk_s": 0.12,
  "ttft_s": 0.20,
  "model_total_s": 3.40,
  "decode_s": 3.20
}
```

Trace에는 `message.extra.model_timing` 원본도 남는다. 다만 CSV와 주 분석에서 쓰는 `message.extra.token_timing.model_call`에는 아래 최소 세트만 저장한다.

```json
{
  "instance_id": "astropy__astropy-7671",
  "model_call_index": 4,
  "prompt_tokens": 12345,
  "completion_tokens": 678,
  "total_tokens": 13023,
  "finish_reason": "tool_calls",
  "ttft_s": 0.20,
  "model_total_s": 3.40,
  "decode_s": 3.20
}
```

이 값은 assistant message가 agent history에 추가될 때 `TokenTimingProgressAgent.add_messages()`와 `annotate_model_usage()`를 거쳐 `message.extra.token_timing.model_call`에 들어간다.

## Tool Call Timing

Tool timing은 `TokenTimingProgressAgent.execute_actions()`에서 시작된다.

1. assistant message의 `extra.actions`를 읽는다.
2. 각 action을 `execute_timed_action()`으로 실행한다.
3. submission marker인 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`가 들어간 command는 계측하지 않고 기존 `env.execute()`로 우회한다.
4. 일반 command는 `execute_streaming_action()`으로 실행한다.
5. 실행 결과와 timing record를 observation message로 렌더링한다.
6. observation message의 `extra.token_timing.tool_calls`에 tool metric을 붙인다.

현재 runtime tool metric은 의도적으로 작다.

```json
{
  "instance_id": "astropy__astropy-7671",
  "tool_call_id": "call_...",
  "command_category": "pytest",
  "duration_s": 12.34,
  "time_to_first_output_s": 0.56,
  "returncode": 1,
  "raw_output_chars": 10000,
  "raw_output_bytes": 10000,
  "output_events": []
}
```

Runtime에는 token count, rendered observation token count, 50ms sample row를 저장하지 않는다. 그런 값은 후처리에서 계산한다.

### Streaming command 선택

`execute_streaming_action()`은 환경을 보고 streaming execution이 가능한 command를 만든다.

Docker environment에서는 대략 다음 형태로 직접 subprocess를 띄운다.

```text
docker exec -w <cwd> -e KEY=VALUE <container_id> <interpreter...> <command>
```

Local environment에서는 다음 형태다.

```text
bash -c <command>
```

지원하지 않는 environment라면 fallback으로 기존 `env.execute(action)`을 한 번 호출하고, command가 끝난 시점에 output 전체가 한 번 도착한 것처럼 record를 만든다. 이 fallback은 중간 output timing을 제공하지 못하지만, duration과 final output size는 남길 수 있다.

### Pipe 읽기와 output event 생성

`run_streaming_command()`는 `subprocess.Popen`으로 command를 실행한다.

현재 설정은 다음과 같다.

```python
STREAM_SELECT_TIMEOUT_S = 0.01
STREAM_READ_CHUNK_BYTES = 64 * 1024
```

실행 구조는 다음과 같다.

1. `stdout=subprocess.PIPE`, `stderr=subprocess.STDOUT`로 stdout과 stderr를 하나의 pipe로 합친다.
2. `selectors.DefaultSelector()`에 pipe를 등록한다.
3. 최대 10ms마다 pipe readiness를 확인한다.
4. 읽을 데이터가 있으면 `os.read(fd, 64 * 1024)`로 읽는다.
5. UTF-8 incremental decoder로 bytes를 text로 바꾼다.
6. text가 생기면 output buffer에 append한다.
7. 누적 `output_chars`, 누적 `output_bytes`를 갱신한다.
8. 현재 elapsed time과 누적 output 위치를 `output_events`에 append한다.
9. process가 끝났으면 pipe에 남은 데이터를 한 번 더 drain하고 종료한다.

`output_events`는 chunk text 자체를 저장하지 않는다. 대신 "이 시각까지 raw output의 몇 글자/몇 byte가 관측됐는지"를 저장한다.

```json
[
  {
    "t": 0.041,
    "output_chars": 120,
    "output_bytes": 120
  },
  {
    "t": 0.083,
    "output_chars": 320,
    "output_bytes": 320
  }
]
```

이 구조는 후처리에서 원하는 간격으로 재구성할 수 있다.

- 첫 event까지는 output length가 0이다.
- 각 event의 `output_chars`는 누적값이다.
- event i에서 새로 나온 text는 `raw_output[previous_output_chars:output_chars]`다.
- 25ms sample을 만들고 싶으면 0.025초 간격으로 event cursor를 전진시키면 된다.
- 50ms sample을 만들고 싶으면 0.05초 간격으로 같은 raw event를 다시 걷는다.
- 100ms sample도 마찬가지다.

즉 sample interval을 바꾸기 위해 benchmark를 다시 돌릴 필요가 없다. Runtime trace가 보존해야 하는 정보는 raw output과 cumulative event timeline이고, 현재 trace는 그 정보를 저장한다.

### 종료 감지

종료 감지는 `proc.poll()`로 한다. Loop는 pipe readiness를 최대 10ms 단위로 기다린 뒤 `proc.poll()`을 확인한다. process가 끝난 것이 확인되면 남은 pipe output을 drain하고 최종 duration을 기록한다.

이 때문에 tool 종료 인지에는 최대 select timeout 수준의 지연이 생길 수 있다. 현재 값은 10ms다. 이 지연은 tool 자체를 느리게 만드는 지연이라기보다는 agent가 다음 model call로 넘어가는 시점을 아주 작게 늦출 수 있는 polling granularity다.

### Setup command 제외

아래 command category는 setup command로 보고 tool metric에서 제외한다.

```text
cd, export, source, ., alias, unalias, set, unset
```

비어 있는 command도 실행 결과 observation은 만들지만 tool metric은 남기지 않는다. 목적은 agent 동작은 그대로 두면서 분석에 의미 없는 setup-only action을 CSV에서 제외하는 것이다.

## Observation Rendering

Tool output은 model의 `format_observation_messages()`를 거쳐 tool observation message로 들어간다.

기본 observation content는 대략 다음 형태다.

```xml
<returncode>0</returncode>
<output>
... raw output ...
</output>
```

Observation message의 `extra`에는 raw output도 저장된다.

```json
{
  "raw_output": "...",
  "returncode": 0,
  "timestamp": 1780000000.0,
  "exception_info": "",
  "token_timing": {
    "tool_calls": []
  }
}
```

Rendered observation token count는 runtime에서 계산하지 않는다. 후처리에서 rendered content를 읽고 tokenizer가 있으면 `rendered_observation_tokens`를 계산한다.

## Trace에서 중요한 raw fields

현재 replay와 후처리에 가장 중요한 raw trace 정보는 다음이다.

### Model side

- `messages[*].extra.response.usage`
- `messages[*].extra.response.choices[0].finish_reason`
- `messages[*].extra.model_timing`
- `messages[*].extra.token_timing.model_call`

### Tool side

- `messages[*].extra.raw_output`
- `messages[*].extra.returncode`
- `messages[*].extra.exception_info`
- `messages[*].extra.token_timing.tool_calls[*].duration_s`
- `messages[*].extra.token_timing.tool_calls[*].time_to_first_output_s`
- `messages[*].extra.token_timing.tool_calls[*].raw_output_chars`
- `messages[*].extra.token_timing.tool_calls[*].raw_output_bytes`
- `messages[*].extra.token_timing.tool_calls[*].output_events`

`output_events`는 "tool call 동안 언제 output이 나왔는지"에 대한 원본 timestamp 기록이다. 이 값만 있으면 stream curve와 interval sample은 offline에서 다시 만들 수 있다.

## Offline Summary 생성

실험 실행이 끝나면 runner가 아래 command를 호출한다.

```bash
python scripts/summarize_token_timing.py "$RUN_DIR" --output-dir "$REPORT_DIR"
```

직접 실행할 수도 있다.

```bash
python scripts/summarize_token_timing.py runs/<RUN_NAME> \
  --output-dir reports/<RUN_NAME> \
  --stream-sample-interval-s 0.05
```

sample 간격을 바꿀 때는 같은 trace에 대해 옵션만 바꾸면 된다.

```bash
python scripts/summarize_token_timing.py runs/<RUN_NAME> \
  --output-dir reports/<RUN_NAME>_25ms \
  --stream-sample-interval-s 0.025

python scripts/summarize_token_timing.py runs/<RUN_NAME> \
  --output-dir reports/<RUN_NAME>_100ms \
  --stream-sample-interval-s 0.100
```

후처리 스크립트는 `run_dir` 아래의 `**/*.traj.json`을 모두 읽는다. Tokenizer는 우선순위가 있다.

1. CLI의 `--tokenizer-path`
2. trajectory의 `info.config.agent.tokenizer_path`
3. 없으면 token metric 일부를 비운 채 진행

`transformers`가 설치되어 있지 않고 tokenizer path도 명시하지 않았다면, 스크립트는 가능한 CSV를 만들되 token count 기반 field는 비워둘 수 있다. 명시적으로 `--tokenizer-path`를 줬는데 tokenizer를 로드할 수 없으면 실패하는 쪽이 맞다.

## CSV 출력

후처리는 세 개의 CSV와 하나의 JSON summary를 만든다.

```text
reports/<RUN_NAME>/
  model_calls.csv
  tool_calls.csv
  problem_timings.csv
  summary.json
```

### model_calls.csv

Model call 단위 row다.

```text
instance_id
trajectory
message_index
model_call_index
prompt_tokens
completion_tokens
total_tokens
finish_reason
ttft_s
model_total_s
decode_s
```

### tool_calls.csv

Tool call 단위 row다.

```text
instance_id
trajectory
message_index
tool_call_id
command_category
duration_s
time_to_first_output_s
returncode
raw_output_tokens
rendered_observation_tokens
was_truncated
raw_output_chars
stream_max_tokens_per_sample
stream_mean_tokens_per_sample
```

`raw_output_tokens`, `rendered_observation_tokens`, `stream_max_tokens_per_sample`, `stream_mean_tokens_per_sample`는 offline tokenizer가 있을 때 계산된다.

### problem_timings.csv

문제 단위 row다.

```text
instance_id
trajectory
problem_e2e_s
serving_relevant_e2e_s
agent_overhead_s
model_calls
tool_calls
sum_ttft_s
sum_model_total_s
sum_tool_duration_s
ttft_share_of_e2e
model_total_share_of_e2e
tool_duration_share_of_e2e
ttft_share_of_serving_relevant_e2e
```

여기서 `serving_relevant_e2e_s`는 `sum_model_total_s + sum_tool_duration_s`다. `agent_overhead_s`는 `problem_e2e_s - serving_relevant_e2e_s`다.

## Runtime Critical Path

현재 runtime critical path에 남아 있는 일은 다음이다.

- model streaming chunk 순회와 response 재구성
- model timing timestamp 기록
- subprocess 실행
- pipe readiness polling
- pipe read와 UTF-8 incremental decode
- raw output string 조립
- cumulative `output_events` append
- 최소 tool metric 저장
- trajectory JSON 저장

Runtime critical path에서 빠진 일은 다음이다.

- raw output tokenization
- rendered observation tokenization
- 50ms sample row 생성
- stream token curve 생성
- 종료 1초 전, 5초 전, half-duration token fraction 계산
- command duration 전체에 대한 dense sample materialization

따라서 현재 방식은 일반적인 agent 동작을 크게 해치지 않는 쪽으로 정리되어 있다. 남은 비용은 output을 실제로 읽고 raw event를 남기는 비용인데, 이것은 "tool output이 언제 나왔는가"를 측정하기 위해 필요한 최소 비용에 가깝다.

## 해석할 때 주의할 점

1. `time_to_first_output_s`는 process가 stdout/stderr pipe로 flush한 뒤 runner가 읽은 시점이다. 프로그램이 자체 buffer를 flush하지 않으면 output은 늦게 보인다.
2. stdout과 stderr는 합쳐서 기록된다. 이 실험은 "agent가 관측하는 combined observation output"의 timing을 본다.
3. `output_events`는 token event가 아니라 byte/text read event다. Token count는 offline tokenizer로 재구성한다.
4. `output_chars`는 Python string length 기준이고 `output_bytes`는 UTF-8 byte 기준이다.
5. process 종료 감지는 현재 10ms select timeout granularity의 영향을 받는다.
6. Docker command는 `docker exec` wrapper를 통해 실행되므로, 측정 duration에는 wrapper overhead도 아주 조금 포함된다.
7. fallback environment에서는 중간 output event가 없고 final event만 생긴다.
8. 기존 과거 trajectory에는 legacy metric field가 남아 있을 수 있다. 현재 summarizer는 새 trace를 기준으로 하되 일부 legacy sample format도 읽을 수 있게 되어 있다.

## Trace 재구성 예시

Raw output이 아래와 같다고 하자.

```text
hello
world
```

그리고 `output_events`가 아래와 같다고 하자.

```json
[
  {"t": 0.02, "output_chars": 6, "output_bytes": 6},
  {"t": 0.12, "output_chars": 12, "output_bytes": 12}
]
```

그러면 event별 delta text는 이렇게 복원된다.

- `0.02s`: `raw_output[0:6]`
- `0.12s`: `raw_output[6:12]`

50ms sample을 만들면 다음처럼 해석한다.

- `0.00s`: 0 chars
- `0.05s`: 6 chars
- `0.10s`: 6 chars
- final `0.12s`: 12 chars

100ms sample이면 다음처럼 해석한다.

- `0.00s`: 0 chars
- `0.10s`: 6 chars
- final `0.12s`: 12 chars

이 과정에서 raw trace는 바뀌지 않는다. sample interval만 후처리에서 바뀐다.

## 재현 체크리스트

1. vLLM server가 OpenAI-compatible endpoint로 떠 있어야 한다.
2. `/health`와 `/v1/models`에서 `SERVED_MODEL_NAME`이 확인되어야 한다.
3. Docker daemon이 접근 가능해야 한다.
4. token timing config가 포함되어야 한다.
5. model kwargs에 `stream: true`와 `stream_options.include_usage: true`가 있어야 한다.
6. run output directory 아래에 `*.traj.json`이 생성되어야 한다.
7. `scripts/summarize_token_timing.py`를 run directory에 대해 실행해야 한다.
8. sample interval 실험은 재측정 없이 summarizer 옵션만 바꿔서 수행한다.

현재 trace 생성의 핵심은 단순하다. Runtime은 "언제 output length가 증가했는지"만 정확히 남기고, token 기반 분석은 trace가 끝난 뒤 offline으로 수행한다.
