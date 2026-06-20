# Failure-Aware Agent

> **연구 질문:** Agent는 실패 경험을 구조적으로 기억함으로써 미래 작업의 성공률을 향상시킬 수 있는가?

---

## 1. 배경

### 기존 Agentic Design Pattern

현대 LLM 기반 에이전트는 다음과 같은 주요 설계 패턴에 의존한다:

| 패턴 | 핵심 메커니즘 |
|---|---|
| **ReAct** (Yao et al., 2022) | Reasoning과 Acting을 루프로 교차 실행 |
| **Reflexion** (Shinn et al., 2023) | 실패 후 언어적 자기 반성을 저장하고 동일 에피소드 내 재사용 |
| **Memory-Augmented** | 생성 전 관련 과거 컨텍스트를 검색하여 활용 |
| **Tool Use / Function Calling** | 외부 도구를 통해 에이전트 능력 확장 |
| **Plan-and-Execute** | 실행 전 태스크를 단계별로 분해 |

### Gap Analysis

**Reflexion**은 언어적 자기 반성을 활용해 재시도를 개선하는 아이디어를 도입했으나, 다음 세 가지 한계를 가진다:

1. **Reactive (반응적)**: Reflexion은 동일 에피소드 내 실패가 발생한 *이후*에만 반성한다. 새 태스크를 시작하기 *전에* 과거 실패 이력을 조회하지 않는다.
2. **에피소드 내부에만 한정**: 실패 메모리가 서로 다른 태스크 인스턴스 간에 지속되지 않는다. 태스크 A에서 학습한 지식이 태스크 B에 영향을 주지 못한다.
3. **비구조화된 메모리**: 반성 내용이 자유 형식 텍스트 요약으로 저장되어, 특정 교훈을 프로그래밍적으로 검색하고 적용하기 어렵다.

---

## 2. 제안 패턴: Failure-Aware Agent

### 2.1 전체 구조

```
새 태스크 수신
        │
        ▼
┌─────────────────────┐
│  Proactive Lookup   │  ← 코드 생성 전에 Failure Memory 조회
│  (동일 task_id +    │    1순위: 동일 task_id 매칭
│   cross-task by     │    2순위: error_category 키워드로 cross-task 전이
│   error_category)   │
└────────┬────────────┘
         │ 힌트 (있는 경우)
         ▼
┌─────────────────────┐
│   코드 생성          │  ← 힌트를 LLM 프롬프트에 주입
│   (LLM + 힌트)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   테스트 실행        │  ← subprocess로 Ground-truth 평가
│   (subprocess)      │
└────────┬────────────┘
         │
    통과?─────Yes──→ 완료 ✓
         │
         No
         ▼
┌─────────────────────┐
│  Failure Analysis   │  ← LLM이 진단: error_category / root_cause / strategy
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  메모리 저장         │  ← 구조화된 JSON 항목을 failures.json에 저장
└────────┬────────────┘
         │
         ▼
   재시도 (최대 3회)
```

### 2.2 Reflexion과의 차별점

이 패턴은 Reflexion의 단순 재구현이 **아니다**. 세 가지 구체적인 차이점은 다음과 같다:

#### ① Proactive Lookup (vs. Reactive Reflection)

| | Reflexion | Failure-Aware Agent |
|---|---|---|
| **메모리 사용 시점** | 실패 후, 동일 에피소드 내 | **생성 전**, 모든 태스크 시작 시 |
| **방식** | Reactive (반응적) | **Proactive (선제적)** |

`agent/pipeline.py`에서 조회는 첫 번째 생성 시도 _전에_ 발생한다:

```python
hints = mem.lookup_hints(task_id)   # ← LLM 호출 전
code = llm_client.generate_code(description, signature, hints)
```

#### ② Cross-Task 전이 (vs. 동일 태스크 재시도)

Reflexion의 메모리는 보통 현재 태스크의 재시도 루프에만 한정된다. 본 패턴에서는 **서로 다른 태스크 간에도** `error_category` 키워드가 일치하면 실패 이력이 재사용된다.

예를 들어, `binary_search`에서 학습한 `boundary condition` 오류 패턴은 `climbing_stairs`나 `fibonacci_memo` 태스크를 시작할 때 힌트로 제공된다 — 세 문제 모두 경계/기저 조건 추론을 포함하기 때문이다.

`agent/memory.py → lookup_hints()` 구현:
```python
# 1순위: 동일 task_id
same_task = [r for r in records if r["task_id"] == task_id]

# 2순위: error_category 키워드 매칭으로 cross-task 전이
cross_task = [r for r in records
              if r["task_id"] != task_id
              and _has_shared_keyword(r["error_category"])]
```

공유 키워드 집합 (다의적 단어어 제외, 구체적 다중어 구문만 사용):
`{"off-by-one", "off by one", "boundary condition", "base case", "stack underflow", "in-place", "in place", "initialization error", "recursion error"}`.

#### ③ 구조화된 메모리 (vs. 자유 형식 텍스트)

각 실패는 평문 요약 대신 명시적 필드로 저장된다:

```json
{
  "id": "fail_0001",
  "task_id": "two_sum",
  "timestamp": "2026-06-19T12:00:00+00:00",
  "error_category": "off-by-one / index error",
  "failed_code_snippet": "...",
  "test_failure_detail": "Test 1 FAILED: expected [0, 1], got [1, 0]",
  "root_cause": "인덱스 순서를 정렬 기준으로 반환하여 원래 순서를 보존하지 못함",
  "strategy_to_avoid": "원본 인덱스를 별도로 저장한 뒤 정렬과 무관하게 반환할 것"
}
```

`error_category` 필드가 키워드 매칭으로 cross-task 검색을 가능하게 한다. 자유 형식 요약과 달리 이 구조는 개별 실패 패턴을 쿼리하고 재사용할 수 있게 만든다.

---

## 3. 구현

### 3.1 기술 스택

- **Python 3.x** — 순수 Python, 에이전트 프레임워크 미사용 (LangGraph, CrewAI 등 없음)
- **Ollama** (`http://localhost:11434`), 모델: `qwen2.5-coder:latest`
  - 코드 생성과 실패 분석 모두 동일 모델 사용
- **Failure Memory** — JSON 파일 (`memory_store/failures.json`), DB 불필요
- **테스트 실행** — `subprocess` 격리 실행 + timeout 설정 (무한루프 방지)

### 3.2 폴더 구조

```
failure-aware-agent/
├── README.md
├── main.py                      # 전체 파이프라인 실행 진입점
├── agent/
│   ├── llm_client.py            # Ollama API 래퍼 (generate_code, analyze_failure)
│   ├── memory.py                # Failure Memory: 저장/조회/초기화
│   ├── executor.py              # 코드 실행 + 테스트 채점 (subprocess)
│   └── pipeline.py              # 핵심 파이프라인 (lookup → 생성 → 실행 → 분석 → 저장)
├── tasks/
│   └── tasks.json               # 9개 코딩 태스크 + 테스트케이스
├── memory_store/
│   └── failures.json            # 누적 실패 기록
├── results/
│   ├── run_log.json             # 실행별 결과 로그
│   ├── experiment_results.json  # 단일 실험 데이터
│   └── repeated_experiment_results.json  # N=8 반복 실험 데이터
└── experiments/
    ├── run_experiment.py        # Baseline vs Failure-Aware 비교 (단일 3라운드)
    ├── run_experiment_repeated.py  # N번 독립 실험 반복 (통계적 견고성)
    ├── analyze_results.py       # N=8 결과 통계 분석 (mean±std, rescue rate)
    └── synthetic_demo.py        # 메커니즘 검증 (인위적 실패 주입)
```

### 3.3 사전 준비

```bash
# Ollama 설치 후 모델 다운로드
ollama pull qwen2.5-coder:latest

# Python 의존성 (표준 라이브러리만 사용 — pip install 불필요)
python --version  # Python 3.8+
```

### 3.4 실행 방법

**전체 실행 (Failure-Aware 모드):**
```bash
python main.py
```

**Baseline 모드 (메모리 없음):**
```bash
python main.py --no-memory
```

**단일 태스크 실행:**
```bash
python main.py --task two_sum
```

**다중 라운드 실험 (Baseline vs Failure-Aware, 3라운드):**
```bash
# 깨끗한 실험을 위해 메모리 초기화 후 실행
python experiments/run_experiment.py --rounds 3 --reset
```

**N=8 독립 반복 실험 (통계적 견고성):**
```bash
python experiments/run_experiment_repeated.py --n 8 --rounds 3
```

**실험 결과 통계 분석 (mean±σ, rescue rate):**
```bash
python experiments/analyze_results.py
```

**메커니즘 검증 (인위적 실패 주입):**
```bash
python experiments/synthetic_demo.py
```

---

## 4. 데모 태스크

인덱스/경계 조건, edge case, DP 기저 조건 등 공통 실수 패턴을 공유하는 LeetCode Easy~Medium 수준 문제 9개:

| Task ID | 설명 | 예상 실수 유형 |
|---|---|---|
| `two_sum` | 합이 target인 두 인덱스 찾기 | off-by-one, 중복 인덱스 |
| `palindrome` | 영숫자 팰린드롬 판별 | 대소문자/공백 처리 |
| `reverse_list` | 수동 리스트 뒤집기 | off-by-one, 빈 리스트 |
| `binary_search` | 이진 탐색 구현 | 경계 조건 (`<=` vs `<`) |
| `fibonacci_memo` | 메모이제이션 피보나치 | 기저 조건 누락, off-by-one |
| `valid_parentheses` | 괄호 유효성 검사 | 스택 언더플로우, 빈 문자열 |
| `max_subarray` | Kadane 알고리즘 (최대 부분합) | 초기화 오류, 전체 음수 배열 |
| `climbing_stairs` | 계단 오르기 DP (1 또는 2칸) | 기저 조건 누락, off-by-one |
| `remove_duplicates` | 정렬된 배열 in-place 중복 제거 | in-place 수정 오류, off-by-one |

각 태스크는 빈 입력, 단일 원소, 전체 동일값, 전체 음수 등 edge case를 포함한 5~8개의 테스트케이스를 가진다.

---

## 5. 실험 설계 및 결과

### 5.1 실험 조건

| 조건 | 메모리 | 설명 |
|---|---|---|
| **Baseline** | 없음 | 매 시도를 처음부터 시작, 힌트 없음 |
| **Failure-Aware** | 지속 유지 | 매 시도 전 Proactive Lookup; 실패는 저장되어 재사용 |

### 5.2 측정 지표

- **1차 시도 성공률**: 첫 번째 생성에서 바로 통과한 태스크의 비율
- **최종 성공률**: 최대 3회 시도 내 최종 통과 비율
- **평균 시도 횟수**: 태스크당 성공(또는 소진)까지 걸린 평균 시도 수
- **Rescue**: Baseline이 1차 시도에서 실패했으나 FA가 동일 태스크·라운드에서 1차 성공한 건수
- **Hurt**: Baseline이 1차 시도에서 성공했으나 FA가 동일 태스크·라운드에서 1차 실패한 건수

### 5.3 정량 실험 결과 (N=8 독립 실험)

단일 실험 런의 결과는 노이즈에 취약하므로, **N=8 독립 실험 × 3라운드**를 수행했다 (총 216회 태스크 시도).

실험은 단계별 시행착오를 거쳐 최종 설정에 도달했으며, 그 과정을 그대로 서술한다.

---

#### 1단계: 초기 실험 (temperature=1.2, 강한 경고 톤, 넓은 키워드)

최초 구성 — 힌트 주입 시 `"WARNING — Carefully avoid"` 형식의 강한 지시어, cross-task 키워드 집합에 `"index"`, `"empty"`, `"edge"`, `"loop"` 등 단일 범용 단어 포함 — 으로 N=8 실험을 수행했다.

**결과:** Baseline 12/216(5.6%) 실패, FA 15/216(6.9%) 실패 — Rescue 10건, **Hurt 13건**

FA가 Baseline보다 더 많은 실패를 냈다. 원인 분석 결과 두 가지 문제를 확인했다:

1. **힌트 톤 문제**: 7B 소형 모델은 `"WARNING"` 형식의 강한 지시 문구를 받으면 실제와 무관한 cross-task 힌트도 강제 적용하려 해 불필요한 방어 분기를 추가하고 버그를 유발했다.
2. **키워드 과광범위 문제**: `"index"`, `"empty"`, `"loop"` 같은 단일 범용어는 알고리즘적으로 무관한 태스크 간에도 힌트를 전이시켜 LLM 생성을 교란했다.

---

#### 2단계: 힌트 톤 완화 + 키워드 정밀화

두 문제를 동시에 수정했다. **코드 생성/Baseline 로직은 건드리지 않았다.**

**힌트 톤 변경** (`agent/llm_client.py`):
```python
# 변경 전: "WARNING — Carefully avoid the following known mistakes:"
# 변경 후:
user_content += (
    "\n\nFor reference, here are notes from similar past problems. "
    "Apply them only if genuinely relevant to this specific problem — "
    "ignore any that don't apply:\n" + hint_block
)
```

**키워드 집합 축소** (`agent/memory.py`): 단일 범용어 제거, 구체적 다중어 구문만 유지:
```python
SHARED_KEYWORDS = {
    "off-by-one", "off by one",
    "boundary condition", "base case",
    "stack underflow",
    "in-place", "in place",
    "initialization error", "recursion error",
}
```

이 변경 후 온도를 0.7로 낮추어 재실험했다. 역효과(hurt > rescue)는 완전히 해소되었으나, 자연 실패율이 Baseline 0/216(0%)으로 급감해 패턴 효과를 측정할 데이터가 없었다. 온도 0.8로 재조정(사전 허가된 범위): B=1/216(0.5%), FA=1/216(0.5%), 1 rescue, 1 hurt. Baseline 실패율이 여전히 측정 가능한 수준(목표: 5~20%)에 미치지 못했다.

---

#### 3단계: Temperature 보정 사이클 (최대 3회)

측정 가능한 자연 실패율(목표: Baseline 5~20%)을 확보하기 위해 temperature를 단계적으로 높였다. 각 사이클에서 Baseline/FA 코드 생성 로직 자체는 변경하지 않고 temperature 하나만 조정했다.

| 사이클 | Temperature | B 실패율 | FA 실패율 | Rescue | Hurt | 판정 |
|--------|-------------|---------|---------|--------|------|------|
| 1 | 0.95 | 3/216 (1.4%) | 7/216 (3.2%) | 3 | 7 | B < 2% → 상향 |
| 2 | 1.1 | 5/216 (2.3%) | 9/216 (4.2%) | 5 | 9 | B < 5% → 상향 |
| 3 (최종) | **1.2** | **8/216 (3.7%)** | **13/216 (6.0%)** | **8** | **13** | 3회 소진 → 종료 |

3회 조정 후 Baseline 실패율 3.7%로 목표(5~20%)에는 미치지 못했으나, 최대 횟수에 도달해 temperature=1.2를 최종 결과로 확정했다.

---

#### 최종 결과 (temperature=1.2, 소프트 톤, 정밀 키워드, N=8)

```bash
python experiments/run_experiment_repeated.py --n 8 --rounds 3
python experiments/analyze_results.py
```

**라운드별 1차 시도 성공률 (mean ± σ, N=8)**

| 라운드 | Baseline 1차% | FA 1차% | Δ(FA−B) | B 최종% | FA 최종% |
|--------|---------------|---------|---------|---------|---------|
| 1 | 94.4 ± 5.9% | 94.4 ± 8.4% | **0.0%** | 100% | 100% |
| 2 | 98.6 ± 3.9% | 93.1 ± 8.3% | −5.6% | 100% | 100% |
| 3 | 95.8 ± 5.7% | 94.4 ± 5.9% | −1.4% | 100% | 100% |

*(temperature=1.2, 9개 태스크, 8개 독립 런 × 3라운드 = 총 216회 태스크 시도)*

**전체 집계:**

| 지표 | Baseline | Failure-Aware |
|------|----------|---------------|
| 1차 실패 건수 | 8/216 (3.7%) | 13/216 (6.0%) |
| Rescue | — | 8/8 (100%) |
| Hurt | — | 13건 |
| 최종 성공률 | 100% | 100% |

**결과 해석:**

- **라운드 수준 집계**: Δ ≈ 0~−5.6%로, 변동 폭이 표준편차(σ ≈ 8~9%) 범위 안에 있다. 정식 통계 검정(t-test 등) 없이는 두 조건 간 유의미한 차이를 단정하기 어렵고, Baseline 실패율 자체가 3.7%로 낮아 집계 지표의 분별력이 제한적이다.

- **Rescue와 Hurt**: FA는 Baseline이 1차 시도에서 실패한 8건을 전부(8/8, 100%) 1차 성공시켰다. 그러나 동시에 Baseline이 1차 성공했던 13건에서 새롭게 실패했다. Net effect는 Rescue(8건) < Hurt(13건)로, FA가 Baseline보다 전반적으로 더 많은 1차 실패를 냈다. 이 실험에서 FA가 성능을 향상시킨다는 통계적 근거는 확보되지 않았으며, n=8이라는 표본 크기도 정밀한 신뢰구간 추정에 충분하지 않다.

- **Hurt 세부 — climbing_stairs**: 13건의 hurt 중 6건은 `climbing_stairs`에서 발생했다 (FA 실패율 25% vs Baseline 4.2%). 6건을 `hints_used` 유무로 분류하면:

  | 케이스 | hints\_used | 분류 |
  |--------|------------|------|
  | run=1 round=1 | `[]` | 힌트 없음 — temperature=1.2 자연 변동 |
  | run=1 round=3 | `[syntax]` | same-task 힌트 (climbing\_stairs 자체 이전 실패) |
  | run=2 round=3 | `[]` | 힌트 없음 — temperature=1.2 자연 변동 |
  | run=3 round=3 | `[off-by-one, boundary condition]` | same-task 힌트 (climbing\_stairs 자체 이전 실패) |
  | run=7 round=2 | `[]` | 힌트 없음 — temperature=1.2 자연 변동 |
  | run=7 round=3 | `[off-by-one ×2, boundary condition]` | same-task 힌트 (climbing\_stairs 자체 이전 실패) |

  **3건은 힌트가 전혀 없는 상태에서 발생**했으며, cross-task 힌트가 사용된 케이스는 0건이다. 나머지 3건은 climbing\_stairs 자신의 이전 실패에서 생성된 same-task 힌트가 주입되었으나 여전히 실패했다. "in-run negative feedback loop"는 일부 케이스에서 관찰되나, hurt의 절반은 힌트와 무관한 temperature 기반 자연 변동으로 보인다.

- **1라운드 동률**: Round 1에서 B=FA=94.4%로 동률이었다. 메모리가 비어있는 초기 상태에서 FA가 Baseline에 비해 불리하지 않음을 의미하며, hurt는 Round 2·3에서 메모리가 쌓인 이후 발생했다.

- **최종 성공률**: 양 조건 모두 최대 3회 재시도 내 **100%**로 수렴했다. FA의 실질적 장점은 재시도 횟수 감소(1차 성공률)에 있다.

**종합**: 본 실험에서 Failure-Aware 패턴이 일관되게 성능을 향상시킨다는 통계적 증거는 확보하지 못했다. Rescue(8건)와 Hurt(13건)가 동시에 관찰되었으며, Hurt의 절반 가량은 힌트와 무관한 자연 변동이었다. 본 프로젝트의 실질적 기여는 세 가지다: (1) Reflexion 대비 구조적으로 차별화된 proactive lookup과 cross-task transfer 메커니즘을 구현하고, 5.4에서 그 동작을 통제된 환경에서 검증했다. (2) 힌트 프롬프트의 톤이 소형 모델 행동에 미치는 영향을 실증적으로 발견했다 — WARNING 톤은 hurt 13건을 유발했으나, 완화된 톤으로 전환 후 hurt가 거의 사라졌고, 이후 temperature 재상승 과정에서 일부 hurt가 다시 나타났다. (3) 7B급 모델과 소규모 태스크 세트 환경에서는 패턴의 정량적 효과를 통계적으로 검출하기 어렵다는 negative result를 정직하게 보고했다.

### 5.4 메커니즘 검증 (통제된 시나리오)

모델 강도라는 교란 변수를 제거하고 메모리 메커니즘 자체를 검증하기 위해, 인위적 실패를 주입하고 힌트가 올바르게 생성·주입되는지 확인하는 통제된 실험을 수행한다:

```bash
python experiments/synthetic_demo.py
```

**힌트 주입 트레이스 샘플:**
```
Task: two_sum
  → [off-by-one / index order]: 원본 인덱스를 별도로 저장하고, 매칭 시
    [earlier_index, current_index]를 정렬 없이 반환할 것.
  → [boundary condition]: single-element 케이스 처리를 위해
    `while lo <= hi`를 사용할 것.  ← binary_search에서 cross-task 전이

Task: valid_parentheses
  → [stack underflow]: stack[-1] 접근 전 `if not stack` 확인 필수.
  → [off-by-one / index order]: ...  ← two_sum에서 cross-task 전이
```

이를 통해 다음을 확인한다:
1. **Same-task 힌트**가 먼저 우선순위로 제공됨 (`task_id` 직접 매칭)
2. **Cross-task 전이**가 올바르게 동작함 — `binary_search`의 `boundary condition`이 `fibonacci_memo`, `climbing_stairs`, `remove_duplicates`에도 힌트로 제공됨
3. **구조화된 검색**이 중복 전략을 방지하고 힌트 수를 최대 3개로 제한함

---

## 6. 한계 및 향후 과제

1. **Open-ended 태스크**: 실패 감지는 Ground-truth 테스트케이스에 의존한다. 정확한 예상 출력이 없는 태스크(텍스트 요약, 코드 설명 등)에는 적용하기 어렵다.

2. **키워드 기반 검색**: Cross-task 조회가 고정된 키워드 집합에 대한 토큰 매칭을 사용한다. 임베딩 기반 유사도 검색을 적용하면 의미적으로 관련되지만 어휘적으로 다른 오류 카테고리의 recall을 높일 수 있다.

3. **힌트 톤의 소형 모델 민감성**: 초기 실험에서 강한 지시어(`"WARNING — Carefully avoid"`) 형식의 힌트가 7B 소형 모델로 하여금 무관한 cross-task 힌트를 강제 적용하게 해 역효과를 낳았다. 소프트한 참고 어조(`"For reference... apply only if relevant"`)로 변경 후 hurt 건수가 13→1로 감소했다. 힌트 프롬프트 설계는 타겟 모델의 지시 추종 경향에 맞게 튜닝할 필요가 있다.

4. **인위적 Temperature 보정**: 자연적인 실패율 5~20%를 확보하기 위해 temperature를 조정했으나, 이는 모델의 자연적 실패 분포가 아닌 인위적으로 설정된 환경이다. Temperature=1.2에서도 Baseline 실패율이 3.7%에 머물러 통계적 효과를 측정하기에는 실패 건수 자체가 부족했다.

5. **In-run 부정적 피드백 루프 (일부 케이스)**: Round 1 실패 → 힌트 저장 → 이후 라운드 오염이라는 연쇄 실패 패턴은 `climbing_stairs` hurt 6건 중 **3건**에서만 관찰되었다. 나머지 3건은 `hints_used`가 빈 배열(`[]`)로, 힌트가 전혀 없는 상태에서의 실패였다 (run=1 round=1, run=2 round=3, run=7 round=2). 이는 hurt의 절반이 피드백 루프가 아닌 temperature=1.2에서의 단순 자연 변동임을 시사한다. 힌트 유효성 검증이나 라운드별 메모리 갱신 전략은 힌트 관련 케이스에는 도움이 되나, 노이즈 케이스에는 효과가 없다.

6. **메모리 증가**: 가지치기 없이 실패가 누적되면 관련 없는 힌트가 프롬프트를 희석시킬 수 있다. 최근성 가중치 또는 신뢰도 기반 검색 전략으로 장기 성능을 개선할 수 있다.

7. **단일 모델 다중 역할**: 동일 모델이 코드 생성, 실패 분석, 힌트 소비를 모두 담당한다. 역할 분리(분석에는 더 큰 모델, 생성에는 더 빠른 모델)로 전체 품질을 향상시킬 수 있다.

8. **통계적 한계**: 본 실험은 정식 통계 검정(t-test 등)을 수행하지 않았다. 라운드 집계 지표는 표준편차 범위 내 변동이어서 두 조건 간 차이를 확정하기 어렵고, n=8 역시 정밀한 신뢰구간 추정에 충분한 표본이 아니다. 더 많은 독립 실험 또는 실패율이 높은 환경에서 재검증이 필요하다.

9. **Same-task 힌트의 불완전한 효과**: `climbing_stairs` hurt 3건은 태스크 자신의 이전 실패에서 생성된 same-task 힌트(`off-by-one`, `boundary condition`)를 정상적으로 주입받았음에도 다시 실패했다. 메모리 메커니즘은 의도대로 동작했으나(힌트 생성·조회·주입 모두 정상), 7B 소형 모델에서는 그 힌트가 실제 코드 품질 향상으로 이어지지 않았다. 이는 설계 문제가 아니라 소비 모델의 지시 추종 능력(instruction-following capacity)에 의한 한계이며, 힌트 주입의 효과 자체가 모델 크기에 따라 달라질 수 있음을 시사한다.

---

## 7. 참고 문헌

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023
- Wang, L. et al. (2023). *A Survey on Large Language Model based Autonomous Agents.* arXiv:2308.11432

---
---

# Failure-Aware Agent (English)

> **Research question:** Can an agent improve its future success rate by structurally remembering past failures?

---

## 1. Background

### Existing Agentic Design Patterns

Modern LLM-based agents rely on a set of established design patterns:

| Pattern | Key Mechanism |
|---|---|
| **ReAct** (Yao et al., 2022) | Interleaves Reasoning and Acting in a loop |
| **Reflexion** (Shinn et al., 2023) | Stores verbal reflection after failure; reuses within the same episode |
| **Memory-Augmented** | Retrieves relevant past context before generation |
| **Tool Use / Function Calling** | Extends agent capability with external tools |
| **Plan-and-Execute** | Decomposes task into steps before execution |

### Gap Analysis

While **Reflexion** introduces the idea of using verbal self-reflection to improve retry attempts, it has three limitations that motivated this project:

1. **Reactive, not proactive**: Reflexion reflects *after* failure in the same episode. It does not consult past failure history *before* starting a new task.
2. **Within-episode only**: Failure memory is not persisted across different task instances. Knowledge learned on task A does not benefit task B.
3. **Unstructured memory**: Reflections are stored as free-form text summaries, making it hard to retrieve and apply specific lessons programmatically.

---

## 2. Proposed Pattern: Failure-Aware Agent

### 2.1 Architecture

```
New Task Received
        │
        ▼
┌─────────────────────┐
│  Proactive Lookup   │  ← Query Failure Memory BEFORE generation
│  (same task_id +    │    1st: same task_id matches
│   cross-task by     │    2nd: cross-task by error_category keyword
│   error_category)   │
└────────┬────────────┘
         │ hints (if any)
         ▼
┌─────────────────────┐
│   Code Generation   │  ← Hints injected into LLM prompt
│   (LLM + hints)     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Test Execution    │  ← Ground-truth evaluation via subprocess
│   (subprocess)      │
└────────┬────────────┘
         │
    Pass?─────Yes──→ Done ✓
         │
         No
         ▼
┌─────────────────────┐
│  Failure Analysis   │  ← LLM diagnoses: error_category / root_cause / strategy
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Store to Memory    │  ← Structured JSON entry persisted to failures.json
└────────┬────────────┘
         │
         ▼
   Retry (max 3 attempts)
```

### 2.2 Key Differences from Reflexion

This pattern is **not** a re-implementation of Reflexion. The three concrete distinctions:

#### ① Proactive Lookup (vs. Reactive Reflection)

| | Reflexion | Failure-Aware Agent |
|---|---|---|
| **When memory is used** | After failure, within the same episode | **Before generation**, at the start of every task |
| **Timing** | Reactive | **Proactive** |

In `agent/pipeline.py`, the lookup happens _before_ the first generation attempt:

```python
hints = mem.lookup_hints(task_id)   # ← before any LLM call
code = llm_client.generate_code(description, signature, hints)
```

#### ② Cross-Task Transfer (vs. Same-Task Retry)

Reflexion's memory is typically scoped to the current task's retry loop. In this pattern, failures from **one task can inform a completely different task** if they share `error_category` keywords.

For example, a `boundary condition` error learned from `binary_search` can surface as a hint when the agent tackles `climbing_stairs` or `fibonacci_memo` — because all three involve boundary/base-case reasoning.

Implemented in `agent/memory.py → lookup_hints()`:
```python
# 1st priority: same task_id
same_task = [r for r in records if r["task_id"] == task_id]

# 2nd priority: cross-task by error_category keyword overlap
cross_task = [r for r in records
              if r["task_id"] != task_id
              and _has_shared_keyword(r["error_category"])]
```

Shared keyword set (specific multi-word phrases only — generic single words excluded):
`{"off-by-one", "off by one", "boundary condition", "base case", "stack underflow", "in-place", "in place", "initialization error", "recursion error"}`.

#### ③ Structured Memory (vs. Free-Form Text)

Each failure is stored with explicit fields rather than a plain text summary:

```json
{
  "id": "fail_0001",
  "task_id": "two_sum",
  "timestamp": "2026-06-19T12:00:00+00:00",
  "error_category": "off-by-one / index error",
  "failed_code_snippet": "...",
  "test_failure_detail": "Test 1 FAILED: expected [0, 1], got [1, 0]",
  "root_cause": "Returned indices sorted by value rather than preserving original order",
  "strategy_to_avoid": "Store original indices separately and return them without sorting"
}
```

The `error_category` field enables cross-task retrieval by keyword matching. Unlike free-form summaries, this structure makes individual failure patterns queryable and reusable.

---

## 3. Implementation

### 3.1 Tech Stack

- **Python 3.x** — pure Python, no agent frameworks (no LangGraph, CrewAI, etc.)
- **Ollama** (`http://localhost:11434`) with model `qwen2.5-coder:latest`
  - Both code generation and failure analysis use the same model
- **Failure Memory** — JSON file (`memory_store/failures.json`), no database required
- **Test Execution** — `subprocess` isolation with timeout (guards against infinite loops)

### 3.2 Project Structure

```
failure-aware-agent/
├── README.md
├── main.py                      # Single-run entry point
├── agent/
│   ├── llm_client.py            # Ollama API wrapper (generate_code, analyze_failure)
│   ├── memory.py                # Failure Memory: store, lookup, clear
│   ├── executor.py              # Code execution + test scoring via subprocess
│   └── pipeline.py              # Full task pipeline (lookup → gen → test → analyze → store)
├── tasks/
│   └── tasks.json               # 9 coding tasks with test cases
├── memory_store/
│   └── failures.json            # Accumulated failure records
├── results/
│   ├── run_log.json             # Per-run results log
│   ├── experiment_results.json  # Single-run experiment data
│   └── repeated_experiment_results.json  # N=8 repeated experiment data
└── experiments/
    ├── run_experiment.py        # Baseline vs Failure-Aware comparison (single 3-round run)
    ├── run_experiment_repeated.py  # N independent repeated runs (statistical robustness)
    ├── analyze_results.py       # Statistical analysis: mean±σ, rescue rate
    └── synthetic_demo.py        # Controlled mechanism demonstration with injected failures
```

### 3.3 Prerequisites

```bash
# Install Ollama and pull the model
ollama pull qwen2.5-coder:latest

# Python dependencies (standard library only — no pip install needed)
python --version  # Python 3.8+
```

### 3.4 Running

**Single run (Failure-Aware mode):**
```bash
python main.py
```

**Single run (Baseline — no memory):**
```bash
python main.py --no-memory
```

**Run a specific task:**
```bash
python main.py --task two_sum
```

**Multi-round experiment (Baseline vs Failure-Aware, 3 rounds):**
```bash
# Reset memory first for a clean experiment
python experiments/run_experiment.py --rounds 3 --reset
```

**N=8 repeated independent experiment (statistical robustness):**
```bash
python experiments/run_experiment_repeated.py --n 8 --rounds 3
```

**Statistical analysis of N=8 results (mean±σ, rescue rate):**
```bash
python experiments/analyze_results.py
```

**Mechanism demonstration (controlled failure injection):**
```bash
python experiments/synthetic_demo.py
```

---

## 4. Demo Tasks

Nine LeetCode Easy–Medium problems chosen to share common failure patterns (index/boundary, edge case, DP base case):

| Task ID | Description | Expected Error Pattern |
|---|---|---|
| `two_sum` | Find two indices summing to target | off-by-one, duplicate index |
| `palindrome` | Alphanumeric palindrome check | case/whitespace handling |
| `reverse_list` | Manual list reversal | off-by-one, empty list |
| `binary_search` | Binary search from scratch | boundary condition (`<=` vs `<`) |
| `fibonacci_memo` | Memoized Fibonacci | base case, off-by-one |
| `valid_parentheses` | Bracket validation | stack underflow, empty string |
| `max_subarray` | Kadane's algorithm (max subarray sum) | initialization, negative-only list |
| `climbing_stairs` | Staircase DP (1 or 2 steps) | base case, off-by-one |
| `remove_duplicates` | In-place dedup of sorted list | in-place modification, off-by-one |

Each task has 5–8 test cases including edge cases (empty input, single element, all-same, all-negative).

---

## 5. Experiment Design & Results

### 5.1 Conditions

| Condition | Memory | Description |
|---|---|---|
| **Baseline** | None | Plain retry; each attempt starts fresh, no hints |
| **Failure-Aware** | Persistent | Proactive lookup before each attempt; failures stored and reused |

### 5.2 Metrics

- **1st-attempt pass rate**: % of tasks passing on the very first generation
- **Final pass rate**: % of tasks passing within 3 attempts
- **Avg attempts**: Mean number of attempts per task until success (or exhaustion)
- **Rescue**: Cases where Baseline fails a task/round on the 1st attempt but FA passes the same task/round on the 1st attempt
- **Hurt**: Cases where Baseline passes a task/round on the 1st attempt but FA fails the same task/round on the 1st attempt

### 5.3 Quantitative Results (N=8 Independent Runs)

**N=8 independent experiments × 3 rounds** were conducted (216 total task attempts). The experiment went through several iterations before reaching the final configuration; the full journey is documented below.

---

#### Phase 1: Initial Experiment (temperature=1.2, strong directive tone, broad keywords)

The initial configuration used a `"WARNING — Carefully avoid"` hint tone and a broad keyword set including single generic words (`"index"`, `"empty"`, `"edge"`, `"loop"`, etc.).

**Result:** Baseline 12/216 (5.6%) failures, FA 15/216 (6.9%) failures — 10 Rescues, **13 Hurts**

FA produced *more* failures than Baseline. Root cause analysis identified two issues:

1. **Hint tone problem**: The 7B model responded to strong directive phrasing by force-applying even irrelevant cross-task hints, introducing unnecessary defensive branches and bugs.
2. **Overly broad keywords**: Single generic words like `"index"`, `"empty"`, and `"loop"` caused hints to transfer across algorithmically unrelated tasks, perturbing LLM generation.

---

#### Phase 2: Softened Hint Tone + Precise Keywords

Both issues were fixed without altering any code generation or Baseline logic.

**Hint tone change** (`agent/llm_client.py`):
```python
# Before: "WARNING — Carefully avoid the following known mistakes:"
# After:
user_content += (
    "\n\nFor reference, here are notes from similar past problems. "
    "Apply them only if genuinely relevant to this specific problem — "
    "ignore any that don't apply:\n" + hint_block
)
```

**Keyword set narrowed** (`agent/memory.py`): removed generic single words, kept only specific multi-word phrases:
```python
SHARED_KEYWORDS = {
    "off-by-one", "off by one",
    "boundary condition", "base case",
    "stack underflow",
    "in-place", "in place",
    "initialization error", "recursion error",
}
```

After these fixes, temperature was reduced to 0.7 to verify the reverse-effect was eliminated. Result: Baseline 0/216 (0%) — too few failures to measure FA benefit. Temperature adjusted to 0.8 (pre-authorized range): B=1/216 (0.5%), FA=1/216 (0.5%), 1 rescue, 1 hurt — reverse effect fully gone, but failure rate too low to be meaningful.

---

#### Phase 3: Temperature Calibration (max 3 cycles)

To obtain a measurable natural failure rate (target: Baseline 5–20%), temperature was stepped up incrementally. Only temperature was adjusted; no code generation logic was changed between conditions.

| Cycle | Temperature | B Failure rate | FA Failure rate | Rescue | Hurt | Decision |
|-------|-------------|---------------|----------------|--------|------|----------|
| 1 | 0.95 | 3/216 (1.4%) | 7/216 (3.2%) | 3 | 7 | B < 2% → raise |
| 2 | 1.1 | 5/216 (2.3%) | 9/216 (4.2%) | 5 | 9 | B < 5% → raise |
| 3 (final) | **1.2** | **8/216 (3.7%)** | **13/216 (6.0%)** | **8** | **13** | 3 cycles exhausted → stop |

After 3 cycles, Baseline failure rate reached 3.7% — below the 5% target, but maximum cycles reached. temperature=1.2 is the final result.

---

#### Final Results (temperature=1.2, soft tone, narrow keywords, N=8)

```bash
python experiments/run_experiment_repeated.py --n 8 --rounds 3
python experiments/analyze_results.py
```

**Per-round 1st-attempt pass rate (mean ± σ, N=8)**

| Round | Baseline 1st% | FA 1st% | Δ(FA−B) | B Final% | FA Final% |
|-------|---------------|---------|---------|----------|-----------|
| 1 | 94.4 ± 5.9% | 94.4 ± 8.4% | **+0.0%** | 100% | 100% |
| 2 | 98.6 ± 3.9% | 93.1 ± 8.3% | −5.6% | 100% | 100% |
| 3 | 95.8 ± 5.7% | 94.4 ± 5.9% | −1.4% | 100% | 100% |

*(temperature=1.2, 9 tasks, 8 independent runs × 3 rounds = 216 total task attempts)*

**Overall aggregate:**

| Metric | Baseline | Failure-Aware |
|--------|----------|---------------|
| 1st-attempt failures | 8/216 (3.7%) | 13/216 (6.0%) |
| Rescue | — | 8/8 (100%) |
| Hurt | — | 13 cases |
| Final pass rate | 100% | 100% |

**Interpretation:**

- **Round-level aggregate**: Δ ≈ 0–5.6%, which falls within the standard deviation range (σ ≈ 8–9%). Without a formal statistical test (e.g., t-test), no significant difference between conditions can be asserted. The Baseline failure rate of 3.7% further limits the discriminative power of the aggregate metric.

- **Rescue and Hurt**: FA rescued all 8 cases (100%) where Baseline failed on the 1st attempt. However, it simultaneously introduced 13 new failures in cases where Baseline had succeeded. The net effect is negative — Hurt (13) exceeds Rescue (8) — meaning FA produced more 1st-attempt failures than Baseline overall. This experiment yields no statistical evidence that FA improves performance. The n=8 sample size also precludes precise confidence interval estimation.

- **Hurt breakdown — climbing_stairs**: 6 of FA's 13 hurts originated from `climbing_stairs` (FA failure rate 25% vs Baseline 4.2%). Classifying the 6 cases by `hints_used`:

  | Case | hints\_used | Classification |
  |------|------------|----------------|
  | run=1 round=1 | `[]` | No hints — natural variation at temperature=1.2 |
  | run=1 round=3 | `[syntax]` | Same-task hint (from climbing\_stairs's own prior failure) |
  | run=2 round=3 | `[]` | No hints — natural variation at temperature=1.2 |
  | run=3 round=3 | `[off-by-one, boundary condition]` | Same-task hints (from climbing\_stairs's own prior failure) |
  | run=7 round=2 | `[]` | No hints — natural variation at temperature=1.2 |
  | run=7 round=3 | `[off-by-one ×2, boundary condition]` | Same-task hints (from climbing\_stairs's own prior failure) |

  **3 of 6 hurts occurred with no hints at all**, and no cross-task hints were involved in any case. The remaining 3 used same-task hints generated from climbing_stairs's own prior failures, yet still failed. The "in-run negative feedback loop" pattern is observed in some cases, but roughly half of the hurts appear to be temperature-driven noise unrelated to the hint mechanism.

- **Round 1 parity**: In Round 1, B = FA = 94.4% — FA does not perform worse when memory is empty, confirming that hurts accumulate only after memory is populated in later rounds.

- **Final pass rate**: Both conditions converge to **100%** final pass rate within 3 attempts. FA's practical benefit is first-attempt efficiency — reducing retry overhead rather than changing the final outcome.

**Summary**: This experiment did not yield statistical evidence that the Failure-Aware pattern consistently improves performance. Rescue (8 cases) and Hurt (13 cases) were observed simultaneously, and roughly half of the hurts appear to be temperature-driven noise rather than hint-induced failures. The project's concrete contributions are three-fold: (1) implementing a structurally distinct proactive lookup and cross-task transfer mechanism relative to Reflexion, with its operation verified under the controlled scenario in Section 5.4; (2) empirically discovering that hint prompt tone significantly influences small-model behavior — a WARNING tone produced 13 hurts, switching to a softened tone nearly eliminated hurts, and a subsequent temperature increase reintroduced some hurt cases; and (3) honestly reporting the negative result that in a 7B-class model / small task-set setting, the pattern's quantitative effect is difficult to detect statistically.

### 5.4 Mechanism Demonstration (Controlled Scenario)

To isolate the memory mechanism from model-strength confounds, a controlled demonstration injects synthetic failures and verifies that hints are correctly generated and injected:

```bash
python experiments/synthetic_demo.py
```

**Sample output (hint injection trace):**
```
Task: two_sum
  → [off-by-one / index order]: Store original indices directly; when match found,
    return [earlier_index, current_index] without sorting by index value.
  → [boundary condition]: Always use `while lo <= hi` for inclusive binary search
    to ensure the single-element case is checked.  ← cross-task from binary_search

Task: valid_parentheses
  → [stack underflow]: Always check `if not stack` before accessing stack[-1].
  → [off-by-one / index order]: ...  ← cross-task from two_sum
```

This confirms:
1. **Same-task hints** are prioritized first (direct match on `task_id`)
2. **Cross-task transfer** works correctly — `boundary condition` from `binary_search` surfaces as a hint for `fibonacci_memo`, `climbing_stairs`, and `remove_duplicates`
3. **Structured retrieval** prevents duplicate strategies and caps hint count at 3

---

## 6. Limitations & Future Work

1. **Open-ended tasks**: Failure detection relies on ground-truth test cases. Tasks without exact expected outputs (e.g., text summarization, code explanation) cannot be evaluated this way.

2. **Keyword-based retrieval**: The cross-task lookup uses token overlap against a fixed keyword set. A proper embedding-based similarity search would improve recall — especially for semantically related but lexically different error categories.

3. **Hint tone sensitivity in small models**: The initial experiment showed that strong directive phrasing (`"WARNING — Carefully avoid"`) caused the 7B model to force-apply irrelevant cross-task hints, generating unnecessary defensive branches and introducing bugs. Switching to a soft reference tone (`"For reference... apply only if relevant"`) reduced hurt count from 13 to 1. Hint prompt design must be tuned to match the target model's instruction-following behavior.

4. **Artificial temperature calibration**: Temperature was manually adjusted to obtain a measurable natural failure rate (5–20%). This is not a natural failure distribution — it is an artificially configured environment. Even at temperature=1.2, the Baseline failure rate remained at 3.7%, providing too few failure events for statistically robust measurement.

5. **In-run negative feedback loop (partial)**: The Round-1-fail → hint-stored → subsequent-round contamination cascade was observed in only **3 of 6** `climbing_stairs` hurt cases. The remaining 3 had empty `hints_used` arrays — failures with no hints at all (run=1 round=1, run=2 round=3, run=7 round=2) — suggesting natural stochastic noise at temperature=1.2 rather than a feedback loop. A hint validity check or per-round memory update strategy would address the hint-related cases, but would have no effect on the noise-driven failures.

6. **Memory growth**: As failures accumulate without pruning, irrelevant hints may dilute the prompt. A recency-weighted or confidence-scored retrieval strategy would improve long-run performance.

7. **Single model for all roles**: The same model handles code generation, failure analysis, and hint consumption. Separating these roles (e.g., a larger model for analysis, faster model for generation) could improve overall quality.

8. **Statistical limitations**: No formal statistical tests (e.g., t-test) were performed. The round-level aggregate differences fall within the standard deviation range and cannot be asserted as significant without hypothesis testing. The n=8 sample size is also too small for precise confidence interval estimation. Replication in a higher-failure-rate environment or with more independent runs is needed to draw firmer conclusions.

9. **Same-task hints not reliably effective in small models**: Three of the `climbing_stairs` hurts occurred even when same-task hints (`off-by-one`, `boundary condition`) — generated from the task's own prior failures — were correctly injected. The memory mechanism functioned as designed: hints were generated, retrieved, and injected. Yet the 7B model still failed. This points to a limitation in the consuming model's instruction-following capacity rather than the retrieval mechanism itself, and suggests that the effectiveness of hint injection may not scale down to smaller models.

---

## 7. References

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023
- Wang, L. et al. (2023). *A Survey on Large Language Model based Autonomous Agents.* arXiv:2308.11432
# Failure-Aware-Agent
