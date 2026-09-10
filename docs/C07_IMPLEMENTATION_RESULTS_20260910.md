# C0.7 개발 결과 · 2026-09-10

계획의 첫 배포 범위인 **R01~R05, 프로파일별 실행 계약과 공통 neural API, 근육·수용체 계측 장치의 최소 버전**을 구현했다. 기준 커밋은 `0cb43cbdd478652f78304115904c2feb874a1f2d`다. 코드는 현재 작업 폴더에 반영했으며 별도 커밋이나 원격 게시를 하지 않았다.

최종 회귀 **245/245 통과**, 실제 MPS 전뇌 계산 gate 통과, BANC 짧은 물리 연결 **6/6 조건 통과**, 단일 관절 근육 장치 **9/9 조건 실행 및 9개 기술 gate 통과**다. 0.6의 실제 FAFB·BANC 체크포인트는 원래 코드와 새 코드로 이어 계산했을 때 버전 필드 외 전체 상태가 정확히 일치했다.

**현재 앱의 BANC 몸은 여전히 기존의 공학적 목표각 경로를 사용한다. 새 Hill 근육은 독립 단일 관절 교정 장치다.** 근육의 전신 연결, 생리 단위 교정, 안정적인 자율 보행, 양방향 자연 먹이 접근, 보상·학습의 행동 통합을 완료했다는 결과가 아니다.

## 1. 리뷰 결함 수정

| 항목 | 구현 | 실제 확인 |
|---|---|---|
| R01 파라미터 누락 | `model_config.py`에서 기본값 < binding < 명시적 override 우선순위를 공유. 엔진, 계산 검증기, 실행 전 진단이 같은 파라미터를 사용 | rest/reset −60mV·threshold −30mV 반례와 dt 0.0001/0.00005의 검증기 결과가 엔진과 일치 |
| R02 적분 의미 불일치 | ResearchLIF의 held-drive, reset-current, voltage-events 세 방식을 구현. pulse 목적지, 불응기, 임계값 비교, 발화 후 h 초기화를 각각 유지 | CPU와 실제 Apple MPS에서 기준 ExpLIF와 전압·전류·queue·발화·불응기·개입을 비교 |
| R03 불가능한 캠페인 | BANC는 strict만 허용. 기본 pilot은 18조건, FAFB는 기존 54조건. 명시적 잘못된 case는 이름과 원인을 포함해 거절 | worker·작업 폴더·lock 생성 전에 거절. UI에서도 BANC의 다른 세 모드를 비활성화 |
| R04 비정상 복원 | count/rate/refractory 시간 상한, 음수·시간 0의 양수 trace, trace의 누적 상한, 가중치·마스크·queue 검사 | 잘못된 복원 및 device 복사 실패 시 기존 상태 유지. 정상 복원 후 계산 재현 |
| R05 선택 의존성 | 연구용 설치 파일과 `regression/research/native/mps` 검사 tier 추가 | Torch import가 없는 조건에서 기본 검사는 통과/skip을 분리하고 연구 tier는 `BLOCKED`, exit 2 |

수정 파일: [공통 모델 계약](/Users/chasoik/Projects/FLY_LAB/flylab/c/model_config.py), [ResearchLIF](/Users/chasoik/Projects/FLY_LAB/flylab/c/research_neural.py), [공통 복원 범위](/Users/chasoik/Projects/FLY_LAB/flylab/c/neural_state.py), [캠페인](/Users/chasoik/Projects/FLY_LAB/flylab/c/campaign.py), [회귀 반례](/Users/chasoik/Projects/FLY_LAB/tests/test_c_contracts.py).

ResearchLIF의 지수 계수는 같은/거의 같은 시간상수의 극한을 계산하고, 허용되는 `dt/tau=5000`에서도 `exp(+5000) × exp(−5000)`으로 NaN을 만들지 않게 했다. 발화율 상태를 초기화하거나 값을 잘라서 기존 반례를 숨기는 방식은 사용하지 않았다.

검증 결과의 `model`에는 전체 파라미터·해시·dt·integration·pulse 의미가 들어간다. 요청 모델 시간으로부터 실제 tick을 계산하고 정수가 아닌 기간을 거절한다. 작은 fixture의 통과와 전체 스냅샷 membership gate는 계속 구별한다.

## 2. 실행 지원 범위와 체크포인트

| 경로 | 지원 모드/적분 | 신경 입력·개입 | 앱 연결 |
|---|---|---|---|
| FAFB + 기준 CPU/Metal MPS | B_COMPAT, C_SHADOW, C_ASSISTED, C_STRICT. 신경 경로는 선언한 세 적분 지원 | 기존 pulse·발화 억제·outgoing/edge mute | 기존 CEngine 사용 |
| BANC neuromuscular v1/v2 | C_STRICT만 지원 | 기존 전신 연결과 감각/운동 차단 | 기존 목표각 어댑터 사용 |
| ResearchLIF CPU/MPS | 세 적분 지원, float32 | pulse, 억제·mute, 이질성·지연·gap·선택 edge reward-STDP | 독립 연구 엔진. CEngine backend 선택에는 아직 통합하지 않음 |
| CUDA | 기존 기준 엔진의 선택지는 유지 | 실제 CUDA 장치 검증 미실시 | 이번에 CUDA 완료를 추가 주장하지 않음 |

[공통 API](/Users/chasoik/Projects/FLY_LAB/flylab/c/neural_api.py)는 `advance`, 개입, readout, summary, region summary, snapshot, restore를 명시한다. ResearchLIF의 외부 보상은 **keyword-only `reward`**이며 pulse와 혼용되지 않는다. 생리/보상 프로파일과 해부학적 그래프 신원도 분리한다.

Research 상태는 `flylab.research-neural-state.v2`로 올렸다. 계산 의미가 달랐던 v1을 새 동역학으로 조용히 복원하지 않고, 보존된 v1 실행 환경이 필요하다는 오류를 반환한다. 반면 기존 CEngine 0.6 checkpoint schema와 baseline 동역학은 유지한다.

실제 기존 체크포인트 `checkpoint-35e11ca625ae`(FAFB), `checkpoint-ad0b7ef644bb`(BANC)를 git HEAD의 원래 소스와 C0.7 소스에서 각각 4 control tick(20ms) 이어 계산했다. 전압·queue·발화·몸 상태·감각 필터·개입·시간·이력까지 포함한 전체 checkpoint 비교에서 **`app_version`만 달랐다**. 이 결과는 해당 체크포인트와 짧은 연속 계산 범위의 검증이며 Research v1 변환 승인이 아니다.

[복원 비교 원자료](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/legacy_continuation_comparison.json)

## 3. 근육·수용체 계측 장치

새 [muscles.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/muscles.py)는 설치된 pinned FlyGym의 FlyMimic XML에서 왼쪽 Fe–Ti 관절과 두 Hill 근육을 추출한 **고정 몸·1자유도·접촉 없는 교정 장치**다. 원 모델은 왼쪽 앞다리만 근육으로 구동하며 현재 NeuroMechFly 보행 몸과 관절 구성이 다르다. [FlyGym 공식 설명](https://neuromechfly.org/tutorials/6_muscle_imitation/)

외형 파일의 기본 다운로드는 연결 시간 초과로 실패했다. 교정 장치는 이미 로컬 XML에 명시된 질량·관성과 힘줄 부착점을 사용하고, 외형 mesh와 충돌을 제외한다. 나머지 관절은 원본 keyframe 자세로 고정한다. 원본 파일·파생 XML·파라미터·library 버전·단위 계약·변환 오차를 함께 기록한다. 원본 전체 몸의 접촉 물리를 검증했다고 표시하지 않는다.

변환 전후의 초기 위치 오차는 `1.11e-16mm`, 방향 행렬 오차는 `2.22e-16`, 힘줄 길이 오차는 0, 모멘트암 오차는 `3.12e-17mm`였다. 힘줄 길이를 관절각으로 수치 미분한 결과도 모멘트암과 일치했다.

추가로 원본의 100µs Euler 간격에서 최대 흥분 1을 넣으면 첫 step의 활성도가 **2**가 되는 반례를 확인했다. 교정 장치는 10µs 간격으로 계산하며 최대/해제 입력의 활성도 0~1 범위를 검사한다. 변경된 적분 간격은 새 모델 신원의 일부다. [반례와 수정 비교](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/activation_timestep_probe.json)

측정 항목은 요청/적용 흥분, 활성도, 근육 길이·속도, 전체/수동/능동 근육 힘, 모멘트암, 관절 토크, 각도·속도·가속도, 수동 관절 토크·외력·제약 토크·관성이다. 실제 토크와 힘의 투영, `M·qacc`와 힘의 합을 독립적으로 대조한다. 기존 보행 몸에도 읽기 전용 `joint_observation()`을 추가해 42관절의 실제 actuator 힘과 토크를 별도 파일에 남긴다.

### 100ms 근육 실험

| 조건 | 최종 관절각 rad | 판정에서의 역할 |
|---|---:|---|
| 기준 흥분 0 | 1.578440 | 수동 힘·중력·관절 탄성 포함 기준선 |
| 굴근 0.1 / 0.35 | 1.686636 / 1.927663 | 기준 대비 양의 q 방향·용량 반응 |
| 신근 0.1 / 0.35 | 0.939962 / 0.676839 | 기준 대비 음의 q 방향·용량 반응 |
| 두 근육 0.2 | 0.799396 | 동시 활성 반응 기록. 중립 자세 보장으로 해석하지 않음 |
| 운동 차단, 요청 0.35/0.35 | 1.578440 | 기준의 전체 물리 상태와 정확히 일치 |
| 외부 토크 +0.01 / −0.01 | 1.589562 / 1.567175 | 양방향 외력 반응. 토크는 upstream model unit |

원본 control 하한 `0.0001`과 수동 탄성을 유지한다. 여기서 운동 차단은 **기준 하한을 초과하는 새 흥분을 전달하지 않는다**는 의미다. 이미 남은 활성도의 감쇠나 수동 힘까지 0이 된다는 의미가 아니다. 실제 SI 힘·토크로의 변환은 `null`, 미교정으로 기록한다.

**추가 교정 필요:** 강한 신근 조건의 저장 기록에서 허용 관절 범위 안의 모멘트암 부호 반전을 검출했다. 첫 관측은 83ms, q≈0.608rad이고 신근 모멘트암은 −0.000503mm다. 총 10개 표본에서 음수였으며 최소값은 −0.001775mm였다. 따라서 이 경로를 전체 관절 범위에서 일정한 신전 작용으로 가정할 수 없다. 힘줄 경로·좌표·생리적 사용 범위를 재검토해야 한다. 사전에 고정한 기술 gate는 변경하지 않고 [별도 재평가](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/muscle_assay/reassessment.json)에 `CALIBRATION_REQUIRED`로 기록했다. 검증기에 이 현상의 자동 보고를 추가한 뒤 새 `muscle_assay_checked` 폴더에서 재실행했다. 9조건의 trace는 첫 실행과 바이트 단위로 같고, 기술 PASS와 생리 교정 필요 상태를 분리해 출력한다.

새 [receptors.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/receptors.py)는 양/음 각도·속도, 속도 고주파 성분, 지지 하중, 비지지 접촉을 분리하고 저역 필터·명시적 지연·차단·저장/복원을 제공한다. 입력과 raw/filtered/output을 모두 저장한다. 이 값들은 **0~1의 공학적 수용체 proxy**다. 실제 BANC 뉴런 ID를 임의로 연결하지 않았으며 근육 장치의 q 부호를 생물학적 flexion/extension 세포형에 자동 대응하지 않았다. 정지·지속 운동·진동·반대 방향·하중·차단을 별도로 검사했다.

[9조건 원자료](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/muscle_assay_checked/report.json) · [반응 그래프](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/muscle_assay/response.png)

## 4. 검증 결과의 정확한 범위

| 검증 | 결과 | 해석 |
|---|---|---|
| 최종 로컬 회귀 | **245/245 PASS, skip 0** | Torch CPU, 실제 Apple MPS, 설치된 FlyGym/MuJoCo 포함 |
| Torch import 차단 조건 | **222 PASS + 23 SKIP / 245, error 0** | 동일 macOS에서 선택 의존성 부재를 모사. 별도 Linux 설치 검증은 아님 |
| 연구 tier + Torch 부재 | **BLOCKED, exit 2** | 필수 연구 검사를 skip 성공으로 바꾸지 않음 |
| 연구/native/MPS preflight | 모두 PASS | 실제 로컬 runtime 기준 |
| FAFB 전체 스냅샷 계산 | membership·compute gate PASS | 139,255개, 500tick/0.05모델초, 실제 Metal MPS, v6 reset-current 파라미터 |
| BANC 실제 몸 연결 | **6/6 조건, 7개 기술 gate PASS** | 감각/감각 차단/운동 차단/굴근/신근/DNg100, 각 0.3초. 42관절 힘 기록 포함 |
| 별도 근육 장치 | **9조건, 9개 기술 gate PASS** | 합계 0.9모델초 + 복원 비교 0.018모델초. 접촉·보행 검증 아님 |
| 0.6→0.7 실제 저장 상태 | FAFB/BANC 모두 정확한 연속 계산 | 20ms씩, 버전 외 전체 상태 일치 |
| 브라우저 | FAFB/BANC 모드·진행·저장·복원 확인, console error/warning 0 | 별도 8774/8775 서버. 기존 8772/8773 실험은 계속 유지 |

같은 검사를 여러 번 실행한 숫자를 더하지 않는다. 근육 9조건과 BANC 6조건은 서로 다른 몸과 승인 범위다. 검증기의 최상위 `COMPLETE_WITH_LIMITATIONS`를 전체 생물학 PASS로 바꾸지 않는다.

초기 검사·실패한 검증 보조 스크립트 로그도 보존했다. 기존 checkpoint 비교의 첫 보조 스크립트는 존재하지 않는 `decoder` 저장 키를 읽어 실패했으며, 전체 checkpoint를 비교하도록 고친 후 새 폴더에서 재실행했다. 최초 screenshot 보조 명령의 CLI 문법 오류도 로그에 남아 있다. 이는 앱 console 오류와 구별한다.

BANC 물리 실행 후 제품 소스의 유일한 변경은 독립 ResearchLIF의 극단적 시간상수 계수 보호였다. CEngine의 ExpLIF/MetalLIF 실행 경로에는 사용되지 않는다. 물리 실행 시점의 소스도 원자료에 보존하고 [소스 차이](/Users/chasoik/Projects/FLY_LAB/verification/c07_implementation_20260910/post_native_source_changes.json)를 남겼다. 최종 회귀는 이 마지막 수정을 포함한다.

## 5. 실행 방법

검사 결과 폴더는 기존 결과를 덮어쓰지 않도록 새 이름을 사용한다.

```sh
# 기본 C 의존성 / 선택 연구 의존성
.venv/bin/python -m pip install -r requirements-c.txt
.venv/bin/python -m pip install -r requirements-research.txt

.venv/bin/python tools/run_c_checks.py --tier regression --out verification/my-c07-check
.venv/bin/python tools/run_c_checks.py --tier mps --out verification/my-c07-mps-check
.venv/bin/python tools/verify_c_muscles.py --seconds .1 --out verification/my-c07-muscles

# BANC 프로파일에는 strict 18조건 pilot을 생성한다. 물리 실행은 하지 않는다.
.venv/bin/python tools/run_c_campaign.py \
  --graph data/acquisitions/banc888-v2-20260909/bundle \
  --bindings data/banc888/bindings-neuromuscular-v2.json \
  --out verification/my-banc-campaign \
  --write-pilot verification/my-banc-pilot.json
```

현재 새 앱: [FAFB C0.7](http://127.0.0.1:8774/) · [BANC C0.7](http://127.0.0.1:8775/). 둘 다 실제 MPS/물리를 사용하고, 검증 후 저장한 0.050모델초 상태로 일시 정지돼 있다. 프로파일 선택에 따라 지원하는 모드만 활성화한다.

## 6. 다음 개발의 선행 조건

1. **힘·좌표 교정과 뉴런 연결:** 약 0.61rad에서 확인한 신근 모멘트암 부호 반전의 기하학적 원인과 생리적 사용 범위를 먼저 검토한다. SI 힘/토크 변환, 실제 측정 근육 파라미터, 새 Fe–Ti 관절과 BANC 근육/감각 ID의 대응 근거를 확보한다. 42관절 목표각을 Hill 힘으로 일괄 교체하지 않는다.
2. **국소 신경 폐루프:** 검토한 한 관절에 rate→activation→force→movement→receptor→neuron을 연결하고 피드백 차단·극성 교환·지연·외력 및 미사용 자세에서 검사한다. 현재 독립 proxy와 두 근육 실험만으로 이 단계가 완료되지는 않았다.
3. **보행과 자연 행동:** 국소 교정 뒤 한 다리→양측→6다리로 확대하고 기존 BANC 10초/5mm·고착·회복 기준을 유지한다. 좌우 고정 감각 probe와 자연 먹이·정지·위험·장애물 대조를 이어간다.
4. **내부 상태·학습:** 공통 API를 기반으로 ResearchLIF를 CEngine에 통합하고 섭취/냄새/영양 스키마, 지연 보상·eligibility, 신경조절 대상과 행동 대조를 별도 검증한다.

후속 순서와 승인 기준은 [C07 상세 계획](/Users/chasoik/Projects/FLY_LAB/docs/C07_DEVELOPMENT_PLAN_20260910.md)에 유지했다. 기존 C0.6의 보행·양방향 접근 실패 원자료는 이번 기술 검사 통과로 대체하지 않았다.
