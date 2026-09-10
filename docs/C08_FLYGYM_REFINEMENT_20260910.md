# C0.8 개발 계획 보강: FlyGym 공식 구현과의 대조

작성일: 2026-09-10 · 상태: **소스 조사·계획 반영 완료 / 구현·실험 미실행**

**FlyGym에서 우선 가져올 것은 운동학 자료, 근육 구동 대조군, 접촉·시각 관측 API, 물리 단계별 계측이다.** 이들을 이용해 현재의 감각→운동 실패가 신경 입력, 운동 출력 변환, 몸의 역학 중 어디서 발생하는지 분리한다. 버전 교체나 GPU 전환만으로 실패한 보행이 해결된다는 근거는 없다.

이 문서는 [기존 C0.8 계획](/Users/chasoik/Projects/FLY_LAB/docs/C08_DEVELOPMENT_PLAN_20260910.md)의 M1–M6를 보강한다. 기존 실패 판정, 보행·피드백 기준, 확인용 데이터 분리 원칙을 유지한다. 아래 구현 위치와 실행 묶음은 후속 개발 항목이다.

## 1. 확인한 버전과 증거 범위

| 항목 | 확인 결과 |
|---|---|
| 로컬 FLY_LAB | HEAD `45309118cb4ae3c452bad289a729efdc15cfa176` + 별도 진행 중인 미커밋 변경 |
| 설치된 FlyGym | **2.1.0**, commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`. requirements와 설치 메타데이터 일치 |
| 조사 시 upstream main | `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`, 2026-06-28 UTC 커밋 |
| 두 커밋의 차이 | upstream이 7개 커밋 앞섬. 변경 18개 파일은 문서·뷰어·개발용 자산 생성·노트북이며 핵심 Python 구현 변경 없음 |
| 직접 파일 대조 | 57개 소스·문서 등을 보존. 그중 `src/` Python 48개가 설치본과 바이트 일치. 현재 core `src/flygym/` Python 38개 전체 포함 |
| 구형 공식 예제 | `NeLy-EPFL/flygym-gymnasium`, commit `d285260a1c8a7b3494150cd1590f2c9fe4b5e06b`의 후각·경로 적분 관련 6개 파일 확인 |
| 이번에 실행한 것 | 원격 소스 취득, 설치 파일 대조, 문서·JSON 점검. 신경 계산·물리·학습·브라우저 실험은 실행하지 않음 |

근거: [공식 커밋 비교](https://github.com/NeLy-EPFL/flygym/compare/ca65a510c2afe6ac61c51df4f274c8d190c2f95f...38c8ec61034cd59bc5ba0de20688d4a3c0000d60), [로컬 의존성](/Users/chasoik/Projects/FLY_LAB/requirements.txt), [이번 조사 근거 목록](/Users/chasoik/Projects/FLY_LAB/docs/C08_FLYGYM_EVIDENCE_20260910.json).

따라서 FlyGym 2.x의 성능 개선을 앞으로 받을 개선분으로 계산하지 않는다. 조사 결과만으로 pin을 변경할 이유도 없다. 실제 실행 비용과 모델 지원 범위를 계측한 뒤 필요한 변경만 선택한다.

별도 진행 중인 코드는 운동단위별 포화 응답 합산, 발 들림 예상에 따른 접착 해제, 네 프로파일 비교 도구를 포함한다. `EXPERIMENTAL_NOT_VALIDATED` 상태이며 여전히 관절 **위치 목표**를 만드는 후보다. 이 조사에서는 해당 코드의 검증 결과를 판정하거나 수정하지 않았다. 기존 계획의 수치는 원래 기준선의 저장 결과로 유지한다. [후보 코드](/Users/chasoik/Projects/FLY_LAB/flylab/c/neuromuscular.py:130), [비교 도구](/Users/chasoik/Projects/FLY_LAB/tools/compare_c_banc_gait.py:28).

## 2. 보강된 개발 우선순위

| 순서 | 기존 단계·문제 | 추가할 작업 | 다음 결정을 위한 산출물 |
|---|---|---|---|
| 1 | M1 / U14 성능 | 정상 실행 구간의 native profile와 MuJoCo 단계별 시간 분해 | Python 제출, MPS 계산·대기, 충돌·제약 해법, 관측·기록의 비용표 |
| 2 | M2·M3 / U01–U06 | 기록 운동학의 관절명·축·좌표 대조, 같은 장치의 기계 교사 구동 | 신경 없이 가능한 운동 범위와 BANC 출력의 부족 위치 |
| 3 | M2 / U01–U03 | 위 결과와 q/v replay를 이용한 감각→전류→운동단위 진단 | 기존 18/72mV 실패를 설명하는 원인별 후보 |
| 4 | M3 / U04–U06 | 접촉 좌표계·바닥 지지·접착 분리, 현재 v3 후보의 요소별 대조 | 접착 고착과 운동 부족의 구분, 한 다리 접촉 검증 |
| 5 | M4 / U07–U09 | 실제 복안 관측 및 다지점 후각장을 먼저 관측 전용으로 통합 | 좌우 감각의 단위·시계·가시성 검증 후 실제 ID 연결 |
| 6 | M3–M6 | 미사용 지형, 시간변화 냄새, 보행 감각 기반 경로 적분 | 일반화 범위와 신경 대조 결과 |
| 조건부 | M1·M3 | NVIDIA 장비에서 CUDA 물리 지원·수치·처리량 검증 | 단일 세계 지연과 병렬 처리량을 분리한 채택 판단 |

FlyGym의 CPG/반사 제어기는 기계적인 보행 가능성을 확인하는 대조군으로 사용한다. 이미 FAFB 경로에 있는 HybridTurningController를 새로 구현된 신경 회로로 세지 않는다. 기계 교사 구동과 전체 커넥톰 구동은 서로 다른 실험 조건으로 기록한다.

## 3. M1: CPU 잔여 비용의 원인부터 좁히기

공식 profiling 안내는 초기 import·모델 생성·warm-up을 제외한 실행 구간을 native stack으로 측정하고, `mjcb_time`과 `mjData.timer`로 충돌·제약 해법·적분 비용을 분해한다. 내부 timer는 callback 비용을 추가하므로 절대 처리율 측정과 분리해야 한다. [공식 profiling 문서](https://neuromechfly.org/tutorials/7_performance_profiling/).

개발 항목은 다음과 같다.

1. [profile_c_runtime.py](/Users/chasoik/Projects/FLY_LAB/tools/profile_c_runtime.py)에 워밍업 이후 구간 선택과 MuJoCo 단계별 집계를 추가한다. callback은 별도 실행 프로세스에서 사용하고 종료 시 원래 callback을 복원한다. Mac에서 native stack 수집이 불가능하면 timer와 기존 구간 계측의 결과만 보고한다.
2. 기존 신경/몸 합계에서 충돌 탐색, 접촉·제약 풀이, 적분, Python 제어, 장치 제출·동기화, 센서, JSON·기록을 구분한다. 신경과 물리가 겹쳐 실행되는 구간의 시간은 단순 합산하지 않는다.
3. 전체 뇌+몸, 몸+기록된 명령, 신경+기록된 감각을 진단용으로 비교한다. 분리 실행의 빠른 처리율을 전체 폐루프의 처리율로 표시하지 않는다.
4. 채택 속도는 계측을 끈 기존 5워크로드에서 다시 측정한다. 모델 dt·접촉 margin·solver iteration·noslip 변경은 보존 최적화에 포함하지 않는다. 관측 주기 변경도 감각 입력까지 바꾸면 별도 모델 후보가 된다.

M1의 기존 동등성 검사와 10% 개선/다른 조건 5% 악화 제한을 유지한다. Python이 지배하면 호출·버퍼·집계를, 물리 제약 풀이가 지배하면 실제 접촉 구성과 모델 비용을 먼저 검토한다. 원인이 확인되기 전 추가 MPS 이식이나 물리 dt 증가는 우선 작업이 아니다.

### CUDA 경로의 정확한 위치

| 경로 | 조사로 확인한 지원 | 계획상의 취급 |
|---|---|---|
| 현재 Mac | CPU MuJoCo + 기존 MPS 신경 계산 | 주 개발 환경 유지 |
| FlyGym `GPUSimulation` | NVIDIA CUDA 장치를 검사하는 Warp 경로 | Apple MPS 물리 구현으로 사용할 수 없음 |
| 근육 모델 호환성 검사 | `mjw.put_model` 수용 여부 확인 | 실제 적분·힘·접촉 안정성 통과와 별개 |
| 대량 병렬 물리 | 공유 모델과 여러 세계의 배치 실행 | 단일 사용자 화면의 지연과 별도 성능 지표 |

특히 GPU 생성 경로는 지원하지 않는 `noslip_iterations`가 양수면 0으로 변경한다. 적용 전후 모델 옵션 diff를 남기고, 원래 값이 이미 0인지 확인해야 한다. 옵션이 바뀌는 후보를 기존 물리와 같은 모델로 간주하지 않는다. [CUDA 장치 검사](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/warp/utils.py#L195), [옵션 변경 구현](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/warp/simulation.py#L433), [근육 모델 호환성 검사](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/compose/fly/musculoskeletal.py#L466).

향후 해당 장비를 사용할 때만 compile → 실제 step → 힘·접촉·복원 대조 → 배치 확장 순으로 확인한다. 배치는 1/16/64 세계를 상한 후보로 두되 전뇌 상태의 메모리를 먼저 계산한다. 공식 README의 약 10배 CPU/300배 GPU 성능 수치는 이 앱의 전뇌 단일 세계 속도 예측에 사용하지 않는다. [공식 저장소](https://github.com/NeLy-EPFL/flygym), [GPU 튜토리얼](https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/).

## 4. M2: 운동학·기계 교사·BANC의 세 가지 대조

### 4.1 관절 의미와 시간축을 먼저 검증

Spotlight `MotionSnippet`은 6다리×7 DOF의 기록 운동학과 관절·다리 이름, ego-frame keypoint, fps, 원래 trial/frame 범위를 제공한다. 로더는 오른쪽 다리의 roll/yaw 부호를 해부학적 규약으로 **한 번** 변환한다. 현재 모델의 좌우 문제를 진단할 때 유용한 참조이지만, 변환 규약이 이미 적용된 배열에 다시 부호를 바꾸면 오류가 된다. [공식 운동학 로더](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym_demo/spotlight_data/preprocessing.py#L11).

- 숫자 인덱스 대신 다리·부모/자식 link·회전축으로 데이터와 모델을 연결한다. 각 DOF에 작은 ±변위를 주어 FK/Jacobian과 발 끝 방향을 확인하고 변환 전후 해시를 기록한다.
- 몸 원점 `xpos`, 질량 중심 `xipos`, ego/world 좌표를 구분한다. 서로 다른 신체 모델의 keypoint를 그대로 같은 좌표라고 비교하지 않는다.
- 데이터 fps, 필터, 보간법, 끝점, 지연을 고정한다. 공식 모방학습은 기본 2ms 제어이고 현재 단일 관절은 1ms 제어·10µs 물리다. clip frame 증가와 실제 모델 시각을 맞추는 adapter가 필요하다.
- 현재 사용한 FlyMimic `0002` clip은 탐색 자료다. 새로운 확인 자료로 재사용하지 않는다. 7 DOF clip의 Fe–Ti 열만 사용하는 시험과 7 DOF 전체 추적을 따로 기록한다.

### 4.2 같은 물리 장치에 다른 구동기를 연결

공식 `ImitationEnv`는 근육 활성도를 action으로 받고 관절·몸 위치 및 속도 추적 오차를 보상에 사용한다. observation에 clip 잔여 시간도 포함한다. 이 학습은 목표 운동을 알고 있는 공학 제어기 훈련이며 실제 BANC 신경 계산이나 초파리의 보상학습 구현은 아니다. [공식 모방학습 환경](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym_demo/muscle_imitation/env.py#L36).

| 대조 | 허용 입력·구동 | 확인할 것 |
|---|---|---|
| 운동학 기준 | 기록된 목표 q/v. 강제로 q를 설정하면 운동학 전용으로 표기 | 축·기하·가동 범위, 요구 토크의 크기·부호 |
| 기계 교사 | 같은 몸의 Hill 활성도에 목표 추적 제어를 적용 | 이 근육·힘줄·수동 역학으로 목표 반응을 만들 수 있는가 |
| BANC | 동일 몸·외력·초기 상태, 감각→실제 ID→운동단위 출력 | 기계적으로 가능한 반응을 감각 회로가 모집하는가 |

먼저 현재 **2 MTU/1 DOF 장치**에서 기계 교사를 만든다. 공식 15 MTU/LF 다리의 성공을 2 MTU 장치의 대조 결과로 대신하지 않는다. 15 MTU 장치로 확장할 때는 새 기준선과 동일 장치의 BANC 연결을 만든다. 공식 근육 몸은 현재 LF만 근육 구동한다. [근육 몸 소스](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/compose/fly/musculoskeletal.py#L1), [공식 근육 튜토리얼](https://neuromechfly.org/tutorials/6_muscle_imitation/).

교사와 BANC가 모두 실패하면 기하·수동 역학·모집식·교사 설계를 먼저 점검한다. 이것만으로 해당 몸이 원리적으로 제어 불가능하다고 단정하지 않는다. 교사는 성공하고 BANC만 실패하면 FeCO 파형, presynaptic 전류, 느린/빠른 운동단위 모집을 우선 진단한다. 신경 출력 차단 후에도 같은 반응이면 수동 역학 또는 교사 개입의 영향이다.

초기 교사는 제한된 목표 추적 제어로 시작한다. 대규모 PPO 훈련은 우선순위에서 제외한다. 공식 reward 증가나 clip 재생만으로 피드백 성공을 선언하지 않고, 기존 양 외력 방향·미사용 자세·IAE 20% 개선 기준을 별도로 적용한다. q≈0.61344rad의 모멘트암 반전 guard를 유지하며 guard로 잘린 성공 구간만 골라 평가하지 않는다.

수정 대상: [single_joint.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/single_joint.py), [muscle_calibration.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/muscle_calibration.py), [probe_c_joint_feedback.py](/Users/chasoik/Projects/FLY_LAB/tools/probe_c_joint_feedback.py). 기록 q/v 재생과 목표 제어는 기존 장치·저장 경로의 모드로 추가한다.

## 5. M3: 접촉·접착을 검증한 뒤 지형을 확장

공식 `get_ground_contact_info`의 힘·토크는 **contact frame**, 접촉점·법선·접선은 world frame이다. `get_bodysegment_contact_forces`는 world frame의 합력이다. `ground_only`는 등록된 ground geom을 뜻하므로 수평 바닥만을 뜻하지 않는다. [공식 접촉 API](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/simulation.py#L223).

1. 현재 [body.py](/Users/chasoik/Projects/FLY_LAB/flylab/body.py:156)의 캐시된 접촉 집계를 유지하고 공식 API를 독립 비교값으로 사용한다. 같은 MuJoCo 상태에서 접촉쌍별 힘 방향, 벽 접촉, 바닥 지지, 접착 제어 입력을 대조한다. 힘의 norm 하나로 체중 지지와 벽 충돌을 합치지 않는다.
2. 현재 v3의 운동단위 합산과 접착 해제를 **v2/v3 합산 × v2/v3 접착의 네 조건**으로 비교한다. 이미 작성 중인 [compare_c_banc_gait.py](/Users/chasoik/Projects/FLY_LAB/tools/compare_c_banc_gait.py)를 활용한다. 실제 발 clearance, 접촉 지속시간, 미끄럼, 접착 명령과 적용 여부, 기계 출력과 신경 발화를 함께 저장한다.
3. v3의 `target_lift_mm`는 관절 Jacobian으로 예측한 변위이며 실제 미래 발 높이가 아니다. 현재 +z 바닥 법선 가정은 경사면·벽에 그대로 적용하지 않는다. 해당 지형 지원을 추가할 때 접촉면 법선과 접촉 집합의 시간 변화부터 정의한다.
4. 한 다리 접촉 → 양측 → 6다리 평면 보행을 통과한 뒤 작은 블록·틈 → 혼합 지형으로 확장한다. 위상 관계, stance/swing 비율, 발 미끄럼과 몸 속도도 기록 운동학과 대조한다. 기계 교사/CPG의 지형 통과와 BANC의 통과를 따로 기록한다.

FlyGym의 `GappedTerrainWorld`, `BlocksTerrainWorld`, `MixedTerrainWorld`는 사용할 수 있지만, 여러 geom으로 구성된 지형은 요청하더라도 기존 다리별 ground contact sensor를 자동 추가하지 않는다. 이 경우 contact 목록이나 bodysegment API로 감각을 구성해야 한다. 이는 예제를 그대로 교체할 때 생길 수 있는 구체적인 통합 누락이다. [지형 구현과 제한](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/compose/world/complex_terrain.py#L16).

평면 보행의 기존 10초·순변위 5mm·전방 이동 5mm·고착 없음 기준은 유지한다. 지형 난이도와 미사용 seed는 평면 통과 뒤 고정하며, 지형 추가를 기존 BANC 평면 실패의 해결로 계산하지 않는다.

## 6. M4: 복안 API를 실제 감각 경로로 연결

FlyGym은 눈 카메라를 구성한 몸에 대해 `get_raw_vision`과 `(2, n_ommatidia, 2)` 형태의 yellow/pale readout을 제공한다. 색 유형에 맞지 않는 채널의 0은 유형 mask이므로 어두운 픽셀·발화 없음과 구분해야 한다. 렌더러는 기본적으로 geom group 1과 2를 가린다. [공식 시각 API](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/simulation.py#L408).

현재 [body.py](/Users/chasoik/Projects/FLY_LAB/flylab/body.py:117)는 ray에서 몸을 제외하려고 몸 geom 전체를 group 2로 두고 있다. 그대로 눈 렌더러를 붙이면 원래의 선택적 자기 가림 설정과 달라진다. 이는 현재 시각 오류를 재현했다는 뜻이 아니라, **시각 API 추가 시 해결해야 할 설정 충돌**이다.

- **첫 구현:** 모델 compile 전에 눈 카메라를 추가하고 ray 제외용 그룹과 눈의 자기 가림을 분리한다. 좌·우 눈, 시야 방향, mm/rad, 색 유형 mask를 검증한다. 사용자 관찰 카메라 영상은 감각 영상과 구분한다.
- **관측 전용 단계:** 같은 물리 상태에서 원시 영상·ommatidia·단순 움직임 특징을 저장한다. 이 단계에서는 기존 감각 입력을 유지해 관측 호출 유무가 신경·물리 상태를 바꾸지 않는지 확인한다.
- **시간 계약:** 후보 100/200Hz 중 처리 비용과 자극 시간 분해능을 보고 한 값을 먼저 고정한다. 물리/제어 tick의 정수 배수로 샘플링하고 새 영상 도착 전에는 이전 표본을 유지한다. 마지막 영상 시각·필터·optical history·다음 sample tick을 checkpoint에 포함한다.
- **신경 연결:** 현재 64개 ray의 좌우 coverage proxy와 실제 복안을 별도 감각 profile로 둔다. 광수용·시각 회로의 실제 ID와 retinotopy 근거를 확보한 범위부터 연결한다. ommatidium 인덱스를 곧바로 뉴런 ID로 대응시키거나 영상 전체를 LC4 균일 자극으로 바꾸지 않는다.
- **확인:** 정지/움직이는 표적, 좌우 교환, 몸 회전, 가림, looming 속도를 교차해 입력 변화와 해당 회로 반응을 기록한다. 이후 유효한 장애물 노출 조건에서 시각 정상/절단 대조를 수행한다.

첫 통합은 현재 NeuroMechFly 몸에서 진행한다. 근육용 FlyMimic 몸에도 눈을 붙일 수 있으나 공식 구현은 그 몸에서 Retina의 보정을 근사라고 명시한다. 몸을 바꿀 때 eye placement 보정을 다시 검증한다. [근육 몸의 시각 제한](https://github.com/NeLy-EPFL/flygym/blob/38c8ec61034cd59bc5ba0de20688d4a3c0000d60/src/flygym/compose/fly/musculoskeletal.py#L339).

수정 대상: [body.py](/Users/chasoik/Projects/FLY_LAB/flylab/body.py), [sensors.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/sensors.py), [ports.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/ports.py), [pathways.py](/Users/chasoik/Projects/FLY_LAB/flylab/c/pathways.py). 세포 매핑이 없는 단계는 영상 관측 구현까지 완료했다고 보고한다.

## 7. M4·M5: 후각은 구형 예제의 환경 계약을 이식

현재 2.x core Python 전체에서 전용 후각 API는 확인되지 않았다. 공식 구형 패키지의 `OdorArena`는 두 antenna·두 maxillary palp 위치에서 여러 냄새 성분을 계산하며 기본장은 거리의 역제곱이다. plume 예제는 별도 HDF5 냄새장과 모델 시각을 연결한다. 현재 FLY_LAB의 두 고정 상대 위치·Gaussian 냄새장과는 다른 모델이다. [공식 OdorArena](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/arena/sensory_environment.py#L68), [공식 plume 환경](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/examples/olfaction/plume_tracking_arena.py#L11).

1. 설치 패키지를 구형 Gymnasium으로 되돌리지 않고, 현재 `sensors.py`의 감각 생성 경계에 `농도 = field(모델시각, 기관의 world 좌표, odor 종류)` 계약을 추가한다. 실제 site와 근거 없는 위치 offset을 구분한다.
2. 기존 Gaussian은 그대로 기준선으로 보존한다. 다지점·다성분 정적장부터 비교하고, 그다음 단일 고정 plume 자료를 replay한다. 냄새장의 grid/mm, fps, 보간, 원점, 영역 밖 처리, 끝 프레임 처리와 해시를 기록한다.
3. 원천과 센서가 같은 위치일 때의 발산, 농도 0에서 정규화, NaN/경계·자료 종료를 명시적으로 처리한다. clamp/바닥값을 추가한다면 새 모델 가정과 변화량을 기록한다. 이를 생체 농도 단위라고 표시하지 않는다.
4. 기관 농도→수용체 transduction→실제 후각 ID→회로의 중간 기록을 남긴다. 냄새 종류와 영양량·재고를 분리하는 M5 음식 스키마에 연결한다. 양측 후각 평균만으로 모든 기관의 개별 기능을 구현했다고 하지 않는다.
5. 공식 좌우 농도 기반 odor taxis와 냄새 접촉 이력 기반 plume 제어기는 환경과 기계 구동의 대조군으로 활용한다. 후자는 heading 정보도 사용하므로 공학 대조군의 입력 권한을 기록한다. BANC에 실제 세계 방향·목표 좌표를 우회 전달하지 않는다. [odor taxis 제어](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/examples/olfaction/simple_odor_taxis.py#L127), [plume 제어](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/examples/olfaction/plume_tracking_controller.py#L134).

기존 왼쪽 편향 원인 분석을 먼저 수행한다. 감각장을 복잡하게 바꿔 편향을 감추지 않는다. 새 장에서의 행동 검증도 기존 M4의 좌우·감각 정상/절단/교환과 먹이 접근 후 고착 검사를 유지한다. 공식 예제는 수용체 생리, 신경조절, 허기, 섭취 동작, 지연 보상 학습을 자동으로 제공하지 않는다.

## 8. M6: 경로 적분은 감각 가능성 대조부터

구형 공식 경로 적분은 다리의 몸 기준 이동과 접촉 정보를 사용해 방향·이동량을 선형 모델로 추정한다. 초기 방향과 위치는 실제 궤적에서 정렬한다. 중앙복합체의 실제 신경 연결망이나 자율 지도 구현으로 사용할 수는 없다. [stride 관측](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/examples/path_integration/controller.py#L137), [선형 적분 모델](https://github.com/NeLy-EPFL/flygym-gymnasium/blob/d285260a1c8a7b3494150cd1590f2c9fe4b5e06b/flygym_gymnasium/examples/path_integration/model.py#L49).

평면 보행이 성립한 뒤 접촉과 고유감각만으로 방향·거리 변화가 얼마나 추정 가능한지 먼저 측정한다. 선형 모델은 감각의 정보량을 확인하는 기준 대조군으로 두며 최적 성능의 상한이라고 가정하지 않는다. 초기 자세 정렬만 허용하는 조건과 감각으로 초기 방향을 잡는 조건을 구분하고, 이후 실제 위치를 재주입하지 않는다. 계수 추정 trial과 확인 trial을 나누고 시각 차폐·미끄럼·방향 전환별 누적 오차를 기록한다.

이 결과를 중앙복합체 경로의 신경 모델과 비교한다. 그전에 실제 위치를 신경 상태에 저장해 공간 기억을 구현했다고 판정하지 않는다. 이번 FlyGym 조사로 기존 비행 rig의 신경 제어·안정성·이착륙 미검증 상태가 달라진 것은 없다.

## 9. 추가 작업의 범위·검증·중단 조건

아래는 **계획상 추가 상한**이다. 기존 M2의 E2–E4, M4 확인 72회·2,160모델초, M5 확인용 seed 계획을 자동 증액하지 않는다. 동일 자료를 재사용할 때는 한 실행으로 계수하며 후보 선택에 노출된 자료를 확인용으로 세지 않는다.

| 묶음 | 구현·진단 범위 | 제안 상한과 판단 |
|---|---|---|
| FG1 계측 | 기존 5워크로드, warm-up 분리와 MuJoCo phase 계측 | 계측 켜짐/꺼짐 각 0.3모델초 1회, 총 3모델초의 진단. 개선 채택은 기존 M1의 교대 3쌍으로 별도 판정 |
| FG2 축·기계 교사 | 6×7 DOF의 ±FK 대조 후 현재 단일 관절에서 교사/수동/BANC 비교 | FK 최대 84점. 이미 사용한 자세 2개×목표 방향 2개×구동 3조건×외력 유/무 = 24회×0.3초, 총 7.2모델초. 교사 추가 튜닝은 이 결과 뒤 별도 가설·상한 등록 |
| FG3 접촉·v3 | 공식/로컬 집계 동일 상태 대조, 진행 중인 네 후보 비교에 관측 추가 | 기존 v3 실행 프로토콜을 유지. 추가 접촉 준비 시험은 바닥/벽/모서리×접착 켜짐/꺼짐×0.3초 = 1.8모델초. 새 지형에서는 별도 승격 필요 |
| FG4 복안 | 관측 전용 눈·mask·좌표·시간·복원 확인 | 좌우·가림·회전 등을 포함해 최대 12개 고정 장면. 운동학 replay는 최대 2개×0.5초. 영상 신경 연결 후 행동은 기존 M4 묶음으로 진행 |
| FG5 후각 | 다지점 정적장·단일 plume의 수치·시계·복원 | 몸을 움직이지 않는 위치/시각 최대 24조합. 폐루프 행동은 기존 M4 기준과 자원 상한 안에서 별도 고정 |
| FG6 경로 적분 | 이미 확보된 유효 보행 궤적의 오프라인 정보량 평가 | 최대 6개 서로 다른 trial, 추정/확인 분리. 부족하면 일반화 미판정. 새 장시간 보행은 먼저 실행하지 않음 |

실행은 전뇌 MPS 한 프로세스와 묶음당 최초 60분 벽시계 상한을 우선 유지한다. 자료/장비 부족, 모델 변환 실패, 실제 계산 실패, 가설 미지지, 행동 기준 미달을 구분한다. 신경 차단 대조, 운동학·힘 범위, 저장·복원이 실패하면 후속 행동 행렬을 시작하지 않는다.

추가 기록에는 FlyGym commit, 설치 코드 hash, 데이터 trial/frame, body/actuator/접촉 옵션, 참조 좌표, 교사/신경 구동 구분, 눈·냄새 sample clock을 포함한다. 새 문서와 코드의 존재가 구현 완료를 뜻하지 않으며, 결과는 실제 실행 ID와 연결한 뒤 갱신한다.

최초 개발 산출물은 **물리 비용을 분해한 계측기, 관절·좌표 대조기, 같은 2 MTU 장치의 기계 교사와 BANC를 비교하는 진단 모드**다. 그 결과를 바탕으로 기존 M2 원인별 수정과 M3 접촉 확장을 진행한다.
