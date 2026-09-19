**FlyLab 비행 구현 계획 — GPT Pro Web 응답 대조 후**

2026-09-12. 이 문서는 [외부 분석 답변](https://chatgpt.com/c/6aa556af-3dfc-83e8-a1d9-1deea3c3ceda)을 소스와 로컬 환경에 대조한 구현 계획이다. 기존 `FLIGHT_IMPLEMENTATION_PLAN_DRAFT_20260912.md`를 대체한다. 이번 작업은 분석과 계획 작성이며, 제품 코드 수정·패키지 설치·학습 정책 실행은 수행하지 않았다.

권장안은 **원본 TuragaLab FlyBody의 물리 환경과 학습 정책을 같은 별도 worker에서 실행하고, FlyLab에는 독립 `FlightSession`으로 연결하는 것**이다. 먼저 원래 조건에서 정책을 재현하고, 실제 날개 표시와 지속 비행을 완성한다. 이후 조향·시각 자율 비행으로 확장하고, 다리 접촉이 있는 하나의 물리 몸에서 이착륙과 걷기 전환을 검증한다. BANC는 먼저 상위 조향 연결을 검토하고, 직접 날개 제어는 별도의 신경 연결 연구로 진행한다.

공식 [FlyBody 저장소](https://github.com/TuragaLab/flybody)는 비행 모방과 시각 비행 예시, 정책/데이터 및 관련 노트북을 제공한다. 로컬에서는 원본 소스 commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26`을 확인했다. 공개 예시가 존재한다는 사실과 현재 Windows에서 학습 정책이 재현되었다는 주장은 구분한다. 후자는 P1에서 검증해야 한다.

**현재 구현과 채택 판단**

| 경로 | 확인된 기능 | 구현 계획에서의 용도 |
|---|---|---|
| 일반 웹 화면/보행 | `src/ui/world-view.js:50`의 도식 날개는 정적이다. 일반 보행 모델은 날개 구동 경로가 없다 | 기존 보행 유지, 비행 화면 추가 |
| FlyGym 전신 FlyBody | `flylab/body_options.py:79`의 POSITION 서보로 날개 6축을 구동할 수 있다. 기존 50 ms 관절 실험에서 최대 목표 오차 약 0.000957 rad | 관절/자세 실험. 학습 비행과 물리적으로 동일하다고 보지 않음 |
| `flylab/c/flight.py::FlightRig` | 원본 XML과 ellipsoid 유체력을 사용하는 독립 실험. 유체 on/off 궤적 차이 및 자체 저장·복원 검증 있음 | 공기력·디버깅 대조 실험. 공식 정책 기준선에는 사용하지 않음 |
| `flylab/c/flybody_flight.py::make_flight_env` | 원본 Flying/WPG를 쓰는 66차원 자체 과제. 다리와 지면 접촉 활성. CLI는 0 residual 입력 | 접촉 과제 실험 출발점. 안정 비행/이착륙 제어기는 아직 없음 |
| C 서버/엔진 | `flylab/c/server.py:104`는 `flight=False`; 기존 C 시계와 상태 계약은 보행 중심 | 비행 세션용 facade를 추가하고 고속 적분은 worker 소유 |

기존 0.3초 실험의 flight/takeoff/landing은 모두 최종 success가 false였다. takeoff는 약 0.1472–0.1556초에만 기존 판정이 true였고, 최종 upright는 -0.399였다. 해당 실험은 이번 외부 검토 전에 로컬에서 실행한 자료이며, 이번 검토에서 새로 반복 실행한 결과가 아니다. 근거: [초기 조사](C:/Users/ckthd/Dev/Projects/fly_lab/docs/FLIGHT_IMPLEMENTATION_AUDIT_20260912.md), [실행 요약](C:/Users/ckthd/Dev/Projects/fly_lab/docs/FLIGHT_ANALYSIS_EVIDENCE_20260912.json).

| 선택지 | 장점 | 주요 비용과 채택 조건 |
|---|---|---|
| 원본 물리 + 공식 정책 + 독립 worker | 원래 관측/구동/WPG를 유지하여 실패 원인을 분리하기 쉽다 | 별도 의존성 환경, IPC와 상태 저장 필요. 우선 채택 |
| FlyGym 전신에 공기역학과 정책 이식 | 기존 장면/전신 인터페이스 활용 가능 | 유체 형상·단위·관절 프레임·구동기·관측을 대조해야 한다. P7 선택 분기 |
| 자체 WPG와 새 제어기 개발 | 작은 제어 실험을 직접 설계하기 쉽다 | 안정화·조향·접촉 정책을 새로 해결해야 한다. 공식 재현이 불가능할 때 별도 연구 범위로 재평가 |

독립 비행 데모에서 성공한 뒤에도 두 모델 사이에 root 상태를 복사하는 방식으로 걷기↔비행 연속 전환을 구현하지 않는다. 연속 전환에는 같은 자유 관절과 물리 시간을 유지하는 단일 전신 모델이 필요하다. 그 모델은 검증된 원본 전신 FlyBody 또는 동등성이 확인된 FlyGym 포트 중에서 선택한다. FlyGym 이식을 필수 선행 단계로 고정하지 않는다.

**외부 지적의 로컬 검증과 설계 반영**

| 지적 | 검증 결과 | 반영할 작업 |
|---|---|---|
| F1. 공식 정책과 자체 과제의 규격 불일치 | 두 환경을 생성/reset하여 행동 12/66, 날개 인덱스 3–8/9–14, 관측 구성 차이를 재확인 | P1은 원본 12차원 과제부터 시작. 배열 패딩은 해결책으로 채택하지 않음 |
| F2. 학습 정책 의존성 격리 | upstream의 NumPy 1.26.4/optional TF 2.8 계열과 현재 환경이 다름. 정책 가중치·서명은 미검증 | 기존 `.venv`와 별도의 실행 환경. 실제 정책 로드에 필요한 패키지만 선정 |
| F3. FlyGym 유체 이식과 단위 문제 | 설치된 변환기는 fluid geom을 제외한다. global 변환은 중력만 처리하고 density/viscosity의 cm 수치를 유지 | 새 비행 프로파일에서 물리 단위 전체 점검. 기존 보행 기본값 변경은 이 계획의 선행 작업이 아님 |
| F4. prototype 패턴 차이 | `FlightRig` roll은 `0.25*sin(2*phase)-0.1`, 원본 기본 WPG는 `0.25*sin(1.5*phase)-0.1`. phase=π에서 -0.1/-0.35 rad | 기존 주석의 “같은 패턴” 표현은 후속 구현 시 정정. 공식 실측 패턴과 공식 WPG를 기준선으로 사용 |
| F5. 성공 판정의 시간 조건 | `flybody_flight.py:80`은 모든 지면 접촉을 합치고, 85–91은 접촉만 누적한다. 고도/자세/속도는 순간 검사 | 모든 성공 조건을 동시에 만족한 연속 시간을 누적. 발 접촉/몸 충돌/공중 상태를 분리 |
| F6. 기존 엔진에 비행 몸을 바로 끼우기 어려움 | C 시간·provenance·복원 계약이 보행과 연결됨. BANC와 C의 신경/몸 진행 순서도 다름 | 독립 세션과 flight 전용 상태/시계. 향후 결합 순서는 명시적으로 버전 관리 |
| F7. 날개 기록과 체크포인트 보완 필요 | 기존 저속 몸 로그는 약 218 Hz 날개 위상 관찰에 부족. 공식 WPG에는 주파수 필터·궤적 인덱스 상태가 있음 | 고속 날개 기록 및 WPG/정책/Composer 상태 수집. 기존 FlightRig 복원 성공의 범위는 인정 |
| F8. BANC 다리 연결은 날개 연결이 아님 | 현재 rate body의 42개 다리 출력은 날개 6축과 다른 계약 | 상위 BANC 명령 + MLP 하위 제어와 BANC 직접 날개 구동을 구별 |

F5의 순수 판정식 반례도 확인했다. 60.2 ms 동안 공중에 있었지만 마지막 0.2 ms만 허용 자세였던 구성에서 현재 takeoff 식은 true였다. 이는 판정식 결함의 증거이며 새로운 물리 이륙 실험은 아니다.

추가로 `flylab/body.py:447`, `:585`는 적분마다 `xfrc_applied`를 지운다. 후속 FlyGym 포트가 외부 공기력을 사용할 경우 호출 바깥에서 한 번 넣으면 사라진다. 적분 내부의 명시적 힘 적용 지점이 필요하다. 원본 ellipsoid 유체력과 같은 힘을 중복 적용하지 않는다.

MuJoCo는 전역 density/viscosity로 관성 형상 기반 유체 저항도 계산할 수 있다. 따라서 날개 fluid geom이 없다는 사실을 “모든 공기력이 0”이라는 뜻으로 해석하지 않는다. Ellipsoid geom을 활성화하면 해당 부모 body의 관성 기반 유체 모델은 꺼진다. 근거: [MuJoCo 3.9 유체 모델](https://mujoco.readthedocs.io/en/3.9.0/computation/fluid.html).

**정책·물리·시간 인터페이스**

| 행동 그룹 | 공식 기본 flight_imitation | 현재 자체 FlightTask |
|---|---|---|
| adhesion | 없음 | 0–5 |
| head | 0–2 | 6–8 |
| wings | 3–8 | 9–14 |
| abdomen | 9–10 | 15–16 |
| legs | 없음 | 17–64 |
| frequency user action | 11 | 65 |

위 표는 현재 고정 소스의 기본 설정을 실제 생성하여 확인한 값이다. 배포 정책의 서명을 확인하기 전까지 모든 공식 정책이 이 규격이라고 일반화하지 않는다. `disable_legs`는 출력 개수뿐 아니라 다리 관절·센서·구동기를 바꾼다.

공식 기본 관측의 정렬 순서는 accelerometer → actuator_activation → gyro → joints_pos → joints_vel → ref_displacement → ref_root_quat → velocimeter → world_zaxis다. 이번 기본 환경 reset에서는 관절 위치/속도 각각 25개, 참조 변위 6×3, 참조 자세 6×4였다. 자체 과제는 관절 위치/속도 각각 91개이고 해당 참조 필드가 없다. 배포 정책은 실제 dtype, batch, activation/filter 설정, flatten 순서와 입출력 서명을 검사한다. 기본 함수의 `joint_filter=0`을 학습 정책의 설정으로 단정하지 않는다.

CanonicalSpecWrapper를 쓰는 정책이라면 clipping과 실제 action 범위 변환을 정확히 한 번 적용한다. 그 뒤 원본 과제가 수행하는 `WPG target - current wing qpos` 보정도 한 번만 적용한다. 이미 wrapper를 통과한 행동에 다시 정규화를 적용하지 않는다. 정책의 분포 평균 추론과 샘플링은 실행 프로파일로 구별한다. 우선 재현은 원본 예제와 같은 추론 방식을 사용한다.

worker 하나가 정책, WPG, Composer task, 적분용 `MjModel/MjData`, RNG, 시계를 소유한다. 최초 프로파일은 물리 50 µs, 정책 200 µs를 사용한다. UI/상위 명령용 5 ms batch 안에서 **정책 25회와 물리 100회**를 수행한다. 관측 갱신과 task callback 순서는 원본 Composer를 따른다. 동일 행동을 5 ms 유지하는 대체 루프를 사용하지 않는다. 별도 프로세스 통신은 batch 경계에만 두고 정책과 물리 사이에는 두지 않는다.

제안 API는 `create/reset`, `advance`, `set_command`, `observe`, `snapshot/restore`, `close`다. `advance` 요청에 `session_id`, `epoch`, `request_id`, `expected_physics_tick`, 정수 batch 크기를 포함한다. 응답 유실 시 같은 요청 ID의 적용 결과를 조회하며 무조건 재실행하지 않는다. 잘못된 요청은 상태를 바꾸기 전에 거부한다. pause 응답은 이미 승인된 유한 batch가 끝난 뒤 반환하며, worker timeout 이후 세션 상태를 성공으로 추정하지 않는다.

상태에는 physics/policy tick, simulation time, model/policy/pattern/runtime hash, root pose, thorax pose, CoM, 속도, 날개 6축, 접촉 종류, 명령 버전·출처, 실패 이유가 필요하다. 렌더링·구독·관찰은 적분을 진행하지 않는다. worker 종료 시 환경과 renderer를 닫는다. 별도 렌더용 `MjData` 복제는 허용하되 읽기 전용 관찰 상태로 다룬다.

**단위와 좌표 변환**

초기 worker 내부의 cm/g/s 물리 단위는 원본대로 유지한다. API 표시 경계에서 FlyLab의 mm/y-up으로 바꾼다. `flylab/common.py:7`의 `S=[[1,0,0],[0,0,1],[0,-1,0]]`를 사용하면 표시 위치·선속도는 `10*S*x`다. 원본 모델의 로컬 메시 축을 유지하면 회전은 `S*R`, 로컬 축까지 S로 변환했으면 `S*R*Sᵀ`이다. quaternion 성분을 위치처럼 재배열하지 말고 회전행렬을 경유하여 scalar-first 등 저장 규격을 확인한다. root, thorax, CoM의 원점 차이를 별도로 유지한다.

| cm→mm, 질량/시간 유지 | 수치 배율 |
|---|---|
| 길이·선속도·가속도·중력 | 10 |
| 질량·시간·각도·각속도 | 1 |
| 힘 | 10 |
| 토크·회전 관성·hinge stiffness/damping/armature | 100 |
| 유체 밀도 | 0.001 |
| 동점성이 아닌 dynamic viscosity | 0.1 |
| 무차원 fluidcoef | 1 |

각 배율은 차원 분석에 따른 값이며 모든 필드에 이름만 보고 일괄 적용하지 않는다. 예를 들어 길이 좌표를 쓰는 선형 tendon/slide의 강성·감쇠는 hinge와 차원이 다르다. 원본 density 0.00128과 viscosity 0.000185는 mm/g/s에서 각각 0.00000128, 0.0000185에 해당한다. 현재 설치본 global YAML은 cm 수치를 유지한다. 새로운 포트의 형상·관성·기어·구동 범위·유체 응답을 함께 검증해야 한다.

**단계별 작업과 완료 조건**

아래 새 경로는 구현 후보다. P0에서 기존 모듈 충돌 여부를 확인하여 확정한다. 예상 개발 시간이나 실시간 성능은 아직 측정하지 않았으므로 약속하지 않는다.

| 단계/선행 조건 | 목적·후보 파일·인터페이스 | 검증·중단 조건·산출물 |
|---|---|---|
| P0. 재현 계약 고정 | `contracts/flight-protocol.d.ts`, flight 실행 프로파일 명세. 소스·정책·실측 패턴 해시와 라이선스, 대상 과제, 평가 집합 고정 | 실제 배포 자산 존재/접근성 및 버전 확인. 기존 보행의 상태/시간 회귀 기준 보존. 누락 자산이나 서명 불명확 시 P1 성공을 선언하지 않음. 산출물: 자산·환경·인터페이스 명세 |
| P1. 공식 정책 재현 / P0 | `flylab/flight_runtime/official.py`, `policy.py`, `tools/verify_official_flight.py`. 원본 환경과 최소 어댑터 비교 | 별도 환경에서 실측 WPG+학습 정책 로드. 동일 초기 관측/첫 행동/10 control step(2 ms) 비교 후 원래 episode 전체 실행. 차이가 있으면 wrapper/필터/관측을 먼저 교정. 산출물: 잠근 환경, 입출력 규격, 대조 궤적·영상·처리율 |
| P2. 세션·저장·통신 / P1 | `flylab/c/flight_session.py`, `flight_runtime/worker.py`, `state.py`, C 서버 routes. create/advance/observe/restore/close | 50/200 µs 정수 시계, 5 ms batch, pause·중복 요청·응답 유실·timeout 처리. 복원 전후 미래 상태 대조. 렌더 없이도 같은 궤적. 산출물: 비행 세션 API와 전용 체크포인트 |
| P3. 실제 날개 표시 / P2 | `src/c/flight-view.js`, `src/c/app.js`, template/build 등록. native pose/mesh 기반 표시 | 좌우 날개 6축과 표시축 대조. normal/slow motion에서 실제 각도와 시간 확인. UI RAF가 물리 진행률을 바꾸면 실패. 산출물: 실행·일시정지·관찰·종료 가능한 화면 |
| P4. 지속 비행 / P1–P3 | `flight_runtime/reference.py`, `metrics.py`, `tools/verify_flight_stability.py`. reference/metric 계약 | 원래 짧은 episode와 별도로 충분히 긴 참조를 제공. 미래 참조 buffer와 종료 시간 정합성 확인. 연속 위치·속도·목표 대비 자세 조건, 충돌·수치 실패 검사. LAST 뒤 숨은 reset은 실패. 산출물: seed별 지속 비행 보고서와 고속 날개 기록 |
| P5. 조향 / P4 | `flight_runtime/commands.py`와 reference generator. 목표 속도·고도·yaw rate 명령 | 위치를 직접 쓰지 않고 참조를 연속적으로 갱신. 좌우 회전·가감속·고도 변경에서 추종 오차/포화/실패율 측정. 산출물: 명령 기반 자유 비행 시나리오 |
| P6. 시각 자율 비행 / P5 | `flight_runtime/vision_policy.py`, 시각 과제 adapter. 공식 카메라 관측/상위 목표 입력 | 공식 시각 정책 규격 확인 후 bumps/trench 등의 예제 재현. 미학습 장면·지연·가림에서 회피 검증. 목표 정보와 실제 감각의 출처 추적. 721 낱눈 입력은 별도 관측 모델 검증 필요. 산출물: 자율 비행 시나리오와 입력 차단 대조 |
| P7. 접촉 가능한 단일 몸 / P1·P4 | 원본 whole-body 확장 또는 선택적 `flight_runtime/port_flygym.py`. 명시적인 flight body profile | 관절·관성·토크·날개 프레임·유체력·관측 차이 측정. 다리 활성 후 정책 성능 재평가. 단순 규격 변환만으로 성공하지 않으면 별도 정책 학습 범위로 전환. 산출물: 채택 모델과 원본 대비 물리/제어 비교 |
| P8. 이륙·착륙 / P7 | `flight_runtime/contact_tasks.py`, `metrics.py`. airborne/touchdown/stable_landed/failure_reason | 발 접촉과 몸 충돌 분리. 자세·속도·고도/접촉 조건을 동시에 연속 유지. 첫 모델은 다리·날개 상호작용과 충격을 포함. 산출물: 이륙·착륙 과제별 정책과 성공률 |
| P9. 걷기↔비행 / P8 | `flight_runtime/transitions.py`, 기존 보행 controller adapter | 같은 freejoint/물리 시계로 WALK→PREPARE_TAKEOFF→TAKEOFF→FLIGHT→APPROACH→TOUCHDOWN→STABLE_LANDED→WALK. 경계에서 root 순간 변경, 속도 초기화, 숨은 모델 교체가 있으면 실패. 산출물: 반복 전환 궤적·접촉 증거 |
| P10. BANC 연결 / 상위 P4·P5, 직접 구동은 추가 연구 | `flight_runtime/neural_bridge.py`, 검토된 신경 출력 binding | 상위 BANC + 하위 학습 정책과 직접 날개 제어의 모드를 명시. 직접 구동에는 회로/근육 매핑, 단위와 시간 계약, 차단·복원 대조 필요. 산출물: 제어 출처와 생물학적 검증 범위를 명시한 연결 보고서 |

P0–P4가 첫 비행 MVP다. 이 단계의 완료에는 화면에 실제 날갯짓이 보이고, 정책이 물리 상태를 유지하며, 일시정지·기록·복원이 동작하는 결과가 포함된다. P7은 P6 완료를 기다릴 필요는 없지만, 각 단계가 사용하는 평가 환경을 섞지 않는다. 원본 전신 모델을 채택하면 FlyGym 포트 작업은 생략할 수 있다.

기존 보행 물리 100 µs를 새 단일 몸에서 50 µs로 세분하고 같은 행동을 두 번 유지해도 수치 동등성이 자동으로 성립하지 않는다. 통합 몸에는 새로운 모델/실행 프로파일 해시와 보행 회귀 검증을 적용한다. BANC와 C의 현재 결합 순서를 몰래 하나로 바꾸지 않는다.

**상태·관찰·실패 계약**

체크포인트는 MuJoCo integration state뿐 아니라 WPG의 phase/sequence/frequency/filter, task 시계와 성공 dwell, reference cursor와 미래 참조, Composer 관측 buffer, RNG, stateful policy 상태를 포함한다. 모델/정책/패턴/환경 버전이 다르면 복원을 거부한다. 기존 `StateStore`의 무피클 NPY·해시·원자적 저장 원칙을 활용하되, TensorFlow 등의 flight 버전을 기존 전역 체크포인트 필수 조건으로 추가하지 않는다. 별도 후보 worker에 복원하고 무결성을 확인한 후 교체한다.

고속 기록은 정책 주기의 날개 각도/구동과 필요한 물리 관찰 상태를 별도 스트림으로 저장한다. 물리 주기 신호가 필요한 비교는 50 µs로 기록한다. 전체 메시 데이터를 매번 복제할 필요는 없다. 정상 속도 화면은 일부 프레임만 표시하고, 관찰용 slow motion은 기록 시점을 촘촘히 선택한다. 화면을 느리게 만들기 위해 218 Hz 물리 주파수를 낮추지 않는다. 기록 실패, 렌더 실패, IPC 실패와 물리 발산을 구별한다.

날개 주파수 변경 시 현재 `FlightRig`의 `time*frequency` 식은 위상 점프를 만들 수 있다. 원본 WPG의 위상 유지·주파수 필터를 사용하는 경로와 실험용 경로를 구별한다. phase=π의 패턴 차이 0.25 rad를 그대로 수정하면 안정 비행이 해결된다고 추론하지 않는다.

**측정 기준과 첫 실험**

| 대상 | 완료 판정 | 수치의 현재 상태 |
|---|---|---|
| 공식 재현 | 원본과 어댑터가 같은 자산/seed/reference/wrapper에서 초기 관측·행동과 짧은 궤적을 허용 오차 내 재현하고 원래 episode를 완료 | 허용 오차는 동일 환경 반복성 측정 후 사전에 고정 |
| 지속 비행 | 허용 고도·속도·목표 자세·무충돌 조건을 동시에 연속 유지 | 기존 계획의 1초 기준을 최소 점검으로 유지. 외부 제안의 5 seeds×5초, 위치 RMSE 1 mm, 속도 오차 5 mm/s, 자세 오차 20°는 P1 후 채택 여부를 결정할 후보 |
| 조향 | 목표 방향·고도·속도 변화에서 과도 응답과 추종 오차를 측정 | ±45° 회전/2초/최종 오차 10°는 후보. 참조 pitch가 약 47.5°인 비행에 무조건 수평 자세 기준을 쓰지 않음 |
| 이륙 | 지면에서 시작해 접촉을 벗어난 뒤 충분한 고도와 허용 자세/속도를 지속 | 연속 1초는 후보. 접촉 반발이나 순간 success는 불충분 |
| 착륙 | 발로 허용 충격 접촉을 만들고 몸 충돌·반동 없이 안정 상태 유지 | 연속 0.5초는 후보. 관찰 고속 로그로 충격과 dwell 확인 |
| 연속 전환 | 같은 물리 몸에서 걷기→비행→걷기 반복, 상태 연속성·접촉 조건 만족 | 5회 반복은 후보. 모델 교체/자세 주입은 인정하지 않음 |

가장 먼저 실행할 실험은 **공식 실측 wing pattern과 배포 정책을 고정한 원본 환경 대 최소 어댑터의 10 control step(2 ms) 대조**다. 이 비교는 연결 정확성을 검사하며 안정 비행 증거가 아니다. 원본 자체가 실패하면 자산/환경 문제를 먼저 해결한다. 원본만 통과하면 어댑터 문제를 수정한다. 둘이 맞으면 원래 episode를 완료하고 장시간 과제로 확장한다. 공식 기본 flight episode는 0.6초이므로 time_limit만 늘리고 reference를 그대로 두는 방법은 사용하지 않는다.

공식 정책이 재현되지 않으면 사용할 수 없는 정책을 대체했다는 사실을 명시하고 새 제어기 연구로 범위를 재평가한다. 반대로 FlyGym 포트의 물리·관측·정책 동등성이 측정되고 단일 장면 통합 비용이 더 낮으면 이후 모델 선택을 바꿀 수 있다. 현재 계획은 네이티브 Windows GPU/WSL, 실시간 성능, 추가 학습 비용을 확정하지 않는다.

**검토 범위와 남은 불확실성**

외부 답변은 222개 제공 파일 중 51개의 전체 또는 일부를 내용 검토했고 171개는 표적 내용 검토하지 않았다고 명시한다. 코드/테스트 실행, 배포 정책 가중치와 실제 서명 검사는 하지 않았다. 답변의 ZIP 해시 반복만으로 실제 전체 검토를 증명했다고 보지 않는다.

로컬에서는 222개 스냅샷 파일의 해시 일치를 확인했다. 작업 트리는 220개가 같고 `flylab/c/banc_walking.py`, `flylab/c/rate_body.py` 2개가 이후 변경되었다. 차이는 다리별 gain과 load receptor gain 관련이며, 이번 검토에서 읽고 보존했다. 비행 핵심 소스는 스냅샷과 같다. BANC 후속 통합에서는 변경된 parameter/schema를 기준으로 다시 검사한다.

이번 추가 검증은 환경 생성/reset, 행동/관측 규격, 스냅샷 대조, 판정식 반례, 패턴 수식 비교다. 학습 정책이나 새로운 비행 rollout은 실행하지 않았다. 외부 첨부 coverage JSON 다운로드는 Chrome에서 차단되었으므로 파일 자체의 내용/무결성은 검증하지 못했다. 대화 본문에서 확인한 범위 진술과 실제 로컬 검증을 구별해 기록했다.

검증 자료: [발견별 로컬 대조](C:/Users/ckthd/.codex/analysis-runs/flylab-flight-web-20260912-01/local-verification.md), [기계 판독 검증 결과](C:/Users/ckthd/.codex/analysis-runs/flylab-flight-web-20260912-01/return-verification.json), [외부 응답 수집 기록](C:/Users/ckthd/.codex/analysis-runs/flylab-flight-web-20260912-01/web/returned-response-review.md).
