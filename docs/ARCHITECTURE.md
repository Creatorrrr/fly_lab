# FLY LAB B 아키텍처 · v0.2.1

## 목적과 경계

A의 관찰·개입·재현 흐름에 실제 물리 엔진 어댑터를 추가한다. 신경 데이터의 범위와 가정은 바꾸지 않는다. B는 전체 뇌나 사실적 비행이 아니다. 현재 실행 검증 범위는 LOCAL_VALIDATION.md에 기록한다.

```
Browser: observed frame / intervention / selected signals
  └─ BTransport: versioned JSON, ordered request queue
      └─ aiohttp: loopback / token / host-origin / single client
          └─ one ThreadPoolExecutor worker (physics + GL thread affinity)
              └─ Dispatcher / Engine
                  ├─ SensorAdapter ← MuJoCo pose, ray hit, analytic odor field
                  ├─ NeuralController / Circuit (A model, Python port)
                  │    └─ MotorCommand (drive reference, yaw request)
                  ├─ MotorAdapter (left/right amplitude mapping)
                  ├─ HybridTurningController (CPG + contact/retraction reflex)
                  └─ FlyGymBody → 42 joint targets + adhesion → mj_step
```

## 파일과 책임

| 파일 | 책임 |
|---|---|
| flylab/brain.py | sparse COO 연결 전달, 하이브리드 rate 상태, 개입, 신경 상태 복원 |
| flylab/sensors.py | 물리 형상 광선과 후각장 → 상위 신경 관측. XYZ/yaw는 출력하지 않음 |
| flylab/body.py | 공식 API 호출·native 좌표·관절/접촉/CPG·무조코 적분 상태 |
| flylab/engine.py | clock, 설정·환경 검증, 기록, 비교·복원·리플레이, 구독 |
| flylab/server.py | localhost 인증/단일 세션과 요청 순서. 신경 방정식은 없음 |
| flylab/contracts.py | NeuralBackend·BodyBackend·MotorCommand 인터페이스 |
| src/runtime/b-transport.js | same-origin 인증, 응답 라우팅, 요청 직렬화. 로컬 대체 엔진 없음 |
| src/ui/world-view.js | 물리 분절 위치 기반 도식적 몸, 카메라, WebGL/Canvas 표시 |
| src/ui/monitor.js, app.js | 선택 신호·모니터·실험 조작·다운로드·물리 상태 표 |
| src/core/brain.js | B 신경 모델·회피 상태의 수치 대조용. B 배포 HTML에서는 실행하지 않음 |
| contracts/protocol.d.ts | v2 외부 계약의 주요 필드 |

## 시간 모델

- physicalDt = 0.0001 s, controlDt = 0.005 s, 정수비 50.
- 신경 업데이트는 현재 물리 관측을 읽어 다음 50개 물리 하위 스텝의 상위 명령을 정한다.
- 저수준 CPG·반사는 매 물리 스텝에 관측을 읽고 관절각·접착 입력을 갱신한다.
- `frame.tick`과 `simTime`은 제어 스텝의 완료 시점. `physics.physicsTime`은 초기 0.2 s 정착 시간을 뺀 물리 시간이다.
- `sensorTick`은 마지막 신경 입력이 관측된 시점이다. 일반적으로 완료된 tick보다 1 작다. 초기 시점은 둘 다 0.
- `advance` 한 요청은 최대 10 control ticks = 50 ms. UI는 한 요청만 진행한다. 계산이 느리면 모델 시간이 느리게 흐르고 정확도를 희생해 스텝을 건너뛰지 않는다.
- 기록은 20 control ticks마다 10 Hz. 1201개 ring buffer.
- 페이지 숨김/일시정지는 요청 중단이다. 이미 요청한 최대 50 ms 계산은 끝날 수 있다.

## 단위와 좌표

MuJoCo/FlyGym은 `(X,Y,Z)`의 Z-up/mm 좌표. UI는 `(x,height,z)`의 Y-up/mm 좌표.

```
p_ui = S p_native
S = [[1,0,0],[0,0,1],[0,-1,0]], det(S)=+1
```

이는 반사가 아니라 회전이며 양쪽 좌표계의 오른손성을 유지한다. native +X가 몸의 앞, +Y가 왼쪽인 자세를 기준으로 UI yaw = atan2(-forward_native_y, forward_native_x). 양의 UI yaw는 native 평면에서 시계 방향이다.

각도 rad, 시간 s, 위치 mm. 신경 상태는 [0,1] 무차원. 접촉력은 기준 BW로 정규화한다. 액추에이터 출력은 MuJoCo 모델 단위 그대로이며 N·m나 근육 힘이라고 부르지 않는다. GUI의 목표 조향은 실제 측정 yaw rate가 아니라 상위 제어 요청이다.

## 신경 모델과 실제 연결

실제 ID 98개 및 177개 방향 연결은 A의 동일 데이터이다. 192개 ROI 원본 행을 뉴런 쌍별로 합산했고 원본 행 정보가 data에 남아 있다. 모든 실제 연결의 부호를 +로 두고 sqrt(count) 수신 정규화한 것은 설계 가정이다. 신경펩타이드·수용체·실제 발화 데이터가 추가된 것이 아니다.

HD16/GOAL16/PFL3-L16/PFL3-R16/DN2/SENSE4/ALT2는 72개의 보조 모델 노드. ALT는 A와의 신호 비교를 위해 유지하지만 보행에서 수직 명령을 만들지 않는다. 상위 HD는 단일 표식과 회전 감각을 사용하는 관측기이며 입체 지도/SLAM이 아니다.

실제 EPG 모델 상태가 조향 읽기에 관여하지만 HD scaffold와 저수준 CPG/반사도 작동한다. 따라서 전체 신경계에서 보행이 자연 발생한다는 주장은 금지한다.

## 감각과 privileged information

상위 `NeuralController.step`는 센서 packet, dt, 과제 설정만 받는다. XYZ, 실제 yaw, 목표의 세계 좌표는 들어오지 않는다. `validate_sensor_packet`은 허용 필드 외의 키를 거부한다.

64개 광선은 수평 방향의 밝은 벽 표식을 읽는다. 9개 거리 광선은 주변 환경과의 충돌 거리를 읽는다. fly 자신의 형상은 ray mask로 제외한다. 광선은 실제 모델 형상에서 가림을 계산하되 표식 밝기는 범주적 설계값이다. 이 센서가 실제 복안에 해당한다고 주장하지 않는다. 하나의 유한한 거리의 표식이므로 위치 변화에 따른 시차가 방향 추정 오차를 만든다.

저수준 HybridControllerObservation에는 실제 thorax 높이·발 위치·접촉 힘·진행 단위벡터가 들어간다. 이 정보는 공학적 운동 제어기의 관측이며 상위 신경계에 몰래 제공되는 입력과 구분한다. 자체 난수와 내부 목표를 가진 모델이므로 생물학적인 판단·예측 실험의 대체품이 아니다.

## 몸 업데이트와 환경 편집

보행은 joint position actuators와 발 adhesion에 의해서만 구동된다. loop에서 몸통 qpos를 직접 수정하지 않는다. 실제 접촉과 중력을 통해 몸이 움직이거나 넘어지게 둔다. 넘어짐/유효 영역 이탈 시 멈추며 자동 복구 teleport가 없다.

정적 장애물 편집 시 새 모델을 컴파일한 뒤 같은 layout의 integration/controller 상태를 이전한다. 이로써 native broadphase 구조와 geom 위치가 불일치하는 위험을 피한다. 실패 시 기존 모델을 유지한다. 신규 객체는 몸과 겹치지 않도록 검사한다. world bounds는 B에서 고정이다.

신경–운동 연결을 끊으면 CPG drive가 0이고 관절은 중립 자세를 목표로 한다. 이는 관절 제어기까지 제거한 수동 몸과 다르다.

## 체크포인트

`flylab.checkpoint.v2`에 seed/tick/graph/config/world, MuJoCo mjSTATE_INTEGRATION, CPG phase/magnitude/intrinsic rates·RNG, reflex 보정 및 persistence, 신경·감각·개입·RNG, 미완료 가상 외력, 직전 위치/거리 통계를 저장한다.

- 코드 버전·시간 간격·몸 모델 layout 및 dependency hash를 검사한다.
- Python 객체 pickle은 사용하지 않는다.
- 새 인스턴스에서 복원/검증을 끝낸 뒤 기존 인스턴스를 교체한다.
- CPU/OS/라이브러리 차이까지 비트 동일성을 보장하지 않는다. 물리 gate가 실제 연속성 오차를 측정한다.
- 기록 버퍼/관찰 카메라는 물리 상태가 아니며 복원하지 않는다.

## protocol.v2와 C

`init`, `advance`, `frame`, `command`, `subscribe`, `catalog`, `checkpoint`, `restore`, `exportReplay`, `replay`, `initExperiment`, `paired`, `csv`, `physicsCsv`, `preview`, `graph`가 구현되어 있다.

C를 위해 마련한 경계:

1. 숫자로 변환하지 않는 문자열 neuron ID와 데이터 provenance.
2. 단위/범위가 있는 신호 descriptor와 `subscribe`.
3. `catalog` offset/limit/query. B UI는 아직 전체 소규모 카탈로그를 받는다.
4. Body/Motor와 neural controller가 다른 모듈에 있다.
5. 물리·신경·UI의 다른 시간 간격, 완료 tick 기반 명령.
6. body telemetry와 neural state가 다른 packet 영역에 있다.

**C에 필요한 변경을 숨기지 않는다.** 현재 Engine은 A의 rate 모델 배열/그룹 메트릭을 사용하는 부분이 있고, CSV도 소규모 전체 활성 배열을 기록한다. C에서는 NeuralBackend 구현·frame projection·checkpoint serializer·recording sink를 교체해야 한다. `ready`에서 모든 카탈로그를 보내는 경로와 UI network 도식도 페이지/영역 요약 방식으로 바꿔야 한다. 바이너리 신호 스트리밍과 GPU 전뇌 엔진은 아직 구현하지 않았다.

현재 hybrid B backend의 가져오기 상한은 실제 ID 440개 + 보조 72개 = 512개, 연결 100,000개이다. 이 상한을 단순히 풀어서 C라고 하지 않는다. C의 독립된 loader는 FlyWire/hemibrain/BANC의 ID·개체·버전을 구분하고 결측 입력/부호/가중치 가정을 따로 관리해야 한다.

## 검증 단계

B acceptance는 (1) 설치 및 실제 모델 compile, (2) 관절/접촉 검증, (3) 안정 보행, (4) 감각-신경-운동 개입, (5) native checkpoint 연속성, (6) 브라우저 결합 순으로 확인한다. v0.2.0 제작 환경에서는 실제 물리 구동 경로가 막혔다. v0.2.1의 로컬 검증은 별도 기록한다. C의 시작 전에 로컬 physics gate와 더 긴 복수 seed 실험 결과를 확보해야 한다.


## v0.2.1 회피와 관측 수정

- `fusestatic=False`로 고정된 머리 프레임을 보존하고, 컴파일 뒤 모든 분절 ID가 존재하는지 검사한다. 누락된 ID로 음수 인덱싱하지 않는다.
- 바닥 지지와 벽/장애물 접촉을 구분한다. 전자는 정상 발 접촉이며 후자는 발도 상위 접촉 감각에 포함한다.
- `turnMemory`는 감각에서 선택한 회전 방향과 남은 회전량이다. 근접 광선과 접촉으로 활성화하고 각속도 감각으로 갱신한다. `contactTime`, `recoveryTime`은 지속 접촉과 짧은 후진 구간을 저장한다. 모두 체크포인트에 포함된다.
- 이 상태는 공학적 행동 선택기다. 실제 연결 데이터 98 IDs / 177 edges를 바꾸지 않으며 전체 뇌나 생물학적 회피 회로라고 주장하지 않는다.
- 회피 조향도 PFL3/DN 읽기를 통과한다. 운동 연결 차단은 전진/회전/후진 출력을 모두 0으로 한다. MotorAdapter는 음수 CPG 위상 속도를 이용해 제자리 회전과 후진을 구현한다. 루트 자세를 직접 갱신하지 않는다.
- `verify_navigation.py`의 네 벽 배치는 시험 시작 시에만 루트 초기 자세를 정한다. 보행 루프 및 신경 입력에 평가용 좌표를 전달하지 않는다.
