# FLY LAB C 0.3.0

2026-09-09 추가: Apple silicon의 `exp_lif_mps` 백엔드와 상태 보존 장치 전환을 지원합니다. 설치·알고리즘·실측 결과는 [C_MPS_VALIDATION.md](C_MPS_VALIDATION.md)를 참고하세요. CPU 기준 계산과 CUDA 경로도 유지합니다.

구현 기준은 [최근 설계 대화](C_DESIGN_REFERENCE.md)입니다. 원래 설계서 첨부 파일은 대화 조회 도구에서 제공되지 않아, 조회된 최신 답변 전체의 요구사항을 기준으로 구현했습니다.

## 실행과 범위

```sh
.venv/bin/python -m pip install -r requirements-c.txt
.venv/bin/python build_c.py
.venv/bin/python run_c.py --doctor
.venv/bin/python run_c.py --port 8766
```

`http://127.0.0.1:8766`에서 C를 관찰합니다. 기본 모드는 C_SHADOW이고 초기화 후 일시정지합니다. C_STRICT를 선택하고 **새 실험**을 누르면 C 출력만 몸을 구동합니다. **DNp09 전진 출력 직접 자극**은 20mV/0.5모델초 실험적 자극을 예약하며, 재생이나 한 단계로 시간을 진행해야 적용됩니다. 자연 감각 처리의 성공을 보여주는 버튼이 아닙니다.

기존 `run.py`, `FLY_LAB_B.html`, A, 원자료는 유지했습니다. `flylab/legacy.py`의 readout 어댑터로 B의 수치 모델과 체크포인트 형식을 보존합니다. B의 5ms 신경 계산 순서는 B_COMPAT와 C_SHADOW의 실제 몸 구동에서 동일합니다.

## 데이터 계약

공식 Codex의 `neurons` master roster를 기준으로 모든 ID를 포함했습니다. `classification`, `consolidated_cell_types`를 결합하고 `connections_princeton`의 영역별 연결 행을 뉴런 쌍별로 집계합니다. 중복 pair/neuropil과 total 행, 미지의 endpoint, 중복 master ID, 잘못된 개수는 거절합니다. 연결의 양 끝점에서 뉴런 목록을 역으로 만들지 않습니다.

이번 acquisition은 뉴런 139,255개, 원본 연결 행 5,342,446개, 뉴런 쌍 3,732,460개, 시냅스 접촉 50,666,648개입니다. 원래 Codex의 pair-total 5개 이상 필터가 적용된 자료이며, 전체 접촉 원시 테이블과 동일한 범위라는 주장은 하지 않습니다.

외부 ID는 `flywire:fafb:783:<root_id>` 문자열, 내부 인덱스는 정수입니다. JSON과 브라우저에서 root ID를 Number로 변환하지 않습니다. bundle의 `indptr.npy`, `indices.npy`, `counts.npy`, `weights.npy`는 **W[post, pre]** 순서이며 로드 후 읽기 전용입니다. 단일 행렬의 숫자를 바꿔 생물학적 연결 개수로 보고하지 않습니다.

해시와 파일 목록은 bundle/manifest.json과 identity.json에 있습니다. 원본 해시는 `raw/download_manifest.json`에 보존합니다. 전뇌 크기를 코드에 하드코딩하지 않으며, `fullBrain`은 선택한 스냅샷의 전체 포함 뉴런을 계산한다는 뜻입니다. 전신 신경계 복원 여부를 뜻하지 않습니다.

재구성 명령:

```sh
.venv/bin/python tools/build_c_graph.py --download --unknown-policy mask_zero --out data/fafb783/bundle
.venv/bin/python tools/build_c_bindings.py --graph data/fafb783/bundle --out data/fafb783/bindings.json
```

기존 bundle과 binding 출력은 덮어쓰지 않습니다. 새 다운로드에서 주석이 바뀌면 별도 디렉터리와 binding 버전을 만들어야 합니다. 다운로드 endpoint의 현재 FAFB 기본 snapshot은 공식 문서에서 v783으로 확인했지만 endpoint는 영구 불변 URL이 아니므로, 재현에는 파일 해시가 기준입니다.

## 전달물질과 수치 모델

`weight = synapse_count × 0.275mV × sign`입니다. ACh +1, GABA/GLUT −1; DA/SER/OCT는 수용체 모델이 없으므로 명시적으로 0입니다. 미확정 전달물질은 기본 빌더에서 BLOCKED이며, 이번 탐색용 bundle은 명시적인 `--unknown-policy mask_zero`로 생성했습니다. 미확정 뉴런 19,658개와 신경조절성 뉴런 1,677개의 상태도 계속 계산합니다. 효력 0인 연결은 311,234개, 유효한 동역학 연결은 3,421,226개입니다.

**C_STRICT는 회피 보조가 없다는 제어 모드**입니다. 미확정 전달물질이나 포트의 생물학적 검증까지 해결되었다는 데이터 승인 표시는 아닙니다.

신경 방정식은 `tau_m dv/dt = E_L - v + h + x`, `dh/dt = -h/tau_s`입니다. h와 x는 mV입니다. 휴지·리셋 −52mV, 임계 −45mV, 막 20ms, 시냅스 5ms, 불응기 2.2ms, 전도 지연 1.8ms, 발화율 필터 50ms입니다. 문헌 초기값을 참고한 C 모델이며 Shiu 구현의 정확한 재현은 아닙니다.

각 0.1ms tick 시작에 예약 도착과 외부 펄스를 h에 적용하고, 고정 입력에 대한 지수해로 v와 h를 적분합니다. 끝 시각에서 임계 판정을 하여 스파이크를 기록합니다. 불응기 동안 v는 reset에 고정하고 h는 감쇠·유입을 계속합니다. 스파이크 끝 시각+delay_ticks에 출력 전류를 예약하며 delay queue와 슬롯은 체크포인트에 포함됩니다. CPU와 CuPy CUDA가 동일한 연산 순서를 사용합니다. CUDA 하위 스텝 안에는 CPU 복사나 `.item()` 동기화가 없습니다.

## 감각·출력 포트

`PortBindings`는 실제 ID, 그래프 해시, 단위, gain/cap/filter/delay, 선정 근거와 불확실성을 검증합니다. 미해결 ID·미검토 포트는 `BLOCKED_PORT_BINDING`입니다. 대체 뉴런이나 B 설계 노드를 만들어 통과시키지 않습니다. 감각 packet에 yaw/position/goal을 추가하면 거절합니다.

첫 binding의 자연 입력은 좌우 냄새 평균을 **ORN_DM1 68개**에 전달하는 이상화한 전류입니다. 이 스칼라 냄새와 특정 receptor tuning 사이의 관계는 공학적 가정입니다. 광선 영상, 근접, 접촉, 자가 회전 등은 관찰·보조 제어에 남아 있지만 C의 자연 신경 입력으로는 미연결이라고 공개합니다. 검토되지 않은 retinotopy를 임의로 채우지 않았습니다. 추가 포트는 고정 전압 또는 Poisson 입력, 필터와 지연을 지원합니다.

조향 출력은 검토한 DNa02 실제 쌍의 right-left 발화율 차이, 전진은 DNp09 쌍, 후진은 MDN 4개입니다. stop 집단은 미선정으로 명시되어 있고 항상 전진하는 바이어스는 없습니다. DNa02 측은 문헌의 ipsiversive steering과 수정된 soma 주석을 이용한 추론입니다. 개별 axon의 출력 projection을 독립적으로 복원한 것은 아닙니다. 실제 몸의 좌·우 자극 검사는 별도 원자료로 남깁니다. 디코더의 gain은 공학적 스케일이며 실제 근육 신경지배가 아닙니다.

## 제어 모드와 시간

| 모드 | 적용 명령 | 신경 계산 |
|---|---|---|
| B_COMPAT | 기존 B | 기존 B rate |
| C_SHADOW | 기존 B | 전체 C LIF도 계산 |
| C_ASSISTED | C 출력, 필요 시 명시적 회피/후진 | 전체 C LIF |
| C_STRICT | C 출력만 | 전체 C LIF |

상위 경계는 5ms입니다. t_k의 예약 개입 → 몸 관측 → **t_k의 신경 출력** → 현재 명령으로 몸 [t_k,t_k+5ms) 적분 → 같은 구간의 LIF 적분 → clock 일치 확인 → 기록 순서입니다. 새 관측의 신경 효과는 다음 구간부터 적용됩니다. `u_neural`, `u_assist`, `u_legacy`, `u_final`, `command_source`, `assist_reason`, `sensor_tick`, `neural_readout_tick`, 구동 구간을 기록합니다. 주기 생략이나 몸통 위치 보정은 없습니다.

회피·후진은 `RecoverySupervisor`에서 C_ASSISTED일 때만 작동합니다. C 출력이 0이라고 B가 대신 구동하지 않습니다. 넘어짐·신경 오류·기록 오류에서는 실험을 중단합니다. 이동 기대 `motion_expected`는 명시적 실험 조건이며 구동이 0이라고 자동으로 꺼지지 않습니다.

## 개입과 저장

개입은 5ms 경계의 정수 neural tick에 예약됩니다. `stimulate`는 mV 입력, `suppress_spiking`은 새 발화 억제, `mute_outgoing`은 이후 시냅스 출력 전달 차단, `mute_edges`는 선택 CSR edge 효력 차단입니다. 이미 예약된 도착은 유지합니다. 발화율 필터는 즉시 0이 되지 않습니다. `mute_outgoing`은 운동 디코더 자체를 끄지 않으며 신경–몸 차단은 `motor_disconnect`입니다. `sensor_off`, `assist_off`도 별개입니다.

체크포인트는 npy 배열과 JSON의 디렉터리입니다. `pickle`을 사용하지 않고 파일 해시·런타임·데이터·모델·포트·몸·환경·clock을 검사합니다. 신경 전압/전류/불응기/queue/rate, 인코더 지연과 RNG, 활성·예약 개입, 보조 상태, B 호환 상태, MuJoCo 적분·CPG·반사·외력을 포함합니다. 대체 인스턴스 검증 실패 시 현재 인스턴스를 교체하지 않습니다. B 활성도를 C 전압으로 변환하는 기능은 없습니다.

`Recorder`는 고정 기록 집단을 화면 구독과 분리합니다. 선택 전압·발화율 100Hz NPY 청크, 몸·접촉 10Hz JSONL, 모든 운동 명령 200Hz JSONL, 선택 집단 스파이크와 개입 이벤트를 저장합니다. 전체 전압·전체 스파이크의 무제한 기록은 제공하지 않습니다. 용량 초과는 실패로 중단합니다. 시작 체크포인트와 사용자 명령으로 장시간 기록을 streaming replay하고, 대조군/개입군은 같은 초기 상태에서 순차 실행합니다. 기록·체크포인트는 `artifacts/c/`에 저장합니다.

## 화면과 API

기존 3D 몸 렌더러를 재사용했습니다. 브라우저 렌더링은 계산 응답과 독립적이고, 선택 신호의 색상 스케일은 원시 mV/Hz와 분리합니다. 검색은 ID·세포형·영역·전달물질·soma/output 측을 대상으로 서버에서 페이징합니다. 최대 512개 신호만 전송하고 차트는 최대 8개를 그립니다. 전체 신경 graph를 브라우저에 보내지 않으며 이웃은 제한된 `neighbors` API로 조회합니다.

`flylab.protocol.v3`: JSON 제어 + `FLC3` binary 신호. binary는 magic 4bytes, little-endian uint32 JSON header 길이, UTF-8 header, planar float32 voltage/rate, `(uint64 tick,uint32 subscription index)` 이벤트 순서입니다. `sequence`, `subscription_epoch`, tick 범위, dtype, 단위, payload 길이를 포함합니다. 구독 변경 시 epoch를 올려 이전 ID 배열의 신호가 새 ID 아래 표시되지 않도록 합니다.

서버는 기존 localhost Host/Origin/token/단일 제어 클라이언트/단일 worker 경계를 유지합니다. 브라우저가 임의 파일 경로를 읽거나 실행하는 API는 없습니다. API 저장·복원은 로컬 artifact 디렉터리의 이름만 허용합니다.

## 검증 도구

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python tools/verify_physics.py --seconds .5 --output verification/new_b/physics.json
.venv/bin/python tools/verify_c.py --physics --cuda --out verification/new_c
.venv/bin/python tools/c_topology_control.py --out verification/new_topology
```

`verify_c`는 실제 전뇌 계산, CUDA 여부, 직접 DN 자극의 폐루프 기여, 실제 복원, 여러 시드·마찰·조향·장애물·감각 조건, 3.2초 고착 검사를 구분합니다. 짧은 물리 사례의 PASS는 통합 경로가 오류 없이 완료되었다는 뜻이고, MDN/DNa02의 행동 역할까지 자동 승인하지 않습니다. 기존 fix_2 `NavigationMonitor`를 그대로 사용하며 오래된 실패 기록은 보존합니다. `c_topology_control`은 전달물질별 presynaptic identity permutation과 감각 차단을 같은 입력으로 비교합니다. 이는 모델 민감도 검사이며 원본 graph의 행동 우월성 검사는 아닙니다.

## 출처

- [FlyWire Codex FAQ](https://codex.flywire.ai/faq): static products, FAFB v783, pair threshold, neuropil partitions, 주석과 ID 처리.
- [FlyWire 공개 v783 원자료](https://zenodo.org/records/10676866): proofread roster와 연결 집계의 의미.
- [Shiu et al. 모델 코드](https://github.com/philshiu/Drosophila_brain_model): LIF와 초기 파라미터의 참고 연구. 이 구현의 정확한 재현을 주장하지 않음.
- [Transforming a head direction signal into a goal-oriented steering command](https://www.nature.com/articles/s41586-024-07039-2): DNa02 right-left 조향 후보.
- [Descending networks transform command signals into population motor control](https://www.nature.com/articles/s41586-024-07523-9): DNp09/MDN 후보와 network dependence.
- [FlyGym](https://github.com/NeLy-EPFL/flygym): B에서 고정한 NeuroMechFly 몸과 HybridTurningController.
