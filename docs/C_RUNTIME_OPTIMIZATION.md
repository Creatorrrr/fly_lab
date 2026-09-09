# C 실행 경로 추가 최적화 · 2026-09-09

기존 MPS 전뇌 구현 이후 남은 몸 제어, 센서, CPU–GPU 동기화, 기록 전달, 브라우저 갱신 비용을 줄였습니다. MuJoCo의 물리 방정식·메시·충돌 설정·0.1ms 적분 간격과 139,255개 뉴런의 LIF 커널은 유지합니다. 원자료는 [c_remaining_20260909](../verification/c_remaining_20260909/)에 있습니다.

## 변경 내용

| 경로 | 적용한 최적화 | 계산 보존 기준 |
|---|---|---|
| 다리 제어 | DOF 매핑·반사 보정 벡터·위상 매듭 캐시, 6개 다리의 42개 cubic spline을 Numba 네이티브 함수로 묶어 평가 | 원래 SciPy 계수, 항의 계산 순서, float64, fastmath=False |
| 물리 관측 | 컴파일된 body/geom 인덱스로 자세·접촉을 조회하고 같은 순서로 힘 합산 | 원래 접촉 선택 조건과 MuJoCo mj_contactForce 유지 |
| 근접·시각 센서 | 9개 근접 광선과 64개 시각 광선을 mj_multiRay 한 번으로 처리 | 같은 출발점·방향·메시·가림·거리 제한, 센서 RNG 유지 |
| 전뇌와 몸 | 5ms 구간의 MPS 작업을 제출한 뒤 독립적인 CPU 물리 계산 수행, 두 작업 완료 후 다음 구간 진행 | t_k의 출력과 센서를 함께 고정, 신경·몸 각각 50개 하위 스텝 |
| GPU 관측 | 상태 검사·선택 스파이크·선택 전압/발화율·전체 통계를 한 버퍼에 모아 구간마다 한 번 읽음 | 전압 float32, 스파이크 int64; 출력 디코더는 같은 tick의 원시 발화율 사용 |
| 화면 | 상태가 바뀔 때 형상 생성·GPU 버퍼 업로드, 카메라만 바뀌면 버퍼 재사용, 정지 화면의 반복 draw 중단 | 추적·전체·상단·개체 시점과 크기 변경 시 필요한 갱신 유지 |
| 전달·기록 | 선택 신호·발 접촉 DOM 재사용, JSON 공백 제거, 이미 읽은 GPU 값으로 기록 | 신호 100Hz, 몸 10Hz, 운동 명령 200Hz 및 선택 스파이크 보존 |

물리 solver 자체는 CPU의 MuJoCo입니다. 작은 순차 제어·감각 값은 이 물리 상태와 인접한 CPU에서 계산합니다. GPU에 옮기는 대신 반복 Python 호출·검색·할당을 제거했습니다. Numba가 없는 B 설치에서는 캐시된 원래 spline 평가를 사용합니다. C 설치에는 `requirements-c.txt`의 Numba 0.67.0이 포함됩니다.

신경 적분 커널 `flylab/c/lif.metal`의 SHA256은 변경 전과 같은 `fe22d24951461ea39dbfe054422d4af8222dc20be05c3f7e1c321284f1fe6ef6`입니다. 새 `observation.metal`은 상태를 읽기만 합니다. 표시용 전체 평균 발화율은 GPU 블록 합산 순서가 달라 마지막 소수 자릿수에 차이가 날 수 있습니다. 각 뉴런의 발화율·전압·정수 스파이크·운동 디코더 입력은 이 표시용 평균을 사용하지 않습니다.

`performance.neural_s`는 GPU 제출과 CPU 물리 종료 후 남은 대기 시간의 합입니다. 전체 GPU 실행 시간이 아니며, `physics_s`와 GPU 실행은 겹칩니다. 성능 판단에는 전체 `wall_seconds`를 사용합니다.

## 재현 가능한 비교

비교 도구는 변경 전 [동결 소스](../verification/c_remaining_20260909/baseline/)와 현재 소스를 각각 별도 프로세스로 실행합니다. 모델·가상환경·시작 체크포인트를 맞추고 실행 순서를 교대합니다. 초기화와 JIT 예열을 제외한 실제 `CEngine.step(1)` 전체 시간을 측정하며 모든 결과 체크포인트와 각 제어 시점의 몸·센서·명령·선택 스파이크를 저장합니다.

```sh
.venv/bin/python tools/benchmark_c_runtime.py \
  --baseline-root verification/c_remaining_20260909/baseline \
  --checkpoint artifacts/c/checkpoints/checkpoint-73604742ca6d \
  --strict-checkpoint verification/c_local_20260909/acceptance/active_checkpoint \
  --seconds .3 --repeats 3 --out verification/new_runtime_benchmark
```

사용자 체크포인트는 9.920모델초의 C_SHADOW/MPS 상태입니다. manifest SHA256은 `5243e435bce0d7b5be496ac1c8b1133b753ec070964b482fdaa492e0040b37fc`입니다. C_STRICT는 별도로 저장된 활성 체크포인트를 MPS로 동일하게 옮깁니다. C_ASSISTED 사례는 사용자 체크포인트의 제어 모드만 명시적으로 바꾼 실험이며, 사용자 실험을 변경하지 않습니다.

### 실제 측정 결과

M1 Max 32GB, macOS 26.6.2, Python 3.12.7, MuJoCo 3.9.0, PyTorch 2.14.0에서 실행했습니다. 각 조건·버전마다 같은 상태로 복원해 0.3모델초를 3회 계산했습니다. 아래 시간은 벽시계 중앙값입니다. [전체 결과](../verification/c_remaining_20260909/benchmark_v1/report.json)와 [실행 로그](../verification/c_remaining_20260909/benchmark_v1.log)에 모든 반복 측정값이 있습니다.

| 조건 | 변경 전 MPS | 최적화 후 | 추가 속도 향상 | 모델 시간 / 실제 시간 |
|---|---:|---:|---:|---:|
| C_SHADOW | 6.336초 | 2.076초 | **3.05배** | 0.145배속 |
| C_STRICT 활성 상태 | 6.238초 | 1.991초 | **3.13배** | 0.151배속 |
| C_ASSISTED | 8.541초 | 2.205초 | **3.87배** | 0.136배속 |
| C_SHADOW + 기록 | 10.587초 | 2.345초 | **4.51배** | 0.128배속 |

**24회 실제 전뇌·물리 실행 모두 PASS**입니다. 각 조건의 모든 반복에서 최종 체크포인트 전체와 제어 시점별 물리·센서·명령·선택 발화 이벤트가 변경 전 결과와 정확히 일치했습니다. 기록 조건에서도 전압/발화율 30행, 몸 3행, 운동 명령 60개 및 선택 스파이크가 일치했고, 기록 누락은 0입니다. 기록 시작 체크포인트와 종료 시 파일 내보내기는 위 시간에서 제외합니다.

일반 데스크톱의 다른 작업을 중단하지 않고 순차·교대 측정했으므로 지연 변동이 있습니다. 예를 들어 최적화 후 기록 조건 3회는 2.300, 2.345, 5.079초였습니다. 고정된 실시간 처리율을 보장하는 수치가 아니며, 현재도 실시간보다 느립니다. 초기 MPS 도입 때의 CPU 대비 수치와 이번 추가 개선 수치를 곱해 전체 향상으로 보고하지 않습니다.

### 실행 서버와 브라우저

기존 서버와 교체한 서버에 같은 체크포인트를 복원해 2개 제어 구간씩 20회 요청하고, JSON 응답과 FLC3 바이너리 수신까지 3회 측정했습니다. 0.2모델초의 중앙값은 **6.413초 → 2.401초, 2.67배** 개선입니다. 이 측정에는 브라우저 렌더링은 포함되지 않습니다. [통신 비교 원자료](../verification/c_remaining_20260909/live_comparison.json)에 반복값과 바이트 수가 있습니다.

세 번의 최종 신경·물리 체크포인트가 이전 서버와 정확히 일치했습니다. 원래 9.920초 상태로 복원한 체크포인트도 프로토콜에서 의도적으로 증가시키는 구독 epoch를 제외한 모든 값이 일치했습니다. 신호 바이너리의 총 크기는 동일하며, JSON의 총 크기는 969,326 → 919,403바이트입니다.

실제 인앱 브라우저에서 MPS 재생으로 9.920 → 11.300모델초까지 움직임과 신호 갱신을 확인한 뒤, 원래 체크포인트의 **9.920초·MPS·일시정지** 상태로 복원했습니다. 추적·상단·개체 시점이 표시됐고 브라우저 오류 로그는 0건이었습니다. 정지 화면의 초기 추적 수렴 후 draw 수는 108에서 멈췄습니다. 상단 카메라 전환은 draw만 109로 증가하고 형상 생성 1회/업로드 2회는 유지됐습니다. 개체 시점에서는 몸 표면을 숨기기 위해 필요한 형상을 다시 만들었습니다. 재생 종료 뒤에도 형상 생성 141회/업로드 282회가 유지됐고 카메라 수렴 후 draw가 730에서 멈췄습니다.

브라우저 관측값은 [browser_audit.json](../verification/c_remaining_20260909/browser_audit.json)에 저장했습니다.

## 검증 범위

- [전체 회귀 검사](../verification/c_remaining_20260909/all_tests_v2.log): **154/154 통과**, 실제 MPS 검사 포함.
- [실제 MuJoCo 물리 검사](../verification/c_remaining_20260909/physics_gate.json): **14/14 통과**. 3모델초 보행, 접촉 지지, 수평 이동, 외력, 운동 연결 차단 및 체크포인트 뒤 연속 계산을 검사했습니다. 복원된 적분 상태 오차는 0입니다.
- 새 회귀 검사에서 600회 합성 반사/전후진 제어, 주기 경계·큰 위상의 spline 평가, 실제 몸 50개 제어 구간의 관절·접촉·센서·복원 상태, 100개 위치/방향의 광선 결과가 원래 경로와 정확히 일치합니다.
- MPS 관측 검사에서 중복·빈 구독, 513개 뉴런의 마지막 불완전 블록, 큰 int64 스파이크 값, 실패한 입력 뒤 캐시, 복원 뒤 캐시, 물리 예외 시 GPU 작업 정리, 3개 C 모드의 동기/겹침 실행 인과성을 검사했습니다.

이 검증은 계산 경로의 보존과 짧은 실제 물리 실행을 다룹니다. 기존 장시간 33조건 캠페인을 이번 변경으로 모두 다시 실행했다는 뜻은 아니며, C_STRICT 자연 보행이나 생물학적 타당성을 새로 승인하지 않습니다.

## Apple GPU 물리 실행 시도

주 실행 환경과 분리된 `.venv-mjx`에 MJX와 JAX Metal을 설치하고 **실제 M1 Max의 METAL:0**에서 확인했습니다. [최종 원시 결과](../verification/c_remaining_20260909/mjx_explicit_jax_probe.json)는 **BLOCKED / physical_executed=false**입니다.

1. 최신으로 해석된 JAX 0.11.1 + jax-metal 0.1.1은 기본 JIT에서 StableHLO 호환 오류가 났습니다.
2. JAX 0.5.3도 기본 JIT에서 memory-space 오류가 났습니다.
3. JAX/jaxlib 0.4.34 + jax-metal 0.1.1은 실제 Metal 기본 JIT를 통과했습니다. MJX 3.9.0에 `impl='jax', device=METAL:0`을 명시해 현재 몸 모델을 전달하자 다음 오류가 발생했습니다.

```text
NotImplementedError:
(mjtGeom.mjGEOM_PLANE, mjtGeom.mjGEOM_MESH) margin/gap not implemented.
```

따라서 이번 결과는 **현재 모델의 충돌 기능 미지원**입니다. GPU 물리 속도를 재어 느리다고 판정한 결과가 아닙니다. 충돌 여유값이나 메시를 바꾸지 않았으며, GPU 물리 실행 성공으로 보고하지 않습니다. 이 경로를 채택하려면 지원되지 않는 물리 기능의 별도 구현과 동등성 검증이 필요합니다.

```sh
python3.12 -m venv .venv-mjx
.venv-mjx/bin/python -m pip install -r requirements-mjx-probe.txt
.venv-mjx/bin/python tools/probe_mjx_metal.py \
  --model verification/c_remaining_20260909/baseline_profile_v2/body.mjb \
  --state verification/c_remaining_20260909/baseline_profile_v2/body_state.npz \
  --out verification/new_mjx_probe.json
```

기본 JIT·장치 탐색·모델 변환·물리 실행 여부를 따로 기록합니다. 물리 실행 전 차단이면 종료 코드 2이며 CPU 대역으로 바꾸지 않습니다. 실패했던 설치·탐색·초기 프로파일·신규 검사 로그도 원자료 디렉터리에 그대로 보존했습니다.

## 공식 구현 참고

- [MuJoCo MJX 문서](https://mujoco.readthedocs.io/en/latest/mjx.html): JAX 가속 경로와 단일 장면/다중 환경의 특성. 이 프로젝트의 실제 성능은 로컬 측정으로 판단합니다.
- [Apple JAX Metal](https://developer.apple.com/metal/jax/): Metal 플러그인 설치와 호환성.
- [FlyGym GPU 시뮬레이션](https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/): NVIDIA/Warp 기반 다중 환경 경로.
- [SciPy PPoly 구현](https://github.com/scipy/scipy/blob/main/scipy/interpolate/_ppoly.pyx): 원래 cubic 계수 평가 순서.
- [PyTorch MPS event 구현](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSEvent.mm): GPU 작업 제출과 완료 대기의 구분.
