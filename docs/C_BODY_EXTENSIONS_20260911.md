# 몸통 고정·힘줄·공유 공간·공식 GPU 비교 — 2026-09-11

요청받은 네 가지 기능을 선택 가능한 실험으로 구현했다. 기존 기본 신체와 CPU 물리 선택은 유지했다. 이 문서의 완료는 기능 구현·실행·검증을 뜻하며, 자연 신경회로의 보행·비행 성공을 뜻하지 않는다.

## 화면에서 사용

`python run_c.py`로 실행하는 C 화면에 다음 항목을 추가했다. 이번 실제 브라우저 검증 서버는 **http://127.0.0.1:8770**이며, BANC v888 / CUDA 신경 / CPU 물리를 사용한다. 기존 8766 서버는 별도로 유지했다.

1. **새 실험 몸 → FlyBody · 전신 (CPU 물리)**를 선택한다.
2. **몸통 → 고정 · 평면 실험**, **힘줄 구동 → 발마디 / 복부 / 전체**를 선택하고 **새 실험**을 누른다. 고정 모드는 평면만 지원한다. 힘줄은 FlyBody 전용이며 복부 힘줄에는 전신 몸이 필요하다.
3. **비다리 관절·힘줄 직접 구동**에서 대상을 선택하고 입력을 적용한다. 관절 입력은 rad, 힘줄은 해당 모터의 제어 입력이다. **한 단계** 또는 **재생**으로 진행한다. 다리는 엔진의 보행·신경 제어기가 관리하므로 이 패널에서는 비다리 관절만 직접 지정한다.
4. **한 공간에서 여러 초파리 실험**을 열고 2–4개체를 생성한다. 100ms 진행, 개체별 좌우 구동, 상태 저장·복원·종료를 지원한다. 영상은 실제 MuJoCo 렌더다. 이 공간은 개체별 FlyGym hybrid 보행 제어기를 사용하며 전뇌 회로는 연결하지 않는다.

직접 구동 명령은 엔진 시계를 진행시키지 않고 입력만 보존한다. 엔진의 다음 제어 시각에 신체가 진행된다. 이 명령은 사건 기록·체크포인트·재생에도 포함된다. 힘줄의 신경 출력 매핑은 명시적으로 `unmapped`다. 고정 모드에서는 이동 기대 판정을 끄며, 고정 상태를 자유 보행 성공으로 집계하지 않는다.

## 신체 구성

| 전신 FlyBody 옵션 | 관절별 위치 서보 | 힘줄 입력 | 힘줄이 구동하는 관절 | 수동 관절 |
|---|---:|---:|---:|---:|
| `none` | 78 | 0 | 0 | 24 |
| `tarsi` | 78 | 6 | 24 | 0 |
| `abdomen` | 66 | 2 | 12 | 24 |
| `all` | 66 | 8 | 36 | 0 |

힘줄은 공식 `FlyBody.add_tendons()` / `add_tendon_actuators()`로 구성한다. 발마디 힘줄 하나가 네 말단 관절에 힘을 전달한다. 복부 두 힘줄은 각각 pitch/yaw 여섯 관절을 구동하며, 해당 12관절의 위치 서보는 생성하지 않는다. 흉부–첫 복부 마디 연결은 별도 위치 서보로 남는다. 중복 구동은 컴파일 후에도 검사한다. 실제 근육의 활성·수축을 재현하는 Hill 근육 모델은 아니다.

고정 모드는 공식 `TetheredWorld`의 mocap 신체 부착을 재사용하고, 기존 평면 접촉을 유지한다. 자유 관절을 weld로 부드럽게 고정하는 방식이 아니며 고정 신체에는 몸통 자유 관절이 없다. 힘줄 입력과 관절 반응은 MuJoCo 적분 상태로 저장한다. 기본값은 모델 식별자에서 제외하므로 기존 신체 옵션의 기본 모델 해시 계산은 유지한다. 신체 구성이 달라지는 옵션끼리의 체크포인트는 거부한다.

```python
from flylab.body import FlyGymBody
from flylab.body_options import BodyOptions
from flylab.engine import config_values
from flylab.sensors import default_world

body = FlyGymBody(42, default_world(), config_values(), body_options=BodyOptions(
    model="flybody", actuation="whole_body", servo_profile="tracking_all",
    attachment="tethered", tendons="all",
))
try:
    body.step_body_targets(
        {"c_thorax-c_head-yaw": 0.1},
        tendon_inputs={"lf_tarsus": 0.1, "abdomen_pitch": 0.1}, dt=0.005,
    )
    print(body.whole_body_observation())
finally:
    body.close()
```

이 직접 진행 API는 독립 신체용이다. `CEngine`이 소유하는 몸은 `engine.command('body_actuation', ...)`와 엔진의 `step()`을 사용한다. 자유 신체에서 힘줄을 쓰려면 `attachment="free"`로 변경한다. 힘줄·고정 모드는 현재 CPU 물리에서만 허용하며 기존 Warp 캠페인에서는 명시적으로 거부한다.

## 실제 물리 검사

원자료: [기능·보행 보고서](../verification/body-extensions-20260911/run-01/report.json), [실험 사양](../verification/body-extensions-20260911/run-01/spec.json).

- 고정 모드 네 구성의 생성과 17개 양방향 반응 대조를 완료했다. 모든 반응이 사전에 정한 부호·최소 반응 기준을 통과했으며, 몸통 위치 변화는 0이었다. 각 대조의 CPU 체크포인트 후 미래 상태도 정확히 일치했다.
- 브라우저에서 `lf_tarsus=0.1` 적용 후 50ms를 진행했을 때 네 발마디 각도가 약 0.360rad, 모터 힘이 모델 단위 4.0으로 표시됐다. 이 입력은 관절 목표각 0.1rad를 뜻하지 않는다.
- 공유 공간 접촉을 끈 0.5초 대조는 개체 간 접촉 표본 0개, 켠 대조는 474개였다. 이는 시간에 따라 누적한 접촉 표본 수이며 독립 충돌 사건 수가 아니다. 두 조건 모두 유한 상태·경고 없음·CPU 미래 복원 일치를 확인했다. 3개체와 4개체의 짧은 실행·복원은 [별도 결과](../verification/body-extensions-20260911/shared-counts.json)에 기록했다.

평면에서 같은 보행 입력을 2초 동안 적용한 전방 변위(mm):

| 시드 | 힘줄 없음 | 발마디 | 복부 | 전체 |
|---|---:|---:|---:|---:|
| 42 | 4.756 | 5.251 | 4.740 | 5.480 |
| 43 | 4.683 | 21.465 | 4.638 | 21.440 |

8조건 모두 2초를 완료했고 물리 경고·넘어짐 판정은 없었다. 발마디를 사용한 조건에서 전방 변위가 늘었지만 시드 효과가 크다. 접촉 하중이 0.05BW를 넘을 때 말단 분절의 수평 속도도 기록했으며 발마디 조건에서 감소했다. 이 지표에는 구름·접촉 위치 변화도 포함되므로 순수 미끄러짐이라고 해석하지 않는다. 두 시드의 짧은 평면 실험으로 지형·장시간·신경 보행 개선을 확증하지 않았고 기본값으로 채택하지 않았다.

재현 출력 폴더는 새 경로를 사용한다.

```powershell
.venv\Scripts\python.exe tools\verify_body_extensions.py --out verification/body-extensions-new --seconds 2
```

## BANC 고정 신체 대조

[신경 대조 결과](../verification/body-extensions-20260911/neural-02/report.json): 실제 BANC 158,706개 뉴런을 CUDA에서 계산하고 전신·전체 힘줄·고정 신체에 기존 다리 신경 디코더를 연결했다.

- 감각 유지와 다리 감각 차단을 각각 0.1초 진행했다. 관측 운동뉴런의 최대 발화율은 두 조건 모두 0Hz이며 관절 궤적 차이도 0이었다.
- 기존 검토 주석의 왼쪽 앞다리 tibia flexor 운동뉴런 5개를 직접 자극한 양성 대조는 최대 약 126.83Hz, 감각 유지 대비 최대 관절 차이 약 0.2284rad를 보였다.
- 세 조건 모두 몸통 위치 변화 0, 엔진·신체 오류 없음이었다. 직접 명령이 포함된 기록을 재생해 신경 상태와 물리 상태의 정확한 일치를 확인했다.

이는 고정 모드로 전달 병목을 분리할 수 있다는 검사다. 0.1초 무반응은 모든 조건의 감각 피드백 불가능성을 증명하지 않으며, 직접 자극 성공도 자연 감각 운동 제어의 성공이 아니다. 힘줄에 새로운 신경 매핑을 추정해 넣지 않았다. 첫 실행 `neural-01`은 대조를 완료했으나 재생 호출 인자 누락으로 최종 검증이 실패했고, 수정 후 `neural-02`를 새로 실행했다.

```powershell
.venv\Scripts\python.exe tools\verify_tethered_neural.py --out verification/tethered-neural-new
```

## 공식 GPU 비교

`OfficialGPUComparison`은 실제 공식 `flygym.warp.GPUSimulation`의 입력·진행 메서드를 사용한다. FlyLab이 컴파일 이후 적용한 환경 설정까지 맞추기 위해 동일한 실행 모델을 GPU에 업로드한다. 기존 `WarpWorldBatch`와 CPU에도 동일한 모델·초기 적분 상태·중립 관절 명령을 사용한다. 자유 다리 위치 서보 모델만 비교 대상으로 허용한다.

[2세계 / 50ms 비교](../verification/body-extensions-20260911/gpu-02/report.json)는 커널·그래프 준비 이후 동기화한 10개 표본이다. 같은 시간에 실행하는 전뇌 계산은 포함하지 않는다.

| 경로 | 2세계를 5ms 진행하는 벽시계 중앙값 |
|---|---:|
| CPU | 10.40ms |
| 기존 Warp 배치 | 48.68ms |
| 공식 GPU 어댑터 | 45.30ms |

공식/기존 Warp 간 최대 차이는 qpos 약 `1.20e-5`, qvel 약 `0.1671`이었다. 공식/CPU의 qvel 차이 약 `0.1605`는 이 실험의 허용값 0.1을 넘었다. 공식 경로도 적분 상태 복원 후 다음 진행에서 qpos 약 `9.06e-6`, qvel 약 `0.01462`의 차이를 보였다. qpos/qvel은 자유 몸통과 관절을 포함하는 혼합 모델 단위다. 복원 과정의 접촉·솔버 작업 공간 재구성까지 완전히 같다고 주장하지 않는다.

따라서 공식 클래스로 바꾸는 것만으로 기존 정밀도·복원 문제가 해결된다는 근거는 얻지 못했다. GPU 기본 경로를 교체하지 않았다. 공식 어댑터에는 호스트 입력 전송, 기존 배치에는 GPU 제어·감시 비용이 포함되며, 이 두 세계 결과를 대규모 배치의 일반적인 성능 차이로 확대하지 않는다. 첫 실행 `gpu-01`은 Warp 벡터 배열의 논리 shape와 NumPy shape를 혼동한 복원 검사 오류로 중단됐다. 수정 후 `gpu-02`는 비교를 완료했지만 동등성 기준은 미달이었다.

```powershell
.venv\Scripts\python.exe tools\compare_official_gpu.py --out verification/official-gpu-new --worlds 2 --steps 10
```

## 코드·화면 검증

- 전체 Python 회귀: 370개 실행, 51개 선택 의존성/네이티브 검사 건너뜀, 실패 없음.
- 명시적 네이티브 검사: 22개 통과. 기존의 의도적 불안정 서보 거부 검사에서 발생한 MuJoCo 경고는 해당 검사의 예상 결과다.
- 새 파일 Ruff 검사·포맷, 브라우저 JavaScript 구문 검사, C HTML 빌드를 수행했다.
- 실제 브라우저에서 BANC/CUDA 연결, 전신 고정·전체 힘줄 생성, 입력 적용과 물리 진행, 공유 공간 렌더·개체별 구동·저장·복원을 확인했다. 100ms 공유 상태를 저장하고 200ms까지 진행한 뒤 100ms로 복원했다. fly_1의 구동만 `[0.2,0.8]`로 바꾸었을 때 fly_2는 `[0.6,0.6]`으로 유지됐다. 페이지 오류는 관측되지 않았다.

UI 원자료: [힘줄·공유 공간 스크린샷](../verification/body-extensions-20260911/browser-shared-polished.png). 실제 CLI 실행 원자료는 위 링크의 JSON/궤적 파일에 보존했다.

현재 소스와 원자료 SHA-256은 [검증 manifest](../verification/body-extensions-20260911/manifest.json)에 기록했다. 기존 프로세스가 새 HTML을 제공하는 경우 새 확장 입력을 비활성화하여 지원하지 않는 API 호출을 방지한다.
