# FLY LAB C · FAFB v783 전뇌 실험 플랫폼

실제 뉴런 139,255개와 뉴런 쌍 연결 3,732,460개를 사용하는 희소 LIF 엔진을 추가했습니다. 실제 ID 기반 후각·하행 출력, B_COMPAT/C_SHADOW/C_ASSISTED/C_STRICT, 선택 신호 모니터, 개입 예약, 청크 기록·체크포인트·대조 비교·리플레이를 지원합니다.

```sh
.venv/bin/python -m pip install -r requirements-c.txt
.venv/bin/python run_c.py --doctor
.venv/bin/python run_c.py
```

C 화면: **http://127.0.0.1:8766**. 기본 C_SHADOW에서 B가 몸을 구동하고 C가 병행 계산합니다. C_STRICT를 선택하고 **새 실험**을 누르면 C 신경 출력으로 구동합니다. 초기화 후에는 재생 또는 한 단계로 진행하세요. `./launch_c.sh`도 사용할 수 있습니다.

Apple silicon에서는 **MPS 가속**을 사용할 수 있습니다. `.venv/bin/python -m pip install -r requirements-mps.txt` 후 `./launch_mps.sh`로 실행하거나 화면의 **신경 계산 → MPS · Apple GPU**를 선택하세요. 전환 전 체크포인트를 저장하고 현재 모델 시간·신경·물리 상태를 그대로 옮깁니다. 이 M1 Max의 측정 결과는 전뇌 계산 11.56배, 짧은 실제 몸 연결 3.58–4.35배 향상이며 실시간 속도는 아닙니다. [MPS 검증 보고서](docs/C_MPS_VALIDATION.md)에 원자료와 수치 오차를 기록했습니다.

후속 최적화에서는 다리 제어·접촉·광선 조회를 묶고 MPS와 CPU 몸 계산을 겹쳐 실행하며, 정지 화면의 반복 렌더링을 제거했습니다. 현재 C_SHADOW의 같은 상태에서 **기존 MPS 버전 대비 추가 3.05배** 향상을 측정했습니다. 실제 전뇌·몸·기록 비교와 Apple GPU 물리 경로의 미지원 사항은 [실행 최적화 보고서](docs/C_RUNTIME_OPTIMIZATION.md)에 있습니다.

**제한:** 미확정·신경조절성 연결 출력은 명시적으로 0 처리합니다. 현재 자연 입력은 이상화한 ORN_DM1 후각 연결이며, 운동 디코더는 공학적 매핑입니다. 전뇌 계산·직접 자극 폐루프·안정적인 자연 탐색·생물학적 타당성·CUDA 실장치 검증은 별개의 결과입니다.

설계·명령·데이터 재구성: [C_ARCHITECTURE.md](docs/C_ARCHITECTURE.md). 실제 실행 결과: [C_VALIDATION.md](docs/C_VALIDATION.md). 기존 B 실행은 `python run.py`, 포트 8765로 유지됩니다. 아래는 B 이력입니다.

---

> 로컬 반영: fix_2를 현재 작업 폴더에 적용했습니다. 이 환경에서 새로 실행한 검증 결과는 [FIX2_LOCAL_VALIDATION.md](docs/FIX2_LOCAL_VALIDATION.md)에 별도로 기록합니다. 아래 첨부본의 설명·수치는 당시 기록입니다.

# FLY LAB B · fix_2 보완판

고착 검증의 감시 공백, 장애물 ±135도 후보 누락, 체크포인트 복원 범위/원자성을 수정했습니다. 기존 v0.2.1의 보행·신경 동역학, 데이터, 의존성, UI는 유지했습니다. 따라서 유효한 v0.2.1 체크포인트를 사용할 수 있습니다.

**현재 확인 범위:** 회귀 95개 PASS, 제공된 실제 물리 기록의 재분석. 탐색 8개와 캠페인 33조건의 원자료를 새 기준/일관성 검사로 확인했습니다. **fix_2의 새로운 FlyGym/MuJoCo 물리 실행은 이번 턴에서 수행하지 않았습니다.** 아래에 남긴 v0.2.1 로컬 검증은 과거 원자료이며 새 실행 결과가 아닙니다.

전체 설명: [docs/REASSESSMENT_FIX2.md](docs/REASSESSMENT_FIX2.md)

ZIP에는 원시 `verification/wall_fix_20260909/`, 변경 전 코드 ZIP, 새 회귀 로그/재평가 결과가 함께 들어 있습니다. 입력 파일의 FAIL/PASS를 덮어쓰지 않았습니다.

```bash
cd FLY_LAB_B_fix_2
# 기존 Python 가상환경에서 실행; 의존성 변경 없음
python -m unittest tests.test_core tests.test_server tests.test_validation tests.test_avoidance tests.test_fix2 -v
python tools/reevaluate_evidence.py --out verification/fix2_recheck/reevaluation.json
python run.py
```

새 물리 실험의 결과는 `verification/fix2_native/` 등 새 디렉터리에 저장하세요. 복원 범위를 넘는 타이머는 조용히 보정하지 않고 거절합니다. 원래 버전의 UI/설치/모델 설명은 아래에 보존했습니다.

---

# FLY LAB B · 물리 보행 통합본

**v0.2.1 · Python / FlyGym 2.1.0 / MuJoCo 3.9.x · C 확장을 위한 B 단계**

A의 관찰 화면·부분 신경 모델을 Python 백엔드와 물리 몸 어댑터로 확장한 프로젝트입니다. 기본 신경 데이터는 **98개 실제 ID·177개 연결 + 72개 설계 노드** 그대로입니다. B의 핵심 변경은 뉴런 수 증가가 아니라 몸의 관절·액추에이터·접촉 역학 연결입니다.

> **검증 범위:** v0.2.1은 머리 좌표, 벽 접촉 감지, 가까운 벽에서의 회피 및 접촉 후진을 수정합니다. 실제 물리·벽 회피·다조건 검사 결과는 [로컬 검증 기록](docs/LOCAL_VALIDATION.md)을 확인하세요. 생물학적 재현이나 무제한 장시간 안정성을 뜻하지 않습니다. 의존성이 없으면 `BLOCKED / exit 2`를 남기며 대역으로 전환하지 않습니다.

## 1. 실행

Python **3.12–3.14**와 Git이 필요합니다. 첫 설치에는 인터넷이 필요합니다. 시스템 Python에 패키지를 섞지 말고 가상환경을 사용하세요. HTML은 물리 계산기가 아니라 관찰 클라이언트입니다.

### macOS / Linux

```sh
cd FLY_LAB_B
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py --doctor
python run.py
```

3.13 또는 3.14가 설치되어 있으면 `python3.12` 대신 해당 명령을 사용합니다.

### Windows PowerShell

```powershell
cd FLY_LAB_B
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py --doctor
.venv\Scripts\python run.py
```

브라우저에서 **http://127.0.0.1:8765** 를 엽니다. 다른 포트는 `python run.py --port 8770`입니다. 환경을 설치한 뒤에는 `launch.sh` 또는 `launch.bat`로 다시 실행할 수 있습니다. 서버 종료는 Ctrl+C입니다. HTML 파일 더블클릭만으로 B를 실행하지 않습니다.

기본 관찰 화면은 브라우저에서 그리므로 Python에 OpenGL 창을 만들지 않습니다. `MuJoCo 실제 메시 카메라 캡처`는 선택적 offscreen 렌더링 기능이며 OpenGL/EGL 환경에 따라 따로 실패할 수 있습니다. 기본 물리 계산과 브라우저 표시를 그 실패로 A 방식으로 바꾸지는 않습니다.

## 2. 실제 물리 시작 검사

```sh
python tools/verify_physics.py --seconds 3 --output physics_verification.json
```

이 검사는 실제 모델 구성, 능동 관절 수, 상태 유한성, 관절 변화, 접촉력, 이동, 체크포인트 뒤 100 ms 연속성, 가상 외력, 운동 출력 차단을 검사합니다. **의존성 누락은 exit 2, 검사 실패는 exit 1, 전부 통과한 경우 exit 0**입니다. 결과 JSON의 상세 오류를 확인하세요. 통과해도 장시간 안정성이나 생물학적 타당성을 보장하지 않습니다.

OpenGL도 점검하려면 `--render`를 추가합니다. Linux headless 시스템에서는 환경에 따라 `MUJOCO_GL=egl` 설정과 EGL 라이브러리가 필요합니다. 이 선택 기능의 Mac/Windows/Linux 호환성은 제작 환경에서 시험하지 못했습니다.

## 3. 무엇이 바뀌었나

- A의 운동학적 위치 갱신 대신 **NeuroMechFly의 관절/접촉/중력**을 계산하는 어댑터를 작성했습니다. 상위 신경 출력 → 좌우 CPG 구동 → 42개 목표 관절각·발 접착 → MuJoCo라는 경로입니다. 몸통 위치를 매 스텝 덮어쓰는 이동 제어기는 없습니다.
- 신경 활성 모니터 옆에 **6개 발 접촉력, 42개 측정/목표 각도, 액추에이터 출력**을 추가했습니다. 측정이라는 표현은 가상 물리 엔진에서 읽는 값이라는 뜻이지 실물 초파리 센서 기록이 아닙니다.
- 신경–운동 연결 끊기, 마찰 배율, 측면 외력, 물리 CSV, 네이티브 메시 카메라 캡처가 추가되었습니다.
- 실제 B 비행은 없습니다. A 파일은 `legacy_a/FLY_LAB_A.html`에 별도로 보존했습니다.

## 4. 관찰과 조작

마우스 드래그는 카메라 회전, 휠은 확대/축소입니다. 추적·전체·상단·개체 시점을 제공합니다. 개체 시점은 관찰용 일반 카메라입니다. 복안 영상이 아닙니다. `Space`는 일시정지, `F`는 추적, `R`은 초기화입니다.

뉴런을 클릭하거나 검색해서 선택합니다. `+2초 자극`은 모델 입력을 추가하고, 억제는 해당 모델 출력을 0에 고정합니다. `PFL3-L 전체 억제`의 PFL3는 실제 ID가 아니라 A에서부터 사용한 설계 노드입니다.

`신경–운동 연결`을 끄면 좌우 구동은 0으로 되고 다리 제어는 중립 자세 목표로 바뀝니다. 몸의 속도를 강제로 0으로 덮어쓰지 않으므로 관성·접촉에 의한 움직임은 남을 수 있습니다. 이는 모든 액추에이터 전원 차단과 다릅니다.

발 접촉력은 기준 체중 대비 **BW**로 표시합니다. BW는 모델 질량×초기 중력 가속도의 크기로 정규화한 값이며 실제 생체 힘 측정과 다릅니다. 마찰 배율은 기본 접촉 파라미터에 대한 비율입니다. `0.5 BW / 50 ms` 밀기는 명시적인 가상 몸통 외력입니다.

장애물은 반경 0.8 mm의 바닥 구로 추가됩니다. 입력 높이는 장애물에는 적용되지 않습니다. 몸과 겹치는 배치는 거부합니다. 물리 장애물은 최대 12개입니다. 변경 시 정적 충돌 형상을 다시 컴파일하고 기존 몸·신경·제어 상태를 복원합니다. 장애물 재구성 경로는 다조건 검사에 포함되어 있습니다. 실행 결과는 로컬 검증 기록을 확인하세요.

벽이나 장애물이 가까우면 설계한 회피 상태가 선택한 회전 방향을 유지합니다. 회전량은 자기 운동 감각으로 적분하고, 전방 여유 공간을 확인한 뒤 전진을 재개합니다. 접촉이 지속되면 짧게 후진합니다. 화면에 `회피 회전`·`접촉 후진`으로 표시하며, 조향은 PFL3·DN 설계 노드의 출력을 거칩니다. 실제 생물의 신경 회로를 복원했다는 뜻은 아닙니다.

`충돌 진입 횟수`는 몸통의 바닥 접촉과 다리를 포함한 몸의 벽·장애물 접촉 시작을 셉니다. 발의 정상 바닥 지지는 제외합니다. 누적 이동 거리는 진동도 포함하므로 탐색 성공 여부는 경로와 회피 검사로 별도 평가합니다.

## 5. 비교 실험

같은 상태에서 대조군/개입군을 각각 **4 모델초** 실행하고 **1초**에 개입합니다. 원래 관찰 중인 모델을 변경하지 않습니다. 두 조건 모두 고정 방향 과제와 냄새 차단 조건을 적용한 뒤 비교합니다. 실시간 벽시계 4초 만에 끝난다는 뜻이 아닙니다.

PFL3-L 억제, 해부 연결 전달 차단, 시각 기준 제거, 신경–운동 연결 차단을 제공합니다. 넘어짐이나 수치 오류가 발생하면 결과를 성공으로 꾸미지 않고 중단합니다. 이것은 모델 민감도 실험이며 실제 광유전학 검증이 아닙니다.

## 6. 저장·복원

신경 CSV는 최근 120초의 10 Hz 기록이며, 기본 170개 활성도와 몸 상태/조향 명령을 합쳐 177열입니다. 물리 CSV는 시간, 42개 각도·목표각·출력, 6개 접촉력, 좌우 구동값으로 135열입니다. CSV 출력은 **시뮬레이션 결과**입니다.

체크포인트에는 MuJoCo `mjSTATE_INTEGRATION`, CPG 상태·난수, 반사 보정 상태, 신경·감각·난수, 환경·설정을 저장합니다. 다른 앱 버전·모델 구조·MuJoCo/NumPy 버전·A 체크포인트는 거부합니다. v0.2.0 체크포인트는 머리 프레임과 회피 상태가 달라 v0.2.1에서 직접 복원하지 않습니다. 복원은 대체 인스턴스에서 검증하므로 실패 시 기존 인스턴스를 유지합니다. 실제 MuJoCo 연속성은 로컬 physics gate로 확인해야 합니다.

리플레이는 초기 체크포인트와 모델 tick 단위 명령입니다. 단일 가져오기에서 최대 60 모델초, 명령 20,000개입니다. 장시간 전체 기록/청크 재생은 C에서 확장할 부분입니다. 체크포인트로 이어서 관찰하는 것은 별도로 가능합니다. 체크포인트에는 과거 그래프 버퍼를 포함하지 않으므로 복원 후 그래프는 새 기록 창에서 시작합니다.

A 실험 설정을 B로 옮기려면:

```sh
python tools/migrate_a.py A_checkpoint.json B_experiment.json
```

그 뒤 화면의 `체크포인트 / 리플레이 열기`에서 `B_experiment.json`을 선택합니다. **A의 seed·연결망·호환 행동 설정만 이전**하고, 몸/신경의 동적 상태와 A 환경 좌표는 옮기지 않습니다. B의 표준 바닥 환경에서 새로 시작하며 이를 출력 JSON에 명시합니다.

## 7. 계산 범위와 한계

신경 계산은 5 ms, 물리는 0.1 ms의 고정 시간 간격입니다. 1 신경 스텝에 물리 50 스텝을 사용합니다. 이 값은 계산 주기이지 실시간 성능 측정값이 아닙니다. 이번 경로는 **CPU 단일 물리 세계**이며 GPU/Warp 계산기는 연결하지 않았습니다.

상위 신경 모델은 실제 XYZ/yaw를 입력으로 받지 않습니다. 대신 64방향 광선 수광값, 9방향 거리, 분석적 냄새장, 이상화한 자기 운동 감각을 받습니다. 광선은 MuJoCo 환경 형상을 사용해 가림을 계산하지만, 복안/광학/시각 신경망은 아닙니다. 하나의 벽 표식이므로 이동에 따른 시차도 발생하며 절대 방향이나 세계 좌표 지도를 보장하지 않습니다.

저수준 공개 HybridTurningController는 자세·발 위치·접촉을 이용하는 **공학적 CPG/반사 제어기**입니다. 이 제어기가 사용하는 시뮬레이터 관측과 상위 신경 모델의 입력 경계는 다릅니다. 전체 커넥톰만으로 보행이 자연 발생하는 구조가 아닙니다.

브라우저 몸 표면은 도식적입니다. 다리 연쇄는 엔진에서 읽은 분절 위치를 연결합니다. 날개는 고정 시각화이며 공기력을 발생시키지 않습니다. 네이티브 카메라만 실제 FlyGym 메시를 렌더링합니다. 네이티브 카메라 캡처는 신경 입력으로 사용하지 않습니다.

## 8. 운영·보안

서버는 `127.0.0.1`에만 바인딩합니다. Host/Origin 검사, 첫 WebSocket 메시지의 임의 세션 토큰, 단일 제어 클라이언트, 요청 직렬화를 사용합니다. 코드를 업로드해 실행하는 API, pickle, 외부 임의 경로 읽기는 없습니다. 의존성 설치 뒤 실행 데이터는 외부 서비스로 전송하지 않습니다.

브라우저 요청이 있을 때만 계산합니다. 숨겨진 탭에서는 신규 계산 요청이 멈추고 이미 처리 중인 제한된 묶음은 끝날 수 있습니다. 브라우저를 닫아도 Python 프로세스는 남지만 자동으로 계속 계산하지 않습니다. 종료하려면 Ctrl+C를 누르세요. 인터넷 공개 서비스용 인증·다중 세션 설계는 아닙니다.

## 9. 개발과 테스트

```sh
python build.py
python -m unittest tests.test_core tests.test_server tests.test_validation tests.test_avoidance -v
python tests/browser_smoke.py
python tools/verify_physics.py --seconds 3
python tools/verify_navigation.py --jobs 4
python tools/run_physics_campaign.py
```

브라우저 테스트에는 Playwright 및 Chromium이 별도로 필요합니다. `tests/browser_smoke.py`의 실행 파일 경로를 로컬 환경에 맞추거나 Playwright 기본 브라우저를 사용하세요. 제작 환경에서는 관리 정책으로 localhost 브라우저 탐색이 막혀 같은 HTML을 `set_content`로 넣고 명시적인 대역 RPC를 사용했습니다. Python 서버의 실제 loopback WebSocket은 별도 검사했습니다. 이 설명은 v0.2.0 제작 당시의 이력입니다. 현재 브라우저→실제 물리 엔진 검증은 로컬 검증 기록에 따로 기록합니다.

`tests/fixture_body.py`와 `fixture_server.py`는 오직 테스트 대역입니다. 화면에 `UI TEST FIXTURE`를 크게 표시하고, `physics=false`를 반환합니다. production `run.py`에는 이 모드를 고르는 옵션이 없습니다.

## 10. 문서

- `docs/ARCHITECTURE.md`: 모듈, 좌표·시간·출력 규격, C 변경 지점.
- `docs/TESTING.md`: 통과한 검증과 막힌 경로의 정확한 구분.
- `docs/DATA_PROVENANCE.md`, `data/`: A에서 이어받은 발췌 데이터와 출처.
- `docs/MODEL_CARD.md`: 검증 한계와 설계 가정.

FlyGym: https://github.com/NeLy-EPFL/flygym (고정 커밋은 requirements.txt)
공식 보행 예제: https://neuromechfly.org/tutorials/4d_turning_controller/
MuJoCo API: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html

이 프로젝트는 NeuroMechFly/FlyGym/Janelia/SEISMIC의 공식 배포물이 아닙니다.
