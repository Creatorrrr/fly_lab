# CUDA 적용 및 Windows 실행 검증 — 2026-09-10

이 개발본은 실제 사용 가능한 CUDA를 우선 선택한다. NVIDIA RTX 4080 16 GB에서 설치, 커널 실행, 전체 FAFB 계산, 실제 FlyGym/MuJoCo 연결, 기록과 복원을 확인했다. 실행 의존성과 검증 자료는 현재 작업 폴더에 저장되어 있다.

## 설치와 실행

실측 환경은 Windows 11, Python 3.12.7, NVIDIA 드라이버 591.86이다. 프로젝트 `.venv`에 CuPy `14.2.0`, CUDA toolkit 패키지 `13.0.2`, PyTorch `2.14.0+cu130`을 설치했다. CuPy가 보고한 CUDA runtime은 `13020`, PyTorch CUDA build는 `13.0`이다. 시스템에 설치된 CUDA 12.8은 변경하지 않았다. 런타임과 NVRTC 등은 가상환경 패키지로 공급한다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.venv\Scripts\python.exe -X utf8 run_c.py --doctor
.\launch_c.bat
```

기본 `--backend auto`의 우선순위는 `exp_lif_cuda`, `exp_lif_mps`, `exp_lif_cpu_reference`이다. CUDA 탐지는 장치 수뿐 아니라 실제 CuPy 커널 컴파일과 실행을 확인한다. 명시적으로 요청한 GPU가 실패하면 오류를 표시한다. 화면에서는 사용할 수 없는 장치를 비활성화한다.

```powershell
.\launch_cuda.bat
.\launch_c.bat --backend exp_lif_cpu_reference
```

macOS/Linux의 명시적 CUDA 진입점은 `launch_cuda.sh`이다. 이 문서의 실장치 검증은 Windows에서 수행했다. 의존성 설치는 [CuPy 공식 설치 문서](https://docs.cupy.dev/en/stable/install.html)와 [PyTorch 공식 설치 경로](https://pytorch.org/get-started/locally/)를 기준으로 구성했다. 이 고정 CUDA 13 패키지 조합은 해당 런타임을 지원하는 드라이버가 필요하다.

현재 PC의 `flylab.local.json`:

```json
{
  "backend": "auto",
  "graph": "data/acquisitions/local-windows-20260910/bundle",
  "bindings": "data/acquisitions/local-windows-20260910/bindings.json"
}
```

로컬 설정은 Git에서 제외되며 `flylab.local.example.json`은 공유 예제다. CLI 인자가 우선한다. 체크포인트 복원은 저장된 계산 장치를 유지한다. 다른 장치로 옮길 때는 화면의 **신경 계산** 선택으로 명시적으로 전환하며, 전환 전 상태를 저장하고 신경·물리 배열의 보존을 검사한다.

## 구현 범위

- `CudaLIF`와 `lif.cu`: float32 세 적분 방식, 불응기, 지연 큐, 중복 pulse, 억제·전달 차단·개별 연결 차단, 선택 발화 기록, 체크포인트, 관측 집단과 운동 판독 집단을 지원한다. float64 요청은 기존 CuPy 연산 경로를 유지한다.
- 행마다 CSR 연결을 정해진 순서로 합산한다. 부동소수점 atomic add와 fast math를 사용하지 않으며 FMA 수축을 끈다. 발화가 전달되는 행만 계산한다.
- 반복 제어 주기를 CUDA Graph로 실행하고, 재사용 장치 버퍼와 pinned host memory로 입력·관측을 전송한다. 상태 검사, 선택 발화, 전압·발화율·64비트 누적 계수, 전체 요약을 한 번에 내려받는다.
- 독립 CUDA stream의 `begin_advance`/`finish_advance`를 기존 엔진의 CPU 물리 계산과 겹쳐 실행한다. 상태 저장·읽기·개입은 완료된 신경 주기에서 수행한다.
- ResearchLIF는 PyTorch CPU/MPS/CUDA와 `auto`를 지원한다. 이질적 세포 파라미터, 개별 지연·수용체 효과, 전기적 연결, 보상 기반 STDP, 개입과 복원 경로를 CUDA에서 검사했다. 이벤트·전기적 입력 버퍼를 재사용하고 상태 검사와 영역 통계의 장치 전송 횟수를 줄였다.
- 캠페인, 경로·조향·관절 피드백 진단, 수치·감각운동·단일 관절·기록·오류 복구 검증, 백엔드·실행 성능 비교 도구에 `auto` 및 CUDA 선택을 적용했다. 체크포인트 기반 프로파일러는 저장된 장치를 사용한다.
- Windows의 파일 잠금, 독립 캠페인 자식 프로세스와 취소, 메모리 매핑 파일의 삭제 제한, UTF-8 자식 프로세스 출력을 처리했다. CUDA 소스도 실행 코드의 해시·캠페인 증빙 복사에 포함한다.

MuJoCo 전신 물리와 Hill 단일 관절 물리는 CPU에 남는다. 종전 `probe_mjx_metal.py`는 실제 C 몸 계산기가 아닌 MJX 호환성 진단이다. 파일 이름은 유지하고 `--device auto/cuda/metal`을 추가했지만, 이 Windows 가상환경에는 JAX/MJX가 없어 `BLOCKED`를 기록했다. 이 결과를 GPU 물리 검증으로 해석하지 않는다. JAX의 NVIDIA GPU 지원 환경은 [공식 지원 표](https://docs.jax.dev/en/latest/installation.html#supported-platforms)를 참고한다.

## 측정 결과

FAFB 그래프는 뉴런 **139,255개**, 연결 **3,732,460개**, 해시 `c914dc5ea54fc1744ed7bf03b56ec190ae26b0d418722c057a70f4f7e75d8a8f`이다. 활성 실험의 2.295초 체크포인트에서 같은 입력·시드·상태로 50 ms를 3회 실행했다. GPU 외의 병렬 검증 작업은 성능 측정 중 실행하지 않았다.

| 신경 계산만 | 벽시계 중앙값 | 비교 |
|---|---:|---|
| 기존 CuPy ExpLIF | 221.60 ms | 기존 CUDA 경로 |
| 전용 CUDA 커널 | 60.52 ms | CUDA Graph 제외 |
| 전용 커널 + CUDA Graph | 58.58 ms | 기존 CuPy 대비 3.78배 |

이 조건에서 CUDA Graph 자체의 추가 향상은 약 1.03배다. 전체 3.78배 향상을 CUDA Graph만의 효과로 돌리지 않는다. 신경 계산 표는 최초 컴파일·그래프 구성을 워밍업에서 제외했다. 별도 CPU 대비 3회 측정은 1.9443초 대 0.06325초로 30.74배였으며, 별도 측정의 배수를 위 표와 곱하지 않는다.

| 실제 신경·물리 연속 실행 | CPU 중앙값 | CUDA 중앙값 | CPU 대비 |
|---|---:|---:|---:|
| C_SHADOW | 2.1496 s | 0.1854 s | 11.59배 |
| C_STRICT | 2.1235 s | 0.1946 s | 10.91배 |
| C_ASSISTED | 2.1719 s | 0.1894 s | 11.47배 |

물리 표는 엔진 구성 시간을 제외하고 주기 내 신경·몸·관측 계산을 포함한다. CUDA Graph 최초 구성도 이 구간에 포함한다. 최종 몸 상태와 반복 실행의 몸·추적 기록은 정확히 일치했다. 선택 발화·정수 계수·불응기·전압·발화율·지연 큐는 CPU와 같았고, 극소 전류 값에서 최대 `2.39e-40` 차이가 있었다. 별도의 초기 상태·Poisson 입력 500틱 비교에서는 세 적분 방식의 전체 배열과 발화가 모두 정확히 일치했다.

실시간보다 빠르다는 의미는 아니다. 위 물리 비교는 모델 50 ms에 실제 약 185–195 ms가 걸린 짧은 연속 실행이다. 장시간 행동, 자연 탐색, BANC 전신 보행, 생물학적 정확도의 개선은 이 실험에서 평가하지 않았다.

## 검증과 원자료

- 전체 회귀: `regression-fixed/report.json` — 284개 중 263개 통과, 21개 건너뜀, 실패·오류 0. 건너뜀은 Apple MPS 실장치 검사 20개와 별도 검증기로 위임된 기존 CUDA 검사 1개다.
- CUDA 최종 검사: `checks-final/report.json` — 실제 NVIDIA에서 19개 통과, 건너뜀·실패·오류 0. float64 일반 CuPy 경로, 64비트 계수의 관측 버퍼, 오류 시 비동기 계산 정리, 실제 단일 관절 물리와 복원도 포함한다. 전체 회귀 실행 후 추가한 CUDA 검사 3개를 이 실행에서 확인했다.
- ResearchLIF 전체 FAFB: `full-research/report.json` — 이질적 파라미터·개별 지연·전기적 연결·STDP를 함께 적용한 합성 생리 프로파일, CPU/CUDA 각 200틱의 모든 배열과 선택 발화가 정확히 일치했다. CUDA 상태 복원 후 별도 100틱 반복도 정확히 일치했다. 이 짧은 조건의 결과가 모든 CUDA scatter 연산의 장기 결정성을 보장하지는 않는다.
- 전체 FAFB 세 적분 방식: `full-integrations/report.json` — 각 500틱, 전체 신경 상태와 선택 발화 일치.
- 반복 성능 및 결과 비교: `cuda-ablation/report.json` — 기존 CuPy/커널/CUDA Graph와 세 물리 모드, 각 3회. `inputs.npz`와 최종 체크포인트·추적 기록을 동봉했다.
- 초기 CPU 비교: `full-fafb/report.json` — 별도 신경 계산 3회와 단일 C_SHADOW 연속 실행.
- 실제 CUDA 기록·환경 편집·리플레이: `workbench/report.json` — 512채널 기록, 누락 0, 개입 취소·기록 재생의 상태 차이 0, 감각 차단 대조의 기술적 검사 통과.
- 실제 CUDA 오류 복구: `recovery/report.json` — 명시적으로 주입한 오류의 진단 저장, 새 실험 복구, 500틱 진행, 체크포인트 저장 통과. 자연 발생 오류 빈도는 측정하지 않았다.
- Windows 캠페인: `windows-worker-tests.log` — 실제 FlyGym 자식 프로세스 실행, 작업 간 잠금, 취소 후 해제 통과.
- 최초 실패 자료도 보존했다. 패키지 추가 전 체크포인트의 의존성 불일치와 Windows mmap/인코딩 실패를 성공 결과로 덮어쓰지 않았다.

모든 상대 원자료 경로의 기준은 `verification/cuda-20260910/`이다. 이 폴더는 로컬 검증 산출물로 Git에서 제외된다. 직접 확인한 `pip check`, Python 미정의 이름 검사, `git diff --check`도 통과했다.

## 현재 앱에 적용한 상태

포트 8766 서버를 새 코드로 재시작한 뒤, 원래 체크포인트 `checkpoint-0107c5b8c01f`의 2.295초 상태를 복원했다. 브라우저에서 CUDA로 전환하고 `checkpoint-d18e8cacfdad`를 저장했다. `live-transfer.json`은 전압·전류·지연 큐·발화율·누적 발화·불응기·개입 마스크와 몸 상태가 전환 전후 정확히 같음을 기록한다. 원래 C_SHADOW 모드와 환경을 유지했다.

그 후 실제 앱 재생에서 `LOCAL · CUDA / PHYSICS`, 17.885 모델초, tick 178850, 수신 공백 0회를 확인했다. 이는 짧은 UI 연속 실행 확인이며 위 반복 성능 비교와는 별도다. Windows 서버의 현재 실행 정보는 `live-runtime.json`, 로그는 `artifacts/c/runtime/server-cuda.stdout.log`와 `server-cuda.stderr.log`에 남겼다. CUDA로 전환한 체크포인트를 다시 열려면 다음 명령을 사용한다.

```powershell
.venv\Scripts\python.exe -X utf8 run_c.py --restore-checkpoint checkpoint-d18e8cacfdad
```

새 검증을 실행하는 예:

```powershell
.venv\Scripts\python.exe -X utf8 tools/run_c_checks.py --tier cuda --out verification/my-cuda-checks
.venv\Scripts\python.exe -X utf8 tools/benchmark_c_cuda.py --graph data/acquisitions/local-windows-20260910/bundle --bindings data/acquisitions/local-windows-20260910/bindings.json --checkpoint artifacts/c/checkpoints/checkpoint-0107c5b8c01f --seconds .05 --repeats 3 --physics --out verification/my-cuda-benchmark
```

검증 도구는 기존 출력 디렉터리를 덮어쓰지 않는다. 새 환경에서는 해당 데이터와 체크포인트를 준비하거나 자신의 검증된 경로로 바꿔 실행한다.
