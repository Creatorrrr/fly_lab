# BANC CUDA 최적화 적용 — 2026-09-12

**마스크 사전 계산을 실제 packed CUDA 경로에 적용했다.** 현재 BANC 모델 대비 전체 처리 속도는 1.48배이며, 10초 보행 기록이 기존 결과와 정확히 일치했다. 실행 중인 8770 서버도 갱신했다.

## 변경

`flylab/c/adaptive_rate.py`에서 매 RK4 단계마다 `rate * output_mask`를 뉴런별로 계산하고, packed 희소 행렬곱은 그 결과를 읽는다. 연결마다 반복하던 마스크 접근과 곱셈을 줄인다. 이 연산도 CUDA Graph에 캡처되므로 재생·신경 차단·저장 상태 복원 후마다 최신 값으로 다시 계산된다.

뉴런 158,706개, 연결 쌍 11,584,852개, 원래 float32 가중치, 고정 warp 합산 순서, RK4 방정식과 시간 간격은 유지했다. 현재 신경 0.25ms·물리 0.1ms·감각 및 운동 5ms 설정을 변경하지 않았다. 모델 해시는 기존 `fbd46016a3a294e9f099427aca2e364a343690a6eaf4ec2189fac2c04cf9c3a2`이며 기존 체크포인트를 그대로 복원한다.

이전 packed 커널은 `tools/banc_cuda_reference.py`에 비교용으로 보존했다. 행렬곱·전체 세션 비교 도구도 현재 실행 코드와 이 기준선을 비교하도록 수정했다. 운영 모델은 이 비교용 클래스를 사용하지 않는다.

## 확인한 효과

같은 저장 상태에서 0.05모델초 워밍업 후 시작 상태를 복원하고, 0.15모델초를 5쌍 측정했다. 실행 순서를 번갈아 배치했고 화면의 기본 실험과 BANC는 정지했다. 초기화·컴파일·저장·브라우저 통신은 제외했다.

| 항목 | 변경 전 | 변경 후 |
| --- | ---: | ---: |
| 전체 처리 시간 중앙값 | 1.185207초 | 0.799596초 |
| 모델 시간 / 실제 시간 | 0.1266 | 0.1876 |

속도 **1.482배**, 처리 시간 **32.5% 감소**. 신경·운동 어댑터·신체 상태는 양쪽 구현 및 반복 실행 사이에서 비트 단위로 일치했다.

적용 후 별도 프로파일에서는 신경 계산 60.97%, 물리 29.00%, 감각 6.56%, 운동 디코딩 2.72%였다. 신경 계산 시간이 줄어 물리 처리의 상대적 비중이 커졌다. 여전히 실시간 실행은 아니다.

## 검증

- 관련 단위 검사 **33개 통과**. CUDA Graph 재생 중 입력·차단 변경, 초기화 후 복원, 적응 gain 0, 작은 신경 활동 값, 서로 다른 캡처 길이, float32 출력 일치를 포함한다.
- **10모델초 평지 보행 PASS**: 수평 순이동 5.864893mm, 몸 방향 순전진 5.352871mm. 오류·비정상 접촉·고착이 없고 외부 CPG 상태가 유지됐다. 기존 보행 판정 기준을 변경하지 않았다.
- 이전 0.25ms 보행 결과와 **2,000개 시점의 운동신경 출력, 관절 목표·실측 각도, 하중·접촉, 마지막 전체 rate·adaptation이 비트 단위로 일치**했다. 2,001행의 위치·방향·접촉 추적 파일도 바이트 단위로 같다.
- 저장·복원 후 0.3모델초의 신경·물리 적분·운동 어댑터 상태 오차가 모두 0이었다.
- 감각 및 하행 출력 차단은 기존과 같은 영향을 보였다. 전체 회로 전달 차단 후 운동신경 최댓값은 0.00006116, 최종 운동 출력 차단 후 관절 목표 변화는 정확히 0이었다.
- Ruff 검사와 포맷 검사 통과. 이번 변경은 단일 CUDA 계산 경로이며 CPU 및 기준 CUDA 커널은 유지했다.

10초의 같은 조건에서 결과를 보존한 검증이며, 모든 지형·행동·장시간 조건의 안정성이나 생물학적 타당성을 새로 입증한 것은 아니다.

## RK4 연산 통합 후보의 판단

마스크 최적화 이후 원소별 연산 통합도 비교했다. 설치된 PyTorch의 스칼라 나눗셈과 clamp 의미를 따라 구현하고 기존 `torch.tanh`를 유지했지만, 전체 BANC 비교에서 rate 배열의 비트 일치가 실패했다. 추가 속도 향상은 약 3%였다. **이 후보는 적용하지 않았다.** 수치 허용오차를 완화하지 않았으며 실패 로그와 후보 소스를 보존했다. 정확한 차이 발생 지점은 이 실험에서 확정하지 않았다.

참조한 설치 버전의 공식 소스: [스칼라 나눗셈](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu), [clamp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/TensorCompare.cu), [tanh](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu).

## 실제 페이지

8770 서버를 갱신하고 기본 실험 150.650초와 BANC 8.750초를 복원했다. 실제 BANC 재생 버튼으로 11.250초까지 진행하며 CUDA 표시, 운동신경 출력 변화, 이동 및 실제 메시 영상 갱신을 확인했다. 화면의 최근 진행 속도는 약 0.18모델초/실제초였다. 기본 실험 시간은 그대로 유지됐다. 시험 후에는 사용자의 BANC 8.750초 정지 상태로 되돌렸다. 브라우저 오류 로그는 없었다.

- 기본 실험 저장 상태: `checkpoint-a8300940e5f8`
- 적용 전 BANC 상태: `banc-00c7ed0b2c71`
- 실제 재생 시험 결과: `banc-6dee06360013`
- 적용 커널 소스 SHA256: `d87a365b812b617f676ec8406cfb6d8573c4de4024bc20bb07e4b206efd6e74e`

## 재현 자료

`verification/banc-cuda-apply-20260912/`에 다음을 보존했다.

- `throughput-01/`: 변경 전·후 5쌍 비교, 소스 해시, 반복 및 구현 간 비트 일치.
- `walking-01/`, `walking-exactness.json`: 10초 보행, 기존 기록과 전체 신호·궤적 비교.
- `session-01/`: 저장·복원과 동일 시작 상태의 네 종류 차단 실험.
- `regression.log`, `profile-01/`: 33개 검사 및 적용 후 병목 측정.
- `fusion-01/`, `fusion-01.log`, `fusion-rejected/`: 채택하지 않은 연산 통합의 실패 기록과 소스.
- `baseline/`, `server-launch.json`: 적용 전 소스와 실제 서버 갱신 기록.

재현 명령은 저장소 루트에서 실행한다. GPU 실행은 순차적으로 진행하며 출력 폴더 이름은 새로 지정한다.

```powershell
.venv\Scripts\python.exe tools/probe_c_banc_premask.py --out verification/banc-cuda-new-throughput
.venv\Scripts\python.exe tools/verify_c_banc_rate_walking.py --out verification/banc-cuda-new-walking --seconds 10 --case 4:18 --neural-dt 0.00025 --cuda-implementation packed
.venv\Scripts\python.exe tools/verify_c_banc_walking_session.py --out verification/banc-cuda-new-session --neural-dt 0.00025 --cuda-implementation packed
```
