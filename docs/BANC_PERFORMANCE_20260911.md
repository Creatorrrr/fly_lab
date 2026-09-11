# BANC 보행 속도 진단과 개선

## 적용 결과 · 2026-09-11

새 BANC 실험의 기본값을 **무손실 packed CSR + 신경 RK4 0.25ms**로
바꿨다. RTX 4080에서 신경·물리·감각·운동을 포함한 처리율은 기준선의
**2.16배**였다. 같은 저장 상태에서 0.15모델초를 5쌍 실행하고 순서를
교대했다. 중앙값은 2.690887초에서 1.245047초로 줄었다.
시작·컴파일·예열·브라우저 전송은 제외한 수치이며 실시간 실행은 아니다.

등록된 158,706개 뉴런, 11,584,852개 연결과 원래 float32 가중치,
적응 방정식·감각·운동 매핑은 유지한다. 물리는 0.1ms, 감각·운동은 5ms다.
신경 적분 간격이 달라지는 0.25ms 후보는 별도 모델 해시
`fbd46016a3a294e9f099427aca2e364a343690a6eaf4ec2189fac2c04cf9c3a2`를
사용한다. 이전 0.1ms 저장 상태는 0.1ms로 복원된다.
화면에서 다음 BANC 실험의 계산 설정과 현재 실행의 간격을 구분한다.

| 검증 | 직접 결과 |
|---|---|
| 가중치 저장 형식만 변경, 0.1ms 유지 | 전체 신경·감각·운동·물리 미래 상태 차이 0. 전체 속도 1.065배. |
| 동일 held input 100ms, 0.25ms 대 0.1ms | rate RMS 0.000359 / max 0.028656, adaptation RMS 0.001108 / max 0.024414. 사전 RMS≤0.2 / max≤2.0 통과. |
| 10초 자유 보행, seed42 | net 5.864893mm, signed forward 5.352871mm. 기존 각각≥5mm 통과. 오류·낙하·고착 없음, 여섯 다리 운동 지속, CPG 불변. |
| 감각/하행 출력 절단 | 실제 관절 차이 0.043425rad / 0.001907rad. |
| 전체 회로 전달 절단 | 0.3초 후 motor max 0.00006116 모델 단위. |
| 최종 운동 연결 절단 | 신경 활동 유지, 관절 목표 변화 정확히 0. |
| 저장·복원 후 0.3초 미래 | rate·adaptation·drive·mask·신체 적분 상태·감각·운동 버퍼 차이 모두 0. |
| 관련 회귀 | 50개 중 49통과, 선택적 검사 1생략. 빌드·문법·Ruff 통과. |

0.25ms 후보의 보행 경로는 기존 0.1ms와 같지 않다. 기존 seed42 결과
net6.638688mm/forward5.966815mm에서 위 값으로 바뀌었으며 두 모델 모두
고정 보행 기준을 통과했다. 이 결과를 여러 지형/행동에 대한 동등성이나
생물학적 검증으로 확대하지 않는다.

채택 계획: [BANC_PERFORMANCE_GOAL_PLAN_20260911.md](BANC_PERFORMANCE_GOAL_PLAN_20260911.md).
직접 원자료는 `verification/banc-speed-20260911/`의 `numerics-01`,
`throughput-exact-01`, `throughput-dt025-01`, `walking-dt025-10s-01`,
`session-dt025-01`, `regression.log`에 있다. 아래 진단은 변경 전 기록이다.

```powershell
.venv\Scripts\python.exe tools/compare_c_banc_numerics.py --out verification/banc-speed-new-numerics
.venv\Scripts\python.exe tools/benchmark_c_banc_session.py --out verification/banc-speed-new-throughput --neural-dt 0.00025 --cuda-implementation packed
.venv\Scripts\python.exe tools/verify_c_banc_rate_walking.py --out verification/banc-speed-new-walking --seconds 10 --case 4:18 --neural-dt 0.00025 --cuda-implementation packed
.venv\Scripts\python.exe tools/verify_c_banc_walking_session.py --out verification/banc-speed-new-session --neural-dt 0.00025 --cuda-implementation packed
```

각 실행은 새 출력 폴더를 요구한다. GPU 실험은 순서대로 실행한다.

## 변경 전 진단

RTX 4080에서 실제 저장 상태 `banc-757d5d9fe381`를 별도 세션으로 복원해
측정했다. 초기화와 예열을 제외하고 50ms 진행을 세 번 실행했다.
BANC 적응 rate 모델의 병목이며 과거 C_STRICT LIF 성능과 구분한다.

## 측정

| 구간 | 0.15모델초 처리 시간 | 비중 |
|---|---:|---:|
| 신경 계산 | 2.374s | 87.09% |
| MuJoCo 물리 | 0.261s | 9.58% |
| 감각 변환 | 0.0569s | 2.09% |
| 운동 변환 | 0.0250s | 0.92% |
| 운동 출력 읽기·진단·기타 | 0.0088s | 0.32% |
| 합계 | 2.726s | 100% |

순차 반복의 실제 시간은 0.902, 0.921, 0.903초였다. 모델/실제 시간은
약 0.055배다. 별도로 측정한 영상 생성 중앙값은 0.58ms, PNG·JSON
변환은 5.24ms였다. 네트워크와 브라우저 렌더는 위 합계에 포함하지 않는다.
화면은 50모델ms마다 갱신하므로 이 영상 비용만 줄여서는 큰 가속을 얻기 어렵다.

뉴런 158,706개와 쌍 연결 11,584,852개를 0.1ms 적분 간격, RK4 네 단계로
계산한다. 모델 1초당 희소 행렬곱은 40,000회다. 현재도 CUDA Graph를
사용하므로 Graph 활성화만 추가하는 해결책은 적용되지 않는다.

원자료: `verification/banc-speed-20260911/profile-01/profile.json`.
재현 도구: `tools/profile_c_banc_walking.py`.

## 수치를 보존하는 저장 형식 실험

시냅스 수에서 float를 다시 계산하면 기존 float32 값과 최대
1.91e-6 차이가 생겨 해당 재구성은 채택하지 않았다. 대신 기존 가중치의
비트 패턴을 사전에 저장하고, 뉴런 인덱스와 사전 인덱스를 32비트 하나에
넣는 별도 프로토타입을 만들었다. 뉴런·연결·가중치를 삭제하거나 반올림하지 않는다.

| 행렬곱 레이아웃 | GPU 참조 버퍼 | 연산 중앙값 |
|---|---:|---:|
| 기존 64비트 인덱스 | 140.29MB | 0.3461ms |
| 32비트 인덱스 | 93.31MB | 0.3372ms |
| 정확한 가중치 사전과 32비트 묶음 | 46.98MB | 0.3204ms |
| 위 방식과 행 길이별 실행 순서 | 47.61MB | 0.3217ms |

CUDA Graph 안에서 같은 행렬곱을 100회 묶고 레이아웃 순서를 교대해
7회 측정했다. 원래 저장된 신경 상태와 무작위 rate·출력 마스크 모두
기존 커널의 계산 결과와 비트 단위로 일치했다. 가장 빠른 후보의 시간
감소는 약 7.4%, 처리율 향상은 약 1.08배다. 전체 신경 적분이나 보행의
가속을 측정한 값은 아니다. 실행 중인 제품 모델에는 아직 적용하지 않았다.

원자료: `verification/banc-speed-20260911/layouts-03/result.json`.
재현 도구: `tools/benchmark_c_banc_layout.py`.

## 권장 순서

1. 신경 GPU 커널의 연산 통합·버퍼 재사용·희소 행렬곱 실행 방식을 비교한다.
   원본 회로와 적분 간격을 유지하는 후보부터 동일 미래 상태를 검사한다.
   합산 순서가 달라지는 커널은 별도 수치 모델로 식별하고 행동도 재검증한다.
2. 별도 후보로 신경 적분 간격을 0.1ms에서 0.2ms 또는 0.25ms로 늘린다.
   물리 0.1ms와 감각/운동 5ms는 유지한다. 0.25ms이면 신경 적분 횟수가
   2.5분의 1이 된다. 현재 비중만 적용한 전체 가속의 이론값은 약 2.1배다.
   실제 속도와 수치·행동 정확도는 아직 측정하지 않았다. 체크포인트 시간
   단위 및 현재 `50 neural steps / control` 가정도 함께 바꿔야 한다.
3. 신경 계산을 줄인 뒤 물리와 감각 변환을 다시 측정한다. 현재 비중에서
   물리 비용을 전부 없애도 전체 속도의 이론 상한은 약 1.11배다.
   단일 개체의 GPU 물리 전환을 첫 해결책으로 선택할 근거는 약하다.

0.25ms 후보는 같은 초기 상태·감각 입력에서 신경 오차와 리듬을 먼저
비교하고, 10초 보행 기준·넘어짐·전체 전달/운동 출력 차단·저장 복원을
검사해야 한다. 표시 시간만 앞당기거나 모델 tick을 생략하지 않는다.
뉴런을 줄이는 방식은 사용자가 요청한 등록된 BANC 회로 전체 계산의 범위를
바꾸므로 기본 가속안에서 제외한다.

현재 약 18–20배의 간격이 있으므로 소폭의 저장 형식 개선이나 단일
GPU 교체로 실시간을 보장할 근거는 없다. 먼저 검증 가능한 단계별 가속을
측정한 뒤 실시간 목표에 필요한 추가 구조 변경을 결정하는 것이 적절하다.

프로파일링과 메모리 접근을 우선하는 근거는
[NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)와
일치한다. 희소 연산 변경 시 결정성 조건은
[cuSPARSE 재현성 문서](https://docs.nvidia.com/cuda/cusparse/index.html#reproducibility)를
함께 확인한다. 제품의 실제 효과는 위 로컬 측정과 후속 비교로 판정한다.
