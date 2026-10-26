# E246: 저장된 감각·신경 상태의 운동 상태 구별 진단

E245의 유효한 수동 관측을 사용해, 다음 수정이 수용기 입력과 기존 신경 연결 중 어디를 대상으로 해야 하는지 좁힌다. E242 적합 실패와 E244 변화량 표현 기각은 그대로다. 이 실험은 저장 자료 분석이며 새 모델·신체·CUDA 실행이나 계수 적합은 없다. 최종 자율 보행의 F1–F8, M2, E150의 부분 성과와 미사용 F6는 변경하지 않는다.

## 질문과 비교 의미

H246: 원래 MN 입력 합계 표현보다, 직접 수용기 또는 그 이후의 실제 감각·내부 뉴런 상태가 같은 제어 시점의 교사 명령을 더 잘 구분하는 표현을 제공한다. 경쟁 설명은 현재 신체 특징부터 구분이 부족하거나, 선택한 거리와 상태만으로 어느 단계도 개선되지 않는다는 것이다.

공통 판정은 상태 벡터가 가까운 두 제어 시점의 교사 q 차이다. 최근접 시점의 q를 실제 제어에 사용하지 않는다. 표현별 차원·가중치·표준화가 다르므로 결과는 **이 고정 거리에서의 구별성 진단**이다. 정보이론적 손실, 뉴런의 인과 역할, 구현 가능한 신경 수정 또는 보행 성공을 증명하지 않는다. 별도 개입 없이 감각→신경 사이의 인과적 병목을 확정하지 않는다.

## 고정 입력과 다섯 표현

원본 `e245-passive-signals/teacher/signal-raw.npz`, metadata, control 기록 및 검증된 roster를 사용한다. 모든 제어 행은 E241의 target 명령 및 E243의 저장 기준과 일치해야 한다. 전체 2,001행의 MN 및 q/tendon/counter가 같은 E240 자료임을 먼저 확인한다. 391MN 순서와 free343/fixed48, 각 근육·다리 소속은 E241 parameters/groups에서 가져오고, 원래 anatomy sensitivity는 E243 방법으로 재계산하여 저장 weights와 대조한다.

| 표현 | 실제 열과 가중치 |
|---|---|
| R0 `receptor_features` | 각 다리의 실제 241 port 중 해당 다리 `last_features`. 다리 안에서 각 port 가중치 1/P. |
| R1 `injected_filter` | 같은 port의 `old_filtered`. 원래 drive를 만들 때 사용된 값. 가중치 1/P. 실제 sensory_drive의 float32 주입도 대조한다. |
| R2 `afferent_rates` | 각 port에 속한 실제 sensory neuron의 post-neural rate. 각 port가 전체에서 1/P를 갖고, port 안의 N개 뉴런에 1/(P·N)씩 배정. |
| R3 `configured_internal_rates` | 기존 `core_adaptation` 목록에서 MN391을 뺀 실제 51개(49 VNC intrinsic, 2 descending)의 post-neural rate, 각 1/51. 모든 다리에 같은 전체 상태 벡터를 사용하며 담당 다리를 추정하지 않는다. NN 탐색은 한 번만 하고 q를 다리별로 평가한다. |
| R4 `native_motor_rates` | 각 다리에 속한 실제 free MN rate. 가중치는 E243의 해당 다리 원래 anatomy sensitivity를 그대로 사용한다. fixed48은 이 표현에 섞지 않으며 원래 데이터에서 삭제하지도 않는다. |

R0의 raw 특징은 encode c에서 측정되고, R1은 같은 encode가 이전 필터에서 꺼내 사용한 입력이다. R2–R4는 c의 신경 20 ticks 후다. 모두 c의 교사 명령과 대조하지만 단계별 지연이 다르다. raw 특징이 좋더라도 현재 신경 출력이 그 값을 같은 제어에서 이미 이용한다고 주장하지 않는다. trace c의 몸은 명령 적용 후이며 분석에서 이를 입력으로 대신 사용하지 않는다. row0은 NN 계산에 넣지 않는다.

각 표현의 열마다 TRAIN controls 1..1200만 사용해 float64 mean과 population std(ddof=0)를 계산한다. scale=max(std,1e−4). 행 x의 거리 벡터는 `(x−mean)/scale * sqrt(weight)`이다. 평균·표준편차·floor mask·weights·실제 열 ID를 저장한다. tail controls1201..2000은 표준화나 선택에 사용하지 않는다. 다만 이 tail은 이미 노출된 개발 구간이며 독립 확인 자료가 아니다. 일정한 DNg100 입력/발화, 0인 touch 등은 실제 값 그대로 처리하고 생물학적 역할을 추정하지 않는다.

질의 controls1..2000, 참조 controls1..1200, `abs(query_control−reference_control) <= 20` 제외, 동률이면 가장 이른 참조를 선택한다. 제곱 Euclidean distance를 float64 norm/dot 방식으로 계산해 roundoff의 음수만0으로 제한하고, 저장 거리는 제곱근이다. batch128, CPU BLAS1. 각 선택 참조 j에 대해 `q_target[c]−q_target[j]`의 7관절 다리별 RMS와 전체42관절 RMS/최대절댓값을 계산한다. R3는 여섯 다리의 NN indices가 같다.

비교 기준 R5는 **E243 저장 원래 9좌표 MN 입력 표현**의 NN indices/distance/q 차이다. 이를 재탐색하거나 새 정규화하지 않고, 현재 target으로 저장 q 차를 exact 재구성해 확인한다. baseline의 거리와 새 표현 거리의 숫자 크기를 서로 비교하지 않는다. 물리 의미가 같은 q 차만 비교한다.

## 사전 판정과 다음 분기

기록 구간은 TRAIN1..1200, startup1..99, steady100..1200, observed_tail1201..2000 및 연속200-control 구간10개다. 각 표현에 대해 steady와 tail에서 모두 전체 q RMS가 R5의 0.8배 이하이고, 여섯 다리 RMS가 각각 R5 이하일 때 `separates_better_under_fixed_metric=true`다. epsilon으로 통과시키지 않는다. startup/최대오차/거리분포는 기술하며 판정 문턱을 사후 바꾸지 않는다. 실패한 기준도 모두 남긴다.

- R1 또는 R2가 위 기준을 통과하고 R4가 실패하면, 기존 신경 전달/읽기 경로를 보존하면서 해당 정보를 받는 실제 synapse의 유효 반응을 바꾸는 개입 설계를 우선한다. 합계 E/I 표현으로 식별 정보가 뭉개졌다고 확정하지는 않는다.
- R0만 통과하고 R1/R2가 실패하면, 기존 필터 지연과 감각 세포 반응을 구분하는 개입을 우선한다. 이 관측만으로 필터를 제거하지 않는다.
- R3만 또는 R3도 통과하면, 기존 해부학 연결을 따라 정보를 전달할 수 있는지 검토한다. 51개를 모든 근육에 직접 연결하거나 외부 위상으로 바꾸지 않는다.
- 아무 표현도 통과하지 않으면 RBF 계수를 다시 맞추지 않는다. 수용기 관측 범위와 이 거리 진단의 한계부터 검토한다. 표준화·가중치 sweep은 이번 실험에 포함하지 않는다.

이 분기는 다음 설계의 우선순위를 정한다. 어떤 경우에도 실제 후보 채택/F2/F6/전체 완료는 false이며, 교사 없는 폐회로 및 F1–F8은 별도 검증한다.

## 실행·검증·자원

부모 설계 → 실제 gpt-6-sol/medium 구현 → 부모 FIRST → 같은 Sol 첫 수정 → 부모 SECOND → 남은 수정은 부모 순서를 적용한다. medium 선택 이유는 이미 검증된 저장 NN 연산을 새 포트/뉴런 그룹에 적용하는 정형적 CPU 분석이고, CUDA나 신경 적분 구현을 바꾸지 않기 때문이다. 소유 파일은 `tools/analyze_c_banc_m3_signal_discrimination.py`, `tests/test_c_banc_signal_discrimination.py` 두 개다. E243 anatomy helper를 재사용할 수 있으나 과거 파일은 바꾸지 않는다.

생성 테스트는 서로 다른 port 크기와 여러 MN의 가중치, exact 제어 정렬, 동률/20-control 제외, 미래 tail 변화가 TRAIN mean/std에 영향을 주지 않음, R3 공통 참조, 오류/부분 기록을 점검한다. 부모는 원자료에서 정규화/가중치/targets를 재계산하고 고정된 질의 표본의 거리를 직접 뺄셈으로 검산한다. 오류시 계산된 부분과 완료 mask, 최초 오류·후속 오류를 보존한다. 결과/manifest/기한의 늦은 실패에도 가설 판정이 남지 않게 한다.

E243의 비슷한 저장 NN 계산은8.9011초였다. 이번 총 좌표 수와 원자료 크기를 고려해 원본 CPU 분석 한 번: internal40초(마무리5초 포함), external50초, supervisor60초, CPU1/GPU0, RSS3GiB, output128MiB, free disk512MiB, fit/solver/body/network/CUDA/retry0으로 제한한다. E245 이후 누적6736.780708699622 /6800초, 잔여63.219291300378245초 안에60초를 예약하며 추가 배정은0이다. 저장 분석 비용도 원본 회계에 보수적으로 더하고, 생성 테스트/부모 검산은 별도 측정한다. 결과를 본 뒤 한도나 기준을 바꾸지 않는다.
