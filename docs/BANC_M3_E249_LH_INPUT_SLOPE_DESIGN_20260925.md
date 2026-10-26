# E249 저장된 LH 입력 기여와 전달함수 기울기 — 2026-09-25

E248은 유효한 신경 대조였지만 정규화 R와 D 모두1e−4보다 작았다. 다음 결정은 감각 입력의 작은 직접 기여와 내부 뉴런의 포화를 구분하는 것이다. 현재 M3/F1–F8 계약과 E248 기각을 유지한다. 이 분석은 저장 상태의 국소 기술이며 신경 모델을 새로 실행하거나 미래 보행을 예측하지 않는다.

## 고정 자료와 계산

E150의 정확한 native CSR, E248의 새 neural-start 및 고정6개 arm/disk_intact_low 최종 상태를 사용한다. 모두 원래 network identity a31f899cd948a4297a73db66569e5edc2210ac9aa3601aa83e96cb9ff68ce244와 frozen 과학적 tick40000 또는40800이어야 한다. 분석 행 순서는 baseline, intact_low, intact_high, bridge_cut_low, bridge_cut_high, sensory_cut_low, sensory_cut_high, disk_intact_low다. 원래 E248 선정 감각23개, 내부4개, 전체free LH MN56개를 이 순서로 합쳐83개 postsynaptic 행을 분석한다. 각 그룹 안에서는 원래 node_row 오름차순이며 세 집합은 겹치지 않는다. 원래 ID와 annotation을 보존한다.

각 실제 CSR 행의 모든 선언 간선을 유지하고 source23, bridge4, 나머지 세 집합으로 나눈다. source의 output_mask를 적용한 rate와 native weight의 float64 곱을 원래 CSR 순서로 합한다. 각 집합별 signed 합, positive 합, negative 합, 절대합을 저장하고 합집합이 전체 입력을 재구성하는지 확인한다. full rate/mask/drive/adaptation은 해당 저장 상태의 실제 값이다. drive는 postsynaptic 외부 전류이며 presynaptic rate에 합하지 않는다. 총 recurrent 입력에 drive를 더하고 threshold7.5와 adaptation을 뺀 margin m을 구한다.

원래 a1/cap200과 각 행의 E150 membrane_tau를 사용해 phi=max(200*tanh(m/200),0), active=m>0, relative slope=active*(1−tanh(m/200)^2), dr_dt=(phi−rate)/tau를 계산한다. m=0의 기울기는0으로 정의한다. positive active이면서 relative slope≤0.1인 경우만 saturated, m≤0은 rectified로 구분한다. E198의 같은 정의를 재사용하되 고정2000행 입력 함수 자체는 새8개 endpoint에 억지로 맞추지 않는다.

float64 기여도/phi/slope는 수학적 정적 재구성이다. CUDA stage에서 관측한 값이나 새로운 실제 신경 반응으로 부르지 않는다. 추가로4개 bridge에 대해서만8개 상태의 전체 CSR current를 기존 E178 warp_current(mode=packed)로 원래32lane/f32 순서와FTZ를 재구성하고 float64 합과 차이를 저장한다. 이 함수는 순수CPU이며 native stage를 실행하지 않는다. 신규 CUDA observer, model, solver, fit은0이다.

## H249와 결과에 따른 결정

별도의 두 한정된 가설을 사전에 둔다. H249_saturation은 intact_low/high 양쪽 최종 상태에서 동일한4개 bridge 중3개 이상이 active이고 relative slope≤0.1인 경우 지지된다. 기울기0인 rectified 셀은 saturation에 세지 않는다. H249_small_direct_input은 intact high−low에서 source23 집합이4개 bridge에 주는 float64 전류 차이의 cap200 정규화 RMS가≤1e−4이면 지지된다. 이 값은 E248의0.02Hz 단위 반응 기준과 대응하는 입력 크기의 기술이며 전체 회로의 최대 출력 한계는 아니다. 두 가설은 동시에 참일 수 있고 둘 다 실패할 수도 있다.

각 bridge별 source 입력 차이, 전체 입력/drive/adaptation/margin/phi/rate 차이를 보존한다. 원래 간선별 w*(rate_high*mask_high−rate_low*mask_low)의 signed/gross 합과 net/gross를 계산하여 작은 합을 작은 개별 가중치와 자동으로 동일시하지 않는다. actual observed rate endpoint 차이와 정적 phi 차이를 분리한다. 별도의 신경 집합 평균이나 단일 간선의 인과효율로 일반화하지 않는다.

포화만 지지되면 해당 bridge의 현재 입력/threshold/adaptation 작동점을 바꾸는 최소 기존 회로 개입을 다음에 설계한다. 작은 직접 입력만 지지되면 실제 source→bridge 간선의 전달 범위 또는 감각 인코더 크기에 대한 최소 고정 개입을 설계한다. 둘 다 지지되면 두 요인을 분리하는 대조부터, 둘 다 미달이면 현재endpoint로 설명되지 않는 동역학/다른 경로를 검토한다. 결과 확인 전 추가 fit/body rollout을 시작하지 않는다. 어느 결과도 E150/M2 교체나 자연보행/F2 전체 완료를 뜻하지 않는다.

## 구현과 검증

부모 설계 후 실제 gpt-6-sol/medium이 tools/analyze_c_banc_m3_lh_input_slope.py와tests/test_c_banc_lh_input_slope.py만 작성한다. 부모 FIRST→같은 Sol 첫수정→부모SECOND→남은 수정은 부모 순서를 적용한다. 기존 E178 순수CPU warp_current와 E198 정의, StateStore, 기존 gate/ledger/finalization 패턴을 재사용한다. 과거 frozen 파일을 수정하거나 새 범용 프레임워크를 만들지 않는다. 생성 fixture로 W[post,pre], 마스크, 음수/0/상쇄/threshold0 경계, group partition, 두 가설의 독립성, 실패 후 support 무효화를 확인한다. 원본 자료의 숫자 계산은 검토 후 지정한 단일 원본에서만 한다.

출력 e249-lh-input-slope에는 분석 raw.npz, 전체 관련 incoming-edge ID/pre/post/weight/group, 각 상태의 모든 기여/정적값, 원래 annotation, summary/result/ledger/errors/manifest를 보존한다. 실패하면 끝난 배열/작업량은 남기되 가설 verdict는null/not_tested로 하고 stale success/manifest를 남기지 않는다. 부모가 별도의 scalar 합/수식/선정/상태 검산을 한다. 표본과 결과가 적으므로 한정된 source snapshot 및 단일 저장자료 분석이며 holdout/독립적 보행 검증이라고 하지 않는다.

원본 누적6752.545765099629/6800초, 잔여47.454234900371초에서 supervisor45초를 예약한다. 추가배정0, internal30초(최종저장5초 포함)/external40/supervisor45, CPU1/GPU0, 원본1/retry0, RSS2GiB/output64MiB/free disk512MiB다. 작업 전 gate에서 전체 freeze/input/source/package hash를 확인한다. model/body/physics/neural/CUDA/fit/solver/F6/UI는0이다. 미측정 개발 비용은null이며 기존 실패와 자원 누적을 초기화하지 않는다.
