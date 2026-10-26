# E250 LH 기존 감각 연결 복원 대조 — 2026-09-25

M3의 고정 F1–F8을 향한 제한된 신경 전달 진단이다. E248의 작은 운동뉴런 반응과 E249의 작은 직접 감각 전류를 보존한다. E249의 양수·음수 상쇄 부재 및 세 bridge의 반응 가능한 영역은 전달 가중치 하나만 바꾼 대조를 정당화하지만 그 효과를 보장하지 않는다. 자연보행 후보 채택이나 생리학적 가중치 검증이 아니다.

## 한 가지 변경과 정확한 시작 조건

E248에서 선택한 port85의 전체23 감각 뉴런과 bridge4[20156,63026,111970,152280] 사이에 존재하는 25개 선언 CSR 간선만 바꾼다. 간선 목록/원래 graph row/현재 float32값은 parent-e250-edge-preparation.json에 고정한다. 목표값은 원래 rate_weights의 float32(sign(bundle.weights)*bundle.counts*0.03)이다. 준비 단계에서 현재값이 float32(float64(목표값)*.001)과25개 모두 exact임을 확인했다. 원래 접촉수·부호·간선 존재를 유지하고 추가 뉴런/연결/부호 반전/전역 gain 변경은0이다. 0.03은 기존 rate 모델 단위이며 측정된 시냅스 생리가 아니다. 현재 가중치를 단순 float32*1000으로 계산해 반올림을 바꾸지 않는다.

E150의 native CSR에서 이25개 data entry만 교체한다. 전체 indptr/indices와 모든 비대상 data를 exact 비교하고 실제 CUDA 모델에 들어간 CSR을 다시 추출해 확인한다. 원래 E245 최종 full network tick40000의 rate/adaptation/drive/output_mask를 그대로 전달한다. 후보의 새 가중치는 새 model identity를 가져야 하므로 새 생성자 snapshot의 model_hash 및 core_adaptation/motor_adaptation/motor_timescale identity metadata만 명시적으로 이식한다. metadata 내부의 ID·gain·시간상수·그 외 값이 같음을 검증하고 바뀐 identity 필드 목록을 저장한다. 모델 hash를 과거 값으로 위장하거나 단순 restore 검사를 우회하지 않는다. 후보 disk 상태는 이 새 identity의 실제 상태로 저장/복원한다.

threshold7.5/cap200/a1, 모든 membrane/adaptation/motor timescale/tonic drive/마스크와 실제 입력 low/high는 E248과 같다. 신체 생성이나 과거 teacher command 실행은 없다. 전체158706뉴런, CUDA packed CSR/capture20/dt.00025, CPU1을 실제 runtime에서 검증한다.

## 순서와 대조 의미

단일 원본 프로세스에서 다음 순서만 수행한다. 첫 baseline 모델의 intact_low/high 각40controls×20tick를 E248의 같은41행 full rate와 최종 full state에 exact 비교한다. 두 baseline이 모두 일치한 뒤에만 후보 모델을 실행한다. 후보는 intact_low/high, bridge_cut_low/high, sensory_cut_low/high의6 arms를 각각 같은 후보 시작 상태에서200ms 실행한다. cut과 전체 held drive, 저장41행/full endpoint는 E248과 같다. 마지막 별도 세번째 후보 모델을 새 disk candidate-start에서 복원해 intact_low의41행/full endpoint 미래 exact를 확인한다.

감각 출력 차단 low/high에서 모든 비감각 뉴런의 전체41행이 exact여야 한다. 추가로 후보 sensory_cut 두 팔의 전체 발화와 동적 최종 상태는 각각 E248의 같은 팔과 exact여야 한다. bridge_cut 두 팔에서는 후보와 E248의 해당 팔에서 bridge4를 제외한 모든 뉴런의41행/동적 끝상태가 exact여야 한다. 이는 변경 간선이 source→bridge 범위 밖으로 새어나가지 않는지 검증한다. 후보와 기준선의 identity metadata 차이는 예상한 새 CSR identity로 별도 감사한다.

원본은3 constructors/180 setup ticks와9 arms/7200 scientific ticks다. 각 requested/entered/completed와 부분작업을 진입 전에 ledger에 기록하고 오류시 이미 실행한 작업을0으로 만들지 않는다. fresh baseline/후보/disk 모델은 순차 소유하며 불필요한 GPU 객체를 해제한다. Trial.from_checkpoint/FlyBody는 사용하지 않는다.

## H250, 유효성 및 다음 결정

기존 E248의 고정 H조건을 후보에 그대로 적용한다. 56개 전체free LH MN의 controls1..40 rate/cap200에 대해 delta_intact=high−low, delta_bridge=bridge_cut_high−bridge_cut_low, R=RMS(delta_intact), D=RMS(delta_intact−delta_bridge)다. H250은 R≥1e−4 AND D≥1e−4 AND D≥0.2R일 때 지지된다. 기준 R/D, 후보 R/D, 그리고 RMS(candidate_delta−baseline_delta)를 함께 보고하여 약화·증가·상호작용을 구분한다. 상호작용은 반드시 감쇠를 뜻하지 않는다. 개별25간선의 기여를 따로 입증하지 않는다.

유효성에는 정확한 단일 가중치 변경, fresh baseline2의 재현, cut의 전체 신경 경계, 새 disk 미래, 모든 rate/adaptation finite 및 기존 유효 영역, 고정 full drive/mask/시계/입력 해시가 필요하다. 오류/자원 초과는 invalid/not_tested이며 완료된 raw와 비용은 보존한다. 저장 실패 뒤 metrics/result의 H지원 표시를 남기지 않고 최종 manifest가 실제파일과 같거나 제거된 미완료 상태여야 한다.

유효하고 H250 지지이면 다음에는 해당 경로를 유지한 감각-신체 폐루프의 최소 비교를 별도로 설계한다. H250 미달이면 배율을 추가 탐색하지 않고 이번 기존값 복원만으로 충분하지 않다는 한정된 결과를 유지한다. source/bridge/MN 시계열에서 반응이 어디서 제한되는지와 감각세포 자체의 작동범위를 다음 결정으로 검토한다. 유효성 실패는 과학적 기각과 구분한다. 어느 결과도 E150/M2/UI 교체나 F2/F6/자연보행 전체 완료가 아니다.

## 구현·증거·자원

부모 설계 후 실제 gpt-6-sol/high가 새 tools/probe_c_banc_m3_lh_edge_restoration.py와 tests/test_c_banc_lh_edge_restoration.py만 구현한다. 부모 FIRST → 필요시 같은Sol 첫수정 → 부모SECOND → 남은 수정은 부모 순서를 따른다. E248 마스크/drive/수치 판정 및 기존 MotorTimescaleNetwork/StateStore를 재사용하고 과거 frozen 파일은 수정하지 않는다. 생성fixture로 정확한25간선 패치/부호·0·비대상 보호, 새 identity와 동적 상태 전달, 절단/metrics, 실패 후 결과 무효화를 검증한다. 부모는 별도 저장원자료 감사로 실제 CSR/상태/판정을 확인한다.

출력 verification/banc-natural-gait-m3-20260919/e250-lh-edge-restoration에 원래/새 identity와 패치/실제CSR 증거, 시작/각끝 상태, 전체 rate41행, drive/mask, metrics/result/ledger/errors/manifest를 보존한다. freeze와 source review가 소스·입력·패키지를 검증하기 전 새 원본 신경 실행은0이다.

기존 원본 누적6753.757826499630/6800초, 잔여46.242173500370초에서 supervisor45초를 예약한다. 추가 배정0. internal34초(마감5초 포함)/external40/supervisor45, CPU1/GPU1, host6GiB/추가GPU6GiB/output512MiB/free disk2GiB/free GPU7GiB, 원본1/retry0, fit/solver/body/physics/F6/UI0이다. 새 과학적 실행은1.8모델초, 생성자 setup .045모델초를 별도 계상한다. 저장출력 간선열람.0151462초와 원래 bundle 값 대조 준비.1054854초, 생성/검산/미측정 개발·자문비용은 별도 기록하고 과거 비용을 초기화하지 않는다.
