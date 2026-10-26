# E241: 같은 물리 조건에서 얻은 입력의 반응 기저와 시간 분리

E240는 E85 기록 q와 같은 control의 원래 BANC 힘줄 출력을 함께 사용하여 고정 F3–F5 기준을 통과했다. H241은 이 실제 입력으로 E217 계열의 해부학적 근육 공유 반응 기저를 재구성하고, 실제 원래 decoder 및 affine/direct readout을 검증할 수 있다는 것이다. 이것은 수집 자료를 fit에 연결하는 준비 단계다. 최종 자연 보행 기준과 이전 실패는 그대로 유지한다.

## 입력과 목표 정렬

E240의 positive_recurrent/negative_magnitude/drive float32 (160000,391), control-raw (2001행), observed-start 및 부모의 검산 증거를 사용한다. 원래 E217 parameters의 MN ID/소속/96근육/42관절/pool/moment/rest/free343/fixed48/LTM6/initial rate/tau/cap/inverse adaptation tau를 유지한다. 실제 초기 activation과 held q는 E240 observed-start/control0에서 읽는다. 학습 제외48개 출력은 E240에서 관측한 매 control 원래 MN 출력이다. 제외 계수라는 이유로 출력을 상수나 E150 출력으로 대체하지 않는다.

대상 q는 E240 applied_command[1:2001]과 E85 command의 exact 대응이다. row0은 실제 held command이며 affine rest로 재생성하지 않는다. 학습 control은 1..1200 (0<t≤6초), 시간 분리 검증 control은 1201..2000 (6<t≤10초)다. 학습 속도는 (q[c]-q[c-1])/.005의 c=1..1200, 검증 속도는 c=1201..2000이다. 검증 첫 속도는 경계 control1200을 사용하지만 그 목표로 계수를 맞추지 않는다. 이 구간 분리는 동일 궤적 내의 예측 검사이며 F6 독립 초기조건 확인이 아니다.

## 새 기저 계산

입력 스케일은 학습에 속하는 stage[0:96000]에서만 기존 compute_scales로 계산하고 이후 고정한다. 0 초기 fast/slow 기억, fast tau .016초, 실제 inverse adaptation tau, .00025초 RK4와 기존 gate/RBF 식을 유지한다. 전체10초 history를 한 번 계산하되 prototype은 descriptor[0:1201]만 사용하여 기존 muscle-weighted furthest-point 규칙으로 그룹별64개를 고른다. 시작 anchor0, first-unselected tie, sigma 규칙은 E217과 같다. 뒤4초의 입력/목표를 scale/center/sigma 결정에 사용하지 않는다. 학습/검증 목표 q는 prototype 선택에 사용하지 않는다.

새 scale/prototype으로 기존 run_grouped_basis_cuda를 실행하여 전체2001행 B/H/H_factor 및 fast/slow/descriptor/probe를 저장한다. 별도 history와 basis의 기억 값이 기존 오차 한계에서 같아야 한다. B0=0/H0=initial/H_factor0=1, 유한성·비음수·sum(B)+H_factor≤1 envelope 및 명시적 RK decay를 확인한다. history와 basis는 기존 두 CUDA kernel을 순서대로 사용한다. 별도 context를 순차 해제하므로 channel 업로드는 총6회이고 과거 객체·module global을 바꾸지 않는다.

새 rate intercept는 H의 free343과 E240 actual rate의 fixed48로 만든다. 원래 96근육 pool, 10ms activation EMA, .02 moment, held row0을 적용해 해당 trajectory의 q/activation/tendon intercept를 재계산한다. 이 새 intercept만 group_affine_readout에 전달한다. PROBE=(j+1)/65의 직접 재구성과 affine q를 비교한다. 별도로 모든 원래 MN rate의 canonical decoder를 재구성하여 E240 decoded_command 및 원래6 LTM tendon에 맞는지 확인한다. 이전 E217 B/history/scale/prototype/intercept나 E197/E150 목표 loader는 재사용하지 않는다. 기존 수학과 원래 해부학·생리 계수만 재사용한다.

## 결과와 후속 판정

parameters/prototypes/history/basis/affine-readout/canonical-original/targets NPZ, groups JSON, input/source hash, launch·정리·자원 ledger 및 manifest를 저장한다. targets에는 exact control/time, train/validation/velocity 행, source q/실제 tendon/원래 rate를 넣는다. 원래 decoder 대비 q/velocity/startup(1..99)의 오차를 절대 rad/rad/s와 다리별로 보고한다. beta∈[0,1]의 관절별 box envelope도 저장할 수 있으나 그것이 공동 제약의 실현 가능성 증명은 아니다. 이 단계는 solver/fit0, 후보 계수 생성0이며 오차의 임의 허용치로 물리 성공을 주장하지 않는다.

통과 기준은 E240 사전 근거와 모든 입력 hash, 정확한 train/validation 정렬과 뒤 구간 미사용, 원래 decoder 재구성(q atol2e−10/rtol2e−12, tendon atol2e−12), history atol2e−12/rtol2e−12, probe rate atol2e−9/rtol2e−11, affine q atol2e−10/rtol2e−12, B≥−1e−12와 envelope atol2e−10/rtol2e−12, 해부학/초기/자원 보존이다. 원본 기저가 유효하면 새 q/velocity 학습 목적과 범위를 다음 fit 설계에서 고정한다. E197 .018/.04 startup proxy는 별도의 과거 진단으로 남으며 새 물리 재현 허용치로 자동 승계하지 않는다. 실제 물리 충분성은 후속 동일 신체 재생의 고정 F3–F5, 자율 성공은 교사를 제거한 같은 후보의 F1–F8로 판단한다.

## 구현·검증·자원

실제 gpt-6-sol/high가 새 `tools/prepare_c_banc_m3_matched_input_basis.py`, `tests/test_c_banc_matched_input_basis.py`만 구현한다. 순차 CUDA·실제 fixed48·affine/직접 검산·시간 누설 방지를 결합하므로 high를 선택한다. 부모 FIRST→같은 Sol 첫 수정(필요시)→부모 SECOND→추가 잔여는 부모 순서다. 부모가 생성 GPU 실행과 원본 저장 분석을 각각 감독한다. 생성 CPU 검사는 뒤 구간/목표 변경이 scale/prototype 학습에 영향을 주지 않음, actual fixed 출력/intercept, held0와 목표/속도 정렬, 실패 raw/receipt 보존을 포함한다. 생성 CUDA 한 프로세스(작은 인공 입력, history1+basis1, 내부30/외부40/감독45초)는 원래 NumPy 두 RK 구현과 비교하고 실제 BANC/신체를 만들지 않는다.

원본 저장 분석은 CPU1/GPU1/job1/retry0, history1+basis1(각40000 conditional ticks/160000 RK), 모델/신체/새 BANC advance/physics/solver/fit0이다. E213 history 포함 감독15.12초와 E217 기저 감독35.89초에 근거해 내부110초(저장10초 예약)/외부130초/감독140초, RSS6GiB/GPU3GiB/출력1.5GiB/free4GiB를 고정한다. 부분 오류에서도 완료·요청 launch와 readback/원래 원인을 기록하고 자동 재시도하지 않는다. 실패 후 성공 validity를 남기지 않는다. 기존 누적6537.21311039964/6605초·잔여67.78688960036015초를 보존하고 사전 배정+75→6680초·잔여142.78688960036015초에서140초를 예약한다. 생성 검사·부모 검산은 별도 계측하며 사용자 전역 상한을 만들지 않는다. F6 미노출/M2/E150/UI는 보존한다.
