# E117: 저장된 국소 개선 방향의 비선형 응답 검증

2026-09-20 · 부모 설계 · Sol 구현과 첫 수정 · 부모 재리뷰 뒤 잔여 직접 수정

## 질문과 현재 근거

직전 목표 턴은 progress다. E115의 단일 fit은 두 입력의 초기 제약을 만족했으나 원래 형태 개선 기준에 실패했고, E116의12개 LP는 초기 제약이 RF/RH/LF의 국소 선형 감소를 크게 제한함을 보였다. 두 단계의 독립 검증과 원본을 보존했다. E116은 LM/RM 일부 제곱오차를 음수로 예측했으므로 그 해가 원래 비선형 제약을 만족한다고 볼 수 없다. 전체 F1–F8 및 실제 E113의 F3/F4/F5 실패는 유지한다.

이번에는 **E116에서 계산한 방향을 원래 운동뉴런의 비선형 조건부 응답에 그대로 대입했을 때 실제 손실과 초기 제약이 어떻게 변하는가**를 확인한다. 최대 오차 sample 전환의 기여와 동일 sample에서의 선형 예측 오차를 나누어 기록한다. 이 확인 없이 새로운 생리 매개변수가 필요하다고 결론 내리지 않는다.

## 세 가중치와 여섯 응답

부모 입력 parent-e117-step-inputs.npz에는 순서대로 selected_solver, shape_box, startup_box의3개 delta/가중치가 있다. selected_solver는 E115 여섯 best-feasible solver delta의 합성이다. 나머지는 여기에 E116 해당 다리 LP 해의 h만 더한 것이다. epigraph t는 가중치가 아니다. 원래 E115 delta bounds와 h 절댓값 .01을 모두 **그대로** 만족하며 투영·수정·float32 반올림을 하지 않았다. parent-e117-step-inputs.json의8개 검사와 원본 SHA를 동결한다.

W는 E113 실제 native float32 W를 float64로 확장한 원점에 selected edge별 exp(delta)를 곱한다. 여전히 float64 solver 방향의 진단이며 새 native calibration을 생성하지 않는다. selected edge의 ID/row/column과 나머지 가중치의 동일성을 확인한다. 기존 E114의 source/background 두 문맥을 각각 그대로 사용하고, background를 새 W로 다시 분리하지 않는다. 원래391 MN/49 source/96 muscle, tau/threshold/a/cap/pool/moment/rest/LTM 및0초 초기상태/5ms 시간축을 유지한다.

한 번의 CUDA batch에서 pair_context_indices=[0,1,0,1,0,1], pair_weight_indices=[0,0,1,1,2,2]를 평가한다. 각 pair는 독립0상태이며10조건부초, 합6응답/60조건부초다. 구현은 검증된 ContextResponse와 고정 E32 TorchFootGeometry를 읽기 전용 재사용한다. 원래 BANC 전체158706 신경 및 신체 진행은0이다. 새로운 LP/fit/추가 baseline 응답은 없다.

부모는 실행 전에 세 W를 이미 대수적으로 만들었다(.1519262내부/.4140905외부초,LP/조건부응답/신경/물리0). 이 준비를 응답 실행으로 부르지 않는다. LP 방향은 calibration으로 내보내거나 채택하지 않는다.

## 기록과 비교

응답 current/forcing/motor_rate/muscle_activation/command/tendon,6pair 순서와3W,시간축을 response-raw.npz로 먼저 보존한다. geometry는 기존 고정1828행에 한 번 적용하고 feet_fixed(6,1828,6,3)를 별도로 저장한다. 기존 공통1823행과 각 문맥 own-phase 표적/원래99 mask는 그대로다. 실제 발 경로가 아니라 command에서 계산한 고정 geometry proxy다.

다리별·문맥별 q/foot/velocity/LTM 네 정규화 제곱오차와 합을 원래 E115 정의로 계산한다. 초기99 qRMS/qmax/LTMRMS/LTMmax 원래 값과 내부 ratio²도 계산한다. selected_solver의 각 loss parts/ratios는 저장된 E115 선택 eval 원자료와 각각1e-8 이내여야 한다. 이 검사 전에 모든 응답과 산출 지표 원자료를 저장한다. 실패하면 도구 진단을 실패로 남기고 응답을 재실행하지 않는다.

shape_box/startup_box의 비선형 loss2개 및 startup ratios8개를 E116의 고정 affine 예측과 비교한다. 별도로 E115의 원래99/내부 cap, 공통 foot/velocity20%개선 및 q5%악화 이내 판정을 원래 정의로 보고한다. 재사용 가능한 frozen_output_metrics는 각 variant의2pair에 적용하며 비교 원점은 E114의 같은 문맥/W113이다. float64 진단의 screen 통과도 native 후보나 실제 보행 성공이 아니다. 결과에 candidate_adopted=false와 native_gait_tested=false를 명시한다.

각 다리/문맥의 qmax와 LTMmax에서 baseline selected_solver의 최대 sample을 구한다. q sample은99행×7관절, LTM은99행이다. 각 trial에서 그 baseline sample의 제곱비율 r_same과 trial 전체의 최대 제곱비율 r_max를 기록한다. E116 affine 예측 r_affine과 비교하여 다음 항등식을 저장/검증한다.

`r_max - r_affine = (r_max - r_same) + (r_same - r_affine)`

첫 항은 새 최대 sample이 baseline sample보다 큰 양이다. 둘째 항은 동일 sample 값과 선형 예측의 차이다. q의 sample index/time/global joint, LTM의 sample index/time/leg, baseline과trial의 signed error, baseline/trial 최대 index와 동률 개수, argmax 변경을 모두 저장한다. 동률 판정은 최대 제곱비율의1e-10×max(1,abs(max)) 이내로 고정하며 보행 gate가 아닌 설명용이다. baseline 동률이 있으면 둘째 항을 단일 sample의 순수 곡률로 해석하지 않는다. 두 항은 설명용 분해이며 제약을 수정하지 않는다.

## 개발·실행 한도와 검증

새 tools/verify_c_banc_m3_e117_nonlinear_steps.py와 tests/test_c_banc_nonlinear_steps.py만 구현한다. 기존 source/helper/원자료/UI는 수정하지 않는다. 부모가 설계하고 기존 명시적 gpt-5.6-sol/xhigh가 CUDA·여섯 조건·원자료/오류회계를 함께 구현한다. 작은 CPU 합성검사로 pair 순서·독립초기상태, 세 W 구성, peak 전환/동률/분해, baseline mismatch, CUDA 입력 준비 전후의 회계, raw/primary exception 보존을 검증한다. 첫 수정은 같은 Sol, 두 번째 리뷰 뒤 남은 수정은 부모가 맡는다.

부모 최종 소스검토 전 원래 데이터 CUDA 응답은0이다. 이후 RTX4080/cuda:0/float64, OMP/MKL/OPENBLAS/Torch CPU thread1로 한 프로세스만 실행한다. 모든 입력을 GPU에 준비한 다음 필수 시작 receipt가 성공했을 때에만6pair를 계상한다. started/materialized/raw_saved batch와pair 및 conditional seconds를 구분한다. 오류 시 이용 가능한 수치자료와 metadata를 별도 파일로 복구하고 원래 예외를 유지한다. 출력 e117-nonlinear-steps/는 존재하면 거부하며 자동 재시도0이다.

내부20/외부30초. 기존 E116 보수누적700.734초를701로 올려 이번 별도 누적731초를 기록한다. 과거 E115838/E116730 한도와 실패는 소급 수정하지 않는다. CPU/원자료 검토 및 준비비용은 별도로 남긴다. 원래 neural/physical0,LP0,fit0,calibration0이며 여섯 조건부응답60초만 허용한다.

실행 뒤 부모 독립 NumPy 순차 recurrence로 같은6pair를 한 번만 검산한다(60조건부초,내부30초/외부40초,새 native0/LP0/fit0). current≤1e-8, motor/activation/q/tendon≤1e-9의 기존 수치 기준을 유지한다. 부모는 저장된 raw로 지표/peak 분해/판정과 출처·실제 실행 횟수·시간·원본 해시를 독립 확인한다. M2263/E110201/E113207/E11419/E1153043/E11665 원본 및 F6 미노출을 보존한다.

비선형 검증에서 방향이 실패하더라도 같은 fit을 더 오래 돌리거나 기준을 낮추지 않는다. sample별 제약과 최대값의 미분, 실제 초기 상태 전환을 반영하는 업데이트 방식이 필요한지 이 결과로 다음 설계를 구체화한다. 아직 전체 목표의 완료·생물학적 기전 검증·새 실제 보행 실행을 승인한 단계가 아니다.
