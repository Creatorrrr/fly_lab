# E149 단조 모집 전달의 native 연결 설계

2026-09-21. 부모 설계이며 E148의 고정 계수를 실제 전체 BANC/FlyBody 경로에서 검증하기 위한 후속 단계다. E148 원본은 물리적 적합 조건을 통과했지만 첫 독립 검산에서 선형 포함성의 속도 오차가 1e-12를 넘었다. 같은 Sol/high의 첫 수정과 부모 두 번째 리뷰가 진행 중이다. 이 결함을 해결하기 전 원본 신경/물리 실행은 허용하지 않는다. 첫 실패, 원본 출력, 원래 허용오차를 보존한다.

## 단계 A: 상태를 가진 adapter와 trial

이 단계는 생성자료로 구현·검증하며 원본 NPZ, GraphStore, StateStore, CUDA, MuJoCo 또는 실제 신체를 실행하지 않는다. 새 gpt-5.6-sol/xhigh 생성은 세션 agent thread limit으로 거절됐다. 따라서 원본 CUDA/신체 실행기 개발을 분리하고, 명세가 고정된 두 파일의 adapter/fake 복원 구현은 기존 gpt-5.6-sol/high (/root/m3_sol_e122_support_transfer)에게 맡긴다. 기존 에이전트의 실제 effort를 xhigh로 바꿨다고 보고하지 않는다. 단계 B의 실제 native 실행기에서는 별도로 난이도와 가능한 에이전트 설정을 다시 판단한다.

- tools/banc_m3_monotone_recruitment.py
- tests/test_c_banc_native_monotone_recruitment.py

신규 MonotoneRecruitmentAdapter는 tools/banc_m3_rhythmic_recruitment.py의 기존 AnatomicalRecruitmentAdapter를 감싼다. 기존 calibration, groups, motor_ids, 신체와 신경망은 변경하지 않는다. 원래 391 MN → 96 근육 membership에만 양의 전달을 허용한다. 곡선 모수는 고정 tau [0.005,0.01,0.04], kappa (391,), slopes (96,391,3,2), membership (391,)이다. E148과 같은 실제 min/max 기울기 합 .5–2, 원래 MN 하한 1e-6/MN수 및 비음수성, fixed LTM/singleton weight를 생성 시 검사한다. 허용오차는 기존1e-12다. 입력은 유한 비음수 (391,) 발화율과 0<dt<=.05이다.

동적 상태는 391×3 filter bank, 96 activation, 원래 adapter 상태 및 last_motor다. 원래 adapter.decode를 매 제어 tick 동일 rates/dt/cut으로 먼저 호출하여 fixed LTM/singleton activation과 tendon을 정확히 보존한다. 비고정 근육은 bank += -expm1(-dt/tau)*(rates-bank) 후 아래 식을 원래 해부학 순서로 평가한다.

`activation_m = sum(low_slope*bank) + sum((high_slope-low_slope)*max(bank-kappa,0))`

이는 양의 연속 단조 두 구간 전달과 대수적으로 같다. 두 slope가 같으면 원래 선형 전달 연산으로 정확히 환원한다. 기울기 차가 음수인 보정항은 음의 생리 연결을 도입하는 것이 아니다. tau/kappa/weights는 runtime에서 고정한다. 위상 시계·teacher·target trajectory·root 보정·새 신경 노드/연결은 없다. q=rest+.02*(moment@activation), tendon=.2*tanh(LTMactivation/10)을 사용한다. fixed activation/tendon은 원래 adapter 값으로부터 얻는다.

motor_cut은 bank와 신규/기존 activation을 모두0으로 지우며 body.neutral과 tendon0을 반환한다. 재시작은 그0 상태에서 같은 recurrence를 진행한다. 임의 출력 clipping이나 사후 보정은 하지 않는다. 고정한 입력·상태로 미래 출력이 결정돼야 한다.

identity에 schema, 원래 adapter.identity, 정렬된 motor_ids와 membership, tau/kappa/slopes, calibration 및 그룹, 초기0 규약을 포함한다. 입력 배열은 방어적으로 복사한다. snapshot/restore는 모든 상태와 모수를 보존하며 identity/shape/dtype/finite/nonnegative/support 불일치를 거부한다. activation의 상한을 임의200으로 두지 말고 bank와 고정 전달의 재계산으로 일관성을 검사한다. 신규 상태의 activation은 original fixed activation 및 bank에서 같은 연산으로 재계산 가능하다. 실패 시 이미 열린 trial의 자원을 닫으며 첫 오류를 보존한다.

신규 MonotoneRecruitmentTrial은 MotorTimescaleTrial을 감싼다. 기존 network 객체와 CUDA graph/CSR을 그대로 사용한다. source_adapter_identity를 덮어쓰지 않는다. checkpoint에는 별도 schema와 모집 전달 계층을 추가한다. from_checkpoint는 신규 계층을 벗겨 원래 MotorTimescaleTrial.from_checkpoint를 먼저 정확히 복원한 뒤 신규 adapter/state를 복원한다. 신경·body·원래 wrapper의 saved future를 변경하지 않는다. snapshot 시 class schema와 nested adapter를 명시적으로 처리한다.

생성자료 검증은 비균등96근육/391MN, singleton/LTM, nonlinear와 duplicate slopes, 가변dt, prefix invariance, cut/reset/restart, 파라미터 변조, snapshot의 독립 복사, 전체 wrapper 복원·실패 cleanup을 포함한다. 실제 모델은 fake/stub로 대체하며 CUDA 생성은0이다. numerical helper나 기존 native 소스는 수정하지 않는다. 부모가 첫 리뷰하고 결함이 있으면 같은 Sol/high가 첫 수정, 두 번째 리뷰 후 잔여 수정은 부모가 맡는다.

## 단계 B: 원본 검증 전 별도 고정

E148 수정 후 동일 저장 계수의 독립 결정론적 검산이 모두 통과하고 단계 A가 승인되어야 한다. 그 뒤에만 실행기, 원본 입력 SHA, 새 누적 자원 한도와 source gate를 별도로 고정한다. 이 문서만으로 원본 신경·물리 실행이나 후보 채택을 승인하지 않는다.

원본 후보는 E148 한 번의 반환 slope/kappa와 E142/E141의 동일 whole-BANC neutral state다. 재적합·다른 후보 조합은 하지 않는다. 개발 시작 상태에서 10초 실제 신경/물리, checkpoint 미래 비교, cut/restart를 확인할 계획이며 F6 두 미사용 자세는 계속 비노출이다. runtime feedback이 바뀌므로 저장 적합 성공을 보행 성공으로 간주하지 않는다. F3–F5와 strict startup, 이후 F1–F8은 원래 동일 후보 기준을 유지한다. 실제 실행 시간/수와 준비 CUDA tick은 별도로 계수한다. M2와 활성 UI는 유지한다.
