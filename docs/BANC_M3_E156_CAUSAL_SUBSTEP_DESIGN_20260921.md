# E156 기존 단조 모집의 인과적 substep 전달

## 근거와 변경 범위

E155의 실제 접촉점 재구성은 제어 경계의 RM 관절 움직임이 root 움직임을 상당 부분 상쇄함을 보였다. 사후 저장자료 관측에서1–2초 전체 native slip은1.841493mm/s, 5ms 끝값만 쓰면0.144137mm/s다. 하중×slip 합의54.606%가 각 구간 첫1ms에 있다. 현재 `FlyGymBody.step_joint_targets`는 continuous activation 설정이 없으면 최종 명령을 해당5ms 내내 고정한다. 이 시점의 연관성은 개입 효과를 아직 입증하지 않는다.

과거 M2-E22의 단일30ms 지수 endpoint 보간은 전체 slip을1.315배로 악화시켜 기각됐다. 그 실패를 유지한다. E150은 기존 MN별5/10/40ms bank, 고정 근육의10ms EMA, 단조 입력 모집을 사용하므로 이 실제 동역학을 통해 각0.1ms의 중간 관절 명령을 계산하는 별도 가설을 시험한다. 시정수·gain·knot·해부학·신경 연결·운동 끝값을 적합하거나 바꾸지 않는다. 여섯 LTM·부착·복부 명령은 이번 관절 전달 요인과 분리해 원래 control 값으로 유지한다.

## 단계 A: 순수 중간 관절 명령 계산기

원본 실험을 실행하기 전에 고정 수식의 helper와 생성 테스트만 구현한다. 소유 파일은 `tools/banc_m3_substep_recruitment.py`, `tests/test_c_banc_substep_recruitment.py` 두 개다. 기존 E149 `MonotoneRecruitmentAdapter`, 신체와 모든 과거 파일은 읽기 전용이다. 부모 설계 → 실제 Sol/high 최초 구현 → 부모 첫 리뷰 → 같은 Sol 첫 수정 → 부모 두 번째 리뷰/잔여 직접 수정 절차를 따른다. 기존 상태·수치 전달을 다루는 제한된 함수이므로 실제 high로 위임한다.

입력은 실제 기존 adapter, decode 직전 bank(391×3)와 original fixed activation(96), 이번391 MN rates, decode 후 endpoint target42, motor_cut, 실제 dt/physics_dt다. dt는 기존(0,.05] 범위이며 physics_dt는0.0001, dt/physics_dt는 정확한 정수여야 한다. helper는 adapter/state/body/입력 배열을 변경하지 않는다. 생성자는 해당 adapter schema/identity, 고정 필터tau, moment/rest/pool, 원래 membership/fixed-muscle 정보를 검증해 재사용한다. 잘못된 shape·NaN/Inf·복소수·음의 rate·bool dt·index/mapping은 명시적으로 거부한다.

각 substep 종료시간 t=i×physics_dt에서

`bank(t) = bank_before + (-expm1(-t/tau)) * (rate - bank_before)`

고정 근육의 기존 원본 EMA는 `fixed(t) = fixed_before + (-expm1(-t/.01)) * (pool@rate - fixed_before)`이다. 기존 fixed 근육은 이 값을 사용하고 나머지는 E149와 같은 `low*bank + (high-low)*max(bank-kappa,0)`로 근육 활성도를 구한다. 관절 명령은 같은 `rest + .02*(moment@activation)`다. 새로운 필터·신경 oscillator·시간 위상·외부 목표/teacher/root 수정은 없다. 한 control의 현재 MN값을 유지한다는 기존 분할 적분 가정이며, 실제 intra-control 신경발화를 관측했다거나 생물학적 연속 근육을 복원했다고 부르지 않는다.

helper는 n×42 joint_targets와 상대 substep 시간, endpoint 재구성 오차를 반환한다. 마지막 bank·fixed activation·96 activation·42 target을 이미 decode된 adapter의 현재 상태 및 전달받은 끝값과 atol1e-12/rtol0에서 대조한다. 실패하면 임의 보정하지 않는다. 이 검증 후 마지막 관절 행만 원래 target으로 정확하게 대입하여 원래 control 경계의 float값을 보존한다. motor cut에서는 adapter의 zero bank/activation과 정확한 body.neutral 끝값을 확인하고 모든 substep을 즉시 neutral로 만든다. 중간 상태는 현재 입력과 이전 상태만 사용한다.

계산은 substep과 motor축을 묶은 NumPy 배열을 이용한다. 50×96개의 Python 루프로 호출하지 말고, 근육별 loop가 필요하면 각 근육의 모든 substep을 함께 계산한다. 몸체·MuJoCo·Torch/CUDA를 이 helper가 생성하지 않는다. neural network는 기존 실제 CUDA 경로를 유지할 예정이며 이번 단계가 CUDA fallback을 바꾸지 않는다.

생성 검증은 원래 adapter decode와 끝값 일치, 독립 scalar 중간값, 서로 다른 tau와 knot를 가로지르는 단조 전달, fixed/singleton/LTM 처리, motor cut/restart, 입력·상태 불변, 잘못된 schema/shape/시간 간격 거부, 현재 control 이후 입력을 바꿔도 이전 계산이 바뀌지 않음을 포함한다. 별도 helper 내부 상태를 만들지 않으므로 저장된 기존 adapter를 복원한 뒤 동일 이전 state/rates를 주면 같은 중간 관절 명령이 나와야 한다. 원래 모수·원본NPZ·StateStore·신경·신체·physics 실행0이며 생성자료 검증만 한다.

## 단계 B와 최종 판정 경계

단계 A의 source review 뒤 원래 E150 MN 기록과 checkpoint를 이용해 중간 명령의 원래 endpoint 재현을 검사하고, 같은 신체의 baseline hold와 substep 전달을 대조하는 한정된 원본 실험을 별도 동결한다. 해당 입력·시작·길이·모델 수·자원·효과 판정은 실행 전에 정하고 원본 source gate 전에는 실행하지 않는다. 누적 원본 비용2535.612067초/기존 배정2567초를 그대로 이어받으며 단계 A에 원본 자원을 배정하지 않는다.

중간 계산의 수치 일치 자체가 보행 개선은 아니다. 실제 신체에서 native slip/전진 연속성/접촉·upright·관절 운동을 함께 비교하고, 근거가 있을 때만 전체 BANC 감각 폐루프와 저장 미래/운동 차단을 검증한다. 원래 F1–F8, E150 incomplete/F4·F5 실패, 미사용 F6, M2와 활성 UI를 보존한다. 후보 채택·goal 완료는 여전히false다.
