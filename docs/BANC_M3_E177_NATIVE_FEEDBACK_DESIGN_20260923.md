# E177 사전 설계: 기존 하중 피드백 다섯 연결의 실제 폐루프 대조

부모 설계이며 이전 E177 초안을 대체한다. 초안과 E176 생성 검증은 그대로 보존한다. 사용자 지시에 따라 구현은 실제 `gpt-6-sol/high`, 부모 FIRST 리뷰 → 같은 Sol 첫 수정 → 부모 SECOND 리뷰와 잔여 수정 순서다. 여러 기존 실행기·상태·계측의 통합이므로 high를 사용한다. 새 원본 실행은 이 계약과 두 구현 파일의 별도 부모 source gate가 모두 검증된 뒤에만 시작한다.

## 주장과 실제 대조

주장은 E43에서 남은 RM 하중 감각 연결의 .001 효능이 지속 보행을 제한하는지이다. E150 동일한 0초 checkpoint에서 baseline과 candidate를 독립적으로 복원한다. 둘 다 E176 `LoadFeedbackTrial`로 한 번 감싼다. baseline은 고정 다섯 entry의 target을 source와 같게 한 identity patch, candidate는 E176 고정 target을 사용한다. 선택한 float32 CSR entry 이외의 연결·세포·상태·신체·감각 encoder·근육 mapping·gain·tau·recruitment bank는 바꾸지 않는다. 후보 선택/fit/외부 CPG·phase·teacher·root 보정은 없다. E161의 수직 분해는 이 복원이 발 들기를 개선한다는 예측이 아니므로 긍정 효과를 전제하지 않는다.

| edge | pre | post | source | candidate target |
|---|---|---|---|---|
|844054|107685|12036|.00003|.03|
|2400285|30981|32830|.00006|.06|
|7908675|107685|108052|.00003|.03|
|8578714|30981|117483|.00006|.06|
|11240394|30981|153939|.00003|.03|

소수는 각각 직접 float32로 구성하며 현재 값에 1000을 곱하지 않는다. 원래 graph 단위 .275/contact와 rate-model의 .03/contact를 혼동하지 않는다. 전체 여섯 다리 피드백 복원이나 생물학적 근육 힘 검증을 주장하지 않는다.

## 실행과 원본 입력

최소 입력 경로와 전체 hash는 `e177-native-feedback-freeze.json`에 고정한다. E150 `candidate-neutral`의 모든 파일, 원래 graph bundle, `actual-global-csr.npz`, assembled parameter/calibration와 muscle provenance, E113 observation IDs, E103 startup-inputs, 원래 E150 main 비교 trace/native state 및 네 비교 JSON, E161 exact edge receipt와 E176 최종 생성 리뷰가 입력이다. 원래 E150 전체 결과는 마지막 cut 기술 실패로 incomplete였고, 완료된 main의 재현만 비교한다. 미사용 F6 자세는 열지 않는다. Freeze 작성은 hash·ZIP/NPY header·JSON metadata만 읽고 원본 배열 값을 읽지 않는다.

팔별 별도 프로세스/출력 디렉터리에서 순서대로 실행한다. baseline main10초 → checkpoint future .1초 → 별도 motor cut/restart .1초를 먼저 실행하고 종료한다. 부모가 실제 baseline raw와 재현·상태·건강·예산을 검토한 별도 hash gate가 있어야 candidate를 실행한다. baseline의 알려진 F4/F5 미달은 candidate 시작을 막지 않지만, 미완료·신경/물리 시계 오류·미래 불일치·CSR/초기상태 불일치·strict startup 실패·cut/restart 실패·재현 불일치는 막는다. 재시도는 0이다.

두 팔 모두158706 neurons/391 MN/96 muscles, main2000control+future20+cut4/restart16, neural .00025×20/control, physical .0001×50/control, control .005초다. 팔당10.2모델초/40800신경tick/102000물리step, 합계최대20.4초/81600/204000. 모든 생성·복원·CUDA warmup/capture·storage 진입은 기존 profiled/accounted helper로 별도 기록한다. 과학적 step과 생성 비용을 섞지 않는다.

## 기존 helper 재사용과 소유권

`verify_c_banc_m3_native_timescale_transfer._native_body`의 코드/default/closure를 FunctionType으로 복제한다. 허용 전역 교체는 정확히 `MotorTimescaleTrial`, `_observe`, `_save_body_raw`, `INTERNAL_LIMIT_S`, `source_diagnostics` 다섯이다. 나머지 binding과 원래 모듈 globals는 그대로다. 기존 full runner/old verified_inputs/resource_checks는 호출하지 않는다.

새 factory는 E150 source schema와 E176 wrapper schema만 처리하며, wrapper의 patch가 해당 arm과 정확히 일치해야 한다. source 복원 전후 whole snapshot, 실제 source CSR와 E150 저장 CSR를 대조한 뒤 wrapper를 만든다. whole unwrapped state에서 네 network identity metadata만 provenance로 되돌리면 원래 상태와 같아야 한다. baseline에는 모든 CSR byte가 같고 candidate에는 위 다섯 data entry만 달라야 한다. body의 소유권은 wrapper에 한 번 이전하고 실패/완료 모두 한 번 닫는다. 준비 모델은 main restore 전에 닫는다.

`source_diagnostics`는 기존 LF teacher/다른 실험 배열을 읽지 않는 E177 범위 receipt로 교체한다. 이 교체는 F3/F4/F5/startup/slip 평가를 바꾸지 않는다. 기존 helper의 E121/E141 label이 남는 비교 sidecar는 legacy metadata로 표시하고 E177 자체 result와 비교 기록을 별도 저장한다. `trace.json`의 label만 E177/arm으로 바꾸고 모든 sample 값은 정확히 보존한다.

main이 `e150.full_main_complete`를 만족하면 수정된 `e150.run_cut_restart`를 그대로 호출한다. six-leg LTM cut=0, 고정 abdomen pitch/yaw=.2를 보존하는 기존 검산을 사용한다. 이 motor cut 검사는 F7 자율 정지·재시작을 대신하지 않는다.

## 원자료와 계측

기존 native contact/trace/clock/391-MN recruitment state를 모두 보존한다. `_observe`의 추가 필드는 legacy 저장기에서 버려지므로 별도 `feedback-boundary-state.npz`와 JSON receipt를 저장한다. observer는 기존 full recurrent current를 한 번만 계산하고 동일 tensor에서 기존 detailed IDs와 추가 고정 IDs를 뽑는다. 관측이 dynamics를 진행시키거나 추가 `_current` 호출로 중복 계산하지 않게 생성 검증한다.

추가 IDs 순서는 sensory `[30981,107685]`, motor `[12036,32830,108052,117483,153939]`이다. control/network tick, physical time, 각7cell의 raw rate/drive/adaptation/output_mask/전체 recurrent current, 다섯 edge weight와 `weight*(rate_pre*mask_pre)` 직접 전류, `contact_diagnostics`의 RM normal load BW(index4)를 기록한다. 값은 원래 dtype을 명시하며 저장 shape와 availability/complete/error를 함께 보존한다. 예외 발생 시 앞서 확보된 필드와 부분 행을 먼저 저장한다. Rate/drive는 완료된 control 경계, held drive는 직전 신경 구간, load는 경계 시점의 관측이다. 직접 전류는 경계 표본이며 RK4 20 substep의 누적 전류·간접 총효과·물리 근육 힘이 아니다. cut에는 기존 raw만 기록한다.

## baseline 재현 기준과 판정

원래 E150 main의2001행(tick0..2000)과 새 baseline의 다음을 bit-exact로 비교한다: native state의 control_tick/network_tick/complete/availability,391 MNrate,391×3 bank,96 activation,96 original_activation. physics_time만 절대오차2e-8초, 상대오차0이다. trace 각 행의 control_tick,time_s,command_interval,joint_targets_rad,tendon_inputs,qpos,qvel,ctrl,body.position,body.basis,body.contactsBW,upright를 exact 비교한다. 원래 E150 trace metadata/model identity label은 비교 대상이 아니다. 누락/비유한 값/shape·dtype 변화도 실패다. source/candidate 실제 CSR 전체 sha와 선택 entry raw를 따로 남긴다.

건강 중지/자원·실행 실패와 유효한 보행 실패를 분리한다. raw 저장이 먼저이고 부모 리뷰는 요약보다 실제 raw, 실제 patch/state 및 실패 증거를 먼저 읽는다. strict startup qRMS≤.018/qmax≤.04, LTM RMS≤.009/max≤.025와 기존 F3/F4/F5 평가를 그대로 사용한다. 개발상 긍정 결과는 같은 candidate가 startup/F3/F4/F5 및 기준 baseline 이하 supported slip을 모두 만족할 때만 가능하다. 전진 거리 증가/발화 변화만으로 채택하지 않는다. 이 시험이 긍정적이어도 F1/F6/F7/F8과 생물학적 기전은 미검증이며 M3 완료가 아니다.

효과 없음: 직접 전류·MN 활동이 실제 달랐는지로 무효 개입과 행동 전달 한계를 구분한다. 악화/유효 실패: 결과를 보존하고 같은 다섯 edge의 임의 gain sweep을 하지 않는다. 기술 실패/재현 실패: 실패 원인부터 좁혀 기록하고 candidate를 실행하지 않는다. 개선: 동일 후보의 전체 미완료 기준을 별도로 검증한다.

## 자원과 보존

원래 개별 실행≤300벽초 규칙을 지킨다. 팔당 prepare60/body180/internal250/external270/supervisor285초, CPU1/GPU1, raw4GiB/free12GiB, original프로세스최대2/자동재시도0이다. 기존 보수 누적3085.0114084999077초를 보존하고 ceil3086+2×285=3656초로 이번 prospective 누적 배정을 명시한다. 각 팔의 실제 감독 시간을 원래 누적에 더하며 모르는 과거 생성 비용은0으로 기록하지 않는다. 이 배정은 사용자 토큰/전체 시간 예산이 아니다.

M2 설정·E76 manifest 및 해당 protected artifacts, 기존 source, 활성 UI/서버/제어 client, F6를 보존한다. Git push/merge·외부 전송·설정 채택은 없다. 목표는 active이며 F1–F8을 완화하지 않는다.
