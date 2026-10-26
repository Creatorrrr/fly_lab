# E286 — RF 정체점의 신경 발화·근육 전달 상태 귀속

2026-09-27 root 사전 설계. E285는 실제 peak를 .0394235→.0369392로 줄였지만 최종 .0359964에 못 미치고 마지막 개선은8.16e-9였다. 고정 J의 full-domain peak 하한 .0369363과 control99 RF 첫 관절의 음의 오차가 남는다. 중립의 동일 시점 목표 .0860724801은 decoder rest .049864보다 크며, 기존 저장 중립 자료에서 양의 방향 근육 activation은1.872e-9였다. E285 정체점의 내부 상태는 아직 저장하지 않았으므로 그 값까지 중립과 같다고 추정하지 않는다.

H286은 정체점에서도 해당 관절의 양의 방향 MN593/141493의 bank가 decoder kappa 아래에 있어 높은 gain 분기를 사용하지 못하고, 반대 근육의 잔류 출력과 함께 오차를 제한한다는 것이다. 경쟁 설명은 높은 분기가 활성화됐지만 시점 간 신경 응답/근육 출력 경쟁이 남았다는 것이다. 이는 다음 source 시간 성분 대 decoder 전달 변경을 구분하는 한정 진단이며 모델 개선·불가능성 증명이 아니다.

원래 RF context/engine/decoder/계수를 바꾸지 않고 두 실제 상태만 재현한다. (1) 원래 neutral, (2) E285 고정 마지막 `outer-05-alpha-0-actual.npz`의 f64 계수. Neutral 및 plateau q/LTM 전체101행을 각 저장자료와 abs1e-10로 대조한다. coeff/target/error는 exact, 재현 실패는 invalid다. Optimizer/J/새 계수/새 seed/추가조건/행동·F6 노출은0이다.

각 상태마다 새 RF engine 하나로 `solve(..., derivative=False, host=False)`를 한 번 실행하고 current/rate/adaptation/phi/branch/bank/activation/original_activation/q/ltm를 명시적으로 host_copy해 원자료와 receipt를 저장한다. 원래 response ownership을 유지한 copy를 사용한다. 복사된 arrays로 각 motor의 `sum(low*bank+(high-low)*maximum(bank-kappa,0))`를 재구성하고 membership으로 근육 activation을 합친다. Fixed 근육은 original_activation을 사용하고 모든 q=`rest+.02*activation@moment.T` 및 LTM을 독립 재구성해 abs1e-10로 확인한다. Raw arrays를 수정하지 않는다.

RF 첫 관절의 여섯 motor ID593/74561/97445/121831/141493/156479와 명시적 membership/moment 부호를 사용해 모든101control의 기여를 저장한다. controls61/83/99에서는 각 rate/adaptation/세 bank/kappa/low/high/hinge 활성 여부/근육별 양·음 기여를 기록한다. 해당 control까지의 마지막20 neural ticks×4 stages에서 current/phi/branch 요약도 제공한다. Endpoint adaptation을 내부 RK4 adaptation으로 대신하지 않는다. 아주 작은 양수 값을 0으로 반올림해 침묵이라고 판정하지 않는다.

Control99의 양의 muscle에 속하는 MN593/141493에서 모든 bank<kappa이면 해당 high branch 미활성이라는 좁은 가설만 supported, 아니면 refuted다. 다른 모든 motor/state 값과 반대 증거도 보존한다. Global capacity/자연 보행/제품 채택은 어떤 결과에서도0이다. 다음 변경은 이 원자료를 보고 별도 사전 설계한다.

Actual gpt-6-sol/high가 새 `tools/verify_c_banc_m3_rf_state_attribution.py`, `tests/test_c_banc_rf_state_attribution.py`만 작성한다. Root FIRST→필요시 같은Sol 첫수정→root SECOND/잔여수정. 기존32source 불변, E285 freeze/review/root result/plateau-audit/final actual/원래RF case/roster/input/package/E280 qualification을 SHA pin한다. 생성 CPU60초, tiny solver/원본/CUDA0. Generated decode 재구성·identity axes·state row/stage slicing·변경/실패 거절·partial copy receipt/실제 engine 수명·단회2호출 상한을 확인한다.

원본은 source 승인 후 단회 science10초+cleanup5초/supervisor15초, RAM4GiB/GPU추가 추정2GiB/출력256MiB/free disk1GiB, engine2/forward4000/derivative0/J0/QP0. 엔진·response 참조를 풀고 cleanup한 뒤 다음 생성하며 한 번에 실제 engine1개만 살아 있어야 한다. 실행·copy·재현·저장·cleanup 실패의 완료 자료와 비용을 보존한다. 누적7861.705837599529/7905초, 잔여43.29416240047067초 내 배정, 현재예약0이다. F1–F8/이전 모든 기각/M2/E150/UI를 유지한다.
