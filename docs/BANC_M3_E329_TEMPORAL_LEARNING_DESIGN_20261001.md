# E329: 시간 순서와 여섯 다리 관계를 학습하는 BANC 목적함수

2026-10-01 현재 세션 설계. actual gpt-6-astra/max의 `verification/banc-natural-gait-m3-20260919/parent-e329-astra-design-review.md` 리뷰를 반영해 구현 계약을 확정했다. 구현·원본 실행은 아직 없다. 기존 M3 finite-goal과 F1–F8 전체를 유지한다. 이 단계는 실제 신경 가중치 학습과 신체 검증으로 이어질 학습 경로의 변경이며, 학습 손실이나 테스트만 통과해도 목표를 완료하지 않는다.

E327은 실제 신체에서 명령 형태 손실을 0.295329% 줄였으나 전진은 2.82584% 감소하고 slip은 3.17646% 증가해 기각됐다. E328은 전체 관절 추종 오차 약0.09도, 움직이는 다리와 짧게 끊긴 전진 구간을 확인했다. 기존 형태 손실은 정지한 곡선점에0을 주고 시간 순열에 불변이다. E152에서 관측한 하중을 받는 발의 회수와 E229의 반례를 함께 고려하면, 특정 다리·관절을 임의 보상하기보다 운동의 시간 관계를 학습하고 실제 접촉 결과로 선택하는 다음 방법이 타당하다. E206 등의 저장 접선 실패와 E323 장시간 미분 기각도 유지한다.

## 검증 질문과 대안

H329: 이미 검증한 기준 곡선을 현재 자세에서 다음5ms의 자세로 연결하는 학습 신호와 여섯 다리 공통 참조 위상 관계 항을 사용하면, 정지·역순·다리 사이 어긋남을 기존 형태 손실과 구별할 수 있고 현재 SSP 학습 경로에 미분 가능한 목적을 제공한다. 이것은 BANC 보행 개선 가설의 필요한 구현 단계다. 특정 신경 연결의 생물학적 기전을 입증하는 가설이 아니다.

대안은 현재 형태 목적의 반복 최적화 또는 신체 구동기 gain 교정이다. 전자는 E327/E328의 시간 순서 반례를 해결하지 못한다. 후자는 작은 추종 오차만으로 큰 개선을 기대할 근거가 부족하다. 새로운 목적도 저장 감각 입력에서는 좋아지고 실제 신체에서는 악화될 수 있으므로, 후속 후보 선택은 실제 폐루프의 startup·지속 전진·slip·적격 주기를 반드시 포함한다.

## 불변 조건

- 전체158706 BANC의 해부학적 노드/연결과 선택 E29 P→P 효능 그룹, 현재 SSPRK33/u0 계산, E150 monotone decoder 및 신체를 유지한다. 이 단계는 신경/신체를 생성하지 않는다.
- CPG·교사 곡선·phase·이 목적함수는 오프라인 학습과 비교에만 사용한다. 최종 신체 실행은 감각→BANC→기존 decoder→신체이며, 목적함수나 참조 곡선을 실행 controller에 연결하지 않는다.
- E103 시작99control의 q RMS≤.018/max≤.04, tendon RMS≤.009/max≤.025 및 공식 F1–F8·short body evaluator·M2/UI/F6 비노출은 그대로다. 새 목적의 통과를 공식 보행 통과로 바꾸지 않는다.
- 이전 원본·실패·source·숫자·비용은 불변이며 새 파일만 소유한다. reset/clean/commit/push·외부 송신·다른 목표 생성은 없다.

## 목적함수의 확정 계약

E319와 같은 E85 time≥5초 표본을 LF 기준 global phase로 정렬해6×128×7 closed polyline C_l(p)를 만든다. 같은 global phase로 여섯 다리의 표본을 만든다는 점이 중요하다. 이미 읽은 E85의 공통 phase 속도는2.382760007990865Hz이며 source hash는 parent-e329-reference-inspection.json에 남겼다. 빈도는 자료에서 확인한 값이며 새 후보에 맞춰 조절하지 않는다. Δp=f×.005, 약.0119138cycle이다.

현재 명령 x_l의7차원 거리로 가장 가까운 polyline segment m와 clamp된 투영계수 u를 구하고, p_l=(m+u)/128을 계산한다. 동일 거리 tie는 최초 segment를 고르는 결정적 규칙이며 endpoint는 modulo1로 처리한다. 이것은 명령 공간의 참조 phase proxy로서 실제 다리 위상 관측과 구분한다. 실제 phase의 식별 가능성이나 구동 가능한 상태라는 보장은 없다.

한 control 뒤 목표는 y_l=C_l((p_l+Δp) mod1)이다. successor 손실은 각 다리의 mean((q_next-y_l)^2)를 E85 polyline의128개 anchor에서 평균낸 같은5ms 참조 증분의 mean-square로 나눈 뒤 시간·다리를 평균한다. q_prev에 따른 투영과 successor 보간의 미분을 모두 유지하며, 참조 tensor만 고정한다. coordination은 q_next의6개 phase에서 15개 unordered pair를 한 번씩 사용한 mean(1-cos(2π(p_a-p_b)))다. training_loss=successor_loss+coordination_loss, 계수1을 사전 고정한다. 좌우3쌍 항은 진단으로 따로 남긴다. 곡선 자체에 교사 다리 간 시차가 포함되어 있으므로 여기의 목표차0을 공식 실제 다리 위상차0으로 해석하지 않는다. per-leg successor, coordination, LR, 기존 shape/startup 지표도 따로 기록한다.

LR-only에는 세 좌우 쌍의 공통 offset 세 개가 남아 앞·가운데·뒤 사이 관계를 구별하지 못한다. 이번에는 같은 신체에서 실행 가능한 E85의 관계를 학습하는 all-pair 가설을 선택한다. 이것은 교사 보행 선호라는 방법상의 제약이며, 정확한 tripod를 요구하는 새 F4 판정이나 runtime clock이 아니다. 다른 좋은 보행을 배제할 수 있다는 한계를 기록하고 실제 신체 결과로 후보를 판단한다.

완전한 q/tendon 입력은 E319와 같은 control1..300이다. 학습 steady 쌍은 q_previous=q[99:299], q_next=q[100:300]인200개이며 coordination도 q_next, control101..300에서 계산한다. startup은1..99다. 20ms 조건부창 시작 c=100,104,...,296에서 출력 Q가 control c+1..c+4라면 evaluate_pairs(concat(saved_q_c,Q[:3]),Q)를 호출한다. 첫 saved_q_c만 고정하고 Q[:3]와 successor target은 detach하지 않는다. E318 raw control0행을 포함하므로 raw.command[c]가 경계이며, control1부터인 replay_q에서는 c-1이다. 50창의 평균값은 전체200쌍 값과 같아야 하지만 경계 고정 때문에 미분도 같다고 주장하지 않는다. 재생된 bank/r/z와 boundary를 바꿔 새 gain을 숨기지 않는다. 목적함수는 model을 호출하거나 손실 평가 시각을 controller clock으로 주입하지 않는다.

선택 현재 edge e_m과 successor edge e_n이 고정되고 projection clamp 내부인 구간에서 D_x p=e_m^T/(128||e_m||²), D_x F=e_n e_m^T/||e_m||²이다. 잔차 r=q_next-F(q_previous)에 대해 다리별 정상화 s_l 및 평균 인자를 제외한 미분은 next=2r/(7s_l), previous=-2(D_x F)^T r/(7s_l)이다. strict clamp 외부에서는 phase 미분0이다. branch 전환·정확한 clamp endpoint·polyline knot는 미분 불가능할 수 있다. 이 지점은 결정적 값과 유한성만 검증하며 중앙차분 일치를 주장하지 않는다. 4-control 검사는 중간 명령의 양방향 기여와 마지막 명령의 successor 및 coordination 기여를 독립 확인한다.

퇴화 곡선·0 분모·비유한 값·중간 계산 overflow·잘못된 dtype/device/shape/clock은 거부한다. epsilon으로 무효 분모를 숨기지 않는다. CLI에서 control 순서와5ms 간격을 검사한다. 실제 저장 명령의 per-leg 투영거리, 원형 phase jump, clamp 비율, 비국소 competing branch 거리 차이를 기록한다. seam 및 인접 endpoint tie는 정상 원형 위상으로 처리하고 모호함으로 세지 않는다. 비국소는 원형 segment index 거리>1 및 원형 phase 차>2/128로 정의한다. 유효 실제 입력에 거리제곱 차≤1e-12인 비국소 동률이 있으면 E330 진입을 보류한다. phase proxy를 검증 없이 참 위상으로 승격하지 않는다.

생성 CPU 값 비교 rtol=1e-10/atol=1e-10, 안정 branch 중앙차분 h=1e-6 미분 비교 rtol=1e-6/atol=1e-7을 사전 고정한다. 생성 CPU/CUDA f64 값 비교 rtol=1e-10/atol=1e-10, 미분 rtol=1e-8/atol=1e-8이다. 실제 reference round-trip은128개 anchor와 고정 interior/wrap 표본에서 원형 phase 오차≤1e-8cycle, successor q 최대오차≤1e-10rad를 요구한다. 정상 참조 순환의 손실≤1e-10, 고정 전체 anchor 정지 successor 평균≈1 및 역순/비순환 순열/다리 간 불일치가 정상보다1e-8 이상 높아야 한다. 한 LR 쌍 전체만 이동하면 LR 항은 불변이고 all-pair 항은 양수가 되는 반례를 포함한다. cosine의 반주기 및 대칭 배치에는 양의 손실인데도0미분인 지점이 있으므로 항상 비영 gradient라는 거짓 기준을 두지 않는다. coordination의 공통 phase 이동 불변성과 정상 순환의 임의 시작phase 통과를 검사하되 잘못된 유한 궤적 전체의 공통이동 불변성을 요구하지 않는다.

## 실행과 다음 결정

1. actual gpt-6-astra/max가 이 설계의 수학·기존 실패와의 차이·오프라인/최종 경로 분리·판정과 자원을 리뷰한다. 현재 세션이 필요한 변경을 반영한 뒤 구현을 확정한다.
2. actual gpt-6.1-sol/xhigh가 새 목적함수·saved-characterization CLI와 생성 테스트를 구현한다. 비선형 projection/phase/미분과 CUDA device 경로를 함께 다루므로 xhigh를 선택한다. 현재 세션 FIRST→필요시 동일Sol 첫수정→현재 세션 SECOND→잔여 직접수정 순서다.
3. 생성 circle/polyline과 실제 E85 reference에서 위의 고정 반례/허용오차를 검사한다. 경계·wrapped phase·퇴화 곡선·NaN/Inf를 검사하고, 안정적인 projection 구간에서 독립 NumPy 값과 중앙 유한차분 미분을 비교한다. CPU/CUDA의 값·미분 비교도 작은 생성 자료로 확인한다. 독립 root 검사는 Sol의 수식을 그대로 호출하는 자체 일치 검사로 대신하지 않는다.
4. source/input freeze 뒤 E318 중립과 E327 기각 후보의 저장 command만 동일 새 목적에서 성격을 분석한다. 원본 모델·body·새 drive·최적화·BPTT는0이다. 점수 순서는 성격 분석이며 E327이 나쁘게 순위화돼야 한다는 사후 합격 기준을 두지 않는다. 두 궤적의 새 score는 공식 평가를 바꾸거나 기각 후보를 승격하지 않는다. 참조 자격이나 비국소 식별 자격이 실패하면 후속 학습을 시작하지 않는다.
5. 위 자격이 통과하면 후속 E330에서 기존 checkpointed rollout과 50×20ms 조건부창을 재사용해 새 목적의 한 실제 SSP gradient 및 실제 f32 크기비례 방향 하나를 검증한다. full-horizon 미분을 반복하지 않는다. 국소 방향의 실제 유한차분이 맞아야 reset없는1.5초 후보와 시작 네 제약을 확인한다. 조건부 입력 성공 이후에는 동일 theta의 실제 신체 E331에서 지속 전진/slip/적격 주기를 먼저 확인하고, 그 결과가 지지되면 같은 후보의10초 F1–F8 검증·통합으로 이어간다. 각각의 원본 계약과 판정은 실행 전에 별도로 확정한다.

## 자원·소유·기록

E329 Astra는 read-only source/data-metadata 리뷰이며 원본 NPZ 계산·모델·CUDA·신체 실행·하위 위임0이다. Astra 피드백 한 파일만 작성한다. Sol/root 생성 CPU 각각120초, root 생성CUDA30초/host2GiB/additionalGPU1GiB, saved characterization supervisor20초/host2GiB/output20MiB이다. 초기 구현 소유 파일은 tools/banc_m3_phase_free_temporal.py, tools/analyze_c_banc_m3_temporal_objective.py와 대응 tests 두 개다. 기존 source는 읽기 전용이다. 실패를 포함한 비용을 별도 기록한다.

현재 원본 누적9034.148870999521/기존 단계 상한9060초, 예약/활성원본0이다. E329는 원본 신경·신체 실행0이므로 이를 소비하거나 초기화하지 않는다. 사용자 지정 전체 시간/토큰 예산은 없다. 후속 E330/331에는 새로운 사전 단계 배정을 검토하되 기존 비용을 계속 합산한다. 새 원본 계약 없이 오래된 잔여25.851129초를 무제한 자원으로 해석하지 않는다.

E329의 결과물은 학습 가능한 시간 목적 및 실제 저장 명령에서의 진단이다. 이것을 자연 보행 완료로 부르지 않는다. 전체목표/F1–F8은 active·미완료이며 다음 행동은 실제 BANC 가중치 갱신과 실제 신체 결과의 연결이다.

## Astra 리뷰 반영 기록

읽은 리뷰 SHA256와 원래 초안 hash는 `parent-e329-astra-design-review.md` 및 `parent-e329-pre-astra-design.md`에 보존한다. current session은 20ms pair 경계/마지막 명령 미분, all-15-pair 방법 선택, 투영 미분의 branch 유효 범위, 비국소 모호함과 seam 구별, 실제 reference 자격, 저장 순위의 해석 경계를 반영했다. 이 기록은 actual Astra/max 검토 후 root의 설계 결정이며, Astra에게 구현이나 전체 목표 완료 판정을 위임하지 않았다.
