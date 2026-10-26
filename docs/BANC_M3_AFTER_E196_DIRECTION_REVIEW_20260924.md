# E196 이후: 전달 증가와 지속 보행을 구분

전체 finite-goal은 같은 최종 후보의 F1–F8 통과이며 아직 미완료다. E196은 hook 감각의 방향 표현을 바꿔 신경·근육·관절 출력까지 변화가 전달되는 것을 확인했지만 지속전진 .3초에 실패했다. E195/E196을 이유로 같은 감각 gain 또는 문턱을 반복 조정하지 않는다. 미끄러짐의 작은 개선과 많은 감각세포의 반응을 자연 보행 성공으로 합치지 않는다.

## 실제 구동 경로의 재확인

actual gpt-6-sol/high에 원본 실행 없는 읽기 전용 소스·실패 이력 감사를 맡겼고 부모가 핵심 소스와 저장 trace를 확인했다. 현재 경로는 391MN → 근육 pooling/moment·활성 상태 → 42개 목표각 → `tracking_all` position servo → 50개100µs 물리 단계다. 모터 decoder 중간 갱신과 물리 서보 gain 변경은 서로 다른 개입이다.

`flylab/whole_body.py:16–44`에서 다리 Kp는 최소1000이고 Kv는 compiled armature·passive damping과100µs 안정성 상한으로 정한다. `flylab/body.py:425–479`의 기본 제어는 5ms 동안 목표각을 유지한다. 서보는 공학적 액추에이터이며 실제 근육 생리를 복원한 것이라는 의미가 아니다.

E193와 외부 기준 E85는 body hash `1b2a89084311ebd2ccbccbe2c57f2ec29fa937e115bb6cd4c7f6cd16ba7c2e6f`가 같다. E85는10초17.349mm 전진을 기록했다. 부모가 저장 trace에서 계산한0.5초 이후 전 관절 추적 RMS는 E193 .001561726890rad, E85 .002032101333rad다. 서로 다른 명령·초기 상태·시간 길이의 전체 RMS이므로 특정 관절이나 접촉 순간을 배제하는 검사가 아니다. 그러나 신체/서보 자체의 일반적 보행 불가능이나 큰 경계 추적 오차는 현재 근거로 지지되지 않는다.

M2 E8 낮은 서보 강성은 slip3.542배로 기각, E20 torque cap은 품질 실패, E24 활성 강성은 slip 개선에도 전진 감소로 실패했다. E58 수동 damping2배는14개 구간의 두 기준군 동시 통과에 실패했다. E22는 목표 보간(slip+31.5%), M3 E156은 중간 decoder 명령 전달(slip1.8415→2.0519)을 시험했다. 이것을 E150/E193에서 Kp/Kv만 분리한 실험이라고 부르지는 않지만, 명확한 새 근거 없이 물리 gain/damping 탐색을 시작하지 않는다.

## 다음 설계의 범위

기존 E107의 외부 shape-only 진단은 각 다리12개 적격 주기를 회복했고 timing-only는 실패했다. E163에서는 후반 신경의 상대 리듬이 강하게 유지돼도 주기당 두 부근에 전진 중단이 반복됐다. 현재 남은 설계 질문은 **현재·과거 운동신경 상태로 필요한 다리 출력 파형을 만들 수 있는가, 아니면 신경 상태 생성 자체를 바꿔야 하는가**다. 런타임에 외부 교사/phase/time gate를 추가하지 않는다.

E148/E169/E175/E192가 이미 검사한 양수3bank·70ms 이력 endpoint·singleton 확장과 같은 가족의 계수 재적합을 새 구조라고 부르지 않는다. E174/E175의 실패 범위는 그 저장 신호/보호 조건의 전달 가족이며 BANC 전체의 불가능성은 아니다. 신규 동역학을 고르기 전에 기존 전체 신경 상태 생성·활성 이력 구현과 누락된 표현 능력을 비교하고, 실제로 다른 기전·같은 startup/보행 기준·단일 개입과 자원을 사전 정의해야 한다. 이번 감사는 다음 원본이나 fit의 사전 설계를 대체하지 않는다.

새 과학적 신경/물리/fit0, F6노출0, 후보미채택, M2/UI 보존이다. 원본 감독 누적4997.566624599716/5144초 및 잔여146.433375400284초를 유지한다. 부모 저장 산술·소스 확인과 Sol 감사 비용은 원본 감독과 별도이며 미계측 비용을0으로 처리하지 않는다. 개발 순서는 부모 설계→gpt-6-sol 구현→부모FIRST→같은Sol첫수정→부모SECOND/잔여수정을 유지한다.

근거: `BANC_M3_E196_DIRECTIONAL_SENSORY_RESULTS_20260924.md`, `BANC_M3_E163_SIX_LEG_TIMING_RESULTS_20260922.md`, `BANC_M3_AFTER_E175_DIRECTION_REVIEW_20260923.md`, `BANC_GAIT_QUALITY_GOAL_PLAN_20260912.md`, `verification/banc-natural-gait-m3-20260919/e85-space-compatible-cadence-reference/trace.json`, `e193-sensory-transmission/baseline/trace.json`.
