# E158 RM 단일 운동뉴런 근육의 활성화 시간 대조

E157에서 중단 구간 RM의 전방 접촉 속도에 coxa-roll이 크게 기여했고, 제어 목표의 상승과 실제 관절 상승이 거의 동시에 관측됐다. 전진 구간에서는 같은 관절이 반대 방향으로 기여했다. 따라서 전 구간 이득 약화나 물리 추적 개선을 반복하지 않고, 지금까지 고정했던 단일 운동뉴런 근육의 활성화 시간만 한 번 대조한다. 이는 BANC 해부학적 기전의 검증이 아닌 공학적 인터페이스 가설이다.

부모의 저장 자료 점검과 실제 Sol/medium의 읽기 전용 매핑 점검은 parent-e158-design-timing.* 및 parent-e158-sol-mapping-audit.json에 기록한다. 이후 후보 수치나 물리 결과를 보기 전에 이 설계와 입력을 동결한다.

## 고정 개입과 경계

E150의 RM group67 sternal_adductor_muscle은 MN column364 / graph index144018 / flywire:banc:888:720575941644474312 하나로 구성된다. moment row29가 +1이고 이 행의 다른 95개 열은 0이다. 이 근육의 현재 10ms 활성화만 기존 causal bank의 40ms 활성화로 대체한다. 40ms는 E146부터 존재한 고정 branch이며 이번 결과에 맞춰 선택·적합·혼합하지 않는다.

원래 E146–E150의 singleton 고정 계약은 당시 실험에 그대로 적용된다. 이번 실험은 그 중 group67 하나를 명시적으로 해제하는 새 가설이다. 원래 monotone adapter의 singleton validator나 이전 결과를 고치지 않는다. gain·rest·moment·상류 적응·391개 MN 출력·나머지 95개 근육·8개 tendon 입력·여섯 adhesion 값·물리 설정은 보존한다. 기존 MN 적응 0.3초와 이번 근육 40ms를 혼동하지 않는다. 접착은 원래 전 구간 off이며 별도의 lift neuron을 변경하는 실험이 아니다.

후보 EMA는 t0의 저장 bank[:,364,2] 상태에서 시작해 각 저장 motor_rate[t,364]를 사용하여 a += -expm1(-0.005/0.04)*(rate-a)로 갱신한다. 미래 표본, 위상 시계, 외부 궤적, 시간별 gain은 없다. 전체 2001개 저장 bank와 절대오차 1e-10 이내 일치해야 한다. 저장 원래 target 42개 중 row29만 rest[29]+0.02*a로 대체한다. 나머지 41개 target은 bit-exact 복사한다. 원래 전체 target을 rest/moment/activation으로 재구성한 오차도 1e-10 이내여야 한다.

## 한 원본 실행 안의 두 단계

첫 단계는 신체 생성 없이 저장된 2001개 제어 경계의 후보 명령을 계산한다. 원래 99개 시작 표본의 시간 0.005–0.495초와 정확히 정렬하고 원래 E100/E103 target과 비교한다. **전 관절 RMS ≤0.02rad, 최대 절대오차 ≤0.04rad**, 여섯 LTM RMS ≤0.02, 최대 ≤0.04를 모두 만족해야 다음 신체 단계에 진입한다. 기준 미달이면 startup_rejected를 기록하고 body/physics/neural 0으로 종료한다. 이 경우 물리 가설이 반증됐다고 말하지 않는다. 단순히 이 고정 후보가 시작 호환성 계약을 못 넘은 것이다. 같은 E158에서 다른 τ나 blend를 시험하지 않는다.

첫 단계가 통과하면 E150 start의 동일 신체 상태를 한 번 복원하고 0–10초를 재생한다. 신경망은 생성하지 않는다. 2000개 제어 간격, 간격당 50개의 0.1ms physics를 실행한다. E150의 원래 endpoint hold/기타 명령 처리를 유지하고 관절29의 endpoint만 바꾼다. 모든 native 단계에서 joint ctrl 42개가 후보 endpoint, 나머지 40개 ctrl이 원본과 exact인지 검사한다. 시작 상태 qpos/qvel/ctrl과 모델 hash는 원본과 일치해야 한다. root 보정, 추가 baseline 실행, checkpoint1s에서 재시작, 서브스텝 궤적 조작은 없다.

shadow MjData 하나를 사용해 native 접촉과 힘을 관측한다. live actor에 추가 mj_forward를 호출하지 않는다. 이전 도구의 접촉 좌표·단위 규약을 재사용한다. 완료된 적분 단계의 qpos/qvel/ctrl/시간을 먼저 등록한 뒤 관측을 호출한다. 이후 관측 실패에도 완료된 적분 및 부분 원자료를 남긴다. body/shadow/적분/관측은 requested/completed를 구별한다. 발 이외 접촉, upright<0.9, 경고, 비유한 상태가 발생하면 건강성 실패로 종료하고 원자료를 보존한다.

## 사전 효과 판정

완전한 10초 자료에서 기존 analyze_c_banc_biological_gait.py 및 E113의 body_quality/supported_slip 정의로 F3/F4/F5 수치를 계산한다. 최종 F1–F8의 문턱을 바꾸지 않는다. 이번 body-only 시험은 최종 F1/F2/F6/F7의 대체가 아니다.

후속 전체 BANC 폐루프로 진입하려면 아래 조건을 **모두** 만족해야 한다.

- 원본·수치·범위·시작 호환성·2001 control/100000 native health와 자원 검증 통과.
- F3 numeric 통과, 고정 초기 방향 순전진과 net/path 모두 E150의 95% 이상.
- native [.5,10) 하중 가중 slip이 E150 이하.
- F5의 네 무릎 운동폭 기준 유지.
- 기존 정의의 최장 전진 bout가 0.3초 이상이며, 여섯 다리 모두 최소 한 개의 완전한 적격 주기가 새로 발생.
- 고정 시간 (1,2]초의 RM 뒤쪽 힘 적분 크기가 E154보다 10% 이상 감소. 모든 leg/전체 앞·뒤 충격량도 공개한다. 후보의 이동/중단 분류에 따라 창을 재선택하지 않는다.

이 여섯 기준은 최종 자연 보행 성공 조건이 아닌 다음 비용이 큰 검증의 선택 규칙이다. 일부 개선만 있으면 항목별로 기록하되 후보를 승격하지 않는다. 무릎 CV/LR phase 미달을 숨기지 않는다. 시작 단계에서 종료하면 이 물리 지표는 null/미실행으로 둔다.

## 구현·검토·자원

부모 설계 → 기존 실제 gpt-5.6-sol/high 첫 구현 → 부모 첫 리뷰 → 필요 시 같은 Sol 첫 수정 → 부모 두 번째 리뷰와 잔여 수정 순서다. 새 신경/CUDA 구조 없이 고정 EMA, native 물리, 실패 기록을 연결하는 작업이므로 high를 사용한다. Sol은 tools/verify_c_banc_m3_rm_singleton_kinetics.py와 tests/test_c_banc_rm_singleton_kinetics.py만 소유한다. 최초 source gate 전에는 생성된 배열/stub 검사만 허용하고 원본 candidate 계산·StateStore.load·신체·물리·신경은 0이다. 부모는 별도 파일에서 독립 scalar EMA·비변경 항목·시작 오차와 출력 수치 검산을 준비한다.

원본 process 1회, 자동 재시도0, CPU thread1/GPU0, fit0, neural0. 시작 gate를 통과할 때만 body≤1, shadow≤1, constructor settling≤2000, scientific physics≤100000, shadow forward≤100000이다. 내부260초/외부280초/감독295초, 출력1GiB, 여유공간4GiB로 제한한다. 이전 보수누적2619.309456초를 올림한2620초 + 이번 감독295초 = 사전누적2915초다. 실제 종료 시간이 작아도 실패를 삭제하지 않는다. 자원 초과 또는 불완전 자료는 효과 미달과구분한다.

입력·재사용 소스·과거 결과/M2 해시를 전후 확인한다. 최초 구현/수정/부모 리뷰/최종 source gate와 실행·오류·부분raw·manifest를 보존한다. 최초 도구 작성 호출의 JavaScript 인용 오류는 파일 생성 이전에 발생한 부모 준비 오류이며 원본 실행이나 후보 실패가 아니다. 활성 UI/서버, M2, F6 미사용 초기 상태는 변경하거나 노출하지 않는다. 통과 시에만 별도 설계로 실제 신경 폐루프 adapter 상태·차단·미래재현을 검증한다. body-only 성공을 자연스러운 자율 보행 완료로 부르지 않는다.
