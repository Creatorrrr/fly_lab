# E189: 단일 relay→운동신경 연결의 양방향 국소 대조

2026-09-24. 기존 finite-goal의 다음 제한된 진단이다. E188 관측의 원본·실패·검산 정정을 보존하고 F1–F8, M2/UI, F6 노출0을 유지한다. E188은 수치적으로 유효하지만 보행 후보를 선정하지 못했다. 이 단계는 pooled 순위에서 유리한 방향을 골랐다는 주장이 아니다.

## 결정과 경쟁 설명

주장할 수 있는 범위는 **현재 직접 연결의 작은 변화가 MN144018→근육67→RM coxa-roll→물리 반력까지 전달되는지**다. E186은 해당 운동뉴런이 근육67과 coxa-roll 명령의 단독 소유자임을 검산했고, E185는 전진 중단 구간에 RM 음의 전방 반력을 관측했다. E161에서 coxa-roll은 수평 제동과 연관되지만 수직 하강의 주된 항은 femur/tarsus였다. 이 경로가 접지 해제를 해결한다고 가정하지 않는다.

경쟁 설명은 신경 입력 변화가 현 동역학에서 약하게 전달됨, 근육 명령은 변하지만 접촉·다른 관절이 물리 반응을 지배함, 음반력이 줄어도 추진·지지·전진이 함께 악화됨이다. E188의 사건 구성 분해는 gain 방향의 근거가 아니므로 양방향을 사전에 고정한다. 실제 gpt-6-sol/high의 읽기 감사도 이 범위를 독립적인 국소 인과 민감도 시험으로 판단했다.

E95는 상류 E2→IN03A010(54014)의 입력 row/gain을 바꾸고391개 MN 수신 행을 보존했다. 이번 직접 edge의 실패 선례로 바꾸어 쓰지 않는다. E160은 근육67의40ms kinetics가 startup 기준을 넘은 사례, E162는 근육70 전달, E177은 다른5개 감각 연결 복원이다. 이 실패와 현재 양·음 조건의 반례를 보존한다.

## 고정 조작과 비교군

기존 CSR edge10511590, pre54014/root720575941504444354, post144018/root720575941644474312, 원래23 contacts를 사용한다. 현재 source float32 weight는 `0.2087680697441101`이다. 다음 두 값만 허용한다.

- low: `float32(float64(source_float32) * 0.98)`
- high: `float32(float64(source_float32) * 1.02)`

구조·부호·나머지 모든 CSR 값, 신경 drive/mask/초기 rate/adaptation, core·motor gain/tau, 근육 bank/slopes/kappa, body/초기 자세/감각 encoding은 동일하다. 2%는 현재 연결 주변의 작은 고정 섭동이며 생물학적으로 측정한 효능 범위나 최적량이 아니다. 추후 같은 실행에서 값·부호·시간창을 고르거나 backtracking하지 않는다.

비교군은 원자료 exact가 확인된 E188/E185의 동일 theta0 3초다. 새 baseline 물리 실행은 만들지 않는다. 각 팔은 독립된 E150 zero checkpoint에서 시작하며 600 controls,12000 neural,30000 native가 목표다. 한 팔의 최종 상태를 다른 팔의 시작으로 쓰지 않는다. 후보를 최종 M3로 채택하거나10초/F6/목표 접근 성공으로 올리지 않는다.

## 구현 경로와 실행 전 증명

부모가 설계·metadata·감독·독립 검산을 맡는다. 실제 gpt-6-sol/high가 새 `tools/probe_c_banc_m3_single_mn_edge.py`와 `tests/test_c_banc_single_mn_edge.py`를 구현한다. 기존 모듈은 수정하지 않는다. 여러 실행 계층의 상태·단일 CSR 변경·native 기록·실패 보존을 연결하지만 새 신경 수식은 만들지 않으므로 high를 선택한다. FIRST→같은Sol첫수정→SECOND→부모잔여수정 순서를 지킨다.

기존 `tools/banc_m3_load_feedback.py`의 `_validated_patch`, `patch_feedback_weights`, `LoadFeedbackTrial`은 이름과 달리 길이를5로 제한하지 않고 임의의 기존 동일부호 CSR 항목 목록을 검증·복사한다. 해당 상태 이전·packed CUDA 재구성 기구를 재사용할 수 있다. 그러나 이것을 load sensory 개입이라고 부르지 않으며, E189 sidecar에 실제 relay→MN 경로와 단일 항을 명시한다. `fixed_rm_load_patch()` 또는 E177의5개 항 검증을 호출하지 않는다.

원래 `MonotoneRecruitmentTrial.from_checkpoint`로 zero state를 정확히 복원하고 단일 항만 바꿔 새로운 `MotorTimescaleNetwork`를 구성한다. CSR values를 직접 덮어쓴 뒤 stale packed table을 쓰지 않는다. 변경 전후 전체 CSR의 shape/indptr/indices와 모든 비선택 float32 words가 exact여야 한다. 실제 packed rowptr·decoded pre/weight 전체가 새 CSR와 일치해야 하며 원래 표적 값과 새 값도 일치해야 한다. 전체158706/deviceCUDA/packed를 기록한다. body는 같은 객체·model hash이며 neural/body/control tick0과 동적 state가 동일해야 한다. snapshot을 기존 방식으로 되돌려 비교할 때 바뀐 네트워크 정체성4항과 provenance schema만 구분한다. 불필요한 추가신경 step/물리 step으로 검증하지 않는다.

E181의 `run_arm`은 baseline/기존 timing-axis gate 전용이다. **E181 baseline gate를 위조하거나, restore factory를 몰래 바꿔 새 weight 변경을 기존 gate가 승인한 것처럼 실행하지 않는다.** 새로운 E189 typed gate와 명시적 run 함수에 원래 loop·startup99·native health·clock·failure persistence·biological evaluator의 동작을 재사용한다. 필요하면 검토 가능한 정적 함수 사본을 새 파일에 두되 변경 부분은 새 gate/생성기/metadata/native force 연결로 한정한다. E181/E150/shared runtime의 소스나 원래 판정 기준은 바꾸지 않는다. 새 gate는 원래 소스 SHA, 새 arm/weight, 현재 E188 최종 검산, 모든 원본 입력, 부모 승인과 같은 live parent의 검증 영수증을 묶는다.

E185 `install_native_observer`/`shadow_force_sample`/`force_arrays`로 동일 native 상태의 힘을 읽고, E150의 기존 recruitment 관측을 함께 보존한다. 같은 shadow forward를 사용하며 원본 step을 두 번 수행하지 않는다. observer 복원·상태 소유자 close·monitor 종료·부분 raw 저장을 모든 예외 경로에서 수행한다. 실패한 팔을 자동 재실행하지 않는다.

## 원자료와 판정

각 팔에 시작/종료 whole state, 전체601 control trace와391 MN/96근육 recruitment,30000 native 접촉 및 반력, 초기 CSR/packed 검증, 실제 selected weight/current/MN rate/근육67 activation/요청·실제 coxa-roll을 보존한다. 다른 다리 영향도 같은 전체 자료에 남긴다. partial rows, startup/health/resource/저장 실패와 actual neural/physics counts를 구분한다.

주 비교는 `[0.5,3.0)`의 고정 clock, E185의 미리 정한8사건 `[131,219,308,391,433,480,521,564]`의 고정 post10controls, 전체3초와startup99다. 사건은 후보 궤적으로 다시 찾지 않는다. native interval/control endpoint 매핑을 원래 E186 정의와 대조한다. 사건521의 혼합 native 표본은 control530이었다는 E188 정정을 유지한다.

같은 시간창에서 RM 및6다리의 signed AP impulse, 음반력 magnitude impulse, 접촉·지지 표본수, native slip의 분자/분모와 값, upright·비발 접촉·전진거리·net/path·가장 긴 적격 전진 구간을 보고한다. 후보에서 새로 고른 반력 부호 조건의 평균만으로 효과를 판단하지 않는다. 방향별 결과와 baseline 차이를 모두 남긴다. 양팔 모두 유효할 때만 대칭 local response와 비선형성을 기술하며 이를 독립 표본 통계 또는 인체·생체 인과 증명으로 부르지 않는다.

startup99는 원래 q_RMS≤.02/q_max≤.04/LTM_RMS≤.02/LTM_max≤.04를 유지한다. 미완료·건강성 실패·이 기준 미달인 팔은 같은3초 물리 효과 비교의 유효한 팔로 세지 않는다. 신경/명령 변화가 없으면 현재2% 범위의 전달이 약한 것으로, 명령은 변하지만 반력이 같으면 해당 국소 기계 연결이 약한 것으로 기록한다. 반력 개선과 추진·지지·slip·전진이 상충하면 후보를 채택하지 않는다. 개선이 있더라도 후속10초/F1–F8 검증을 별도로 설계한다. 2%에서 무효인 결과를 더 큰 모든 변경의 불가능성으로 일반화하지 않는다.

## 자원과 출발 상태

현재 원본 누적4452.92451379972/4994초, 잔여541.07548620028초에서 팔별150초, 두 팔 합계최대300초를 예약한다. CPU affinity1/BLAS1/GPU1, 순차 child2, 자동재시도0, 각 팔 body process100초(정리10초포함), child 전체140초, 부모145초+owned cleanup5초, 전체원본한도4994초를 유지한다. source/data SHA·신규구성·자원 준비·분석·저장도 해당150초에 포함한다. 각 팔 output2GiB·body raw1GiB·disk여유20GiB를 유지한다.

E188의 승인된 resource helper와 stopped-service proof를 사용한다. 부모의 두 번 준비관측은2초간격이며 영수증에 묶어10초안에 자식이 수락한다. 자식 monitor는 자기 실행 외의 Python을 제외하지 않으며, 부모·현재자신·자기자손만 정확하게 판별한다. 외부 작업을 종료하거나 UI를 바꾸지 않는다. 부모 사전 준비·합성 테스트·독립 검산은 별도 기록하며0으로 숨기지 않는다.

현재 E189 구현·원본 실행은0이다. 소스의 단일 변경 증명과 리뷰를 통과하지 못하면 native 실행 전에 그 원인을 남긴다. 최종 성공 기준을 낮추지 않고 현재 goal을 active·미완료로 유지한다.
