# E301 — 회복 억제식의 원래 신체·시작 기준 검사

Root 설계, 2026-09-28. E299/E300 구현과 자격 확인 뒤 사용할 새로운 body runner를 별도 파일로 준비한다. 지금은 원본 실행 승인이 아닌 구현 설계이며 전체 원본 예약0이다. E298의 same-state proxy를 실제 새 동역학 결과로 바꾸어 주장하지 않는다.

신규 `tools/verify_c_banc_m3_recovery_resource_body.py`, `tests/test_c_banc_recovery_resource_body.py` 두 파일만 실제 gpt-6-sol/high가 구현한다. 고정 모델은 E296 보호 roster의 원래7136 source/725849edge/1513활성source/117973활성edge/core향6172면제, u=.05/tau=.150이다. 파일 `e296-core-afferent-resource-roster.npz` SHA22e04433d8d9e14e89b5a2d575dfdaa68d3e21a9d8fbfb0c5a08c8ea53d7a938을 그대로 사용한다. 변경은 E299 current식/guard와 E300 wrapper/새 identity뿐이다. Neutral u0와 candidate 고정u를 별도 metadata로 계산하며 기존 R 엔진 identity를 재사용하지 않는다.

E296 freeze/review/root original audit/source bytes 및 E298 결과/audit/design의 실제 lineage를 동결한다. E296의 모든 원래 입력과 context를 유지하고 original E150 neutral session만 최초 입력으로 쓴다. BASE context에서 원래 neutral_network_identity도 새 engine identity로 바뀌어야 함을 명시적으로 구분한다. 승인된 E299/E300 네 소스와 root review/tiny qualification의 실제 SHA가 일치해야 하며 결과 파일 존재만으로 승인하지 않는다. 전체 사용 source/preservation/모든 raw input bytes/packages를 동결한다. OWNED/SOURCE_FILES/PRODUCERS/PATHS/LIMITS/STARTUP_LIMITS/FREEZE_SCHEMA/REVIEW_SCHEMA 및 gate/run/execute API와 새schema/diagnostic_id E301/output `verification/banc-natural-gait-m3-20260919/e301-recovery-resource-body`를 제공한다.

E296와 E291의 bounded write/execution/health/startup/short/native logging을 읽기 전용 재사용한다. 새 NativeDependencies에서 load→E299 metadata 검증, construct→E300 RecoveryResourceTrial, initial/neutral compare→새 wrapper layer 제거를 명시적으로 연결한다. E291.execute/run_arm의 deps 주입을 사용할 수 있다. 이전 모듈 전역 monkeypatch/기존 source 변경/metadata를 이전 모델로 가장하는 행위는 없다. E296 frozen load는 원래데이터검증으로만 사용하고 새 freeze identity와 명시적으로 비교한다. 정확한 neutral trace/native/recruitment와 whole endpoint 비교에서는 원래 E291의 독립 wall값 분리 정책만 유지한다. 과학적 수치/배열/상태를 제외하지 않는다.

원본 절차는 neutral .005초→candidate strict .5초→통과 시 같은 session을1.5초까지 지속한다. 원래99 startup targets/controls1..99의 qRMS≤.018/max≤.04,LTM RMS≤.009/max≤.025와 원래 native 건강성(upright≥.9/nonfoot0)을 유지한다. 중간값 fit/교사/CPG/추가 stabilization은 없다. 1.5초 short의 고정 [.5,1.5) 전진≥1.6790085779712631mm,slip≤1.8026436968249955mm/s,최소 upright .9/비발0/eligiblebout≥.3/각6다리완전cycle>0도 그대로다. Gate 실패는1.5·10초로 계속하지 않으며 F6노출은0이다. 모든gate가 통과해도 자동으로 최종 채택/F1–F8 완료로 표시하지 않는다.

원래 raw activity/native recruitment/native contacts/controljournal/trace/whole final/부분실패 state/ledger·source/output manifests를 유지한다. 추가 `recovery-factor-endpoints.npz`에는 완료된 활동 endpoint0..N의 source_ids/control/neural/physical clock/유도g32를 저장한다. 같은 rawrate/R로 재계산한 endpoint 계수이며 실제4개RKstage마다 사용된 계수의 측정으로 부르지 않는다. 모든 u0/core-afferent/negative source의 계수1과 finitebounds, active count를 검사하고 새 모델 안에서 실제 비자명한 endpoint 여부를 진단한다. 이것을 기준 조정/조기선택에 쓰지 않는다. 지표모두identity에대한한후보단위로보고한다.

E299 sticky resource_fault_flags가 발생하면 정상 readout/body 진행으로 넘기지 않는다. 원래 exception/flags/actualtick/nativecount/부분저장 불가와 비용을 보존한다. 새로운 guard 실패를 건강한 과학적 궤적으로 승격하지 않는다. 원래 runner의 오류분류가 technical_or_resource_failure여도 원인을 별도 남겨 신경 유효성 실패와 인프라 실패를 구분한다. 기존 ledger/보조실패/cleanup/allownedclose가 유지되어야 한다. Windows5/32/33 영수증 쓰기의 기존 유한 retry만 재사용하며 신경/물리 advance를 재시도하지 않는다.

구현·검증: root FIRST→필요시같은Sol R1→root SECOND→root 잔여수정. 생성 fake 검증45초/1GiB/CPU1, actualbody/CUDA/원본/fit/solver0이다. 새 module producer가 아직 없을 때는 test fixture에서만 명시적 fake API를 사용한다. 고정정책/identity, neutral대상, 실제run_arm .5gate/연속1.5시계·clockpartial/nativeerrors, endpointfactor표시,source manifests,실패영수증을검증한다. 원래 이미검증한inherited tests를 불필요하게복제하지 않는다.

Root가 E299/E300 tiny source 자격과 E301 실제 passivegate를 확인한 후에만 source-bound original1/retry0, supervisor125초/기존 host/disk/native limits를 예약한다. 현재원본8078.0529733995245/8150초이므로 이 전체예약은 현재잔여71.947초에 들어가지 않는다. 원본실행전 최소8203.0529733995245초이상의prospective 단계배정과누적이월을문서화해야하며기존비용을초기화하지않는다. 전체사용자시간/토큰budget은미지정이다. M2/E150/UI/모든과거실패/F1–F8/F6미노출을보존한다.
