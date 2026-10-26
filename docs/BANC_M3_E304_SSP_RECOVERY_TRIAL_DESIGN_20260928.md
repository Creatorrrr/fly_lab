# E304 SSP 회복 신경망의 whole-session 연결

Root 설계,2026-09-28. E303은 동일 연속 ODE를 SSPRK3로 적분하는 새 engine이며 active/u0 모두 같은solver를사용한다. 이번 작업은 E300 whole-wrapper의 검증된 소유권/체크포인트 계약을 새solver identity에 연결하는 것이다. 새신체보행을 미리 실행하지 않는다. Root 설계→실제gpt-6-sol/high구현→rootFIRST→필요시같은SolR1→rootSECOND→root잔여수정이다.

Agent 소유는 새 `tools/banc_m3_ssp_recovery_trial.py`, `tests/test_c_banc_ssp_recovery_trial.py` 두 파일뿐이다. 기존 E299/E300/E301/E302와 새E303 agent파일은 수정하지 않는다. `SSPRecoveryResourceTrial`을 제공하고 engine은 lazy `network_api()`로 `tools.banc_m3_ssp_recovery_synapses`를 읽으며 생성자는 `SSPRecoveryResourceNetwork`다. `NETWORK_SCHEMA=flylab.m3-ssprk3-recovery-resource.v1`, `TRIAL_SCHEMA=flylab.m3-ssprk3-recovery-resource-trial.v1`, `PATCH_SCHEMA=flylab.m3-ssprk3-recovery-resource-trial-patch.v1`, `PATCH_KEY=ssp_recovery_resource_trial_patch`로 구별한다. metadata/source hash/precision은 새engine의정확한함수로검증한다. E300과 같은 selection keys, recurrent_resource/synaptic_resource snapshot keys를 유지한다.

E300을 참고하되 전역변수변경/monkeypatch로기존production클래스를바꾸지 않는다. 독립된 새wrapper의identity-specific parts를 갖고 공통pure구조검사/cleanup/storage helpers는 재사용할 수 있다. 새 schema/identity를기존schema로바꾸어검사를통과시키는우회는금지한다. 삭제되지 않은 source_model_hash, graph, adapter, original monotone patch, all inherited metadata가 그대로 같아야 한다.

초기 적용은 정확한 열린 MonotoneRecruitmentTrial, controltick0/networktick0/rate0/z0/drive0/mask1만 허용한다. Wrapper 생성은몸상태/제어기/감각/physics시계/metadata를변경하지않고 새network만적용한다. 모든검사후callerbody ownership을이전한다. Precommit 실패시caller열림/상태동일/몸close0이어야한다. R초기는전체1이다. 파라미터/tau/gain/CSR/roster/state를clip·coerce·fit하지않는다.

`checked_checkpoint/remove_resource_layer/make_candidate/metadata_without_state`와whole save/from_checkpoint/restore를 제공한다. Nonzero whole restore는새body와새engine을복원하고 모든dynamicstate 및metadata를검증후원래body를닫는기존transactional순서를유지한다. 실패시새body/engine을해제하고원래body는유지한다. Commit후oldbodyclose실패는commit사실을명시한다. Resource-layer제거는저장복원을위한원래body/adapter재구성용이며그상태로기존RK4와새SSP의미래가같다는주장이아니다.

변경된 solver의모든neural rate/adaptation은finite/nonnegative여야한다. 기존wrapper의 -1e-4 rate 허용치를그대로복사하지않고 **모든rate<0을 body 생성 전에 거부**한다. 양/음zero는동일한0범위, 음수subnormal은거부한다. R는f64,[0,1],u0R1/tick0R1, masksbinary/native dtype/shape, source/identity는정확히검사한다. Hash-covered tuple/NumPy scalar 불안정표현은기존에정한정규화또는거부정책을따르고array를느슨하게바꾸지않는다. Disk복원에대한`require_saved_whole_equal`은기존purehelper를재사용할수있다.

생성검사는 fake engine/body만 사용한다. E300회귀를수정해새schema/hash/constructor연결,nonzeroresource+rate checkpoint,wholefreshrestore후미래동일,모든실패지점ownership,graph/adapter/neural/source/selection/precision오염,원래RK4/E299schema거부,전체negative rate의body생성전거부를검증한다. Fake full metadata와disk실제serializer를사용하되actualengine/model/neural/CUDA/body/physics/solver/fit0이다. 모든실패를포함한외부실행45초/1GiB/CPU1을기록한다. Agent는tinyactualE303도실행하지않는다.

E303소스가준비되기전에는generatedfake network_api 주입으로검사하되 실제original검증을우회하는production분기는넣지않는다. FIRST제출후새2SHA/전체검사·실패·fake호출/시간/peak/actual0을보고하고편집을멈춘다. Root가E303 CPU/CUDA자격후실제tinyneural+fakebody연결을별도정해검사한다. 성공은wholecontract자격이며보행/생물학/F1–F8완료가아니다. 원본예약0·누적8144.108421799498/8210초·M2/UI/F6미노출을유지한다.
