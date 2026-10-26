# E300 — 회복 억제식의 전체 session 상태 래퍼

Root 설계, 2026-09-28. [E299 신경 엔진](BANC_M3_E299_RECOVERY_RESOURCE_ENGINE_DESIGN_20260928.md)을 원래 E150 MonotoneRecruitmentTrial에 통합하는 별도 wrapper만 구현한다. 원본 신체 생성이나 candidate실행은0이며 E290의 전체 상태/소유권/복원 계약을 그대로 유지한다.

신규 파일은 `tools/banc_m3_recovery_resource_trial.py`, `tests/test_c_banc_recovery_resource_trial.py`다. 실제 gpt-6-sol/high가 구현한다. 클래스 `RecoveryResourceTrial`, schema `flylab.m3-recovery-resource-trial.v1`, patch schema `flylab.m3-recovery-resource-trial-patch.v1`, patch key `recovery_resource_trial_patch`를 사용한다. E299 engine API는 `RecoveryResourceNetwork`, `resource_metadata`, `checked_resource_metadata`, `resource_identity`, `checked_selection`이다. Source metadata key는 `recurrent_resource`, state key는 `synaptic_resource`이고 반드시 E299 schema/precision/identity를 검증한다. 기존 R 엔진/patch/checkpoint를 새 식으로 읽지 않는다.

원래 MotorTimescaleNetwork/MonotoneRecruitmentTrial tick0만 최초 입력으로 받는다. Neutral rate/z/drive0 mask1/R1, 모든 metadata, graph/adapter/readout/신체/seed/누적시간의 정확한 보존을 확인한 뒤 호출자의 body소유권을 한번만 넘긴다. Native source topology/CSRweights/ordered391motor/공유source 파라미터를 그대로 사용한다. 새 engine 선택과 identity를 검증하며 원래 source를 편집하거나 전역 API를 교체하지 않는다.

원래 E290의 StateStore tuple/list·NumPy scalar 표현만 정규화하는 whole 비교를 읽기 전용 재사용한다. 배열 dtype/shape/bytes와 wall을 포함한 모든 상태값을 유지하며 exact 외의 수치 오차를 허용하지 않는다. Checkpoint metadata는 hash-stable 표현이어야 한다. E299 유효 상태인 active source r≥0, R[0,1], u0R1을 whole validator에서 확인하여 잘못된 저장값이 새 신체 생성 전에 거부되게 한다. 그밖의 native rate−1e−4 허용은 원래대로다. Faultflag가 있는 engine은 snapshot성공을 허용하지 않는다.

비영 whole disk 저장·복원은 원래 body/native qpos/qvel/contact/RNG/readout/controller/neural R를 모두 복원한다. 원래 session에 neural state만 이식하거나 resource를 초기화하지 않는다. Restore는 fresh 전체 session구성→완전검증→교체 commit→oldbodyclose 순서이며, 실패는 새 소유 자원을 정리하고 이전 session을 보존한다. Commit 뒤 oldclose실패는 실제 commit된 상태를 명시한다. 이 소유권 계약과 모든 실패 paths는 생성 fake body/network로 검증한다.

Root FIRST→필요시같은Sol R1→root SECOND→root잔여수정이다. Agent는 fake·tiny 생성 검증45초/1GiB/CPU1, original/nativebody/CUDA/solver/fit0으로 제한하고 모든 시도와 비용을 보고한다. E299 source가 완료되지 않아도 자체 fake API를 test fixture에서 명시적으로 주입할 수 있으나 이를 실제 E299 자격으로 부르지 않는다. Product 전역 mutation은 금지한다. E299와 함께 root가 실제 source 계약과 tiny CPU/CUDA/diskfuture를 확인한 뒤에만 다음 원본 runner를 별도로 설계한다. 사용자 보행모드/UI/defaults와 F1–F8, 과거실패·8078.0529733995245/8150초·원본예약0·F6미노출은 유지한다.
