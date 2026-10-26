# E157: 전체 물리 구간의 실제 접촉점·관절 속도 기여

부모 설계, 2026-09-21. E156은 전진 +10.598%에도 미끄럼 +11.425%로 기각됐다. 중간 명령이 초기 1ms의 미끄럼 집중을 줄이고 뒤 4ms의 미끄럼을 늘렸으므로, E155의 RM 끝점 관측만으로 어느 관절을 고칠지 정할 수 없다. 이 단계는 원래 E150 명령의 전체 0.1ms 구간을 관측하여 다음 실제 개입을 선정할 근거를 만든다. E156 후보를 재실행하거나 채택하는 단계가 아니다.

## 질문과 범위

보행 중단으로 분류된 구간에서 여섯 다리의 실제 지지 접촉점은 어느 방향으로 움직이며, 그 속도를 root·해당 다리의 7개 구동 관절·나머지 DOF가 각각 얼마나 만드는가? 접촉 힘의 전방 성분과 접선 방향 일의 분해를 함께 기록한다. 현재 상태에서의 선형 속도 분해는 관절 변경 후의 인과적 효과나 근육 생리를 증명하지 않는다.

기준은 E150 t=1초 checkpoint, control201..400, t=2초까지다. 저장된 E154 전체 native10000 및 contact24836을 원래 명령으로 정확 재생한다. 새로운 명령·fit·신경망·CUDA 실행은 없다. E154는 native qvel/Jacobian을 저장하지 않았으므로 새로운 관측을 위해 신체 재생 1회가 필요하다. 기존 E154 상태·힘·접촉 기록과 일치하지 않으면 진단 자료 오류로 중단한다.

## 고정 측정

E154의 기존 `construct_body_safely`, `replay_command`, `_observe_shadow`와 원래 `step_joint_targets`를 이용한다. 실제 body에는 관측용 forward를 호출하지 않고 integration state를 shadow mjData로 복사해 forward한다. 모든 control qpos/qvel/ctrl은 원래 trace와 exact, 모든 native 시간·회전·힘·하중·slip·유효 mask·nonfoot·upright 및 contact id/geom/segment/leg/sign/frame/wrench/point는 E154와 exact여야 한다. 이상적 tick 시간과 누적 float clock의 비교에는 기존 2e-8초 허용폭을 사용한다.

native10000의 qpos/qvel/ctrl, 실제 회전·힘·하중·slip와 각 단계의 진행/관측/검증 mask를 저장한다. 접촉24836 중 양의 normal force를 갖는 **24420개 전부**에서 foot와 support의 실제 contact world point에 대한 `mj_jac`를 각각 호출한다. 총48840회이며 하위 표본을 고르지 않는다. 원래 leg별 개수는 LF3808/LM3720/LH5093/RF2363/RM6156/RH3280이다. 접촉점은 발끝 원점으로 대체하지 않는다.

모델에서 검증한 root6과 전체42 구동 hinge DOF의 매핑을 저장한다. E155 `validate_layout`의 일반 레이아웃 검증을 재사용할 수 있으나 RM 고정 contact 분류는 재사용하지 않는다. 전체 30개 발마디 segment에서 `leg=segment//5`, sign으로 foot/support geom을 식별하고 실제 map과 비교한다. 여섯 다리 순서는 lf/lm/lh/rf/rm/rh이며 각7개 제어 인덱스와 실제 joint 이름·DOF 주소·geom body 주소를 함께 저장한다.

각 contact에서 n=frame[0], P=I−n nᵀ, Jrel=Jfoot−Jsupport, Jtan=P Jrel이다. 다음을 기록한다.

- 전체 접선 속도 `v=Jtan qvel`, slip=`norm(v)`.
- 전방 성분은 해당 native의 수평 단위 heading h를 사용한다. DOF별 성분은 `(hᵀ Jtan)_j qvel_j`, root6 합·소유 다리7 합·나머지 합과 전체 합을 저장한다. 이 수평 접촉 방향은 F4의 물리 forward 속도 정의를 바꾸지 않는다.
- fly에 가해지는 실제 힘은 `f=sign * frame.T @ wrench[:3] / weight0`이다. 접선 힘 ft=P f에 대해 DOF별 접선 power 성분은 `(ftᵀ Jtan)_j qvel_j`다. 전체 합은 `ft·v`와 같아야 한다. 이를 dt=0.0001초로 적분한 값의 단위는 BW·mm이며, SI joule이나 근육 자체의 일로 부르지 않는다. contact 힘의 torque 성분과 회전 마찰 일은 이 선형 접선 power에 포함하지 않는다.
- 접촉별 normal force/weight0로 속도와 slip을 가중한 native/leg 평균은 E154의 load/slip와 절대1e-8 이내로 일치해야 한다. 모든 분해의 합과 직접 계산은 절대1e-10 이내여야 한다. 0하중 leg는 미끄럼 유효값을 만들어내지 않는다.

비음수 normal load로 속도를 평균하지만, 이미 힘을 포함한 power에는 normal load를 다시 곱하지 않는다. 모든 양의 contact를 포함하는 기본 집계와 native leg load>=0.02BW인 지지 부분집합을 구분한다. 제동/추진 부분집합은 추가로 해당 native/leg의 net forward force<0/>0이며, 개별 contact force 부호와 혼동하지 않는다.

그룹은 all, forward, interrupted, early, late, interrupted_early, interrupted_late의 7개다. forward/interrupted는 기존 E152의 전체10초 Gaussian20ms 속도 기록에서 control201..400의 >0.5/<=0.5를 각50개 native에 반복한다. 이는 5ms 해상도의 조건 분류이지 새 native 속도 판정이 아니다. early는 각 control의 처음10단계, late는 뒤40단계다. 각 그룹/다리의 표본 수, 지원 수, 정상 하중합, 하중 가중 속도 성분과 개별7관절 기여, slip, 양/음 전방 충격량, DOF별 접선 일을 함께 보고한다. 빈 그룹/무하중 평균은 null이며 0으로 위장하지 않는다. 집계법·그룹·허용폭을 결과 후에 바꾸지 않는다.

## 구현과 실패 기록

소유 파일은 `tools/analyze_c_banc_m3_native_contact_decomposition.py`, `tests/test_c_banc_native_contact_decomposition.py`다. 기존 source와 입력은 읽기 전용이다. 실제 Sol/high가 FIRST 구현한다. 이 작업은 고정 Jacobian 수식과 물리 관측의 결합이므로 high로 정했다. 새로운 CUDA/신경 동역학을 구현하지 않는다. 부모 첫 리뷰 후 필요한 첫 수정은 같은 Sol, 두 번째 리뷰 이후 잔여 수정은 부모가 담당한다.

입력·소스·보존 SHA, source review gate, 새 output 경로, 실제 BLAS CPU1과 disk reserve를 검증한 뒤 모델을 만든다. 원본 해시를 완료 뒤에도 확인한다. 원본 run은 별도 부모 gate가 없으면 불가능해야 한다. 소유 두 파일의 실제 SHA와 동결 SHA를 gate에 요구한다.

원시 관측과 완료 mask를 수치 assertion 전에 등록한다. native의 실제 qvel과 각 contact의 두 Jacobian 원시값은 실패 시에도 보존한다. 반환된 NaN, 실패한 두 번째 Jacobian, 완료된 적분 후 관측 실패를 구분한다. call 요청/반환/완료 횟수를 나누며 실측 수로 기록한다. body.close, method 복원, raw 저장 실패는 최초 실패와 별도로 남긴다. 부분 자료에 완료 표본24420을 허위로 붙이지 않는다. result/progress 이후 manifest를 확정하고 마지막 게시 뒤에도 내부 deadline을 확인한다. 원본 출력 재시도나 덮어쓰기는 없다.

raw는 전체 native와 contact 전환 인덱스, 실제24420의 source_raw_index·native_index·point/frame/wrench/geom·jfoot/jsupport 및 가용/완료 mask, DOF별 전방 속도와 접선 power·합 오차, 실제201 control 및 모델 레이아웃을 포함한다. qvel은 native index로 연결해 불필요한 중복을 줄인다. 파생 행을 원시 Jacobian과 연결하지 못하면 성공으로 처리하지 않는다.

## 사전 한도와 결정

원본 process1/자동 재시도0, CPU1/GPU0/neural0/fit0, body1/shadow1, 초기 안정화2000+scientific10000, shadow forward10000, 실제 접촉24420·Jacobian48840. 내부90초/외부100초/감독110초, raw 최대768MiB/free4GiB. 직전 실제 보수누적2575.245951900026초/배정2621초를 보존하고 ceil2576+110=**2686초**를 사전 누적 상한으로 정한다. E156 감독39.25초에 전체 Jacobian 수집·직렬화와 시간 여유를 추가한 한도이며, 부족하면 실패 기록을 보존한다.

먼저 생성자료에서 회전 접촉 frame·힘 부호·moving support·DOF 분할과 power 합을 독립 scalar로 확인한다. native 매핑·exact 원래 재생·분해 수식·부분 실패 저장·최종 시간 한도가 모두 통과하면 진단 자료가 유효하다. 이는 보행 성과 판정이 아니다. 결과에서 특정 제동 기여가 반복되는 경우에만 그 해부학적 출력 또는 신체 전달에 대한 다음 제한된 개입을 설계한다. 근거가 없으면 임의로 한 다리를 선택하지 않는다. 후보 채택·F1–F8 완료·F6 노출·M2/UI 변경은 모두 false다.
