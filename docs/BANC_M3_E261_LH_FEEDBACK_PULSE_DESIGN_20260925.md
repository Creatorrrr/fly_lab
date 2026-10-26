# E261 최초 LH 위반 직전의 관절 자극과 폐루프 반환

E260의 고정 감각 대조는111970→47329 억제 연결이 복원망의 muscle44 감소와 관련됨을 지지했다. 연결 제거는 평균 MN rate도3.1배 바꾸므로 후보로 채택하지 않았다. SNpp50 특징은 물리적 굴곡에 비례하고, 같은 신체 hash의 E193/195/196 knee_sign은−1이다. 여섯 감각→111970 연결은 양수,111970→47329는 음수, muscle44→q19 moment는 양수다. 이 조건부 부호 사슬은 양의 반환을 예측하지만 E257 위반 자세에서의 Jacobian과 실제 전체 폐루프 반응은 아직 관측하지 않았다.

이번에는 E248/250/256의 tick40000 고정 high/low를 반복하지 않는다. E257 실패 후보 W6의 원래 control0 whole 상태를 복원해 최초 위반 직전 control55/.275초까지 실제 감각·신경·신체로 진행하고, 정확히 같은 whole 상태에서 짧은 물리적 목표각 자극을 비교한다. 과거 M2 E51의 상쇄 방향 반응은 다른 모델/자세의 결과로 보존한다.

## 고정 비교

모델은 E257 candidate identity `4d5bae3386bed2dce2ca8a2add5130449c60a70e3b0d95a51947feeb03c76354`다. 원래6간선 복원 외에 가중치·마스크·gain·tau·해부학·decoder·감각 mapping·body·navigation을 바꾸지 않는다. E257 candidate/observed-start를 `LHPartitionTrial.from_checkpoint`의 검증된 경로로 한 번 구성한다. prefix는55controls, fork는 control55, 각 branch는20controls로 control75/.375초까지다.

branch 순서는 `sham0`, `plus`, `minus`, `sham1`이다. 매번 동일 fork whole을 제자리 복원하고 전체 snapshot exact를 요구한다. 복원은 기존 checkpoint 필드와 network/body/adapter/navigation의 restore를 명시적으로 사용한다. component identity/patch는 바꾸지 않고 cuts/control_tick/initial_position/signed_forward/wall/rate/tendon state를 복원하며 `_clocks()`와 whole exact로 검증한다. 숨은 상태 영향은 두 sham의 모든 관측/최종 과학 상태가 exact인지로 추가 검사한다. 원래 wall 측정만 과학 비교에서 제외한다.

첫 분기 control56의 decoder 반환 직후 `body.step_joint_targets`에 전달되는 q19 목표 하나에만 ±delta를 더한다. delta는 이미 관측된 E257 control56 candidate와 baseline의 **전체 q19 명령 차이 절댓값**으로 실행 전에 고정한다.0<delta≤.01rad이어야 하며 양·음이 같은 값이다. pulse로 body qpos/root/신경/근육 내부 상태를 직접 덮어쓰지 않는다. adhesion/tendon/다른41목표·모든 kwargs는 원래 반환을 보존한다. 두 sham은 목표를 복사·변경하지 않고 원래 함수를 그대로 호출한다. control57부터 모든 팔은 원래 신경 명령만 적용한다. 이 외부 단발 자극은 진단 전용이며 최종 BANC 제어기에 들어가지 않는다.

## 유효성·관측·가설

prefix0..55와 sham0/sham1의55..75 기록은 원래 E257 candidate 같은 control의 MN391/기존82/core24·bank·activation·command/tendon·qpos/qvel/실제ctrl·몸 좌표·접촉 시계열과 exact여야 한다. 원래 raw가 없는 새로운 관측은 exact 재현 주장에 포함하지 않는다. native physics prefix2750 및 sham각1000행은 E257 원래 native-contact 같은 slice와 exact를 요구한다. sham끼리 최종 whole은 wall_s만 제외하고 exact다. 두 pulse가 정확히 한 control의 한 목표에만 적용됐음과 첫 control의 신경/decoder 상태는 sham과 같음을 확인한다. 건강성 실패·clipping/경계초과·시계 누락은 원자료를 보존하고 판정 불가로 남긴다.

각 branch의 fork/endpoint21행에 control/neural/physics clock, 실제42q·qvel·적용ctrl·원래 및 전달된42목표, rate391, 기존82/core24 rate/drive/adaptation,2MN×3bank와muscle44 activation, 소스23/bridge4, 감각 이전·이후241filter와 실제port85 feature/주입, 물리적 LH 굴곡/속도/Jacobian을 저장한다. 매 encode 앞/뒤와 neural/body 후의 시간 소속을 명시하며 기존 encode/neural/decode/body 호출을 추가 실행하지 않는다. 관측은 읽기 전용이다. 입력 전체 drive 해시 또는 직접 기록으로 navigation5000과 원래 주입 순서를 검증한다. fork와 모든 arm의 전체 초기·최종 상태를 저장한다. 필요한 diagnostic-current probe는0으로 제한한다.

H261은 **현재 시작 위반 근처에서 실제 관절 자극에 대해 같은 방향의 신경 목표 반환이 나타나는가**다. paired physical delta q19는 plus−minus의 control56 값이다. 이것이1e−8rad보다 크고 pulse가 범위 내 실제 전달됐을 때 검출 유효성 조건을 통과한다. 이후 controls58..75의 **원래 decoder q19 목표** plus−minus 평균을 D라 한다. control56의 외부 pulse 목표는 D에 포함하지 않는다. D>1e−8rad면 `supported_positive_return`, D<−1e−8rad면 `refuted_compensatory_return`, 나머지는 `inconclusive_below_resolution`이다. 검출 유효성 미달 또는 기술 실패는 각각 inconclusive/invalid이며 positive support로 바꾸지 않는다.1e−8은 수치 해석의 작은 개발용 하한이고 생리 기준이 아니다.

fork에서 직접 측정한 `d_flexion_dq[2]`와 pulse 전후 굴곡 변화, controls58..75의 port85/source23/bridge111970/MN47329/bank/muscle44·전체 q19 및 각 signed/positive/negative 차이를 기술한다. 특히 다른 감각/근육 경로가 전체 반환을 상쇄하는 대립 설명을 보존한다. 부호 사슬은 직접 측정과 함께 조건부로 해석하고 해당 경로가 전체 반환을 유일하게 매개한다고 주장하지 않는다. 지지해도 특정 수용기 극성 변경의 보행 효과나 자연 보행을 증명하지 않는다. 이후 교정은 별도 설계하며 전역 극성/gain sweep은 자동 진행하지 않는다.

## 구현·자원·기록

실제 gpt-6-sol/high가 새 도구 `tools/probe_c_banc_m3_lh_feedback_pulse.py`와 `tests/test_c_banc_lh_feedback_pulse.py`만 구현한다. 부모 FIRST→필요시 같은 Sol 첫 수정→부모 SECOND→부모 잔여 수정 후 원본을 단 한 번 실행한다. E257 구성/계측/초기 상태 경로를 재사용하되 이전 gate/arm 수·전역 상수·출력 경로를 monkeypatch하지 않는다. 제자리 restore·pulse 격리·실패/부분 step·sham 비교와 판정은 생성자료로 먼저 검증한다. 부모는 원래 입력·clock·각 slice·차이·신호 부호와 모든 판정을 독립 검산한다.

원본1/retry0/CPU1/GPU1, whole construction1/기존 neural constructors7/setup실행280 중capture140/body1/중립settle2000steps. 과학은 prefix55+4branches×20=135controls, neural2700ticks/physics6750steps. 제자리 restore4회이며 추가 모델 구성·current probe·fit·solver0이다. neural constructor의 실제 requested/entered/completed 및 setup/capture, physical 실제시계증분과 partial을 계측한다. 마감/cleanup/쓰기 오류는 성공 해석을 철회하고 원자료·최초/추가 오류를 보존한다.

E257의2회 whole 구성/600controls가81.872초였으므로 이번1회/135controls에 internal45초(최종화8)/external48/supervisor50초를 예약한다. 원본누적7021.415493899611/7074초·잔여52.58450610038926초 안에서50초를 예약하고 배정 증액0이다. host6GiB/추가GPU4GiB/최소GPUfree4GiB, output384MiB/최소disk1GiB. 준비/생성/부모검산은 별도, 미측정 개발/자문은null이다.

출력 `verification/banc-natural-gait-m3-20260919/e261-lh-feedback-pulse/`는 write-once다. E260 양성 신경 진단·E257 보행 기각·과거 실패/비용·E150 best partial·M2/UI·F6 미노출·고정 F1–F8 전체 미완료를 유지한다.
