# E271 — source 계수 후보의 실제 신체 연결과 checkpoint

2026-09-27. E269/E270의 계수 학습과 독립적으로, 그 결과 한 후보를 실제 E150 세션에 연결할 얇은 wrapper를 준비한다. 후보 선택이나 신체 실행은 아직 없고, E270 조건부 gate 뒤 별도 freeze/예약을 거친다. 이 작업은 기존 코드의 중복을 줄이고 신경 계산 외의 개입 없이 원래 감각·근육·신체·저장 경로를 사용할 목적이다.

실제 gpt-6-sol/high는 `tools/banc_m3_source_filter_trial.py`, `tests/test_c_banc_source_filter_trial.py`만 소유한다. 기존 E267/E268/E150·계획·검증자료·UI는 수정하지 않는다. root FIRST→필요시 같은Sol 첫수정→root SECOND→root 잔여수정 순서를 적용한다. 작은 생성 CPU 신경/가짜 세션의 helper 검사와 Ruff만 허용한다. 원본 BANC/신체/물리/CUDA/optimizer/다른에이전트는 실행하지 않는다. 모든 생성 constructor/advance/tick/초/시간을 기록한다.

## 고정 동작

실제 `MonotoneRecruitmentTrial`을 기반으로 하고, **network만** E267 `SourceFilteredSynapseNetwork`로 바꾼다. 기존 source_matrix(old)와 tau/a/threshold/cap/core gain/MN gains/MN log_tau/dt/device/packed/capture_steps를 보존한다. edge_ids/coefficient triples/free_motor_ids는 명시적 입력이며 E267 native validator를 사용한다. 새 간선·부호 변경·외부phase/teacher/CPG·다른신경모수·adapter/body/sensory/world 변경은 없다. 실험용 wrapper이며 현재UI나M2의기본경로를 바꾸지 않는다.

새 후보 생성은 원래 neutral checkpoint에서만 허용한다. root는 E150 candidate-neutral의 rate/adaptation/drive가 모두0, output_mask가 모두1, tick0임을 확인했다. original identity 및 inherited metadata가 candidate의 source_timescale_identity와 일치해야 한다. 같은 초기 배열·tick을 보존하고 source history는0으로 시작한다. E267의 neutral-only import를 비중립 후보에서 우회해 임의중간상태를 주입하는 일반 API를 만들지 않는다. candidate의 template metadata/identity를 유지한 상태로 검증된neutral배열을 이식하고, 실제 candidate.restore가 전체metadata/state를 재검사하도록 한다. neutral계수도 같은candidate초기화규칙을 따른다.

생성 뒤 whole base snapshot과 비교할 때 허용되는 차이는 session schema, network model_hash/source_filter/source_history, 명시적 source_filter_trial_patch뿐이다. body/adapter/감각/기존patch/명령/tendon/navbar/seed/시계는 그대로다. 두 복부 hold=.2와 원래 여섯 LTM/adhesion 경로를 유지한다. 실제 model schema/원래 CSR/선택 간선/MN/adapter/provenance가 일치해야 하며 coefficient 값을 숨긴 채 기존identity를 재사용하지 않는다.

## 저장과 복원

고유 trial schema와 patch schema를 정의한다. patch에는 원래 MonotoneTrial schema/신경 identity/adapter identity, 새 신경 identity, E267 정확한 selection과 기존monotone patch를 묶는 provenance를 담는다. snapshot에 실제 source history와coefficients metadata가 들어가고, E267의 atomic neural restore 검사를 그대로 사용한다. caller의배열/patch를공유하지않는다.

`from_checkpoint(graph,saved)`는 이새schema만받는다. 검증된checkpoint에서 source filter layer만제거한 E150 형식 사본을 만들어 `MonotoneRecruitmentTrial.from_checkpoint`로기존전체세션을복원한다. 이것은body/adapter/기존neural구성복원을위한생성중간단계일뿐,후보실행이아니다. 이후동일coefficients/source identity로E267네트워크를재생성하고 **원래저장된전체source-filter state(비영history포함)**를restore한다. 일반새후보neutral초기화와checkpoint복원을구분한다. 복원된전체snapshot이saved와exact여야한다. 잘못된schema/selection/c/hash/inheritedmetadata/history/motorroster/monotonepatch는검증오류로남긴다.

중립계수전용원본이식API를확장하거나 E267파일을수정하지않는다. 필요한검증·초기화·layer제거helper는새wrapper파일에두고,불변metadata와sourceidentity를명시적으로검사한다. 생성실패시실제소유한body만한번정리하고기존에사용중인세션을실수로폐기하지않도록 소유권 이전시점을분명히한다. caller는성공후wrapper가body를소유하며base는더이상사용하지않는다는계약을갖는다. restore실패도부분세션정리/원래오류를보존한다.

## 검증·후속

생성 CPU 검사: 기존 CSR/고유identity/같은neutral state, 새로운coefficients의차이, 허용외snapshot항목의불변성,외부배열변경차단,잘못된neutral입력·schema·provenance거부,layer해제시원본patch·adapter보존,복원된비영이력의다음계산동일성,실패정리/소유권. actualBody로테스트하지않는다. 테스트간CPU 신경모델수를측정하고필요한작은fixture만쓴다.

wrapper통과는F1–F8이나보행채택이아니다. E270의한후보가확정되면root가E257의startup/short-forward/slip/gait 및원래F3–F5 평가기를재사용해동일초기신체의.5→1.5→10초판정을설계한다. whole-body/fresh-constructor future·motor-cut·F6/F7/F8은조건에맞춰별도명시한다. 현재wrapper개발의원본예약0이며학습/원본/신체가백그라운드에서실행된다고주장하지않는다.
