# E235: 실제 BANC startup의 RK 입력·기억 수집

H235는 E233의 첫25ms 이후 effector 차이가 **동일 실행에서 달라진 실제 E/I/drive 입력과 연결되는지 관측**하는 것이다. 경쟁 설명은 계산/수집기/초기상태 불일치다. E234 생성8뉴런에서 수집기 추가 전후 whole state, graph/direct, 디스크 복원 이후 상태·raw, overflow/stop 경로 모두exact였으며 부모298검산이 통과했다(최대CPU오차1.86e−9). 이것은 전체BANC 검증이나 자연보행 성공은 아니다.

## 고정 실행과 비교

E224 `effector_canonical/start`의 동일 전체158706뉴런/391MN/96근육/343free/48fixed·신체 상태를 복원해 E223 별도 신경 전달 경로를 사용한다. beta/prototypes/근육전달/감각encoding을 변경하지 않는다. 신경은packedCUDA .00025초, physical .0001초, control .005초다. control0..99, 새99controls/.495초/1980neural ticks/4950physics steps/7920RK rows만 실행한다. 시작99행을 넘겨 전진·보행 평가 또는 미사용pose를 시험하지 않는다.

`NeuralCurrentTrial.from_checkpoint`로 한 신체를 복원한 후 E234 `EffectorStageObserverNetwork`로 신경 객체만 교체한다. 원래 source_matrix/tau/a/threshold/cap/core/motor scale·timescale와free/fixed/group/center/sigma/beta를 정확히 전달하고 기존 E223 snapshot을 restore한다. 새 network identity와 whole trial snapshot이 저장 start와exact여야 한다. `session.network`와 기존 `session.adapter.network`를 함께 교체하며 adapter identity/설정/상태는 보존한다. 원래 객체는 별도GPU작업 없이 해제한다. 기존 동결 클래스를 수정하거나 전역 monkeypatch하지 않는다.

관측자는 원래 graph capture 완료 후 한 번arm하고, E224와 같은 `protocol.advance_control`로99control을 실행한다. E224의100개control raw 전 필드·trace100행 및 native4950행과 byte exact 대조한다(원래 trace wall_s만 기존 exact helper의 계약대로 제외). control1의 전체 checkpoint는 기존 checkpoint-5ms와exact다. 비교 전에 raw를 보존하고, 각control 오류/건강 중단 때도 완성행·부분native·GPU관측buffer/counters를 남긴다. 복구/추가실행/재시도0.

E234 관측버퍼6개(positive_recurrent,negative_magnitude,drive float32; effector64,fast64,slow64 float64)를 `.npy` 원시 배열로 저장해 float64 상태까지 보존한다. counters343개는7920,overflow0,armed0,start_tick0,end_tick1980. shape은 scalar채널7920×343, 기억7920×343×3,전부finite. 관측은다른계산의input이되지않는다. 실제fixed48은control motor_rate32/decoder_input32 그대로기록한다.

E208 저장 입력의 같은0..7919stage와free_columns 순서를 비교한다. 각 E/I/drive 채널은 처음80행(5ms) byte exact인지, 처음 byte차이 stage/해당neural_tick/control, 원래 float32값의 max차/RMS를 보고한다. 같다고 가정하지 않는다. 추가로 각채널 변환 `log1p(E)/scale0`, `log1p(I)/scale1`, `asinh(drive)/scale2`의 차이를 보고한다. 변화 시점은 학습상태와 실제입력의 비교이며 신체감각의 어느 경로가 원인인지 아직 분리하지 않는다.

판정: 신체/control/5ms full state exact와관측완전성이 먼저 통과해야 valid_capture다. 처음5ms 입력까지다르면 초기/수집부를 먼저검토한다. 처음5ms같고그후입력차이가있으면 저장자료로 새 실제입력에 대한 effector 재현과 감각 경로 차이를 검토하는 후속을 선택한다. raw입력이 모두같은데기억이달라지면 수식/연산순서가우선이다. 단순precision수정·추가hold·기준완화·candidate채택은하지않는다. 기존 E233 startup표 오류와정정·실패·비용을 보존한다.

## 구현·자원

실제 gpt-6-sol/high는 `tools/observe_c_banc_m3_effector_startup_stages.py`, `tests/test_c_banc_effector_startup_stages.py`만 구현한다. 부모FIRST→같은Sol첫수정필요시→SECOND→부모잔여수정. E234클래스와동결기존파일변경0, 생성CPU검사 후부모만 원본1회 실행. 부모가 실제원시control/native와입력 차이를 별도재계산한다. 이번도최종F1–F8평가가아니다.

CPU1/GPU1/원본job1/99control/retry0/fit0/solver0. 신체생성1, 신경생성은trial복원내부2+교체1이며 setup interval 실행·capture를과학적1980ticks와분리계측한다. 내부45초(마감10초예약)/외부50/감독60초, RSS6GiB/추가GPU3GiB/출력512MiB/free2GiB. input hash와패키지버전을실행전에고정한다. 전체 원본누적6090.0136165996555/6119초·잔여28.986383400344494초를보존하고 실행전단계배정+35→6154초,잔여63.986383400344494초중60초예약. E234 생성검증3.8601507초는별도,전역사용자시간상한은새로만들지않는다. M2/UI/F6미노출/E150최선/F1–F8미완료보존.
