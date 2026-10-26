# E225: 고정 effector 후보를 동일 10초 창까지 연장

## 질문과 범위

E224의 effector는 실제 3초 건강 기준을 통과했지만 prefix에서 전진 bout은 .225초, eligible cycles0이다. 나머지7초에서 안정성·전진·반주기 협응·반복성이 고정 F3–F5를 충족하는지 확인한다. 단기 건강 성공을 자연 보행으로 승격하지 않는다. 새로운 계수/모델 선택·학습·초기상태 탐색은 없다.

E224 effector의 `final/`(control600,neural12000,physics3초)를 `NeuralCurrentTrial.from_checkpoint`로 복원한다. 그 즉시 전체 snapshot을 원본 final과 `wall_s`만 제외해 byte-exact 검산한다. 현재 source/파라미터/graph/hash는 동결한다. control601부터2000까지1400 controls=추가7초만 진행한다. 원본 E224 시작을 다른 초기상태나 후보로 바꾸지 않는다.

## 같은 10초 창의 증거 결합

- E224 effector의601 control rows 및30000 native samples가0–3초 prefix다. 복원 시점 row0(control600)의 body/action/rate/effector/history/activation를 저장 prefix 끝과 정확히 비교하고 seam 검증을 기록한다.
- 새 continuation raw에는 복원 시점 row600부터2000까지1401 rows와3.0001–10초 native70000 samples를 저장한다. prefix 마지막 row600과 중복된 continuation 첫 행은 결합 시 한 번만 사용한다.
- 결합 trace/control은2001 rows(0..2000), native는100000 samples(.0001..10)이어야 한다. 실제 timestamp와 모든 값·shape·dtype를 검증하며 틈/중복/보간/재표본화/변경은 허용하지 않는다. 모델 파라미터 동일성이 없거나 checkpoint seam이 맞지 않으면 기술 실패로 중단한다.
- native100us 기존 건강 guard(upright.9/비발접촉/수치이상), control journal, 실제 부분 advance ledger, warmup/capture 계측, 원자료 보존을 재사용한다.
- 10초 완료 시 기존 `analyze_c_banc_biological_gait`, `e113.body_quality`, `supported_slip` 및 E197 고정99 startup 진단을 그대로 적용한다. 새3초 지표로10초 기준을 완화하지 않는다. F4/F5가 안 되면 자연 보행 기준 미달로 보존한다.
- whole_disk_future_exact는 별도 main/clone 미래 비교를 하지 않았으므로 false/not_executed로 남긴다. 즉시 disk restore exact와 이를 혼동하지 않는다. motor cut, F6 초기자세, F7 목표 시험, F1/UI 채택도 수행하지 않는다.

## 구현과 자원

실제 gpt-6-sol/high가 `tools/verify_c_banc_m3_effector_continuation.py`와 `tests/test_c_banc_effector_continuation.py`만 구현한다. 이전 E223/E224 source는 변경하지 않는다. 부모 FIRST→같은Sol 첫수정(필요시)→부모SECOND→부모 잔여수정 절차와 독립 raw 감사를 따른다. 부모는 감독/결과 감사를 병행한다.

E224 effector의 setup10.15초와3초 실행/저장 약23.8초를 근거로 추가7초+복원+저장/평가가 약80–90초 범위에 들어갈 것으로 예상한다. 이는 예상이며 실제 자원 제한이 우선한다. 원본 잔여97.84008260031624초 중 supervisor95초 예약: internal85초(최종저장10초 예약), external90초, supervisor95초. CPU1/GPU1, RSS6GiB, 추가GPU2GiB, 출력512MiB, free disk2GiB. original process1/retry0/solver0, 추가 과학 neural28000ticks 및 physics70000steps 이하; setup warmup/capture 별도. 기존0–3초 자료 재사용을 새 계산량으로 중복 청구하지 않는다. 전체 원본 누적5906.159917399684/6004초, 배정증가0.

건강 중단·자원 제한·기술 오류가 발생하면 부분 raw와 비용을 보존하며 자동재시도하지 않는다. 시간 부족으로10초에 못 도달하면 full F3–F5는 미검증으로 남긴다. 사후 기준 조정·실패 삭제·기존 M2/E150/UI 변경 및 goal 완료 선언은 금지한다. 향후 실제 저장 미래·차단·미사용 상태·목표 회귀는 다음 별도 판단 대상이다.
