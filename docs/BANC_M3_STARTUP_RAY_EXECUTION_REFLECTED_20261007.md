# Startup ray 실제 Astra 반영 설계

ROOT가 실제 GPT-6 Astra/max의 `parent-startup-ray-execution-astra-review.json`(15084B/SHA750ea557d566e7e46a8cb47151fb7a6a054d7002e5b8097c070bddc24d0a1034)과 MD(5590B/SHA5e9a10403648d842856b4e84a38786563d92d66c069899e5a140de56d763ee54)를 읽고 S1–S3를 반영한다. 기존 startup 연구의 R 수학·네 창·독립 backward와 F1–F8은 유지한다. 이것은 새 과학 구현 FIRST의 입력이며 실행 결과가 아니다.

봉인할 alpha 목록은 `[0.0009,0.00075,0.0005,0.00025,0.0001]`이다. 최대5개, 내림차순, 처음 적격 하나를 선택하고 중단한다. 방향은 합산 R gradient 하나를 최대 절댓값으로 정규화한 음의 방향이다. gradient0이면 후보 없음으로 종료한다. 작은 alpha가 안전하다는 단조성 가정을 사용하지 않는다. 실행 후 목록 확대·재클리핑·direction 변경·body 결과로 후보 재선정을 하지 않는다.

적격은 실제 배포 rounded representation에서 startup99의 네 고정 조건과 기존 R 판정을 함께 만족하는 경우다. R 판정은 기존 `reference_parameter_direction.analyze`와 같은 `-sum(O)>1e-9`, `-sum(T)>1e-9`, O/Q·O/T reconciliation 두 개 passed다. 기존 의미 있는 개선/조건부 지지와 기술 유효성을 분리한다. 수치 quadratic·R·gradient 대응은 기존 rtol1e-10/atol1e-12, neutral 재현 q/tendon atol1e-11·bank/original_activation atol1e-10·rtol0 및 기존 rate/adaptation exact 계약을 유지한다. 새로운 ray schema에서 alpha를 명시하며 기존 alpha0.001 전용 validator나 raw branch를 느슨하게 만들지 않는다.

각 제안의 requested theta, exact float32→float64 theta, 실제 selected-weight bytes, palette/CSR identity와 수치 receipt를 저장한다. requested alpha≤0.001과 실제 max(abs(theta32→64-1))≤0.001을 별도 검사한다. 중복 theta/weights, no-op, nonfinite/nonpositive/sign/unselected edge/palette16384 위반을 제거한다. 실제 forward와 배포 weights의 대응을 검증한다. 파일을 저장했다는 사실만으로 대응을 선언하지 않는다.

neutral의 control0/drive1–99 원본 감사를 먼저 수행하고 full99 재현을 적격화한다. startup용99 no-gradient trajectory는 기존 max_controls4/R backward와 분리한다. 기존 forward16/32 receipt를 여러 alpha에 재사용하지 않는다. baseline99, 실제 평가한 alpha별 startup99와 네 R 창, 기존 neutral R/독립 backward16 및 partial work의 요청/반환을 각각 기록한다. per-alpha 유한 평가 기록과 최종 물리 후보1개의 audit/handoff는 다른 artifact다.

실제 신체는 작은 공통 lifecycle0→99→300을 사용한다. neutral은 `neutral_values`/`initial_neutral`의 동일성을, candidate는 `initial_body`의 distinct identity/CSR를 증명한다. `OriginalFactory.construct`를 재사용하고 E343 전용 `initial`은 호출하지 않는다. 기존 `run_prefix` exactly99와 baseline의 cleanup을 변경하거나 이어붙이지 않는다. 각 arm은 동일 감사된 초기 자산에서 별도 한 인스턴스를 생성하고, arm 내부에서는 reset/reload/재생성 없이 같은 observer/counters로 연속 진행한다.

99에서100 rows/4950 native samples와 controls1–99의 startup를 확인한다. 실제 startup/health가 적격이면300에서301 rows/15000 samples까지 이어간다. 제어마다 neural20/physics50, model time0.005*control, constructor2000 physics/320 setup/160 capture를 별도 확인한다. unknown advance·부분 raw·오류를 보존한다. 두 arm과 raw/body 감사는 순차 실행하고 각 소유권을 종료한다. E358/E359 invalid 결과는 대조군이나 허용 margin으로 쓰지 않는다.

ROOT가 actual grant를 발급하는 범위는 Original +6000, global16200/stage6300/new-local6000이며 새 local offset은 발급 당시 immutable debit length다. 과거 누적10020.115960899504/stage126.361은 초기화하지 않는다. Source/입력/후보/자원/offset을 manifest→request→pre-spawn→pre-array→worker에 동일하게 묶는다. 기존 Original20 admission은 승계하지 않는다. worker1/auto0와 full-return once accounting을 유지한다. 첫 outer ceiling120/1200/120/300/300/120/120/180=2460과 필요한 잔여 audit/close 예약을 유지하고 내부·child·outer 마진을 계약에서 고정한다. 실제 fit/timing은 미확인이다.

구현은 기존 실행 인프라를 복제하지 않는 새 과학 모듈과 작은 과학 테스트로 한정한다. Sol은 canonical이나 원장을 변경하지 않는 SOURCE_ONLY 제안 FIRST를 출판한다. ROOT FIRST→필요 시 같은 Sol sole R1→ROOT SECOND→ROOT 잔여/적용 순서를 따른다. ROOT가 현재 생성 검사·actual grant/manifest·실제 실행을 담당한다. 적격이면 같은 후보로10초/F3–F5→F6/F7/F8을 이어간다. 기존 short screen·해부학 provenance·M2/UI·외부 CPG/교사/위상/궤적/root 보정 금지와 실패 기록은 유지한다.
