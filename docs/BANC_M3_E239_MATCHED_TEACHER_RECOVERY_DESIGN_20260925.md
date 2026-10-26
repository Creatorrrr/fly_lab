# E239: 검증된 parity를 재사용한 교사10초 수집 복구

E238은 교사9.68초에서 사전 마감으로 종료됐다. 부모 독립 saved-prefix 검산은 parity 전체와 교사 완료 구간의 입력/물리 충실도를 확인했지만 10초 성공은 미완료다. 같은 수집 질문의 기술적 두 번째 시도이며 과거 실패·비용을 초기화하지 않는다. E238에서 사용한 물리/신경/명령/입력/건강/완전성 기준은 그대로다. 시간 한도 부족을 신경 가설의 반증으로 해석하지 않는다.

새 과학 비교나 fit 없이, E238의 검증된 parity 원본과 부모 검산 파일 및 그 hash를 먼저 확인하고 교사 arm만 실행한다. E224 start를 명시적으로 unwrap한 같은 원래 `MotorTimescaleTrial`에서 E85의 2000개 q와 all8 tendon=.2, adhesion=false를 신체 action 경계에 적용한다. 기존 E238 `run_arm`을 변경 없이 재사용하고 그 전체 소스와 의존성을 동결한다. 기존 helper의 module global이나 설정을 monkeypatch하지 않는다. parity를 다시 생성하지 않으며 부분1936행을 합쳐10초라고 하지 않는다.

실행 기준은 새 2000control/40000 neural ticks/100000 physics steps/160000 RK stages, 전체391 MN signed E/I/drive finite, counters/overflow/armed 정확, E85 전체 physical control/native/final body exact다. 새 observed-start/final full checkpoint 및 각 raw/trace/advance/construction ledger를 독립 폴더에 저장한다. 원래 신경과 감각 필터를 유지하며 decoder 출력과 실제 교사 출력은 구분한다. 실패 시 부분 증거를 보존하고 이번 프로세스 안에서 재시도하지 않는다. 부모는 전체 자료와 LTM decoder/교사 힘줄 차이를 독립 검산한다. 데이터 수집은 자율 보행 성공이 아니다.

사용자가 정한 Sol 구현→부모 FIRST→동일 Sol 첫 수정→부모 SECOND 이후의 잔여 작업이므로 이번 복구 진입점과 검사는 부모가 작성한다. 기존 Sol 두 파일은 수정하지 않는다. 소유 파일은 `tools/observe_c_banc_m3_matched_teacher_recovery.py`, `tests/test_c_banc_matched_teacher_recovery.py`다. 생성 검사는 재사용 parity gate 및 실패 기록을 확인하며 원본은 부모의 단일 감독 실행으로 제한한다.

원본job1/retry0/fit0/solver0, CPU1/GPU1, 신체1/복원신경6+관측교체1=7, setup280실행/140capture를 과학 시간과 별도 계측한다. E238의 실제 교사9.68초 과학 실행이 약102초였음을 근거로 내부160초(최종 저장15초 예약)/외부175초/감독190초를 정한다. RSS6GiB/추가GPU4GiB/출력2GiB/free3GiB는 유지한다. 기존 원본 누적6278.4757079996525/6325초를 보존하며 원본 전에 배정+150→6475초, 잔여196.524292000348초 중190초를 예약한다. 새 전역 사용자 상한을 만들지 않는다.

전체10초 수집 유효성이 확인된 뒤에만 새 학습 문제를 설계한다. 기존 E197 startup 목표를 자동 교체하거나 E85의 고정 힘줄을 신경 출력으로 가장하지 않는다. 원래 F1–F8, F6 미노출, 최선 E150, M2·UI는 보존한다.
