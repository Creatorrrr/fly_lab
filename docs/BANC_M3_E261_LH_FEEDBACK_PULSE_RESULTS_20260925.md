# E261 짧은 폐루프 자극 시험: 복원 오류로 판정 불가

실험은 무효이며 H261은 미시험이다. 자극 없는 첫 분기에서 `abdomen_pitch`와 `abdomen_yaw` 입력(각0.2)이 빠졌다는 exact 검사가 중단시켰다. plus/minus/sham1은 실행하지 않았다. 양의 되먹임이나 보상 반응에 관한 결론은 없다. 보행 후보도 채택하지 않았다.

원인은 진단 도구의 `restore_in_place()`가 session.tendon_inputs를 비우고 adapter가 관리하는 다리 힘줄6개만 다시 넣은 것이다. 원래 세션 생성은 복부 힘줄2개를 포함한 전체8개를 설정하고, 일반 checkpoint 복원은 update로 기본 항목을 보존한다. session snapshot에는 이 held dictionary가 직접 포함되지 않아 whole exact만으로 누락을 찾지 못했다. trace의 실제 입력 비교가 누락을 검출했다.

부모25,626검사는 prefix0..55가 E257 candidate의 전체 원래 trace·MN391·기존82/core24·bank·감각·native2750 표본과 exact임을 확인했다. sham0의55..75에서도 이들 raw 배열과 native1000 표본은 exact였으며 trace 차이는21행 모두 tendon_inputs 두 key뿐이었다. 따라서 보존된 fork는 재사용 가능한 원래 control55 상태다. 그렇더라도 E261의 전체 입력 유효성 조건을 사후 완화하지 않는다. 직접 기록된 fork의 d_flexion/dq19=-0.999892433180187는 기하 부호 관측일 뿐 폐루프 반환 판정이 아니다.

실제gpt-6-sol/high 구현·FIRST3결함 첫수정·부모SECOND 뒤 필수 관측 누락과 actuator 제한 검사를 부모가 보완했다. pytest11/Ruff/독립 생성730/CLI5를 통과했지만 실제 복원 입력 누락은 생성 fixture가 포착하지 못했다. 다음 복구는 adapter 이외의 held tendon 항목을 포함하는 fixture와 모든 입력 sidecar를 검증한다.

원본1/retry0, whole1/network7/setup280실행+capture140기록/body1/settle2000. 실제 과학75controls=1500neural ticks/3750physics steps, in-place restore1이다. child25.284676099996초/감독29.222378699997초, 누적7050.637872599607/7074초·잔여23.362127400393초. 추가GPU719323136B/호스트RSS합2803683328B/출력91094976B. 소유 실행 프로세스0, 강제 종료0, 재실행0이다.

첫 부모 부분자료 감사는 scalar metadata를 시계열처럼 자른 IndexError로 실패했다(process1.9478909초, 내부시간 미측정null). 소스·실패를 보존하고 v1 감사에서 metadata 분류만 고쳤다. v1감사1.602919100013초/process1.9515301초는 원본과 별도다. 실제 모델이나 원자료를 다시 만들지 않았다.

다음은 별도 E262 복구 설계로 모든 held 입력을 명시적으로 복원하고, 이미 검증된 control55 fork에서 네 분기만 실행한다. pulse 크기/부호/창/1e−8 판정은 그대로 유지한다. E261 무효·E257 기각·E150 best partial·M2/UI·F6 미노출·F1–F8 미완료와 모든 비용을 보존한다.

증거: `verification/banc-natural-gait-m3-20260919/e261-lh-feedback-pulse/`, `parent-e261-failure-review-v1.json`, `parent-e261-supervisor.json`, `parent-e261-failure-audit-recovery.json`.
