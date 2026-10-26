# E233: E232 저장 분석의 식별자 검증 복구

E232는 저장 자료 식별자 검사에서 중단됐다. 전체 parameters membership은391개, 실제 effector snapshot membership은 free_columns 순서의343개다. 기존 검증 코드와 부모 FIRST가 이 차이를 놓쳤다. 저장 snapshot에서 dtype과 `parameters.membership[free_columns]`의 bytes가 실제 native membership과 같음을 확인했다. 수치 분석 결과는 없고 가설은 not_tested다. 신경/물리/CUDA/fit0, 원본 감독3.575347100006초는 누적6079.795094499659초에 포함한다. E232 동결 소스·출력·실패를 유지한다.

E233은 같은 분석 질문의 **기술 복구 두 번째 시도**다. E232 문서의 과학적 비교, 입력, 수식, 기준, 고정 startup1..99, 6D 해석 한계와 분기 모두 유지한다. `tools/analyze_c_banc_m3_effector_distribution_shift_recovery.py`와 `tests/test_c_banc_effector_distribution_shift_recovery.py`만 새로 만든다. 같은 실제 gpt-6-sol/high가 첫 수정하고 부모 SECOND로 검토한다. expected native membership에 free_columns를 적용하고, 실제 snapshot 구조를 모사하는 생성 검사를 추가한다. 잘못된 그룹 값/순서/shape는 계속 거부한다. 결과를 얻기 위한 허용 오차나 평가 기준 변경은 없다. 부모가 재검토했을 때 남는 수정은 부모가 처리한다.

원본 E232 함수를 수정하거나 import global을 바꾸지 않는다. 수정한 복사본과 부모 독립 분석·감독 코드 해시를 실행 전에 고정한다. 신체나 신경 생성 없이 저장자료 job1회, 자동 retry0, E232 포함 기술 시도 누적2회. 원래 CPU1/내부25초(마감5초예약)/외부30/감독35초/RSS2GiB/출력128MiB/free1GiB를 유지한다. 잔여34.204905500341초를 보존하고 실행 전 배정+5→6119초로39.204905500341초 중35초 예약한다. 생성/검토/독립 저장 검산은 별도, 미계측 읽기를0으로 표기하지 않는다.

이 분석은 후보 채택이나 최종 보행 개선이 아니다. E150 최선, E225 startup/10초/주기 미달, E228/E231 hold기각, M2/UI/F6미노출 및 고정 F1–F8 미완료를 보존한다.
