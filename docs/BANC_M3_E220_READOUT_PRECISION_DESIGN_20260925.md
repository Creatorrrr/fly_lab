# E220: E219 입력 스키마 복구와 동일 정밀도 비교

E219 원본 프로세스는 저장 ID 경로 오류로 계산 전에 실패했다. 그 소스와 증거를 그대로 보존하고, 사용자 절차의 부모 SECOND 이후 남은 수정으로 이 세션이 별도 복구 도구를 작성한다. 새 행동 가설이나 profile 선택은 없다.

E219 설계의 여섯 profile, 입력, 정밀도, held row0, 비교식, 원래 proxy, 복원 허용오차를 그대로 적용한다. original_monotone/original_canonical/new_monotone/new_canonical/rate32_readout64/readout32를 한 번씩 계산하며 native 통합 대상은 canonical이다. `adapter.calibration.motor_ids`를 E201 parameter/endpoints ID와 비교한다. 원본 실험 전에 저장 checkpoint의 키·shape·dtype만 확인하고, 생성 테스트도 실제 calibration 구조를 사용한다. 원본 readout 계산을 사전검사로 우회하지 않는다.

원래 E219의 FIRST에서 수정된 정확한 UTF-8 바이트와 result-last 게시를 유지한다. 부모의 독립 반복식으로 여섯 raw와 metrics/hash를 검산한다. E219의 실패 비용·원본 프로세스 수는 소거하지 않는다. Sol에게 새 구현이나 재수정을 맡기지 않으며, 이 복구는 사용자가 지정한 부모 잔여 수정이다.

소유 파일은 tools/analyze_c_banc_m3_readout_precision_recovery.py와 tests/test_c_banc_readout_precision_recovery.py다. 이미 검토한 E219 소스를 별도 파일로 복사하고 ID 경로 및 E220 provenance만 바꾼다. 기존 성공·실패 소스를 변경하지 않는다.

원본 프로세스 1/6고정 readout/CPU1/GPU0/fit0/neural0/physics0/retry0. 내부20/외부30/감독40초, RSS4GiB, 출력64MiB, free disk1GiB. 해시·입력 로드는 내부시간에 포함한다. 누적5791.716809299691/6004초, 잔여212.2831907003092초에서40초 예약하며 총배정 증액0이다. 준비·생성·부모 검산·미측정 agent/설계 비용은 별도 기록한다.

E220은 저장 rate의 끝점·readout 민감도이며 실제 RK·신경 feedback·보행 검증이 아니다. 원래 strict proxy를 완화하지 않는다. 최선E150, M2, UI를 보존하고 F6미노출, F1–F8 active·미완료를 유지한다. 후속 native 실행은 독립 계약과 잔여 자원으로만 진행한다.
