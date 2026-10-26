# E307 동일 SSP의 자원 감쇠 없는 0.5초 대조군

Root 설계, 2026-09-28. E306 실제 신체 실행은 신경 오류 없이 유효했으나 시작 관절 명령 최대 오차 0.06620980268835475 rad로 고정 기준 0.04를 넘었다. 최댓값은 control 99 / joint index 8이며 최초 초과는 control 50이다. 현재 증거는 원래 RK4 → 새 SSP와 자원 계산식 효과를 분리하지 못한다. 이번 한 번의 대조 실행은 같은 새 SSP에서 u만 0으로 바꾸어 0.5초의 신체 피드백을 포함한 차이를 측정한다. 성공 후보 탐색이나 승인 기준 변경이 아니다.

가설: E306의 시작 동작 실패는 활성 자원 변화에 의존한다. 경쟁 설명: 자원 감쇠가 없어도 SSP로 바뀐 신경 궤적과 신체 감각 피드백만으로 시작 기준을 넘는다. 대조군이 원래 시작 기준을 통과하면 u 활성 대비의 후속 분석을 한다. 대조군도 실패하면 자원 파라미터를 먼저 맞추지 않고 적분기 / 폐루프의 차이를 조사한다. 기술 오류, 건강성 조기 종료, 부분 궤적은 충분한 0.5초 비교가 아니므로 가설 판정을 보류한다. 모든 수치 차이와 실패는 보존하며 어떤 결과도 E306을 소급 승인하지 않는다.

실제 gpt-6-sol / xhigh는 새 `tools/verify_c_banc_m3_ssp_neutral_body.py`와 `tests/test_c_banc_ssp_neutral_body.py`만 구현한다. Root FIRST → 필요할 때 같은 Sol R1 → root SECOND → root 잔여 수정 순서다. 기존 E306/E305/E304/E303/E291 및 원자료는 불변이다. 생성 fake 검사 전체 60초 / host 1.5 GiB / CPU 1, 실제 신경·신체·CUDA·원본·solver·fit 실행 0회다. Root 실제 실행은 검토와 passive gate 후 별도로 예약한다.

E306 NativeDependencies의 생성·초기 비교·관측·신경 첫control 비교·자원 오류·저장·cleanup을 재사용할 수 있다. E291.run_arm은 neutral을 1control로 고정하므로 호출해서 candidate로 위장하거나 globals를 바꾸지 않는다. 새 자체 neutral run_arm은 같은 예외/저널/close/부분 상태/clock 비용 처리를 이어받아 정확히 최대 100 control / 0.5초만 실행한다. 새 자체 execute와 source manifest는 실제 호출한 코드와 새 entrypoint를 정직하게 표기한다. 상위 E306.execute는 호출하지 않는다. 같은 E304 SSPRecoveryResourceTrial, E296 보호 roster, 모든 u=0, 기존 tau, 원래 전체 초기 상태, CSR·신경·adapter·세계·감각·물리·startup 평가를 그대로 유지한다. 새 모델 identity는 E305/E306 neutral identity와 정확히 같아야 한다.

전체 초기 state는 원래 E150와 resource layer 제거 후 byte exact이고, E305 neutral initial-neural과 새 neural metadata까지 일치해야 한다. Control 1의 전체 신경 endpoint는 E305 및 E306 neutral과 exact여야 하며, mismatch에서 즉시 중단한다. E306 capture가 이미 수집하는 neutral_end_snapshot과 첫 신경 snapshot을 그대로 재사용한다. 첫control 수치 delta는 첫구간 비교로 명시하고, 이를 0.5초 최종 상태로 오인하지 않는다. 100controls 관측에서 R=1/g=1, source 순서·타입·건강성·보호 edge 조건을 검증하고 전체 final snapshot, native contact, recruitment, activity, factors, trace, 실제 과학 clock과 생성/setup/capture/settling 비용을 남긴다. 추가 매control fault polling 또는 solver 변경은 금지다.

원래 99개 시작 명령 참조와 q RMS≤0.018 / 최대≤0.04, LTM RMS≤0.009 / 최대≤0.025를 그대로 계산한다. 건강성 upright≥0.9 / 비발 접촉 0 기준도 유지한다. 어떤 결과에서도 100controls 다음 진행은 없다. 원래 E150 trace 및 E306 candidate trace와 각각 같은 control/time/joint 배열에서 q 명령, LTM, 측정 관절 각도 차이의 RMS/최대/최대 시점·index를 보고한다. 저장된 E306 후보를 새로 실행하지 않는다. 두 비교 상대의 source/config 차이를 표시하고, u 대비에는 폐루프 피드백의 결과가 포함된다고 설명한다. 원래 startup 수치가 합격 판정의 유일한 명령 오차 기준이며 비교 delta 자체에 새 합격 문턱을 만들지 않는다. stage arrays, 전체 neural history, F3–F8 검증으로 과장하지 않는다.

Gate는 E306 freeze `ed943af832cc4d97a845e53aabecb9b13792adfcec1a8fd9ac391cdc767c08a6`, review `f824fae64ff3faa10e142efac1f6054710c9596d1a2245fce7b4b57719baab6e`, result `6be280252bfa8a0a3978b0b63405f68474c5899a7615a6ad287e9c38a882e406`, output manifest `836c7a7e3fa0d3da01e9e02cf726fdf3965d4a68d52a81c61adc08957d64ccb4`, root raw audit, source manifest, advance ledger, result SHA 및 전체 output manifest의 byte를 고정한다. E306 run_valid=true / 후보 미달 / 모든 F 판정 false, 실제 신체 2 / neural 2020 / native 5050 / setup 560 / capture 280 / settling 4000 / unknown 0을 검사한다. 이전 E301/E296 및 SSP E303–E305의 qualification·identity·packages·input·source·preservation 계보를 원래 gate와 같은 조건으로 검사한다. 이미 존재하는 E306 출력을 새 출력인 것처럼 속이지 않는다. 새 gate는 순수 lineage/load helper를 호출할 수 있지만 실제 호출 소스의 pin을 누락하지 않는다.

Freeze/review/result는 새 schema `flylab.m3.e307.ssp-neutral-body.{freeze,review,result}.v1`를 쓰며, 출력은 아직 없는 `verification/banc-natural-gait-m3-20260919/e307-ssp-neutral-body`로 고정한다.

원본 자원은 최대 1회 / 재시도 0, supervisor 65초 / internal 55초 / science 40초 / finalization 15초, host 6 GiB / 추가 GPU 4 GiB / output 2 GiB / free disk 4 GiB / CPU 1, 신체 1회, science neural 2,000 / physics 5,000 이하로 고정한다. Setup 280 / capture 140 / settling 2,000 예상은 실제 계측으로 확인하며 과학 비용과 분리한다. 생성 실패의 알 수 없는 비용은 0으로 처리하지 않는다. 현재 누적 8221.508572399503 / 단계 배정 8290초, 잔여 68.49142760049654초이며 65초는 잔여 안이다. 이 설계 시점에는 원본 예약 0이다.

생성 검사에서는 neutral만 100controls / 한session으로 도는 경로, 첫control mismatch 즉시 종료, startup 실패/성공 모두 연장 없음, 건강성/신경 오류의 부분 계측, 원래 예외 보존 및 모든 publication/cleanup 실패, exact identity/u0/resource 조건, 저장된 비교의 시계·shape·dtype·비유한 실패, package·source chain, 최종 output manifest와 별도 result SHA를 검증한다. Output publication 실패는 신뢰 가능한 성공 receipt를 남기지 않고 원래 오류와 부수 실패를 모두 보존한다.

실제 결과가 나오면 root는 원자료를 독립 검산한 뒤 다음 설계를 결정한다. 기존 F1–F8 / M2 / 활성 UI / 모든 실패·비용 / F6 미노출을 유지한다. F2 전체, F3–F8, 후보 채택, 자연 보행 완료, 생물학적 기전 주장은 모두 false다.
