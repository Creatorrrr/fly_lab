# E199: 무발화 MN이 실제로 비활성 근육을 만드는지 판별

2026-09-24 부모 사전 설계. 직전 E198 턴은 progress다. 391 MN 중196개의 침묵을 확인했지만, 침묵 세포 수는 부족한 근육 수가 아니다. E179는 같은 E150 수치 궤적에서 RM 굴근의 다른 세 MN이 출력을 제공함을 이미 확인했다. E179의 전체 검산을 새 성과로 반복하지 않고, 저장되지 않았던 **96근육 전체의 지원 상태**를 E198과 연결한다.

## 비교 질문과 변경 없는 관측

H199: 무발화196개로 인해 일부 해부학적 근육이 완전히 무출력인가, 또는 활성 MN이 모든 근육에 출력을 제공하는가? 입력/모터/몸체를 바꾸지 않는다. 같은 E150의 원본96 activation과 원래 MN 소유권, 검산된 E198 per-MN 요약만 사용한다. 무출력 근육이 발견돼도 보행 중 그 근육이 필요하다는 주장이나 활성화해야 한다는 결정은 별도 근거가 필요하다. 작은 출력에 새 문턱을 정해 비활성으로 취급하지 않는다.

입력: E150 `native-recruitment-state.npz`, `adapter-provenance.npz`, `muscle-provenance.json`; E198 `per-mn.npz`, `result.json`, `parent-e198-final-review.json`. 원본 manifest 및 E198 결과/원자료 SHA로 부모가 입력을 봉인한다. 원본 수치 계산 전 현 입력의 schema/header와 과거 결과만 확인했다. F6, E85 참조, 새 신경/물리/fit/GraphStore/StateStore/GPU는 사용하지 않는다.

정합: E198391 motor_ids와 adapter391 motor_ids가 exact, membership391 정수/0..95, pool96×391 비음수 유한, 각 MN의 양수 pool이 자신의 근육에만 존재하고 각 근육에 ≥1 MN. provenance의96개 leg/muscle/columns와 같은 membership을 exact 비교한다. motor_roster391은 유일한 문자열 목록이어야 한다. E198의3창 all/startup/gait 및391개 r/phi 분위수와 active_fraction을 확인한다. E198 result의 native 입력 SHA가 이번 native 파일과 같고 final review가 valid_saved_diagnostic/독립43검사를 확인해야 한다. 원래 native2001행 complete/activation_available/motor_rate_available가 모두 참, clocks0..2000/0..40000/5ms(1e-9), activation2001×96 유한/비음수여야 한다. 원래 time0은 평균에 넣지 않는다.

## 계산과 결정

창은 E198과 같은 all ticks1..2000, startup1..99, gait100..2000이다. 근육별로 전체 MN수, 그 창에서 실제 rate가 항상0인 MN수, 순간 phi가 항상0인 MN수, 이들 global index/root ID를 보존한다. rate0은 E198 r 최대 분위수=0, phi0은 active_fraction=0으로 판별하며 둘을 혼동하지 않는다. 원래 activation의 min/p05/median/p95/max/mean, 정확한0 표본 수, p95−p05를 기록한다. 골라낸 근육이나 유리한 창만 보고하지 않는다.

각 창에서 (1) 모든 소속 MN이 rate0, (2) 일부만 rate0, (3) rate0 MN이 없는 근육 수와 실제 activation 항상0인 근육 이름을 모두 표시한다. 모든 소속 MN이 rate0인데 전체 창 activation이 양수인 경우는 모순 후보로 따로 기록하며 zero-initial-state의 전체 창에서도 이 경우가 있으면 가설 판정을 보류한다. 일부 창에서 과거 필터 상태가 남는 현상은 즉시 현재 MN 출력과 동일시하지 않는다. moment42×96과6개 LTM으로 해당 근육이 기여할 수 있는 joint row를 표시하되, 이 위치 명령 성분을 실제 힘으로 해석하지 않는다.

완전 무출력 근육이 없다면 '침묵 MN 때문에 필요한 근육 전체가 꺼졌다'는 포괄적 설명을 기각하고, 임의의 전체 역치 감소를 선택하지 않는다. 무출력 근육이 있으면 그 이름/보행 역할/기존 목표 파형을 확인해 후속 개입 가치가 있는지 판단한다. 둘 다 지속 전진 실패의 인과 증명이나 F1–F8 통과가 아니다. 기존 E197 전달 가족 기각, E178/E179 국소 음성 근거, E150 최선 보행과 모든 미달을 유지한다.

## 실행과 분업

actual gpt-6-sol/medium이 `tools/analyze_c_banc_m3_muscle_support.py`와 `tests/test_c_banc_muscle_support.py`만 구현한다. 부모 FIRST, 같은 Sol 첫 수정, 부모 SECOND/필요한 잔여 수정 순서다. 정렬/비활성 분류/필터 잔류 해석/오류 출판에 필요한 소수 합성 시험을 사용한다. 이전 E179의 모든 오류 경로를 복제하거나 새 검증 프레임워크를 만들지 않는다. 부모는 별도 소형 계산 및 원본96행 집계를 직접 대조한다.

CPU1/GPU0, 원본1/retry0, 내부15/외부20/감독30초, 출력8MiB. 원본 누적5031.169947799714/5144초에서30초를 예약하며 상한을 늘리지 않는다. 원본 저장 진단 비용과 별도 준비/합성/부모 검산 비용을 구분하고 미계측을0으로 두지 않는다. M2/UI 변경0, F6노출0, 보행 후보 미채택/전체 F1–F8 active·미완료다. 소스 리뷰 뒤 부모가 한 번 실행하며 실패/partial을 보존하고 자동 재시도하지 않는다.
