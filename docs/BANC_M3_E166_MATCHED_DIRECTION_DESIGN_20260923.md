# E166: 같은 BANC 기록에서 전진·중단 순간의 명령 방향 대조

2026-09-23. 부모 설계, M3 finite-goal 안의 저장자료 진단이다. E165는 유효한 정적 계산이지만 정상 지지에도 아래쪽 요청이 필요하다는 경쟁 설명을 해소하지 못했다. 부모와 실제 gpt-6-sol/high의 독립 읽기 전용 검토가 이 한계에 동의했다. 말초 전달 자원 R의 사용 중 감소·휴지 회복은 기존 적응/onset 함수와 구별되지만 이번에는 적합하지 않는다. 성체 다리 NMJ의 해당 실측 상수도 확인되지 않았다.

질문 H166은 **E165의 비양수 수직 방향이 같은 E150의 전진 순간에도 얼마나 나타나는가**다. 전진 순간은 speed>.5mm/s라는 기존 표본 분류이며 지속적인 자연 보행이나 정상 생물학 대조군이 아니다. 몸 자세·접촉 이력 등 관측하지 않은 차이는 남으므로 대응 비교도 인과 추정이 아니다.

## 고정 입력과 표본

E165 복구의 command-lift-raw.npz와 layout.json, E163 raw-derived.npz, 두 실험의 부모 final review를 읽는다. 새로운 신경/물리/기구학 계산은 없다. E165의 world-Z body-origin 투영 단위 mm, effective request index2, canonical LF/LM/LH/RF/RM/RH와 tarsus1–5, 영점대1e−12mm를 그대로 사용한다. 원점은 접촉점/clearance/실제 swing 해제가 아니다.

E165 tick1..2000과 E163 control_time_s[1:]를5ms/atol2e−8로 대응한다. E163 e1_phase는 cycle 단위 [0,1)이다. unavailable 값의 NaN을 유효0으로 채우거나 radian으로 나누지 않는다. [.5,10)이며 여섯 위상이 모두 available인 기존 공통 표본은 tick100..1923의1824개다. freeze에 이 기존 범위를 고정한다. E165 전체 completed/forward_completed/jacobian_completed가 참이고 필요한 z/load/speed/AP 자료가 finite임을 검증한다. 원본의 phase unavailable NaN은 허용된 결측이다.

각 다리에서 support load≥.02BW, 실제 ego AP velocity>0인 표본을 사용한다. interrupted case는 speed≤.5mm/s, forward control은 speed>.5mm/s다. 위상 결측·비접지·AP≤0 제외 수, 전체/공통 표본 수를 보존한다. t0과 상한10초는 포함하지 않는다. 사전에 관측한 사실은 E165 집계와 E163 가용 범위뿐이며 새 대응 결과는 아직 평가하지 않았다.

## 단 하나의 대응 규칙

같은 다리와 같은 기존 시간창 [.5,2), [2,5), [5,10) 안에서 각 case에 control 하나를 중복 허용하여 대응한다. 전진 표본이 없는 경우를 포함해 unmatched를 그대로 남긴다. 매칭 허용폭은 위상 원형 거리≤.025cycle, load 절대차≤.02BW, ego AP 속도 절대차≤.5mm/s다. 이 값은 작은 문맥 차이를 허용하기 위한 고정 공학적 비교 범위이며 생리 측정 오차가 아니다. 결과에 따라 넓히거나 두 번째 조합을 시험하지 않는다.

허용된 control 중 (phase_distance/.025)^2+(load_difference/.02)^2+(AP_difference/.5)^2가 가장 작은 하나를 선택한다. 점수가 정확히 같으면 control tick이 작은 표본을 선택한다. 시간 거리는 최적화 항이 아니며 따로 보고한다. 위상 경계0/1을 원형으로 처리한다. 각 case tick, 선택 control tick(없으면−1), 허용 control 수, score, 세 성분의 부호 있는 차이, 시간차, case/control의5원점 Z와 평균/최소/최대, all-nonpositive 표시를 raw에 남긴다. unmatched의 수치 값은 NaN이며 JSON 요약은 null이다. 선택 control 중복 횟수/고유 수를 저장하고 표본 수를 독립 반복 수라고 부르지 않는다.

## 요약과 해석 결정

다리별 전체와 세 시간창에서 case/control 수, match 수와 case 대비 coverage, 고유 control 수, 중복 분포, 각 요청의5원점/원점 평균 통계, 비양수 비율, paired case−control 차이의 mean/median/p05/p95를 보고한다. 대응 전 전체 적격 case/control 요약도 함께 보존하여 대응으로 생긴 선택을 확인한다. control 요약은 case별 대응 가중값과 고유 control 값 둘 다 보고한다. 외부 정상 보행 기준과 비교한 것으로 쓰지 않는다.

비양수 부호만을 특이 근거로 사용할 수 있는지에 대한 제한된 결정은 다리별 전체에서 한다. matched case≥20, 고유 control≥20, case coverage≥.5를 모두 만족해야 충분한 공통 표본으로 분류한다. 이 수는 독립성/통계 유의성 기준이 아니다. 충분한 경우 paired forward control 비양수 비율≥.90이면 `nonpositive_sign_shared`다. 이 경우 E165의 부호 자체를 중단 특이 결함이나 고갈 모델의 근거로 사용하지 않는다. 충분하고 case 비양수 비율−paired control 비율≥.10이면(앞 조건과 배타적) `sign_separation_in_matched_subset`다. 이는 이 공통 표본의 연관만 나타내며 고갈·인과를 입증하지 않는다. 그 밖은 `inconclusive`다. 효과 크기와 세 시간창 결과를 그대로 보고하며 전체 다리의 하나의 유리한 결론으로 합치지 않는다. 이 진단은 후보를 채택하거나 F1–F8을 통과시키지 않는다. 공통 표본 부족은 과학적 반증이 아니다.

## 구현, 보존, 자원

부모 설계 → 실제 gpt-6-sol/high FIRST 구현 → 부모 FIRST 리뷰 → 필요한 첫 수정은 동일 Sol → 부모 SECOND 리뷰와 잔여 직접 수정이다. 수치 매칭 자체는 작지만 위상/시계/결측/선택 편향과 실패 저장 계약을 연결하므로 high를 선택한다. 소유 파일은 tools/analyze_c_banc_m3_matched_direction.py와 tests/test_c_banc_matched_direction.py 두 개이며 과거 helper와 원본은 읽기 전용이다.

CLI --freeze/--freeze-sha256/--source-review/--source-review-sha256/--out를 제공한다. 부모 승인 source gate 이후 원본 process1회/자동재시도0이다. 직접 script를 무관 cwd/PYTHONPATH 없이 호출하는 생성 fixture 검사도 수행한다. 원본 파일을 생성 테스트에서 읽지 않는다. 생성 full-run fixture, 원형 경계, 허용폭 경계, tie, 중복, 결측과 무대응, 잘못된 time/unit/shape, 저장 오류/사후 hash/최종 시간·용량 초과를 필요한 범위에서 확인한다.

실행 전후 고정 입력·소스·과거 자료와 live freeze/gate/snapshot을 검증한다. 입력 정렬과 수치 residual/mask를 assertion 전에 partial raw로 보존한다. 새 pair 배열도 요약 판정 전에 저장한다. 성공 결과는 최종 보존·자원·출력 기록이 끝난 뒤에만 게시하며 늦은 오류에는 기존 성공을 취소한다. 원래 오류와 partial/temp를 남긴다. JSON NaN/NumPy scalar 오류와 subprocess import를 실제 경로로 확인한다. 필요 이상의 프레임워크나 원본 복제를 만들지 않는다.

CPU1/GPU0, 원본 process1, fit/body/StateStore/GraphStore/neural/physics/Jacobian0. 내부60/외부80/감독90초, 출력64MiB, free1GiB. E165까지 실패 포함 실제 보수누적2862.3110312초를 올림한2863+90=2953초로 배정한다. 이전 배정 종료값은 새 사용량0이 아니며 이번 단계 이후에도 누적을 이어간다. 동일 대응 재평가/허용폭 sweep은 없다. 원본 보존 검산과 생성 테스트 비용은 별도 기록한다. M2/F6/UI/고정 F1–F8과 전체 미완료 상태는 그대로 유지한다.
