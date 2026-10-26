# E161 접지 해제·수직 운동 결과

저장자료 분석은 유효하다. 실제 보행 개선이나 자연 보행 검증 완료를 뜻하지 않는다. E157의1–2초 native10,000단계/양의 접촉24,420개를 E150 신경 출력과 대조했고, 부모의 별도 scalar 계산65개 배열 비교는 최대2.8422e−14 차이로 일치했다. 원본 실행은1회/21.748491초, 신체·물리·신경·fit은0이었다.

RM 하중이 .02BW 위에서 아래로 내려간 전이는384개다. 고정한 앞10ms 연속 지지·뒤10ms 연속 비지지·±20ms 온전한 창을 모두 만족한 사건은0개다. 창 경계44개, 선행 지지 불안정339개, 후행 비지지 불안정1개로 제외됐다. 선행 안정은 전체384개 중1개, 후행 안정은2개였지만 같은 사건에서 함께 만족하지 않았다. 기준을 바꾸지 않았으며 event 평균은 unavailable로 저장했다. 긴 비지지 구간은 관찰되므로 이 결과를 “발이 한 번도 떨어지지 않았다”는 뜻으로 해석하지 않는다. 하중 문턱 자체도 기하학적 발 이탈 증거가 아니다.

## 제동 구간의 수직 기여

기존 interrupted 분류(speed≤.5mm/s), RM 지지≥.02BW, 순전방 힘<0 구간을 하중 가중한 결과다. 단위는 mm/s, world-Z 양은 위쪽이다.

| 실제 운동의 기여 | 값 |
|---|---:|
| root | +1.160971 |
| coxa roll | +.004672 |
| femur pitch | −1.408681 |
| tibia pitch | +.039741 |
| tarsus pitch | −.280090 |
| 나머지 수동 자유도 | +.002817 |
| 전체 접촉점 수직 속도 | −.487420 |

모든7관절을 원자료에 보존했다. 전방으로 복귀하는 coxa-roll AP 기여>0 조건까지 붙이면 femur-pitch는−1.890983, tarsus-pitch−.404944, root+1.538942다. coxa-roll은 수평 제동에는 관여하지만 이 구간의 주된 수직 하강 항은 아니다.

현재 접촉 기하에 투영한 **요청 명령 변화율**에서는 group70 trochanter extensor−.965596, group68 flexor−.545897, group74/75 tibia flexor−.401835/−.560802, group76 tibia extensor+1.000951, group79 tarsus levator−.599675였다. 이것은 실제 개별 근육의 속도·힘 또는 조작 효과가 아니다. 예를 들어 flexor activation이 감소해 extensor 방향의 명령 변화가 생길 수 있으므로 근육 이름만으로 부호를 판단하지 않는다. LTM78은 별도 tendon 경로를 보존했고 position moment0을 물리 효과0으로 해석하지 않았다.

![E161 전체 하중 및 수직 기여](../verification/banc-natural-gait-m3-20260919/e161-parent-figures/rm-unload-and-vertical-contributions.png)

## 소유 경로와 다음 실험

별도 source 감사와 부모 CSR 대조에서 load SNpp53→RM tibia flexor의 기존5개 직접 연결이 E43의 비선택 MN 입력 .001 배율을 유지함을 확인했다. rate-network의 .03/contact와 원래 GraphStore .275/contact는 단위가 달라 구분했다. 이5개만 복원한 과거 시험은 집중 조회에서 찾지 못했으나, 현재 출력 변화는 해당 구간에서 아래쪽 투영을 보인다. 따라서 이를 그대로 “발을 드는 감각 연결”로 복원하는 근거는 부족하다. 자세·간접 회로·반응 변화의 인과 효과는 여전히 미검증이다. 자세한 소유 경로와 기존 실패는 BANC_M3_E161_EXISTING_PATH_AUDIT_20260922.md에 있다.

다음은 group70의 아래쪽 출력 기여를 제한하는 단일 근육 효능 대조를 설계한다. 신경 출력 자체와 근육 구성은 보존하고, 시작99개 명령의 원래 허용 오차로 허용되는 하나의 감쇠량을 사전 정의된 식으로 정한다. 제동 감소와 지속 전진을 실제 신체에서 확인한 뒤 전체 신경 폐루프 적용 여부를 판단한다. 고정 F1–F8과 기존 시작 기준은 바꾸지 않는다. 출력 분해만으로 인과 개선을 선언하지 않는다.

## 구현·검증·보존

첫 구현과 같은 에이전트의 첫 수정은 실제 Sol/medium이었다. 고정된 저장 배열의 선형 분석이므로 medium을 선택했다. 부모 첫 리뷰에서는 contact/native 이름 충돌을 발견했고, 두 번째 리뷰에서는 NaN×0 평균과 마지막 저장 실패의 성공 표시 잔존을 발견했다. 잔여 수정은 부모가 수행했으며, 접촉 없이도 알려진 activation/명령 변화율을 보존했다. 최종20 tests/Ruff/format/compile, 별도 scalar 검산 및 생성 pipeline·저장 실패 검사가 통과했다. 원본 실행 전 실제 동결 계약 점검으로 부모 gate의 누락 필드를 발견해 v2 gate로 바로잡았고, 원본 발송은 늘어나지 않았다.

원본 결과 accepted/diagnostic_valid/publication_complete는 true, body/physics/neural/fit은0, candidate_adopted/whole_goal_complete/F6_exposed는 false다. OpenBLAS 실제 thread1, 자동재시도0, 보수 누적2660.748491/사전2694초다. M2/UI/F6와 과거 실패는 그대로다. 활성 목표는 여전히 미완료다.

주요 증거: e161-unload-timing/result.json, raw-derived.npz, output-manifest.json; parent-e161-saved-output-review.json; e161-unload-timing-supervision/supervisor-result.json. 코드는 tools/analyze_c_banc_m3_unload_timing.py, tests/test_c_banc_unload_timing.py다. 그림과 PDF는 e161-parent-figures에 보존했고 부모가 PNG를 직접 확인했다.
