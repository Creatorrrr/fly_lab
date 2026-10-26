# E310: 고정 자원 감쇠 후보의 지속 협응 이익 없음

저장된 E307/u0와 E309/u=.005의 0.5초 whole 상태를 각각 그대로 복원해 1.5초까지 실행했다. 실제 CUDA 신경·신체 비교는 완전하고 건강했지만 후보가 전진·미끄러짐·지속 보행을 개선하지 못했다. 이 고정 resource branch는 더 확대하지 않는다. E309의 시작 실패도 그대로이며 자연 보행 목표 F1–F8은 미완료다.

| .5–1.5초 고정 평가 | 같은 SSP/u0 | u=.005 후보 |
|---|---:|---:|
| 원래 방향 전진 mm | 1.6857973223001235 | 1.5308455782782253 |
| 하중 가중 미끄럼 mm/s | 1.9427288085211483 | 1.996997490909536 |
| 최장 연속 전진 s | .210 | .205 |
| .3초 이상 적격 전진 구간 | 없음 | 없음 |
| 다리별 적격 완전 주기 | 여섯 다리 모두0 | 여섯 다리 모두0 |
| 전체 native 최소 upright | .9255629161268596 | .9255510569476302 |
| 비발 접촉 | 0 | 0 |

후보−대조군 차이는 전진 −.15495174402189815mm, slip +.05426868238838778mm/s, 연속 전진 −.004999999999999977초다. 후보는 원래 absolute short의 전진1.6790085779712631mm/미끄럼1.8026436968249955mm/s/적격 bout.3초/여섯 완전 주기 조건을 모두 충족하지 못했다. 대조군도 slip·bout·cycles는 미달이다. 기준을 낮추거나 양쪽 후보의 부분 통과를 합치지 않는다. 각자 다른 초기 .5초 폐루프 이력을 포함하므로 이 대비를 단일 순간 자원만의 인과 효과로 주장하지 않는다.

후보의 실제 endpoint minR .876602196848188/minG .9098394513130188, 비unity factor212093표본을 관측했다. Neutral은 R1/g1이다. 자원 교란은 존재하지만 지속 협응의 이익이 없다. 내부 stage 전체의 resource 이력이나 생리 기전은 이 endpoint 기록으로 입증하지 않는다.

## 복원 및 보고 오류 교정

Root가 두 arm의 저장 whole→실제 observed-start를 wall/RNG/신경/감각/held/근육bank/물리warmstart/resource를 포함해 exact 비교했다. 각각 추가200control/4000neural/10000native, 첫101 interval, 저장 prefix와 새201endpoint/10000native 및 평가301endpoint/15000native 연결을 검산했다. 최초 heading을 유지하고 원자료에서 displacement, frozen gait judge, slip, 절대 판정을 다시 확인했다. 독립 미래 disk replay 검증을 새로 했다는 주장은 하지 않는다.

첫 root 감사는 비교 요약의 시간 차이가 null인 오류를 발견해 실패했다. 원래 gait 값은 NumPy float64인데 matched_contrast가 Python float/int의 exact type만 허용했다. 실제 gpt-6-sol/xhigh의 첫 수정(R1) 뒤 root SECOND에서 보고 함수 이외 AST 불변,28개 focused 생성검사/Ruff를 확인했다. 수정은 finite NumPy scalar 처리뿐이며 원래 source/result/manifest와 첫 감사 실패를 보존했다. 별도 parent-e310-derived-contrast-correction.json에 누락된 −.005초만 기록했다. 원본 실행·평가·solver 재실행은0이며 startup/short 판정은 변하지 않았다.

최종 raw audit은 실행 전 보관한 원래 소스 사본과 모든 frozen input/output hash를 검증한 뒤 통과했다. 최초 결과 파일이 완전한 비교표였던 것으로 되쓰지 않는다. 현재 runner의 보고 수정 hash와 실제 실행 당시 hash는 별도로 기록한다.

## 비용과 다음 결정

원본1회/retry0, 신체2/추가neural8000/native20000, 실제 setup560/capture280/settling4000이다. Supervisor87.62893130001612초를 한 번 합산한 누적8416.23501749954/8560초, 잔여143.76498250045915초/예약0. Runner68.6755391000188초, treeRSS3735851008B, output352755600B, 소유PID1712/37564 종료를 확인했다. 별도 root audit 첫 실패6.8222721벽초, v2 내부7.265333499992266초를 기록한다. 생성 검사와 지원 시간의 일부 외부 합계는 미보존으로 unknown이며0이 아니다.

최초 Sol35/48/50 검사(총133), root FIRST50, Sol R1 focused28, root SECOND28 및 각각 최종Ruff가 통과했다. 최초Sol lint1fail/지원조회3error/Windows os206실행전거부1와 원래 감사 실패를 지우지 않는다. 이 검사 수는 실제 신경·신체 반복 수가 아니다.

다음 방향은 신경 상태 생성과 여러 MN의 근육 출력 결합 중 무엇을 바꿔야 하는지 구분하는 것이다. E175 additive history와 E287 affine decoder의 기각을 단순히 반복하지 않는다. E287은 바뀐 E277 source의 startup 조건만으로 불가능했으므로, 우선 실제 고정 근육 경로의 제약인지 확인하고 이 결과를 시작이 통과한 E150/SSP/u0에 일반화하지 않는다. 새 후보·fit·10초/F6 실행은 아직 없다. M2/UI/고정F1–F8/이전 실패·비용을 보존한다.

근거: verification/banc-natural-gait-m3-20260919/e310-resource-continuation/, parent-e310-original-raw-audit.json, parent-e310-audit-first-reporting-defect.json, parent-e310-derived-contrast-correction.json, parent-e310-second-source-review.json 및 두 생성 supervisor. 실제 실행 source archive는 e310-original-reviewed-sources/다.
