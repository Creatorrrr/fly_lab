# D18 잔여 수정 03: 검증 시간의 독립 보완 설계

원래 설계 `M3_28D18_RESIDUAL3.md`를 수정하지 않고 이 보완을 별도로 검토한다. 실제 r3의 초기화 실패4개만98.29초 걸렸다. actual full return123.633은 이미 한 번 정산됐다. 고정4의 실제 본문은 추가 fixture 및 여러 Native 프로세스를 포함하므로 기존 child160에서 완료할 수 있다는 근거는 없다. Astra의 읽기 전용 초기 검토도 HIGH RISK/UNPROVEN으로 판정했다. 이 판정을 성공으로 바꾸거나 검사를 빼지 않는다.

ROOT가 제안하는 우선 대안은 새 r4 bootstrap만540/520/20초로 배정하는 것이다. 전체 Generated caps5794/6530, Original13370, combined19900과 현재 비용·offset·사용량은 그대로다. 현재 Generated local1288.785691600001/shared1296.573676000001 중 r4 maximum540+기존 다음690=1230을 배정할 수 있다. Original1106.9150391004946 중 조건부600도 유지된다. 새로운 총 예산·환불·리셋·resource transfer는 없다. 사용자의 최신 일반 승인 없이 판단하라는 지시 범위에서 기존 잔여 자원을 한 실행에 더 배정하는 제안이다.

이것은 실행 완료 보장이 아니다. source-only 경로 분석과 과거 실측을 바탕으로 기존160보다 여유를 주는 유한 상한이다. 성공 조건은 같은4 pytest nodes, failure/error/skip0, 실제 자식 프로세스와 ownership closure, 진짜 dispatch→첫 FINAL once cost다. child timeout 및 소유권/비용/overrun 처리·failure stop은 그대로다. supervisor/internal stopwatch를 full-return fee 대신 사용하지 않는다. admission 전에 전체 r4+690 envelope를 실제 남은 한도와 다시 비교한다. r4의 실제 once 정산 뒤 remaining이690보다 작으면 후속 실험을 무리하게 입장시키지 않는다. 바뀐 증거와 여유로 정당한 in-scope 대안을 판단한다.

바뀐 값은 미사용 새 r4의 role/request/table/identity 및 현재 bootstrap envelope에만 적용한다. 이미 소비된 r1/r2/r3의 세 proof tuple과 그 requests/limits180/160/20은 불변이다. source-before7ac와 실제 r3 비용·종료·typed/order/binary64 증명은 보존한다. fee 분기에 새 CPG/새 검증 회피 경로를 넣지 않는다. 신규 예산 event/무근거 다음 call 허용은 없다. 기존 고정4/690/조건부Original600 순서를 지킨다.

우선 수정은 시간 상한이며 테스트 fixture 공유는 이번 기본안에 넣지 않는다. fixture나 process/cold/cache/unknown/atomic 조건을 병합하면 실제 독립성·소유권 관측이 바뀔 수 있기 때문이다. 실제 전체 실행이 같은 의미를 유지하며 비용만 줄인다는 별도 설계·검토 근거가 있을 때만 최소 재사용을 추가한다. 현재 일반 루틴의 테스트 삭제/skip/negative weakening은 금지다.

시간 검토는 설계 수용의 조건이며, 실제540에서 완료·실제 한도 충족을 아직 주장하지 않는다. 원문 Python/Native/c5/tests/Source caps와 871/11/860/0 및 모든 기존 조건은 유지된다. 실제 Astra 검토→ROOT 반영→ROOT 잔여 구현/검토/closure가 선행한다. 생물학적 보행 F1–F8은 ACTIVE/INCOMPLETE다.
