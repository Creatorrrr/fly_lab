# E360 기존 공유 생성 예산 재배분과 소스 탐색 비용 개선 설계

현재 M3와 E360의 원본 범위 및 F1–F8을 유지한다. 추가 사용자 예산을 만들지 않는다. ROOT가 이 설계를 작성하고 실제 GPT-6 Astra/max 검토와 ROOT 반영을 거친 뒤 실제 GPT-6.1 Sol이 유한 예산 연결 부분을 구현한다. ROOT FIRST, SAME Sol ONE R1, ROOT SECOND, ROOT 잔여 수정 순서를 유지한다.

E360 ROOT SECOND 후 실제 Ruff 세 번(0.552 실패, 0.585 수정 후 실패, 0.558 통과)과 소스 탐색 두 번(0.677 인코딩 오류, 8.473 성공)을 각각 저장하고 차감했다. inherited42 다음의 비용 5개는 모두 보존한다. 성공 탐색은 프로젝트114파일/1974edges에서 실제88개와 누락15개를 발견했고 설치35개를 대조했다. unresolved1개는 미실행 신체 경로이며 diagnostic unresolved0이다. 원본 배열/header/hash 접근 및 body/neural 실행은 아직0이다.

현재 E360 생성10.845초, E358–E360 공동 생성41.612초, 공유 생성121.8020156/150초, 잔여28.1979844초다. E360의 25초 지역 배정에서는 fixture6초와 후속 suite12초를 동시에 예약할 수 없다. 전체 사용자 공유150초는 남아 있다. 원래 E360 지역25초와 배정되지 않았던14.0429844초 중 **14초만 재배분하여 E360 생성 지역 한도를39초로 변경**한다. E360 시작 공유110.9570156+39=149.9570156<=150이며 원본 지역20초, 공동 원본300초, 전체10200초, 원본 diagnostic최대1회/자동재시도0, candidate1/retry0은 그대로다. fixture6/5/4/3와 suite12/10/8/7, 원본12/11/10/9의 역할별 hard cap과 CPU/raw 범위도 그대로다. 미래 비용 추정으로 기존 실제 비용을 대체하지 않는다.

기존 frozen E360 plan/registration 및 inherited42, E359/E358 source/Q/실패를 수정하지 않는다. 새 amendment registration이 기존 E360 registration descriptor, 실제 비용47개 시점의 immutable snapshot, 재배분 근거 및 실제 Astra 리뷰와 ROOT 반영 descriptor를 연결한다. registration을 읽을 때 old authority와 amendment authority를 모두 검사한 뒤 생성 local limit39를 유도한다. shared budget/local 사용은 inherited42 이후 모든 E360 debit의 실제합이며 reset하지 않는다. live ledger의 최초47비용 전체와 descriptor/실제 full-return 파일을 확인한다. contract/Q/source closure는 새 amendment도 pin하며 재배분 이전 소스 탐색 결과를 현재 qualification으로 재사용하지 않는다.

ROOT는 반복 source-path/모듈 조회·동일 source SHA 계산을 한 scan 내부에서 캐시하도록 코드를 수정했다. 최초 안전 경로 조회, 실제 AST import edges, oracle 비seed, 순환/alias/unknown 검증을 유지하고 publication 전에 모든 파일 SHA와 조회한 파일·namespace·없던 경로의 layout 및 no-reparse를 재확인한다. 프로세스를 넘는 과거 캐시 재사용은 없다. 수정된 현재 source의 실제 탐색 보고서를 새 경로로 한 번 만든다. 예상 속도를 통과 결과로 쓰지 않는다.

같은 프로세스에서 build/CASE/Q가 동일 검증을 반복할 때도 과거 프로세스의 AST를 가져오지 않는다. 해당 프로세스에서 실제 수행한 AST 결과에 한해 등록 전체 identity와 root가 같고 매 사용 시 프로젝트 및 설치 source SHA, 모든 조회 경로의 파일/디렉터리/부재 상태와 no-reparse가 그대로일 때만 AST 재파싱을 생략할 수 있다. 보고서에 조회 layout을 명시해 누락/새 모듈/namespace 변경도 fail closed로 검사한다. source 또는 layout drift는 이전 캐시를 그대로 통과시키지 않고 거부한다. 반환 보고서는 복사하여 호출자가 cache object를 변형할 수 없게 한다. 실측 비용은 실제 반환마다 기존 원장에 한 번 기록한다. contract.source_sha256는 closure_report 파일 descriptor의 path/SHA도 포함해 pre_array 이후 publication까지 보고서 변경을 검사한다.

추가 budget grant나 local 사용 reset, 원본 body 재실행, 원본 GPU/neural, 알파 탐색, 기준 모델 재실행, 과거 결과 qualification 승격은 금지다. real standalone CPU fixture가 먼저 닫혀야 전체 current-case suite를 실행하고, 새 Q가 닫혀야 한 번의 저장 partial diagnostic을 진행한다. 현재20case/60phase는 예상이며 실제 아직 미실행이다. startup 재계산 성공도 자연 보행/생물학/M2/UI/F1–F8 완료가 아니다. 리뷰나 실제 잔여가 부족하면 원본을 열지 않고 모든 실패 비용 및 상태를 보존한다.
