# r4 정산 설계 — ROOT 반영

실제 Astra/max 최종fbfa96/exit0/ALL_WRITES_STOP의review.md7256/16eea47f,review.json14789/579cd28a,seal5441/4e71a45d를 ROOT stdlib로 재인증했다. 실제b5467b/exit0은906개 검토 입력과Source871을 원문 bytes/hash로 확인했다. 전체 근거는residual03/r4-settlement/review-auth.json과 불변Astra audit에 있다. R4S-C1–C8을 아래처럼 반영한다. 이전 Source/검토/봉인을 수정하지 않는다.

ROOT만 구현한다. 현재 account의 전체 원문을 private actor로 복사하고 정확히 세 구역만 바꾼다. copied Native의K-ExitMatch에는 고정 r4 request 경로가 일치할 때 새 helper로 먼저 연결하는 작은 hook을 추가한다. 기존 equal-exit와 legacy suffix 본문은 바이트 그대로 보존한다. helper와 불변 증거 pin 값, 새 ROOT metadata guard의 메모리 controls는 Native 구역 밖에 둔다. 기존Native131072 및account393216 한도 안에서 전량 inverse를 검토한다. 운영13은 그대로다.

공통 transaction 전 진입 guard는PSBoundParameters가정확히Action/TracePath이고Action=commit,TracePath가 고정 genuine-return 경로임을 확인한다. 현재288c raw와before-fee 전체가 일치하고r4 admitted/known closed-pending이며cost와lock이 없음을 확인한다. 현재13의raw를 다시 인증한다. 외부 launch descriptor가 actor 전체해시를 고정하며actor 본문 안에 자기해시를 넣지 않는다. 원래 공통 transaction을 시작한 뒤에는 실제owner.before와 현재288c의raw를 다시 비교한다.

새 r4 helper는 고정 request/Source/review/trace/outer/wrapper/initial/6polls/complete capture 원문을 매 호출 bounded read/hash/readback으로 다시 인증한다. 원문이 별도로 인증된 trace와outer의 전달값 전량 및형·키순서·binary64를 비교한다. 캡처 output의outer는 키순서가 달라서 전량 typed membership/arrays/binary64로만 연결하고 각 원문은 그대로 유지한다. 실제 command 전체와workdir,wrapper의실제254바이트 내용과literal interpreter/argv 경로,dispatch/firstFINAL/session/chunks를 보존한다. 지금 존재하는 interpreter 경로 확인을 과거binary 동일성 검증으로 표현하지 않는다.

새ROOT helper controls는 동일한 one-shot invocation에서 common transaction 전에만 메모리를 변형한다. 정상 r4 proof, 별도 표기된 legacy header control, bool/int·키·duplicate·nonfinite·시각·command·request·Source·alias·size/hash·exit·resume/EOF/PID/unknown negatives와 단일bool 반환을 검증한다. 프로젝트 Native fee/Source/catalog의 독립 무료 probe, fixture 생성, 기존r4 worker 재실행은 없다. controls 결과는 실제 stdout/firstFINAL로 보존한다. controls에 실패하면 정산·운영 쓰기 전에 중단한다.

그 뒤 현재 K-Commit과 common owned transaction을 한 번 실행한다. 최초 trace,provisional cost,paid-tail의예상3회 hook 각각 같은 원문 proof를 검증한다. one cost의fee525.537,exit1,기존mismatch124/1/true만 기록하고cost/ledger에새증거필드를 추가하지 않는다. 원래 Source/Identity/Roster/two predecessors/provenance와 once publication·typed guards를 유지한다. whole paid payload는게시보다먼저 확인한다. actual common PID/FILETIME/CreateNew token/costfsync/uniquebackup/atomicreplace/readback 및 uncertainty 동작은 그대로다.

ROOT가 실제323/205,한번의cost/debit,r4charged,pending제거,실패HOLD 및순차totals 외 모든 과거형·순서·배열·binary64가 같은지 전체정방향/역방향으로 검증한다. 신규backup의before raw와actuallock해제,운영13/Source871도 다시 인증한다. 비용정산 성공은runtime/690호환PASS가 아니다. 원래 운영validator는actualwrapper이력에여전히legacy suffix거부를 한다. HOLD/no-successor/SPENT를 유지하며690/Original/body를 실행하지 않는다. 거부·unknown·uncertainty면 실제원문/소유권을 보존하고 자동재시도하지 않는다.

예상잔여generatedlocal763.248691600001/shared771.0366760000006,Original1106.9150391004946은정적산술이다. 실제정산후독립검증값으로만현재값을보고한다. 환불·리셋·이전·추가grant는없다. F1–F8ACTIVE/INCOMPLETE와M2/UI/frozen8352/eightinputs/fulltheta/alpha0.0001,최종teacher/CPG/phase/trajectory/rootcorrection금지를유지한다. 보행·생물학검증증거는없다.
