# r4 private helper의 실제 JSON 정수형 반영

실제 candidate-v3 반환ba8866/exit1은쓰기에앞서 `scalar type`으로거부됐다. 원문은candidate-v3/actual-refusal.json에있다. actual controls와K-Commit은실행되지않았고비용·원장·lock은변경되지않았다. 새ROOT helper입력만 built-in JSON으로읽은실제cca06a/exit0은limits와trace.exit_code가모두System.Int64임을확인했고canonical288c/cost absent/lock absent를다시확인했다. 프로젝트fee/Source/catalog 함수나worker를독립probe하지않았다.

ROOT가처음쓴expected limits리터럴은작은값에서Int32였고exit명시검사도Int32였다. 실제JSON의Int64를그대로반영해expected limits는전부long,실제3개exit도long으로고정한다. trusted raw와전량typed 비교, bool거부,키·순서·배열·binary64비교를완화하지않는다. 새helper의오류진단에는형·경로를넣어같은실수를추적가능하게한다. ROOT는private candidate-v4의전체inverse와Source871/current13을다시검토한다.

기존운영13/Source/봉인/actualr4자료/실패candidate-v2,v3는보존한다. known no-write helper거부뒤의수정이며r4과학worker재실행이나unknown/persistenceuncertainty재시도가아니다. 실제r4fee525.537은미정산상태다. 성공후에도runtime/690호환PASS로보지않으며실패HOLD/SPENT/no-successor/F1–F8ACTIVE/INCOMPLETE를유지한다.
