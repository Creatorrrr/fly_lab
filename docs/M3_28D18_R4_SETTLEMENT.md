# r4 실제 시간 초과의 좁은 정산 설계

사용자는 일반 승인 없이 판단·진행하도록 지시했다. r4는 실제 실행 후 소비됐다. 재실행·새 이름의 동일 재시도·환불·리셋·추가 자원 배정은 없다. 신규 구현은 ROOT가 맡으며 Sol R2는 없다. 같은 Astra/max에 아래 설계의 읽기 전용 검토를 요청한다.

실제 r4 request1527/77de4fe1, Source262142/872cc9ca, review247577/250081fa, wrapper254/40514eff를 보존한다. 실제 dispatch1791702420346→첫 FINAL 관측1791702945883은525.537초다. 첫 FINAL359901/exit1/session25679 SPENT이다. outer의 supervisor/worker exit124, TimeoutError, owner43740/worker26588, RESUMED, 양 EOF, all_owned_ended/released true와 unknown{}를 실제 확인했다. stdout은FF 두 바이트이고 JUnit은 없다. 고정4 PASS,690,Original600,보행 증거는 없다. 증거는 residual03/r4-fee-evidence.json3112/019debb9와 genuine-return4595/4edcc6f0에 있다.

현재 canonical은 실제 supervisor 종료 후 worker/reservation이 비었고, r4 admitted/closed-pending이며 아직 정산되지 않았다. 이 정확한 raw를 residual03/r4-before-fee-ledger.json에 보존했다.322 debits/205 calls의 모든 과거 배열·유형·순서·binary64·자원 caps와 offsets는 그대로다. pending map 및 실제 raw pin은 검토 전에 파일에서 재확인한다. r4 비용을 한 번 정산하는 것만 이번 범위다.

기존 Python shell_pair 및 Native K-ExitMatch는1/124의 경우 exec_args.cmd가 정확한 기존 상대 Python 직접 실행 suffix로 끝나는지 검사한다. 실제 r4는 인용된 pwsh -NoLogo -NoProfile -File wrapper로 호출됐으므로 이 suffix가 아니다. collector 명령을 가공해 그럴듯한 직접 실행 명령으로 꾸미거나 이 predicate를 전역 완화하지 않는다. 기존 모든 legacy fee/Source/cache/compiler/실험 조건을 유지한다.

추천 구현은 **이번 r4 정산만을 위한 ROOT metadata-only Native actor**다. 현재 실제 account 원문과 inline Native를 보관·인증하고, 별도의 private actor를 생성한다. 기존 common transaction의 PID/FILETIME/token·CreateNew exclusion·원본/replace backup·unknown no-write·uncertainty/no retry·cost then ledger raw readback을 그대로 재사용한다. fee-only path는 recovery/SourceFence/catalog/eligibility/archived execution/worker/NPZ/body를 호출하지 않는다. 운영 코드13, Source/review, 원래 request/wrapper/trace/outer와 sealed reports를 바꾸지 않는다. actor 자체의 실제 원문과 변경점을 별도 pin 및 as-executed evidence로 보존한다.

수정은 K-ExitMatch에 기존 legacy branch를 그대로 두고 **r4 한 건의 actual collector wrapper proof**를 추가하는 것으로 제한한다. 새 branch는 이번 request/call/Source/trace/outer/wrapper의 정확한 기존 pin, 실제 command 전체 문자열과 workdir, 보관된 tool initial/polls/첫FINAL pin의 연결을 확인해야 한다. request alias·hash·size·bool·field 추가·command prefix/suffix 변형·다른 role/call/Source/exit pair를 받지 않는다. wrapper의254 bytes/SHA40514eff, 실제 내용, resolved local interpreter 및 동일 argv 연결을 원문으로 확인한다. 임의 shell 해석이나 wrapper 실행은 없다.1/124 mismatch는 실제 TimeoutError/worker124/RESUMED/EOF/known closure로만 허용한다. actor 안의 다른 guards는 변경하지 않는다.

새 guard는 이미 보존된 genuine trace와 outer만 읽고 둘의 차이를 실제값 그대로 기록한다. trace exit는1이며 cost/debit actual_returned_exit_code도1이다. cost에 기존 native_supervisor_exit_code124/actual_collector_shell_exit_code1/shell_deadline_status_pair_observed true를 유지한다. fee525.537은 진짜 dispatch→첫FINAL 값이고 observed_supervisor_interval525.0240754나 저널 구간으로 대체하지 않는다. actual outer와 EOF/ownership은 이미 끝난 실행의 관측값이며 합성하지 않는다.

ROOT는 metadata-only actor의 전체 before/after 역변환, 이 변경 외 동일 본문, 실제 현재 raw pins/Source871, cost 원문, exact provisional/after ledger 전량을 stdlib로 독립 검토한다. fee 효과는 r4 state charged, cost descriptor 및 debit 하나, pending 제거, r4 실패 HOLD 하나, 실제 used/remaining ordered fold뿐이다. before에서 값을 복원하면 전체 원문값·형·순서·binary64가 일치해야 한다. bool/int 경계, duplicate JSON, finite timers, missing/mutated/alias evidence 거부와 existing legacy behavior 유지의 실제 bounded 검증 방법을 Astra가 제시해야 한다. 무료 project probe 또는 소비된 r4 재실행은 없다.

이 설계의 불확실성은 새 guard가 실제 common fee path와 source-bound metadata guard를 모두 만족하는지, 그리고 자료 형식에 추가 field가 필요한지다. 실제 도구 거부를 없애기 위해 증거 또는 명령을 꾸미지 않는다. Astra가 actor 별도 경로와 original fee-only provenance 검증의 실제 호환성을 정적 원문으로 확인한 뒤 조건을 ROOT가 반영하고 구체 actor를 작성한다. 호환되지 않으면 actual fee evidence와 unpaid 상태를 보존하고 다른 bounded metadata settlement 설계를 리뷰한다. 다른 science는 정산 및 실패 HOLD 처리가 끝날 때까지 진행하지 않는다.

Decision fork: 정확한 정산 성공→실제323/205와 한 번의525.537 비용·known empty ownership/HOLD를 독립 확인하고 FF 원인 및 timeout 단계의 저장된 관측 자료를 조사한다. guard refusal→실제 no-write/owner release 상태와 이유를 기록하며 r4를 재실행하지 않는다. unknown/uncertainty→소유권·예약·실제 원문을 보존하며 자동 재시도하지 않는다. 비용 정산은 도구 회복 결과이며 BANC 보행 가설/실제 물리 결과/자연성·생물학적 의미의 성공으로 취급하지 않는다. F1–F8 ACTIVE/INCOMPLETE와M2/UI/frozen8352/eight inputs/fulltheta/alpha0.0001 및 최종 외부 CPG/teacher/phase/trajectory/root correction 금지를 유지한다.
