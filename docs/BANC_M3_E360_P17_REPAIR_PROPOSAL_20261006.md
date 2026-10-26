# P16 수정본 적용·검증을 위한 다음 작업 제안 — 승인 전

전체 목표는 BANC 신경 출력만으로 자연 보행하는 M3 및 기존 F1–F8이다. 이번 제안은 그 경로를 막는 검증 코드의 실제 오류를 제거해 generated fixture/suite/Q까지 진행할 수 있게 하는 작업이다. generated 검증 통과만으로 전체 목표가 완료되지 않는다. root GOAL_PLAN.md의 기존 변경과 모든 후보/실패·M2/UI·F6·해부학/연결 provenance를 유지한다.

직전 회차는 실제 P16 Source 적용·Native 활성화/요청/once 비용 기록과 Ruff 실패의 발견, 같은 Sol ONE R1·ROOT SECOND Source 검토를 완료했으므로 progress이다. 현재 실제 상태는 closed94/calls44/passes16, shared586.7730156/1400·local475.816/1289·offset42이며 잔여813.2269844/813.184초다. pending/reservations/holds/unreturned/original jobs/reserved0이다. Ruff PID10004/start ticks639268195783398392는 종료·두 EOF·matching live=false로 닫혔다. original cumulative10020.115960899504/stage126.361/E360original0은 그대로다. 기존 한도와 비용을 초기화하지 않는다.

P16 literal 및 인간 승인 객체의 first closed failure stop/automatic retry0/P17 false가 실제 활성화·실행 경계다. 자동 goal continuation은 새 직접 인간 승인 기록이나 재실행 권한으로 사용하지 않는다. 이번 문서는 제안이며 authority literal·인간 승인 객체·활성화 이벤트를 만들거나 실행하지 않는다. Source-only 설계/리뷰는 가능하다. 승인될 때만 새 manual P17으로 한 번 진행한다.

수정 대상은 same Sol R1 tests346811/SHA72bb8e3ccc2b3c63e9e95baa525e72ab8b3265c8b1ec9400666d319adf354179다. 표준 importlib exact-file loader/전후 SHA 검사를 사용하고, P14 fixture에 잘못 복제된 미정의 current/value 블록을 제거해 이전 helper를 정확히 복원했다. ROOT SECOND 수용1679/SHA97ee001ce2a27b3321aaab19a1d9f0f975061e0f2f2c2ac6dbd2b7bfc1774225다. 현재 실행 tests347603/SHAad9defe054f0f486b5308e660a33ceab26d8ff2c293e24ae6f72ee0c794cc852와 실제 P16 freeze85962/SHA9390ee4e57501cd5a5bbd54ba3c28dd0700c5772c0594912ea7102277c431d62는 과거 실행본으로 보존한다. S102/F821 여덟 오류와 actual1.804/cap3/exit1, valid capture/collector를 지우지 않는다.

승인 후 구현 범위는 다음과 같다.

1. ROOT 설계→actual GPT-6 Astra/max 검토→ROOT 반영→GPT-6.1 Sol 적절한 effort FIRST→ROOT FIRST→필요하면 SAME Sol ONE R1→ROOT SECOND→필요하면 ROOT residual→적용/freeze/실제 검증 순서를 유지한다. 검토만으로 actual runtime PASS를 주장하지 않는다.
2. 새 current P17과 immutable completed P16을 분리한다. index93의 P16 Ruff는 actual cost3414/once1.804/exit1/failedcap3/next1 및 whole94의 정확한 과거 입력이다. 현재 Source9/clock/PID/selector로 과거를 다시 해석하지 않는다. P16 Source4는 actual frozen Source로만 읽으며 old54 relocation을 유지하고 actual P16 four를 추가한다. old28 activation fields/grants2와 all94 비용·calls44/pass16을 보존한다.
3. old17 reader의 경로/bytes/SHA는 그대로 두고 승인된 새 P17 reader18을 추가할 경우 현재 capacity18 두 guard만 올린다. 이전 historical policy16/current P16 policy17을 바꾸지 않는다. text512/literal262144/aggregate2097152/own9/hashcycle 금지·현재만의 uncached admission·첫 오류/fence/ownership/qualification 기준을 유지한다.
4. 원래 P14/P15 회귀, actual frozen P16 및 current P17 문맥을 분리한다. 84 named functions/order·252 phases·full11 Source14 callbacks·late fourth nonown drift·original Native15+PROGRAM16·37 primary slots 및 별도 실제 함수 의존성·one subprocess timeout8을 유지한다. 새로운 Source 문맥의 무정의 변수/잘못 복제된 블록/누락 함수·의도치 않은 global default를 ROOT와 Sol이 실제 읽어 확인한다. runtime을 실행하지 않은 채 타이밍 통과를 추정하지 않는다.
5. 재실행/새 scope를 자동 복구로 꾸미지 않는다. fresh namespace·literal·인간 직접 승인·Native4/collector/freeze/request/own9/paid clock을 일치시킨다. 기존 실패·실행본·비용의 변경이나 별도 grant는 없다.

동일 fixed9 caps3/20/25/55/15/30/600/20/30=798, suffix795/775/750/695/680/650/50/30/0, future95/50와 필요한 same-pending fallback120을 사용한다. local813.184−798=15.184초는 산술 여유이며 실행 완료 보장이 아니다. 첫 closed failure/Q once debit 뒤 새 역할을 중단하고 같은 pending의 필요한 seal/once debit/known-owned closure만 수행한다. P18·원본/GPU/body/neural/UI/service/network·추가 grant·자동 재시도는 이번 제안에 포함하지 않는다.

현재 검토해야 할 지식 공백은 R1 fixture의 모든 immutable metadata 의존성, 실제 Native PROGRAM의 함수/변수 의존성, 고정8초 내부 probe가 불필요한 full-history 반복을 갖는지다. Source/JSON/hash/Native ParseFile과 정적인 호출 구조 분석만으로 추가 위험을 식별한다. Python/import/compile/AST/Ruff/helper/account/activation/test/probe/role 실행을 timing probe로 쓰지 않는다. 검토에서 필요한 제한된 교정은 제안에 반영하고, paid 실행이나 새 authority 생성은 직접 승인 이후로 둔다.

직접 입력: actual P16 freeze85962, closed94 139634/SHA58fe659c2ff23ddef0371730e2934a20e753752295d32c397c4c43eba92deb95, P16 cost3414/SHAb198b53e671b1d2577d9539f8a26a5d546a454d2959070400c7bf6fa35d5b211, owned closure1442/SHAed300dcfd056ca5da80cef0c06eb535d257f0812d1a5666ba9c2039c70b38640, closed decision4533/SHA7243c54bc5680740ceea48b6cefc70d1a1b015fd692fbfb8dfb682a916610007, R1 handoff10538/SHA50423c62c7c5fb8b88914454d89ff6be8ebb54ce9fab37bbb5e1b34d0d5e0cf1와 source/diff4234/SHAf3925006556e14351af49c16826cbab5a2364abb4f37297a232832da0837af3a이다. 보고서/리뷰는 참고 증거이고 그 자체는 실행 권한이 아니다.
