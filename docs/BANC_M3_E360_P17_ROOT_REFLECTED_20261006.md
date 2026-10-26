# P17 수정·검증 제안 — ROOT 설계 반영, 승인 전

목적은 BANC 신경 출력만으로 보행하는 M3의 F1–F8 검증 경로를 복구하는 것이다. 현재 수정 대상은 검증 코드이며, 이 작업이나 generated 테스트 통과를 실제 보행·생물학적 기전 검증으로 해석하지 않는다. 실제 신경·신체 실험은 이번 제안 범위에 없다.

이 문서는 [원래 제안](BANC_M3_E360_P17_REPAIR_PROPOSAL_20261006.md)에 [실제 GPT-6 Astra/max 검토](../verification/banc-natural-gait-m3-20260919/parent-e360-p17-proposal-actual-astra-review.md)의 세 가지 필수 교정 M1–M3를 ROOT가 반영한 구체적인 승인 대상이다. 최종 리뷰 JSON25151/SHA34f178ca6e6a9ea740a557fd07fc9528fa1c982cc421101e9ce41b1bc4e659b8와 MD5266/SHAa361fdab21e1602d9381e57ba6ddc385cc28bbdc8e61397c1710bf60d6dd4c3e를 읽어 확인했다. 조건부 제안 수용은 구현 수용·새 실행 권한이 아니다. 현재 새 P17 실행 권한·인간 승인 객체·활성화 이벤트를 생성하지 않았으며, 새 구현이나 테스트도 실행하지 않았다.

## 실제 기준 상태와 승인 경계

P16은 Native 활성화·callbegin·once 비용 기록을 통과한 뒤 첫 Ruff에서 S102 한 건과 F821 일곱 건으로 실패했다. 실제 1.804초/cap3/exit1, valid capture/collector, 당시 Source4/Native4/collector/freeze, 모든 이전 실패를 보존한다. 동일 GPT-6.1 Sol의 ONE R1과 ROOT SECOND 검토는 완료했지만 R1은 실행 코드에 적용하지 않았다. 수정된 R1의 lint·suite·Native probe는 NOT_RUN이다.

기준은 immutable/live closed94 SHA58fe659c2ff23ddef0371730e2934a20e753752295d32c397c4c43eba92deb95, costs94/calls44/passes16, shared586.7730156/1400·local475.816/1289·offset42다. 잔여는 shared813.2269844/local813.184초이며 pending/holds/reservations/unreturned/original jobs/reserved는 0이다. P16 Ruff PID10004/start ticks639268195783398392의 종료·두 EOF·matching-live=false 기록을 유지한다. 현재 own9와 root GOAL_PLAN.md151552/SHA16898e461fede082206f9584c60b3fb11d994a7ab708cdfb19f950e35126eace를 보존한다.

기존 P16 인간 승인과 literal은 첫 closed failure 뒤 새 역할 중단, automatic retry0, P17 false를 명시한다. 새 직접 인간 승인 없이 P17 literal·인간 승인 객체·활성화·paid 역할을 만들지 않는다. 자동 goal continuation이나 리뷰 보고서로 이 승인을 대체하지 않는다. 이번 제안은 기존 잔여 한도에서 새 manual P17을 한 번 수행하는 것이며 추가 grant는 0이다.

## 1. 실행 당시 P16과 새 P17의 문맥 분리

현재 R1 tests346811/SHA72bb8e3ccc2b3c63e9e95baa525e72ab8b3265c8b1ec9400666d319adf354179의 표준 importlib exact-file loader·전후 SHA 검사, 원래 P14 helper 복원을 새 테스트 scaffold의 출발점으로 사용한다. 실행 당시 P16 tests347603/SHAad9defe054f0f486b5308e660a33ceab26d8ff2c293e24ae6f72ee0c794cc852와 freeze85962/SHA9390ee4e57501cd5a5bbd54ba3c28dd0700c5772c0594912ea7102277c431d62에 R1을 덮어쓰지 않는다.

FIRST는 P14/P15/actual frozen P16/current P17의 module·ROOT·literal·own9·ledger·reader policy를 명시적으로 대응시킨다. `_P16_CURRENT=h`로 새 P17을 과거 P16 허용 검사에 넘기는 연결을 제거한다. P15/P16 긍정·부정 검사는 정확한 당시 frozen tool/metadata를 사용하고, 새 P17 검사만 새 현재 Source를 사용한다. 표준 loader의 exact path/전후 SHA, 임시 global 전환과 try/finally 복원, 실행 시 선택하는 함수 인자를 유지한다. 별도 승인 없는 실행이나 검증 성공 cache를 만들지 않는다.

과거17 reader의 path/bytes/SHA와 P16 policy17, historical policy16을 유지한다. 승인 후 새 P17 reader18을 추가하면 현재 두 capacity guard만18로 바꾸고, 현재18-entry/19th rejection/descriptor19와 과거17-entry/18th rejection 문맥을 각각 검증한다. text512/literal262144/aggregate2097152와 own9/hashcycle 금지, uncached current admission을 유지한다.

P16 Ruff index93은 실제 비용3414/once1.804/exit1/failedcap3/next1과 closed94의 과거 입력이다. 새 현재 clock/PID/selector/own9로 과거를 다시 해석하지 않는다. old54 relocations 뒤에 actual frozen P16 Source4를 추가해58로 만들고, old28 activation fields/grants2와 costs94/calls44/pass16을 보존한다. 새 admission은 P17만 허용한다.

## 2. Native PROGRAM의 버전별 의존성과 도달 경로

ROOT 독립 읽기에서 기존 `_P15_PROGRAM_CONTEXT`/observer는 p15HistoricalLedger/p15PriorAdmittedRequest를 사용하지만 current P16 trace slice는 p16HistoricalLedger/p16PriorAdmittedRequest/p16Spec/p16Active를 참조함을 확인했다. 기존 P16 history 가지는 production proof 또는 좁은 typed gate 뒤 return하므로, 뒤 current loop 변수 문제를 실제 실행된 P16 결함으로 주장하지 않는다. 새 P17 PROGRAM에 이 문맥을 그대로 재사용하지 않는다.

FIRST는 각 version의 선언·초기화·observer·function dependency·입력 ledger·early return·도달하는 production slice를 대응표로 낸다. immutable historical END positive와 current loop context를 분리하며, 실제로 도달하는 모든 변수와 함수가 해당 버전에서 정의돼 있어야 한다. 원래 typed CLI Request, old collision negative의 첫 semantic rejection, 37 primary production slots의 순서·marker·출처를 보존한다. 필요한 context dependency는 별도 실제 Source slice로 제시하고 변경된 context bytes/SHA를 정직하게 기록한다. mock 또는 선언만으로 production gate 통과를 대신하지 않는다.

original Native15+PROGRAM16, one subprocess timeout8, historical positive/negative와 current semantic gates를 유지한다. ROOT는 stable FIRST/필요한 SAME Sol ONE R1/SECOND에서 실제 assembly와 version 대응을 읽어 확인한다. Native ParseFile 성공은 문법 확인이며 runtime PASS나8초 통과 증거가 아니다.

## 3. 반복되는 공통 version 검사 축소

현재 `_p15_versioned_history`의 기존10개 named test teardown은 모두 `_p15_current_equivalents(_P15_CURRENT)`를 호출한다. 이 공통 전체 검사의 직접 full-proof 호출만 기존 각 회3번, 총 최소30번이며 nested 호출은 별도다. Native full-positive도 P15 literal 내부 completed proof와 바로 다음 completed proof를 반복한다. 이는 실제 시간 측정이 아닌 Source의 호출 구조다. 공통 version assertion은 미리 지정한 기존 한 named-case 문맥에서 한 번 수행하도록 배치한다. 각 기존 test의 역사적 긍정·부정 본문과 필요한 개별 semantic assertion은 유지한다. assertion을 삭제해 검사 시간을 줄이지 않는다. 같은 호출 문맥의 이미 검사한 immutable evidence를 지역적으로 재사용할 경우에도 정확한 모든 gate를 보존한다.

84 named functions/order와 원래 의미, 252 ENTRY/CASE/EXIT phases, full11 Source14 callbacks·late fourth nonown drift를 보존한다. 좁은 typed negative는 같은 production leaf validator를 사용하고 매 negative마다 전체 immutable history proof를 추가하지 않는다. current admission·clock·budget·source는 매 실제 boundary에서 다시 검증하며 process-wide 성공 cache로 이 최적화를 구현하지 않는다.

정적 반복 경로를 줄이는 설계다. 실제 suite·Native8초 실행은 NOT_RUN이며 완료 시간이나 PASS를 예측해 기록하지 않는다. Source-only 단계에서는 trusted Source/JSON/hash와 Native ParseFile만 사용하고 Python/import/compile/AST/Ruff/helper/account/probe/paid role 실행으로 사전 타이밍을 측정하지 않는다.

## 승인 후 수행 범위

ROOT 설계 → 실제 Astra6/max 리뷰 → ROOT 필수 반영 → GPT-6.1 Sol/xhigh FIRST → ROOT FIRST → 필요한 SAME Sol ONE R1 → ROOT SECOND → 추가 잔여 교정은 ROOT → 적용/freeze/실제 fixed9 검증 순서를 따른다. 이전 P16의 SAME Sol R1은 이미 소비됐으며, 새 승인 후 P17 FIRST는 새 범위다. 리뷰 결과와 구현 결과를 구분한다.

fresh P17 namespace/literal/인간 직접 승인/Native4/collector/freeze/request/own9/paid clock을 일치시킨다. fixed9 caps3/20/25/55/15/30/600/20/30=798, suffix795/775/750/695/680/650/50/30/0, future95/50 및 필요한 same-pending fallback120을 유지한다. local813.184−798=15.184초는 산술 여유이며 성공 보장이 아니다.

첫 closed failure 또는 Q once debit 뒤 새 역할을 중단하고, 같은 pending의 필요한 seal/once debit/known-owned closure만 수행한다. P18·자동 재시도·원본/GPU/body/neural/UI/service/network·추가 grant는 이번 제안에 포함하지 않는다. 모든 과거 once 비용과 failed candidates, M2/UI/F6·해부학과 연결 provenance, original cumulative10020.115960899504/stage126.361/E360original0을 유지한다.

이 generated 경로를 통과해도 전체 목표는 ACTIVE다. 실제 BANC 폐루프·차단/저장·10초 전진·여섯 다리 교대/운동폭·미사용 초기 상태2개·6목표/정지/재시작 및 증거를 요구하는 고정 F1–F8를 바꾸거나 완료로 표기하지 않는다.

ROOT의 [실제 반영·보존 기록](../verification/banc-natural-gait-m3-20260919/parent-e360-p17-proposal-ROOT-design-reflection.json)은 리뷰 입력 pins, M1–M3 대응, 현재 own9/closed94와 변경 전후 계획 문서 pins를 기록한다. Source-only 진전이며 runtime/timing/자연 보행 결과는 NOT_RUN이다. 이전 회차는 실제 P16 구현·실패 발견·same Sol R1 검토로 progress였고, 이번 연속 회차는 새로운 구체적인 Source 병목 확인·설계 교정으로 progress다. 승인 대기만을 진행이나 목표 완료로 세지 않는다.
