# D18 잔여 수정 03: 실제 r3 실패를 보존한 입력 staging 및 새 검증

## 결과와 범위

실제 residual02 Native가 통과했다(615bdc/exit0). r3의 실제 worker44208은 고정4 테스트를 실행했지만, 네 테스트 모두 fixture 초기화 중 과거 r1 요청 파일 누락으로 실패했다(75958f/exit1). 새로운 검사 본문은 미실행이다. 실패 비용123.633초를 한 번 정산했다(7fc6e6/exit0). 독립 stdlib 전체 유형/binary64 비교88e5f8/exit0은322 debits/204 paid calls, 유일한 r3 실패 HOLD, 소유권·예약·pending 없음, 잠금 없음을 확인했다. 이는 자연 보행 성과가 아니다.

현재 운영13 및 Source871을 6b5a88/exit0으로 원문 인증했다. 입력 packet은 `verification/banc-natural-gait-m3-20260919/s28-f18-root/residual03/design-inputs.json`(95925/149b5b163b0f51798af920ad4e409a8d0be460d2400d89ef2b1e2daee24d6f21)이다. Source7acb957d, review392b73fd, c577926dcb, canonical499333/7f836034를 실제 기준으로 사용한다. D18 FIRST/SECOND/Sol sole R1 및 residual01/02 봉인·실패·비용·게시를 보존한다. 이 작업은 ROOT 잔여 수정이다. 동일 실제 GPT-6 Astra/max의 새 읽기 전용 검토를 반영하고 ROOT가 구현·검토한다. Sol R2/새 구현 에이전트는 없다.

## 설계

1. 두 fixture에서 과거 paid step의 명시적 의존 입력을 positive recovery/fence 전에 준비한다. 기존 raw_pins 인증은 유지한다. 각 step의 old_request/old_cost를 exact bounded metadata reader로 읽고 그 비용의 trace/outer도 pin을 그대로 검증·복사한다. 임의 재귀·전체 archive 실행·다른 cap 확대를 하지 않는다. 기존 r1에서 이 네 파일이 raw_pins에 없었음을 실제 유한 의존 대조로 확인했다. r2에는 포함돼 있지만 동일한 명시적 staging을 적용해 목록 구성에 따라 소비 입력이 빠지지 않게 한다. 복사가 중복되면 현재 fixture의 동일 원문 검사로 처리한다.

2. 새 검증은 소비된 r3를 재사용할 수 없다. 기존 두 전체 paid fold의 기준319→320→321을 원문·유형·키 순서·binary64 그대로 인증한 뒤, 실제 r3의321→322를 세 번째 동일 proof body로 인증한다. 새 current `c5.actual320`은 실제 r3의 before/admitted/terminal/request/cost/trace/outer/admission capture/commit capture/failure/old_continuation20/old_failed_hold와 order_relations를 연결한다. 이 이름은 기존 유한 schema 경로 재사용이며322를320으로 잘못 보고하지 않는다. 기존 current c5779 및 이전 c5b8be는 Source/review/contract archive 연결로 별도로 인증해 앞의 두 step을 찾는다. 현재 Source7ac의 c5779 연결과 c5779의 historical Source5d7→c5b8be 연결을 각각 whole three-contract association, Sources 및 reviewed_source와 비교한다. 기존 source5d7를 Source7ac와 원문 동일하다고 취급하지 않는다.

3. 기존 Native KPredecessor scriptblock 및 Python terminal_baseline의 동일한 provenance 검사를 유한 두 조상에 재사용한다. 각 helper는 해당 step의 인증된 adoption_before_Source를 사용하고 current step은 컴파일에 bound된 실제 physical Source7ac에 exact 비교한다. 과거5d7 기반과 새7ac 기반의 검사 경로를 분명히 구분한다. 임의 재귀 해석이나 mutable-current 회피는 없다. 세 paid step의 고정 scalar vector는 r1/115.76/320/202, r2/0.672/321/203, r3/123.633/322/204이며 모든 once 비용과 closure를 확인한다. 새 비용을 과거 contract/review에 역기입하지 않는다.

4. r3 before는 실제 Native recovery의 원장 `residual02/r3-before-admission-canonical321.json`을 사용한다. admitted는 `residual02/r3-admitted321.json`, paid는 `residual02/r3-terminal322.json`이다. r3 old_continuation은 실제 event20이다. before 재구성은 이전 r2 HOLD 제거와 marker20 변경뿐이며, admitted는 단일 r3 call/reservation 추가, paid는 실제 closed outer 기반 once cost/debit/charged/HOLD 및 ordered totals다. 각 전체 raw operand와 독립 유한 key-order permutation을 확인하고 모든 나머지 배열/형/bit를 유지한다. 실제 closed-pending 파일과 전체 tool 반환도 보존하고, NO_CREATE r2와 RESUMED r3를 혼동하지 않는다. 현재 paid322를 복구할 때 r3 실패 HOLD와 marker만 바꾸며 비용·총계·한도를 건드리지 않는다.

5. 이 세 실제 ledger 경로만 기존 Python data/reader_cap 및 Native K metadata/reader의 finite524288 allowlist에 추가한다. Source/review/contract/trace/outer는 기존 bounded JSON caps다. six 기존 ledger 예외·원문 검사·wrong type/bool/alias/hash/size/extension 거부를 유지한다. 일반 prefix나 전체 cap을 넓히지 않는다.

6. 실제 Source-bound event21은 before_ledger=actual322, previous_recovery=actual20, old_source/review/contract=actual7ac/392/779의 logical-before와 immutable archives를 연결한다. 새 Source/review에는 실제 완료된 ROOT 최종 보고서만 들어간다. ROOT5의 마지막 두 슬롯을 새 residual03 final MD/JSON으로 교체하고 나머지 FIRST/SECOND/SECOND.json 및 Sol2를 유지한다. 원래 fit_provenance4, 원래 direct4 및 과거 residual4는 봉인·보존한다. 현재 final의 residual_provenance는 정확히 이번 설계/실제 새 Astra MD/JSON/ROOT 반영4개다. 두 Source-bound fixture의 실제 단계와 기존 positive·negative 검사를03으로 연결한다.

7. 새 미사용 r4만 `residual03/full4-request.json`으로 게시한다. 새Source/새event21, argv의 동일 네 pytest node, 180/160/20초,2GiB/8MiB를 유지한다. 새로운 JUnit/output 경로는 r4, 기존 r3 자료는 불변이다. 새 기준 수는322/204→bootstrap323/205→qualified327/209→body328/210→audit329/211이며 allocation/qualification/conditional-body 및 fee-only 분기의 모든 의존 상수를 함께 갱신한다. 다음690와 조건부Original600의 네 generated 역할과 과학 조건은 그대로다. 실패를 성공 선행 조건으로 사용하거나 failure stop을 완화하지 않는다.

## 크기와 검증

현재 Python98682/99999, Native130809/131072, c5127140/131072, active test261630/262144다. account393216, Source/review/fixture/test262144 caps와 Source871/11 replacements/860 untouched/0 additions, headers43/70, fixture13/14를 유지한다. 동일 proof body의 세 번째 유한 tuple·현재 경로·유한 선행 선택만 추가한다. 반복 call ID 접두사를 기존 base-call 값을 별도 local scalar로 보존해 같은 문자열을 구성하는 의미 보존 축소는 가능하다. 일반 comparator/binder/fee/cache/backend/C# Type guards를 제거·완화하지 않는다. 원문 후보가 실제 모든 cap에 맞을 때만 진행한다. 맞지 않으면 정적 단계에서 최소 공유 표현을 다시 설계하고 새 검토를 받는다. 한도를 늘리거나 과거 증거를 빼서 맞추지 않는다.

기존 parity node 안에서 두 fixture의 실제 신규 staging block을 각각 Source-bound raw에서 compile/exec하고, 세 step의 request/cost/trace/outer가 실제 쓰였음을 확인한다. 누락/변형/alias/hash/size/bool/예상 밖 layout은 기존 negative 검사에 합쳐 확인한다. 기존 6 reader 검사, Type strictness, 두 Source/review drift와 provenance24 case, 레거시133 reason/FNV/counter 기준은 보존한다. 새로운 무료 독립 project probe/test/Native/CUDA/Original/NPZ 실행은 없다. 준비 단계는 own stdlib raw/hash/JSON 검사만 허용한다.

실제 design→실제 Astra/max 검토→ROOT 반영→private current11→독립 whole forward/inverse 및 모든 크기 검토→실제 ROOT final 보고서→review/TRUE/FALSE Source→독립 최종 closure→한 번 common-owned 채택/게시 순서를 지킨다. 현재 canonical7f836034/physical7ac·392를 owned writer의 실제 raw-before로 사용한다. PID/FILETIME/token·서로 다른 before/replace backups·raw readback·uncertainty/unknown no-write·자동 재시도 금지를 유지한다. 새 승인 절차는 만들지 않는다.

## 실험과 결정 분기

남은 실제 Generated local1288.785691600001/shared1296.573676000001/Original1106.9150391004946초와 기존5794/6530/13370/19900 caps 및118/165 offsets를 유지한다. r4 full4=180 및 이후690의 총870은 현재 Generated 한도 안이다. Original600도 기존 별도 한도 안이다. reserve는 비용이 아니며 실제 dispatch→첫 FINAL만 한 번 기록한다. timeout/failure/unknown은 그 실제 결과로 남는다.

- 새 Native가 진짜 Type/fast/BOTH warm/no-refusal, 새 balanced journal의 지표[779,779,1558,1558,779,134528396,5453]/FNV5e419bb433ae9b98 및 정확한 canonical 변화에 성공하면 새 r4를 한 번 입장·실행한다.
- r4에서 실제 네 PASS, failure/error/skip0 및 실제 once-fee/known ownership closure를 확인하면690(40/35/170/420+close25)으로 진행한다. 과거 r3 실패와 검사 본문 미실행은 변경하지 않는다.
- 690의 전체 조건과 비용 종료에 통과하면 조건부Original600의 실제 BANC closed-loop body/audit/closure로 진입한다. proxy/teacher/metadata 성공을 gait로 채택하지 않는다.
- 실패하면 실제 종료·비용·원인과 Source를 먼저 보존한다. 임의 반복은 없다. 가설과 실제 남은 한도로 바뀐 in-scope 최소 대안을 판단한다. 기술 복구의 출구는 실제 body 검사 진입이며 계속 기록만 늘리는 성공 기준은 없다.

F1–F8 전체 목표 ACTIVE/INCOMPLETE, M2/UI와8352/fulltheta/alpha0.0001/eight frozen inputs를 유지한다. 최종 구동에 외부 teacher/CPG/phase/trajectory/root correction 금지. E150 partial, candidate4 unadopted, NativeVJP unverified, biological mechanism unproven이다.
