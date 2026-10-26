**제안: P10의 확인된 소스 오류를 수정한 뒤, 기존 잔여 한도에서 수동 P11 한 번으로 E360의 전체 생성 검증을 다시 진행한다. 현재 P11 실행 권한은 없다.**

전체 목표는 기존 `BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md`의 finite-goal/F1–F8이다. BANC 신경 출력만으로 실제 여섯 다리가 협응해 전진하는 최종 조건을 유지한다. P11은 E360의 원본 신체 평가 진입 전 필요한 생성 검증 복구 단계이며 별도 보행 성공 목표로 대체하지 않는다. 교사·proxy·정적 진단·Ruff/fixture/suite/Q의 성공은 자연 보행 또는 생물학적 기전 검증이 아니다.

현재 authoritative 상태는 `parent-e360-p10-closed-decision.json`4777B/SHA9a017dda09b932d69f9881883b47e1634955f17dcd53c7ffa5e91a66b24be98b와 `parent-e360-p10-closed-ledger83.json`96365B/SHAd2139e26b7cd0e9dc9563fe64bd2d8eba83cdc8b495bae9ae93f668992e5e195다. P10 첫 Ruff 실제0.885초/exit1/B02314건을83번째 비용으로 한 번 차감했다. Native capture proof의 Int64/Int32 비교 결함과 원래 invalid flag를 보존했고, 별도 ROOT 종료 증거에 따라 잘못 남은 debit-backed hold만 닫았다. Pending/reservations/holds/begun/jobs0, 원본/GPU 호출0이다. 활성 own9는 P10 실행 당시 해시 그대로다.

직전 goal turn은 progress다. 실제 P10 실행/전체 로그/실패 원인, 실제 비용83와 독립 종료 증거를 만들었고 다음 변경을 결정하는 증거가 생겼다. 현재 자동 goal continuation은 직접 인간의 새 수동 pass 승인이 아니다. 마지막 직접 승인은 “추가 예산 없이 남은 한도에서 P10을 한 번 진행하도록 승인할게”이며 P10 literal의 `pass11_authorized=false`, `automatic_retries=0`을 바꾸지 않았다.

준비된 `parent-e360-p10-ROOT-residual-off-active-proposal.json`3297B/SHA255d247a5a730ad9cd2f81130bc80b3fb35541ba06f502cf501f53e55ad8680a를 구현 기반으로 사용한다. 테스트 사본220439B/SHAb6a7424a44a96be4178c2e904248dffafc49e08341d714d2af32f3cd42528978은 두 기존 테스트에서 반복문 입력을 default argument로 고정한다. Native account 사본163018B/SHAdcb698fda22ba77d52ce6c6036dbd87cda7f1c2f1f9f1d4787bec902940df5bb는 P10 transport cap3만 명시적인 Int64로 비교한다. Global `Equal-Typed`, lint 규칙/옵션, 테스트 본문·실패 사례·84개 이름과 예정252 phases를 유지한다. Boolean/double cap 거부 사례는 기존 native 회귀 이름 안에 추가했다. 두 수정은 미적용이고 실제 재검증은0이다.

먼저 실제 Astra6/max가 원자료와 source diff부터 위 소스 수정과 다음 권한 설계를 리뷰한다. ROOT가 반영한 뒤 필요한 소스/권한 통합은 실제 GPT6.1Sol/high FIRST→ROOT FIRST→같은 Sol ONE R1→ROOT SECOND→ROOT 잔여 순서로 수행한다. Multi-module Python/PowerShell·historical/current 권한 연결이라 high를 사용한다. 같은 단계의 Sol R2는 없다. 설계/소스 준비가 새로운 pass나 원본 실행을 승인하지 않는다.

P11 승인이 실제로 도착한 후에만 immutable 승인 기록과 새 scope literal을 작성한다. 새 grant0, shared1400/local1289, 기존 비용83 전체와 cost offset42, old16 activation/amendment fields와 두 grants를 그대로 보존한다. 새 P11 activation/amendment pair 둘만 한 번 추가한다. Generated pass 최대11, modified generated10, original1, total12로 한 번만 확장하며 P12/automatic retry/original execution authority는 false다. 승인 이전에는 ledger/literal activation/P11 request/callbegin/유료 실행이 없다. Proposal은 active authority가 아니다.

Historical P10은 live83의 정확한 as-executed pass/call/cost 상태를 고정한 사본으로 검증한다. 새 current own9와 historical P10 own9를 직접 비교하지 않는다. P10의 네 current source를 `parent-e360-p10-root-final-source-{tool,tests,account,helper}` frozen 사본으로 exact pass/path/bytes/SHA 연결하고, 기존 relocation30개 뒤에 네 개를 추가한다. P10 capture leaf·ROOT freeze·full-return·genuine return·manifest·owned closure는 기존 핀 그대로 검증한다. 원래 `internal_completion_valid=false`를 true로 바꾸거나 실패를 소거하지 않는다. New P11 admission은 once-debited closed P10 failure와 별도 실제 owned closure를 모두 요구한다. 현재 예산 선택자는 기존 P9 grant를 유지하고 새 P11 scope 선택은 독립된 uncached whole83 검증이다.

Fixed reader는12개, max16 entries/512 text descriptors/각 literal256KiB/aggregate2MiB를 유지한다. 기존 P2–P10 explicit historical policy와 v2 resource extension을 보존하고 debit index82는 P10,83부터 승인된 P11만 허용한다. Request/bootstrap/clock/helper/checkpoint/CASE/Q/worker/fallback/PS Call-Policy/Future-Admin/begin/actual/admin/finalize에 같은 direct new descriptor를 연결한다. 전체84개 이름·252 phases 및 journal cap4MiB/record8KiB/두 durable fences를 유지한다. 새로운 test function은 추가하지 않는다. P11 회귀는 기존 관련 함수 안에서 검증하고 과거 회귀도 실행한다.

Ruff capture는 새 고정 P11 이름과 output 경로의 function-only module로 작성한다. Module은 fresh P11 own9의 tenth key가 아니라 final freeze/concrete_fix의 leaf다. Independent ROOT freeze로 request/scope/source9/concrete_fix를 전달하고, genuine P11 callbegin 이후 한 번만 호출한다. Stdout/stderr 각각8KiB read buffer와 최대1MiB 보존, 최대16KiB manifest와8KiB compact ROOT return을 유지한다. Fsync/close/independent readback·두 EOF·실제 exit/creation identity를 요구하며 unknown ownership은 한 번 차감된 debit-backed hold로 남긴다. JSON integer 종류와 compiled constant를 명시적으로 맞추고 bool/double을 허용하지 않는다. Nonzero Ruff exit는 wrapper error와 구분하되 pass는 실패한다.

| 순서 | 외부 full-return 상한 | 내부 상한 | 이후 성공 suffix 예약 |
|---|---:|---:|---:|
| Ruff | 3초 | — | 795초 |
| Scanner | 20초 | 18초 | 775초 |
| Fixture build | 25초 | 22초 | 750초 |
| Fixture worker | 55초 | 46초 | 695초 |
| Fixture seal | 15초 | 13초 | 680초 |
| Suite build | 30초 | 27초 | 650초 |
| Suite worker | 600초 | 540초 | 50초 |
| Suite seal | 20초 | 18초 | 30초 |
| Qualification | 30초 | 27초 | 0초 |

전체798초는 현재 shared896.0529844/local896.010초 안에 들어간다. 전체 예약 후 여유는 shared98.0529844/local98.010초다. Fixture child50/science42, suite child560/science500, fixture future administration95/suite50를 유지한다. 실패/초과를 포함한 실제 ROOT dispatch→최종 반환 시간을 한 번 차감한다. 어떤 Python/import/compile/AST/Ruff/helper/test/probe도 무료로 실행하지 않는다. Native text/hash/source parse·metadata bookkeeping만 실행 전 준비로 사용한다. 실제 성공 여부와 시간/descriptor fit은 미검증이다.

현재 source·실제 원장·보호322·나머지 own5를 검증해 final freeze를 만들고 scope를 한 번 활성화한 뒤, 위 순서를 fresh outputs로 진행한다. Ruff/scanner/helper의 첫 실패면 후속 validation을 시작하지 않는다. Worker 실패면 정확히 같은 pending attempt/call의 이미 예약된 seal·actual debit·owned cleanup을 마무리하고 후속 validation/Q/original을 막는다. Q actual once debit 또는 그보다 앞선 실제 닫힌 실패에서 이 수동 범위를 종료한다. 원본20/300/10200, candidate1/retry0/retrylimit1, M2/UI·provenance·과거 실패와 F1–F8을 유지한다. Original build5/4.5·seal2/1.5은 이전 실측보다 작으므로 P11 Q 성공만으로 실행하지 않는다. 원본 신체 진입은 실제 current Q와 별도 실행 가능성/권한 검토 후 판단한다.

공통 실행 원칙: 최신 직접 사용자 지시를 우선하고 변경되지 않은 요구와 무관한 작업을 보존한다. 현재 finite-goal의 자연 보행 효과는 직접 증거로만 인정한다. 기존 진단·teacher·proxy·source/tests/Q를 최종 효과로 승격하거나 실패 후 기준을 낮추지 않는다. 실제 현재 소스·원자료·프로세스·자원으로 재개를 판단하며 문서나 자동 continuation을 새 실행 권한으로 쓰지 않는다. 기존 결과·누적 비용을 보존하고 큰 reversible source 수정도 이미 허용된 개발 범위에서 진행한다. 필요한 추가 agreement는 그 부분만 보류한다. Goal complete는 F1–F8 전체 검증 때만, blocked는 같은 genuine blocker가 세 연속 goal turn에 반복되고 의미 있는 안전한 진행이 없을 때만 판단한다. 이 파일 저장은 P11 승인/activation/실행이 아니다.
