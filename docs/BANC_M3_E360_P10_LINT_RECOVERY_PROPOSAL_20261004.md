# E360 P9 실패 이후 수동 P10 제안 — 미승인

P9는 첫 Ruff 검사에서 exit1 / 39건으로 실패했다. 실제 전체 반환 시간0.607초를82번째 비용으로 한 번 차감했다. 추가800초는 한 번 활성화됐으며 현재 shared503.0620156/1400초, local392.105/1289초다. 남은 allowance는 각각896.9379844초와896.895초다. scanner, fixture, suite, Q, 원본 호출은0회다. F1–F8, M2/UI, 원본20/300/10200, 기존 실패와 실제 비용을 보존한다. 자연 보행과 생물학적 기전은 여전히 미검증이다.

이 문서는 추가 예산을 요청하지 않는다. 기존 잔여 allowance 안에서 새 수동 P10 한 번을 제안한다. 승인된 P9 literal은 P10을 금지하므로 이 문서, 소스 제안과 설계 리뷰만으로 실행 권한이 생기지 않는다. P9를 재개하거나 실패를 수정/재분류하지 않는다.

ROOT에서 설계하고 actual gpt-6-astra/max가 검토한다. 승인 후 actual gpt-6.1-sol이 난이도에 맞는 effort로 새 scoped FIRST를 구현한다. ROOT FIRST에서 필요한 수정은 같은 Sol ONE R1에 맡기며 ROOT SECOND 뒤 잔여 수정은 ROOT가 마무리한다. Sol R2는 없다. 현재 lint 제안은 이전 단계의 ROOT 잔여 소스 수정안이며 활성 소스에 적용하지 않았다.

`parent-e360-ROOT-lint-source-proposal.json`과 proposed tool/tests 사본이 구체적인 소스 제안이다. 규칙 선택, 프로젝트 Ruff 설정과 전체 테스트 요구를 그대로 유지한다. 단일 nested context16개를 순서와 범위를 보존해 결합하고, subprocess.run에 check=False를 명시하고, import 순서/형식과 scanner의 nested if를 정리했다. 실제 고정 production bootstrap을 실행하는 세 개의 finite regression에만 기존 파일의 사용 방식과 같은 S102 주석을 제안했다. 전역 noqa, 검사 선택 축소, 테스트 삭제/skip/xfail은 없다. 현재79개 이름을 보존했다. 출력은 도구의2000-token 제한으로 일부가 유실됐으므로39건 전체 해결이나 Python 문법/런타임 통과를 주장하지 않는다. 승인 후 구현 리뷰에서 알려진 진단과 관련 소스를 모두 점검한다.

P10의 첫 Ruff는 동일한 규칙과 세 개의 활성 Python 파일을 검사한다. stdout/stderr를 bounded fresh text 파일에 보존하고 fsync/close/readback한 뒤 실제 exit code를 반환하여 또다시 진단이 유실되지 않게 한다. 이 저장 시간도 실제 전체 반환에 포함하고 한 번 차감한다. 알려진 실패 소스를 그대로 재실행하는 진단 수집 호출은 하지 않는다. 소스가 수정된 새 P10의 실제 첫 검사만 실행한다. source preparation/AST/compile/Ruff/helper/test를 무료 사전 실행하지 않는다.

실행 권한은 새 direct human scope authorization으로 기록한다. 예산 grant는0이며 P9의1400/1289, 기존 distinct800 grant, 과거 activation13개 및 비용82개는 그대로다. immutable inherited82의 전체 calls/passes/debits/evidence 및 offset42를 보존한다. P9 Ruff의 실제 once debit, exit1, cap3, actual0.607, helper가 아닌 Ruff의 internal_completion_valid=true와 미생성 scanner/worker를 확인한다. 원본/GPU는0이다. P9 failure는 P10 correction의 근거이며 테스트나 과학 결과로 승격하지 않는다.

새 P10 authority를 기존 revalidation_amendment로 직접 고정한다. resource_extension은 기존 v2다. P9 current budget selector는 current1400/1289와 직접800 승인/완전한 pair 및 grant를 uncached/nonrecursive로 검증한다. 명시적 P2–P9 historical policy는 옛 caps를 유지하며 P10만 새 finite pass로 선택한다. P9 old request의 own4는 실제 root-final-v2 사본으로 exact descriptor/path/pass/bytes/hash relocation4개를 추가한다. 기존26개를 보존한30개다. historical P9 own9와 fresh P10 own9를 구분하고 P9 전체 상태를 inherited82와 exact typed equality로 검증한다. P9 full closure510와 own9 projection의 ROOT N3 수정도 보존한다.

reader는 기존10개+P10 한 개다. max16 entries,512 descriptors,256KiB literal/2MiB aggregate, passive reached-only7 checkpoints, genuine ENTRY와 actual captured clock/writer, canonical import bridge, first failure 유지, two durable journal fences를 보존한다. 실제 전체 source collection은79개 이상84개 이하이며 기존79개를 모두 보존한다. N84 journal bound4177920≤4194304를 predispatch에 검사한다. 정확한 descriptor 수/aggregate 및 전체 test collection은 실제 scanner/build에서 검증하며 문서만으로 fit을 주장하지 않는다.

whole fresh9-call schedule은 Ruff3, scanner20/18, fixture build25/22, fixture worker55/50/46/42, fixture seal15/13, suite build30/27, suite worker600/560/540/500, suite seal20/18, Q30/27이다. outer total798와 suffix795,775,750,695,680,650,50,30,0을 보존한다. shared/local 잔여에서 whole reservation을 뺀 여유는98.9379844/98.895초다. 신규 P10 한 개, generated 최대10/modified9/total11/original1, automatic0, P11 권한false, 새 original 권한false다. 각 실제 반환을 지연/실패/초과 포함 그대로 한 번 차감하고 Q actual debit 또는 더 이른 닫힌 실패에서 종료한다. suite 성능과600초 완료는 미검증이다.

첫 검사 또는 이후 단계가 실패하면 그 증거·비용을 보존하고 더 이상 dispatch하지 않는다. 성공하면 실제 closed fixture, 전체 suite receipt와 journal, 동일한 candidate/source authority의 Q를 저장·fsync·close·재독한 범위만 검증 완료로 기록한다. 이 성공도 E359 원본 재실행, H359a 채택,1.5초/10초 보행, F1–F8 또는 자연 생물학적 보행 완료를 뜻하지 않는다.
