# E360 P10 출력 보존 래퍼 사전 구현 설계

이 단계는 이미 승인된 개발 범위의 되돌릴 수 있는 소스 준비다. P10 실행, 권한 literal 작성/활성화, 추가 예산 배정과 검사를 승인하지 않는다. 현재 P9 활성 소스 9개, 원장 82개 차감, 모든 과거 실패 및 F1–F8은 그대로 유지한다. 원장 SHA는 `8735eae1e690d2747828545a500022896e836c467d58c7384439676a2eb51a7f`다.

ROOT 설계 → 실제 Astra6/max의 좁은 설계 리뷰와 반영 → 새 실제 GPT6.1 Sol/high FIRST → ROOT FIRST → 필요하면 같은 Sol ONE R1 → ROOT SECOND → ROOT 잔여 수정 순서를 사용한다. 이전 P9 구현자의 R2가 아니다. 구현 대상은 `verification/banc-natural-gait-m3-20260919/parent-e360-p10-capture-proposed.ps1` 한 개이며, FIRST/R1 사본과 짧은 인계 기록만 추가한다. 활성 dispatcher, Python, 테스트, 설정, 원장, 기존 literal과 기존 제안 사본은 수정하지 않는다.

기존 실제 Astra 리뷰 `parent-e360-P10-astra-design-review.json`의 P10-4와 반영된 `BANC_M3_E360_P10_REFLECTED_SCOPE_PROPOSAL_20261004.md`를 구현 기준으로 삼는다. 이번 좁은 리뷰는 아래 호출 경계와 시간/오류 처리의 구체화를 확인한다. 이전 문서의 “승인 후 구현”은 활성 P10 권한 통합과 실행을 가리키며, 이번 별도 사본 준비는 이를 활성화하지 않는다.

소스는 함수만 정의한다. dot-source만으로 프로세스를 시작하거나 파일을 만들지 않는다. 함수의 고정 실행 대상은 `.venv/Scripts/ruff.exe check tools/verify_c_banc_m3_source_closure_diagnostic.py tests/test_c_banc_source_closure_diagnostic.py verification/banc-natural-gait-m3-20260919/parent-e360-root-metadata-helper.py`다. 임의 명령, flags, target 교체 또는 shell 실행을 허용하지 않는다. native .NET ProcessStartInfo/ArgumentList를 사용하며 Add-Type, Python, 별도 런타임 또는 의존성을 쓰지 않는다.

래퍼는 승인 판정자가 아니다. 미래의 승인된 ROOT callbegin이 만든 live `e360-generous-p10-ruff` row와 exact request descriptor를 확인한 후에만 작동한다. 입력 expected request/scope bytes와 SHA는 미래 ROOT의 immutable final freeze에서 전달해야 하며, request 자신에게서 추출하지 않는다. fixed call/pass/role/mode/action, begun 상태, sequence0, reserved3/future795, failed_cost 없음, expected-output의 고정 P10 경로, request/source pins와 실제 세 Python target 및 래퍼 concrete-fix pin, call/request/pass direct authority 일치, resource_extension의 기존 v2를 확인한다. authority authentication, human scope approval와 모든 inherited82/14 fields/grants 검증은 미래 ROOT admission의 책임이며 integration은 아직 구현되지 않았다. 현재 ledger에는 P10 row가 없으므로 이 경계가 열리지 않는다. 새 scope key/schema 또는 가짜 approval placeholder를 만들지 않는다.

stdout/stderr paths는 각각 `parent-e360-generous-p10-ruff-stdout.txt`, `parent-e360-generous-p10-ruff-stderr.txt`로 고정한다. completion manifest는 `parent-e360-generous-p10-ruff-output.json` 하나다. 모두 task base 내부에서 reparse/traversal을 거부하고 fresh CreateNew로 생성한다. 두 stream을 먼저 열고 한 쪽 생성 실패 시 Ruff를 시작하지 않는다. 기존 파일은 삭제/덮어쓰지 않는다.

stdout/stderr는 별도 8KiB byte buffers와 ReadAsync를 사용해 동시에 drain한다. stream별 retained cap1MiB, 합계2MiB, 완료 manifest16KiB이며 retained 총계2MiB+16KiB다. EOF까지 관측한 count와 retained count를 각각 기록한다. overflow/쓰기 실패에도 메모리 또는 파일에 추가 데이터를 축적하지 않고 두 pipe를 계속 drain한다. 한 pipe만 순차 ReadToEnd하거나 PowerShell background runspace/callback을 쓰지 않는다.

native entry UTC와 monotonic stopwatch는 함수의 첫 executable statements에서 capture한다. 이것은 내부 관측값이며 ROOT의 tool-dispatch→실제 마지막 반환 전체 시간 및 비용을 대신하지 않는다. 3초의 고정 외부 cap을 늘리지 않는다. 기본 soft deadline은 entry+2초로 고정해 소유 종료와 파일 정리에 여유를 두며, 정확한 3초 완료는 런타임 검증 전까지 미검증이다. 2초 이후 직접 시작한 Ruff process를 한 번 종료 요청하고, 별도 고정 cleanup deadline entry+2.5초까지 두 pipe와 exit을 비차단 관측한다. 종료 확인이 안 되면 owned_state/exit/EOF를 unknown으로 기록하고 완료를 만들지 않는다. 재시도, PID 열거, kill tree 또는 다른 프로세스 종료는 없다. OS fsync가 지연되면 외부 실제 시간이 초과할 수 있으며 그대로 비용을 차감하고 실패해야 한다.

직접 Process 객체의 PID와 start time을 capture하고 종료 요청 전 같은 identity를 확인한다. identity 오류가 생기면 kill하지 않고 unknown을 기록한다. 실제 exit code는 알 수 있을 때만 기록한다. Ruff exit1, timeout/start/capture/flush/readback/publication 오류는 구분하며 첫 실패를 뒤 오류로 덮지 않는다. capture_complete는 두 EOF, overflow 없음, 파일 fsync/close/independent readback 일치, Ruff 종료 확인, 오류 없음이 모두 만족할 때만 true다. Ruff exit1은 complete capture일 수 있어도 검사 실패이며 이후 단계로 진행할 수 없다.

각 stream은 Flush(true)→close→별도 readback으로 byte length와 SHA를 확인한다. empty도 같은 절차를 거친다. bounded readback은 stream/file SHA 방식으로 수행하고 raw text를 전체 메모리나 tool output으로 출력하지 않는다. manifest도 CreateNew→Flush(true)→close→independent readback하며 actual exit과 wrapper errors, identities, deadline/EOF/overflow, byte/hash descriptors를 보존한다. manifest publication 실패는 compact 반환에 별개로 남긴다. 함수는 bounded summary를 반환하고 host 전체를 exit하지 않는다. 미래 ROOT 호출자는 Ruff exit과 wrapper failure를 결합한 tool exit을 명시적으로 전달하고 genuine full return을 한 번 차감한다.

이 단계에서 native source parse, 텍스트/해시 및 소스 리뷰만 가능하다. 래퍼 호출, dot-source 실행, 실제 Ruff/compile/AST/Python/helper/test/probe, 원본 IO/GPU, 가짜 ledger/admission 실험은 하지 않는다. PS source parse 성공은 런타임 검증이 아니다. 현재79개 전체 테스트 및 실제 suite/fixture/Q는 실행하지 않는다. 검증되지 않은 동작과 미래 authority integration은 최종 인계에 남긴다.
