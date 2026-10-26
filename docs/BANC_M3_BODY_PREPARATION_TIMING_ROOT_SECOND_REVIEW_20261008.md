# 신체 입력 준비 관측 — ROOT SECOND 및 현재 판정

2026-10-08. Astra/max 리뷰→ROOT 반영→gpt-6.1-sol/high FIRST→ROOT FIRST→동일 Sol sole R1→ROOT SECOND를 실제 수행했다. ROOT는 R1.md5478B/SHA256 1975185666532c2f7ab730cecd04f1985095326aa48c857099f6af708bbeb596, R1.json6239B/SHA256 e60cc0f5f5a8791a868e27b06846c0f76d8cf0eb9f9b0d084e9867645fca3d21를 전체 읽고 입력9개/출력4개 pins를 검증했다. R1 diff와 새 두 fixture를 전체 읽었다.

판정은 **Source 리뷰 완료, 실행 및 적용 미검증**이다. 기존 중첩14 제한을 별도 상수로 유지했고, 단계 이름22와 분리했다. 단계 START 기록이 실패하면 callback이 진입하지 않고 같은 예외를 보존하는 fixture를 추가했다. ROOT의 독립 비교로 journal의 R1 변경은 요청한 상수/guard 두 edit뿐이며, startup은 FIRST와 같고, 기존 FIRST tests59031B 전체가 R1 tests의 prefix임을 확인했다. ROOT FIRST에서 원래 arm_data 문장/검사/callback 순서, 함수 밖 텍스트, 기존 테스트32758B prefix가 같음을 별도 확인했다. 텍스트 일치는 테스트 통과나 실행 동치가 아니다.

재리뷰에서 추가로 필요한 Source 수정은 발견하지 않았다. Sol R2나 임의 ROOT 구현 변경은 수행하지 않는다. 최종 구현안은 다음 R1 세 Source이며 canonical에는 적용하지 않았다.

| 최종 proposal | Bytes | SHA256 |
|---|---:|---|
| verification/banc-natural-gait-m3-20260919/parent-body-preparation-timing-sol-R1-20261008/proposals/tools/verify_c_banc_m3_startup_ray.py | 78610 | c480c002ebb9252a9a77163b4361a3699cb33b1b620b045a246cff730ce5d254 |
| verification/banc-natural-gait-m3-20260919/parent-body-preparation-timing-sol-R1-20261008/proposals/tools/banc_m3_contact_runtime_journal.py | 26358 | 2f9932e4b1465a47b751896c303a05ad3c2c207154dafd16df1b39b3c01221c1 |
| verification/banc-natural-gait-m3-20260919/parent-body-preparation-timing-sol-R1-20261008/proposals/tests/test_c_banc_contact_runtime_journal.py | 63017 | 2ae60174cd43a62befb583a189163f64cb426a248db0a4a7c7b9c023fbfd91ed |

## 실제 확인과 미확인

- 실제: Source/metadata의 bytes/SHA 및 텍스트 비교, 실제 지정 reviewer/implementer의 Source cycle, 현재 canonical 코드와 ledger 불변, 이전 frozen Source716개 모두 exact unchanged. ROOT의 처음 재해시에서는 명시적으로 frozen된 절대 Source 경로를 상대 경로로 취급해 읽기 오류가 났으며, 절대/상대 경로 처리를 수정한 별도 재검증에서716개 전체 일치를 확인했다. 원본 데이터 접근이나 Source 변경은 없었다.
- 미확인: Python import/AST/compile/pytest/Ruff·Native/CUDA·Original payload/header/hash·actual body 실행, 준비 단계 실제 시간, 속도 개선, full qualification, 실제 보행. 모두 NOT_RUN/UNKNOWN이다. 추가 보호 실행0이며 과학 작업 시간/량을0으로 추론하지 않는다.

새 코드의 목적은 기존 검증을 유지한 상태에서8개 bulk 준비 단계의 inclusive time과 nested fence time을 구분해 기록하는 것이다. Extra stream write0, event64/active14/emit16/wire32KiB/stream512KiB/science270/worker285/flush1 및 terminal reserve는 유지한다. Non-opaque 정보는 기존 실제 snapshot이 획득될 때까지 메모리에만 있으며 hard stop 뒤의 진행을 복원하지 못한다. Preamble와 arm_data 바깥 neutral 추가 로딩은8슬롯 범위 밖이다. 시간의 합산이나 신체/neural/physics0 판정을 허용하지 않는다.

## 다음 판단과 한도

Current ledger453918B/SHA256 405ffe995f1241f2ff3263efc916c9e69897b18c37028da3a9aab82c6d37f661는 closed282/164 그대로다. Generated932.2819999999998/local964/shared remaining39.505984400000216/close20/local unused31.71800000000019, Original remaining3936.9150391004946, 모든 owned workers/reservations/pending/holds0이다. 최소 full8+12+96+close20=136초도 들어가지 않는다. 기존 실패 gate68/probe1/Count conformance1과 비용을 보존하고 새 grant/reset/close 차용/Original 대체/축소 qualifier/자동 재실행0이다.

최종 Source에 대해 기존 테스트와 신규 fixture를 모두 포함한 full qualification이 필요하다. 현재 제약에서는 실제 body를 실행할 수 없다. 이후 자원 계약을 판단할 때도 기존 누적 비용/실패와 고정 F1–F8을 보존한다. 다음 관측이 실제로 획득되면 가장 오래 걸리는 준비 단계와 그 내부 fence 시간을 비교해 구조적 비용 개선의 대상을 선택한다. 관측이 없으면 병목 UNKNOWN을 유지한다. 반복된 준비 시간 실패를 CUDA나 보행 문제의 원인으로 확정하지 않는다.

Finite-goal은 ACTIVE/INCOMPLETE다. Candidate-k UNADOPTED, biology UNPROVEN/native VJP UNVERIFIED이며 M2/UI·unused F6·BANC provenance·root GOAL_PLAN을 유지한다. 이번 결과는 실행 자격이 없는 검토된 구현안이며 자연 보행 개선의 완료가 아니다. 사용자 최신 지시대로 추가 승인 질문 없이 판단했다.
