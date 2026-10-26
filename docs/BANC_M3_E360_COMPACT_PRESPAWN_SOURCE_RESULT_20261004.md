# E360 compact pre-spawn SOURCE 수정 결과 — 2026-10-04

ROOT SECOND 완료 상태는 **SOURCE_REVIEW_COMPLETE_RUNTIME_NOT_RUN**입니다. 실제 P11은 첫 닫힌 실패로 종료했습니다. 이번 후속 작업은 그 실패 원인을 고치기 위한 별도 소스 제안이며 활성 코드에 적용하지 않았습니다. Python 구문·import·AST·compile·Ruff·테스트·fixture·Q·원본·신체·신경·CUDA·UI 추가 실행은 없습니다.

ROOT 설계 → 실제 GPT-6 Astra/max 설계 리뷰 → ROOT 반영 → 실제 GPT-6.1 Sol/high FIRST 구현 → ROOT FIRST → 같은 Sol의 ONE R1 → ROOT SECOND를 완료했습니다. ROOT SECOND에서 추가 actionable 소스 결함을 발견하지 않았습니다. Sol R2와 ROOT 잔여 코드 수정은 0입니다. 이는 실행 검증 통과를 뜻하지 않습니다.

## 고정한 수정 제안

최종 소스는 [R1 tool](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-sol-R1-tool.py), [R1 tests](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-sol-R1-tests.py)입니다. tool은 311842B/SHA bb98309cb4e81c660233cda4d61dea404ce33effb56d1067360617173cc10e79, tests는 242632B/SHA 60edbdb56b0dcdc1234bfc0f47e0e4ab6ab6dfa26c1659e3f3a08d76cc486b9e입니다. FIRST 파일과 R1 파일을 각각 불변 산출물로 유지했습니다.

현재 영수증을 v2의 정확한 33개 필드로 바꾸고 중복된 전체 authority 값만 제거했습니다. 전체 계약과 소스 해시 맵을 계약 descriptor로 참조해 두 쪽에서 새로 읽고 완전한 typed equality를 확인합니다. 소스552개를 줄이거나 authority 검사·owner·시각·한도를 생략하지 않았습니다. 계약 읽기524288B, 영수증/기존 bootstrap262144B와 fsync·close·독립 재읽기 조건을 유지합니다.

Astra 설계 리뷰의 지적 F1–F5를 반영했습니다. 현재 소비자는 v2만 받으며, 과거 P8 v1은 고정된 full-exit/inherited81·유일한 차감·계약·영수증·원래 source510·ENTRY·실제 닫힘을 확인하는 별도 역사 경로로만 취급합니다. 이후 seal/closed-exit/SuiteDeadline에서 변경된 live ledger·예약을 대상으로 prospective admission을 다시 수행하지 않습니다. 계약/영수증을 실제 경계의 descriptor에 등록하고 publication/Popen/첫 numeric 실행 경계에서 새 fence를 수행합니다. suite/Q는 전체 closure끼리 비교하며 own9는 정확한 projection으로 따로 확인합니다.

ROOT FIRST의 세 지적을 ONE R1에서 수정했습니다. 작업자의 최초 ENTRY와 나중 검증기의 작업 시계를 분리했고, pre-array가 최초 영수증의 ENTRY를 캡처해 재읽기와 timer에 계속 바인딩하도록 했습니다. 큰 계약이262144B bootstrap에서 거절되면512KiB 이하의 bounded failure-only 읽기로 기존 첫 오류/owned closure 기록을 남깁니다. 반복문에 걸린 회귀 함수의 변동 값을 명시적으로 바인딩했습니다.

새 회귀 소스는 실제 writer/readback/pre-array의552개 구조, 전체 계약·영수증·UTF8·descriptor·type·ENTRY·PID·owner·size·변조, 나중 authenticated verifier에서 이전 worker의 v2 검사, bootstrap 거절 후 no-Popen/closed-failed 처리, suite/Q 전체맵과 own9 분리, 고정 P8 역사 검사를 포함합니다. 기존 **84개 테스트 이름과 소스 순서를 유지**했습니다. **252 planned / 0 executed**입니다. 구조와 seam을 쓴 회귀 소스는 실제 paid admission의 증거가 아닙니다.

## 실제 P11과 자원

[P11 닫힌 결정](../verification/banc-natural-gait-m3-20260919/parent-e360-p11-closed-decision.json)은 계속 CLOSED_FAILED/qualified=false입니다. Ruff1.190, scanner8.237, fixture_build9.822은 exit0이었습니다. fixture_worker8.479는 fresh bounded publication으로 자식 Popen 전에 실패했고 같은 실패 worker의 seal7.602만 수행했습니다. 실제 반환 합계35.330초를 한 번씩 차감했습니다.

정확한 전체 실패 payload 크기는 관측하지 못했습니다. 소스와 실제 계약 구조에서 구한 하한은 authority220181B와 중복 source_sha25679401B의 두 값만299582B로 cap262144B보다 큽니다. 이 SOURCE 하한 진단과 이번 수정 제안은 실행 성공 주장이 아닙니다.

| 자원 | 사용 / 한도 | 남은 한도 |
|---|---:|---:|
| 공유 generated | 539.2770156 / 1400초 | 860.7229844초 |
| E360 local generated | 428.320 / 1289초 | 860.680초 |

추가 배정0, 원장88/call38/pass11, pending/reservation/hold/unreturned/original job/예약0을 유지했습니다. 현재 원장은106985B/SHA b23a5ec9fd07f3e15074181e3c5c9f3ba65554413a2256375c1797f23aef96f8입니다. 이전83비용·옛 activation16·grant2·P10 false cost와 실패를 유지했습니다. 원본20/300/10200, candidate1/retry0/retrylimit1도 그대로입니다.

[ROOT native 보존 확인](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-ROOT-native-preservation.json)은 활성 own9와 보호322개(상대229/절대93), 원장88을 실제 Source hash로 재확인했습니다. current84와 proposal84 이름/순서는 native regex로 확인했습니다. Git Source diff에는 whitespace 진단이 없었고, no-index 변경의 exit1을 테스트 통과로 취급하지 않습니다. ROOT의 native 검증 명령 실수2건과 Sol R1의 경로 구성 실수1건은 각각 리뷰/인계 JSON에 남겼고 활성 코드·원장 변경은 없었습니다.

## 다음 검증 범위 — 새 수동 승인 필요

[P11 직접 승인 기록](../verification/banc-natural-gait-m3-20260919/parent-e360-p11-human-authorization-v2.json)의 범위는 “First closed failure or Q actual debit closes scope.”이고 P12/원본 권한은 false입니다. 실제 첫 실패로 P11이 닫혔으므로 추가 runtime을 P11로 실행하지 않습니다. 검토된 수정안 적용과 다음 generated 체인에는 **새 수동 P12 1회 승인** 및 그 승인에 맞춘 별도 authority 통합 설계·리뷰·고정이 필요합니다. 이 문서는 P12 literal/activation/실행 승인이 아닙니다.

제안하는 한도는 기존798초 전체 체인, 추가 배정0, 원본 실행0, 자동 재시도0입니다. 남은 local860.680초에798초가 산술상 들어가며62.680초가 남습니다. 이 수치는 런타임 적합성이나 실제 완료의 증거가 아닙니다. 통합은 같은 ROOT/Astra-max/Sol6.1/ROOT/ONE-R1/ROOT 방식으로 먼저 검토하고, 이후 ROOT가 genuine admission과 실제 dispatch→full return을 계측·한 번 차감해야 합니다.

| 다음 체인의 고정 순서 | outer cap |
|---|---:|
| Ruff | 3초 |
| scanner | 20초 |
| fixture_build | 25초 |
| fixture_worker | 55초 |
| fixture_seal | 15초 |
| suite_build | 30초 |
| suite_worker | 600초 |
| suite_seal | 20초 |
| qualification | 30초 |
| 합계 | 798초 |

첫 닫힌 실패 또는 Q의 실제 차감에서 새 범위를 종료하고, 실패 worker가 pending이면 같은 worker의 필요한 seal/실제 once debit/owned closure만 수행합니다. 과거 비용·권한·실패를 리셋하거나 원본 실행을 추가하지 않습니다. 고정84/252 전체 경로를 통과하더라도 이것만으로 BANC 자연 보행·생물학적 기전 또는 전체 목표 완료를 선언하지 않습니다.

## 전체 목표

활성 목표는 [BANC M3 목표 계획](BANC_NATURAL_GAIT_M3_GOAL_PLAN_20260919.md)의 고정 F1–F8 전체입니다. BANC-only·해부학적 provenance·M2/UI/F6·기존 실패를 유지합니다. 이번 단계에서 후보를 채택하거나 보행 품질을 개선했다고 판정하지 않았습니다. 실제 Goal API 마지막 조회는 active(tokensUsed142029464/timeUsedSeconds643460)이며 목표는 미완료입니다.

[ROOT SECOND](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-ROOT-SECOND.json), [ROOT FIRST](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-ROOT-FIRST.json), [Sol ONE R1 인계](../verification/banc-natural-gait-m3-20260919/parent-e360-compact-prespawn-sol-R1-handoff.json), [반영 설계](BANC_M3_E360_COMPACT_PRESPAWN_SOURCE_REFLECTED_20261004.md)를 보관했습니다. 각 최종 산출물은 CreateNew → Flush(true) → close → 독립 strict UTF8/bytes/SHA 재읽기로 고정합니다.
