# E338 신체 대조 자료 사전 검사: 실제 실행의 기술적 실패

2026-10-01 ROOT 기록. 자연 보행 목표/F1–F8은 미완료이며 H337은 not_evaluated다. 이 기록은 배열 값·신체·보행의 수치 검산이 아니다.

ROOT 설계 → 실제 GPT6 Astra/max 리뷰와 M1–M6 반영 → 실제 GPT6.1 Sol/high 구현 → ROOT FIRST까지 수행했다. Sol 생성73 PASS/Ruff PASS, ROOT 생성42 PASS였다. 그 뒤 사전 등록된 실제 header-only CLI 한 번은 control 역할의 `full inherited archive metadata differs`에서 실패했다. worker와 result/receipt/stdout은 technical_invalid로 일치하며 engineering_qualified=false다. 실패를 성공으로 바꾸거나 원본 CLI를 재시도하지 않았다.

## 직접 확인한 원인

독립 구현의 한 번짜리 control header 진단은 파일 SHA/크기와 공통6개 항목의 모든 metadata가 과거 descriptor와 일치함을 확인했다. 실제 archive에는 `activation[301,96]/<f8`이 추가로 있어 전체7개였다. 과거 E336 inventory 생성기는 selected6개의 metadata만 기록했다. 이를 완전한 전체 roster로 가정한 ROOT의 E338 설계가 잘못됐다. 최초 scanner는 당시 계약에 맞게 거부했다. 기존 생성 control fixture는6개만 포함하여 이 차이를 포착하지 못했다.

이 진단은 새로 읽은 header observation이며 값 decode는0이다. ZIP 버퍼링/파일 해시는 payload bytes를 읽을 수 있고, CRC는 ZIP field 관측이다. payload의 숫자 해석 또는 CRC 독립 재계산을 수행한 것으로 주장하지 않는다. Native/reference/live/startup 전체 실제 사전 검사는 이 실패로 완료되지 않았다.

## 검산과 자원

ROOT가 저장된 산출물만 순차 검산하여17개 일관성 항목을 확인했다. result/receipt/stdout equality, terminal SHA/bytes, 실패/0 실행 카운터, no-array outputs, 최초 source identity와 감독/PID 종료 증거, 공통 metadata/추가 activation, 미사용 R1을 확인했다. 별도 모델 리뷰나 보행 수치 검산은 아니다.

실제 metadata CLI 감독3.694047099997988초, 별도 진단.22591629999806173초, 저장 기록 검산.11613070000021253초를 실패 포함 합산했다. ROOT E338 누적4.91436899999826/90초, Sol FIRST 누적1.6528338999996777/90초다. 원본 numerical-analysis cumulative9565.21586319952/9660초/예약0, E337 실패와 과거 자원 기록을 유지한다. 관리 명령 wall은 세션 기록에 있으며0으로 주장하지 않는다. 감독 owned PID 종료 후 비용을 기록했다.

## 수정 설계와 남은 절차

추가 설계는 `docs/BANC_M3_E338_R1_SELECTED_INVENTORY_DESIGN_20261001.md`다. 기본 complete descriptor의 exact 검사와 생성 decoder를 유지하고, metadata 전용 historical_selected scope/new full observation을 명시적으로 구분한다. mismatch의 관측/roster/필드 차이 보존과 생성7개/기록6개 회귀를 요구한다. 이것은 E338 실제 header 실패에 소급 적용하지 않는 새 수정 계약이다.

추가 설계의 실제 GPT6 Astra/max 리뷰는 direct spawn, 이전 reviewer follow-up, 기존 Sol의 subtree spawn 모두 `agent thread limit reached`로 실패했다. 리뷰어가 생성되거나 리뷰가 완료됐다고 기록하지 않았다. `consulting-codex-cli/SKILL.md`는 immediate Codex caller의 호출을 금지하므로 CLI 우회도 실행하지 않았다. 같은 모델의 CLI 리뷰에 이 스킬 예외를 허용할지 사용자에게 질의했다. 답변 전에는 지정 리뷰를 우회하거나 source를 수정하지 않는다.

리뷰 완료와 ROOT 반영 뒤 아직 사용하지 않은 **같은 Sol ONE R1 → ROOT SECOND → 남은 ROOT 수정**을 진행한다. 현재 owned2는 최초 제출 SHA 그대로이며 R1 사용0이다. 실제5 archive engineering qualification=false, H337 not_evaluated, candidate adoption=false, F1–F8 complete=false다. 자연 보행 개발 목표는 유지한다.

주요 근거: `parent-e338-root-first-review-supplement.json`, `parent-e338-control-header-postmortem.json`, `parent-e338-saved-failure-audit.json`, 최초 source snapshots/v1 freeze/review/proof와 `e338-body-packet-preflight/` 실패 terminal, ROOT/Sol cost ledgers. 저장 경로는 모두 `verification/banc-natural-gait-m3-20260919/` 아래다.
