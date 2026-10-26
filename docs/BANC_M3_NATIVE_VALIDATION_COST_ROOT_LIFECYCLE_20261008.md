# Native comparator lifecycle 선택 — ROOT, 2026-10-08

실제 Sol FIRST 작성 중 질문에 ROOT가 다음 구현 선택을 답했다. Process 전체 compile1은 필수 조건이 아니다. ROOT reflected와 handoff의 process-local reuse 조건은 재사용하는 경우에 적용하며, 실제로 직접 얻은 owned Type capability와 안전한 수명이 우선이다. 원 design/reflected/handoff/실제 Astra 리뷰는 덮어쓰지 않는다.

FIRST variant는 각 helper extraction 또는 새 `& owner` script scope마다 cheap eligible-large-shape 이후 최대 한 번 lazy compile하고, 같은 scope의 warm3는 그 capability를 재사용하는 보수적 방식이다. Metadata input과 무관한 GUID namespace는 타입 충돌을 피하는 ownership 수단이고 comparator logic/template는 고정 Source다. Capability는 Add-Type -PassThru에서 직접 받은 Type만 사용하며 임의 타입 이름 검색이나 self-claimed marker로 되찾지 않는다. Scope 내부 compile/init 실패 상태도 bounded하게 유지하여 같은 실패를 반복 compile하지 않는다.

이 선택은 process-global marker/공유 data state를 추가하지 않고 기존 scratch 추출과 반복 script 호출을 지원하기 위한 판단이다. 실제 compile/runtime 의미와 성능은 아직 NOT_RUN이다. 각 invocation마다 새로운 cold 비용이 생기므로 기존 metadata Act 그룹 전체를 더 빠르게 만들지 못할 수 있다. 정확한 comparator template identity, namespace/Type provenance, cold/warm/attempt/fallback witness와 repeated invocation별 비용을 FIRST.md/json 및 미래 진단에 구분한다. Process 전체 compile1이나 전체 그룹 속도 개선을 주장하지 않는다.

기존 Native group assertions/timeouts/14–30 catalog 및 account의 다른 함수·Read-Pin·Generated-Budget는 그대로다. Root가 같은 Source 전체를 리뷰하고 필요한 동일 Sol sole R1/ROOT SECOND/ROOT 잔여를 마친 뒤 bounded generated diagnostic 조건을 다시 확인한다. Canonical/ledger/Original 변경과 새로운 보호 실행은 현재0이다. F1–F8 ACTIVE/INCOMPLETE, 후보 미채택과 biology/native VJP 미검증은 그대로 유지한다.
