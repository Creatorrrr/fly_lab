# S18 원시 읽기 경로 범위 — 2026-10-09
B = verification/banc-natural-gait-m3-20260919. ROOT는 실제 S17 311/193 정산을 전체 재구성했고, S18의 사전 자료를 Source/metadata 작업으로만 작성했다. 실제20 이전이나 프로젝트 실행은 하지 않았다.

현재 catalog는 B/s18-root-catalog-v1.json 93088 / 1f471860f97116c186dd39a372c144694aa88c9369194a69276431e2a17bd7d3이다. 전체296행/263경로/기존 raw0 다섯 개를 다시 인증했고, 이전253행은 순서와 전체 값을 그대로 보존했다. 새43행은 이전 Source의 실행된11개 원시 보관본, 이전 b5/t5 JSON 읽기, 실제 S17 Source/review/adoption/recovery/요청/반환/비용/outer/stdout/311, 기존 provenance 및 새 prospective20 event/post311를 포함한다. catalog와 current S18 Source/c5/review/provenance의 순환은 허용하지 않는다.

새 S17 stderr0는 이미 실제 정산과 보관한 원본 및 인증한 outer에서 확인했다. 이를 Source runtime에서 새 raw0 허용 경로로 추가하지 않는다. 실제311을 명시적인 whole trust로 받아들이는 경로이며 과거 실행 의미를 재연산하지 않는다. 다섯 기존 empty 예외와 각 경로의 raw/JSON 여부를 그대로 보존한다.

새로 필요한 정확한 named524288 경로는 다음 여섯 개다. 기존 named 경로와 예외를 모두 보존한다.

- B/s17-bootstrap-settled311-HOLD-20261009.json: 실제 pre20 원본.
- B/s18-anchor311-after-repair20.json: prospective post20, 실제 적용 확인 전에는 정산 증거가 아니다.
- B/s17-recovered310-before-bootstrap.json: 실제 S17 비용 이전 역사 자료.
- B/parent-s18-bootstrap-settled312-before-allocation-20261009.json.
- B/parent-s18-qualification-settled316-before-body-20261009.json.
- B/parent-s18-body-settled317-before-audit-20261009.json.

Module 두 개65536, ordinary262144, 기존 정확한 소유 토큰16384, finite named ledgers524288을 유지한다. catalog kind=json도 기존과 같이 raw 인증이므로 별도 재파싱 제거를 주장하지 않는다.

최종 구현의 모든 실제 callsite가 이 유한한 범위에 포함되는지는 Sol과 ROOT가 최종 Source를 기준으로 다시 대조해야 한다. 현재 행 수/해시 인증은 Source 채택, full4 PASS, actual20 적용, qualification, body, F1–F8 통과를 뜻하지 않는다.

