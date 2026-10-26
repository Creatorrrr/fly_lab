# E338 ONE R1: 과거 선택 inventory와 새 전체 헤더 관측의 구분

2026-10-01 ROOT 수정 설계. 사용자 지정 절차는 ROOT 설계 → 실제 GPT6 Astra/max 리뷰 → ROOT 반영 → 기존 GPT6.1 Sol/high의 ONE R1 → ROOT SECOND → 잔여 ROOT 수정이다. 이 문서는 첫 E338 설계나 실제 실패 기록을 덮어쓰지 않는 추가 계약이다. 자연 보행 F1–F8, H337 판정, E337 원본 job1/retry0, E338 실제 header job1/retry0, 비용과 모든 실패는 유지한다.

## 새 근거와 첫 리뷰 보완

E338 생성 73 PASS와 ROOT 독립 42 PASS 뒤 실행한 실제 metadata CLI는 첫 control 역할에서 `full inherited archive metadata differs`로 technical invalid가 되었다. 감독 3.694047099997988초를 ROOT E338 공유90초에 합산했다. 원본 수치 decode·신체·신경·CUDA0이며 전체 5개 archive 자격은 획득하지 못했다.

별도 사전 등록된 control header 진단은 0.22591629999806173초로 완료됐으며 numeric0이다. `parent-e338-control-header-postmortem.json`의 실제 목록은7개이고 과거 descriptor는6개다. 공통6개의 metadata, 전체 파일 SHA와 크기는 일치한다. 빠진 배열은 `activation[301,96]/<f8`다. ROOT가 현재 읽은 과거 생성기 `parent-e336-inventory-v2.py`는 명시적으로 selected keys만 inventory에 넣었다. 이것은 과거 전체 목록과 불일치를 허용하라는 근거가 아니라, 과거 기록을 전체 목록이라고 부른 E338 설계 가정이 잘못됐다는 근거다. 첫 CLI는 당시 계약대로 실패한 상태를 보존한다.

과거 `archives`의 `{sha256,bytes,members}` 구조는 정확하지만 members의 의미는 **historical selected metadata**다. ROOT가 현재 읽은 E336 verifier `tools/analyze_c_banc_m3_saved_gait_transfer.py:verify_inventory`는 기록된 member 각각을 비교하고, 이 archive member 비교에서 전체 equality를 요구하지 않는다. 이번 Astra 리뷰는 이 verifier를 허용 목록에 포함하지 않아 이 문장을 독립 확인하지 않았다. 확인된 생성기 동작과 ROOT source 확인을 구분한다. 과거 코드/JSON을 수정하지 않는다. 이 추가 설계 작성 당시 ROOT 공유 누적은4.798238299998047/90초였다. 이후 saved-failure 감사 감독.11613070000021253초를 한 번 더 계상한 현재 누적은4.91436899999826/90초이다. 기존 Sol FIRST 누적1.6528338999996777/90초이며 남은 R1과 ROOT SECOND 비용은 같은 각90초에 실패 포함 합산한다. 원본 numerical cumulative9565.21586319952/9660초와 예약0은 유지한다.

## 수정할 API와 의미

1. `scan_archive`의 기존 기본 `complete` 비교를 유지한다. descriptor 키가 누락되거나 추가되거나 metadata가 다르면 거부한다. `decode_generated`는 항상 이 완전 descriptor 모드를 사용한다. 기존 default의 검사를 subset 비교로 몰래 바꾸지 않는다.
2. 별도 명시적 scope `historical_selected`를 metadata 전용 경로에 추가한다. 자동 fallback·roster 오류를 보고 scope 추론0. `make_freeze`는 control/native/reference에만 이 scope를 고정하고 live/startup은 inherited descriptor가 없는 새 observation으로 고정한다. 역할/path/SHA/정책은 gate가 current freeze와 exact 비교한다. 새로운 freeze schema는 v2로 표시하며 이전 failed v1 freeze를 현재 계약으로 받아들이지 않는다.
3. `historical_selected`에서도 container/member의 정확한 필드·타입·SHA·전체 파일 bytes와 기록된 각 member의 모든 metadata를 대조한다. 해당 member가 실제 전체 scan에 없거나 metadata가 다르면 거부한다. 모든 실제 member에 duplicate/name/object/endian/order/NPY/크기/총량 검사를 적용한다. 파일 prehash/posthash와 절대 deadline은 그대로다. 미기록 safe member의 값을 decode하지 않는다. 파일의 완전 SHA가 내용의 provenance이고 실제 전체 scan은 새 관측이다.
4. receipt는 `descriptor_scope`, `historical_selected_metadata_match`, `inherited_roster_match`, `unrecorded_inherited_members`, 전체 `descriptor`를 분리한다. historical_selected에서 `inherited_roster_match=true` 또는 `complete_inherited_inventory=true`라고 표시하지 않는다. scope가 complete이면 기존 exact 비교 의미를 유지한다. live/startup의 inherited match는 null이다. 나중에 전체 관측을 완전 descriptor로 재사용할 때는 별도 현재 source review/계약이 필요하다.
5. mismatch 예외에는 비교 scope, actual/expected roster, 누락/미기록 키, 공통 member별 필드 차이를 bounded structured metadata로 보존한다. 전체 scan이 끝난 mismatch이면 필요한 posthash 후 관측을 남긴다. posthash·deadline·안전성 실패를 qualified observation으로 표시하지 않는다. worker는 실패 역할/metadata를 기록하되 성공 receipt나 과학적 결과로 승격하지 않는다. 첫 오류와 secondary/publication 철회 규칙을 유지한다.

## 불변성과 실행 범위

Sol은 기존 owned2 `tools/qualify_c_banc_m3_body_packet.py`, `tests/test_c_banc_body_packet.py`만 수정한다. frozen E337/E336 source, 최초 E338 source snapshot, v1 freeze/review/proof, 실제 terminal output/감독/진단과 비용을 보존한다. R1은 이 추가 설계·실제 Astra 리뷰·ROOT reflection 및 postmortem provenance를 새로운 current immutable pins로 추가한다. old design의 역사 pins는 유지한다. mutable 비용/checkpoint를 source closure에 넣지 않는다.

이번 ONE R1은 생성 자료와 이미 저장된 header JSON만 사용한다. 원본 archive header 재스캔·원본 값·신체·신경·CUDA·예약·기존 CLI 재실행·새 원본 CLI output 디렉터리0이다. 코드가 수정됐다는 이유로 E338 실제 header 실패를 성공으로 소급하지 않는다. E338은 실제 preflight 기술적 실패이며, R1의 통과는 생성 구현 자격으로만 보고한다. 향후 실제 원본 실행은 다른 과학적 질문과 별도 사전 계약을 ROOT가 설계하고 리뷰받기 전까지 없다.

## 생성 회귀와 ROOT SECOND

생성 control NPZ에 실제 구조와 같은 command/tendon/motor_rate32/control_tick/network_tick/physics_time_s/activation7개를 넣고, 이전과 같이6개만 기록된 **명시적 selected** descriptor를 만든다. 기본 complete 경로가 이 descriptor를 거부하고 structured actual7/expected6을 보존하며, 명시적 historical_selected 경로만 공통6개 exact 및 전체7개 안전 검사를 통과해야 한다. 원본 수치나 원본 압축 payload를 fixture로 복사하지 않는다.

누락 recorded member, CRC/dtype/order/shape/version/size/pin 차이, 미기록 unsafe member, 잘못된 scope/역할 결합, native/reference/live/startup scope, generated decoder의 partial descriptor 거부와 선택 외 neural/activation 값 미해석을 검증한다. 실패 metadata가 publication/readback에 남으며 worker 전체 결과는 invalid인 사례를 포함한다. 기존 endian/Fortran/singleton/slip/원본 deny/publication/시간 검사를 보존한다. 과거 가짜 전체 fixture PASS를 실제 자격으로 재사용하지 않는다.

Sol shared90초 FIRST+R1, ROOT shared90초 generated+이미 실행한 header+진단+SECOND이다. host2GiB/retained64MiB/thread1/GPUhidden, 감독 owned PID 부재 후 비용·산출물 검산. 아직 안 쓴 동일 Sol ONE R1을 이 수정에 한 번 사용한다. ROOT SECOND에서 남은 문제가 있으면 Sol 재수정 대신 ROOT가 처리한다. 최종 engineering_qualified(실제5 archive)=false, H337=not_evaluated, candidate_adopted=false, F1_F8_complete=false를 유지한다.

## Actual Astra/max R1 M1–M5 반영: 위 본문보다 우선

2026-10-01 실제 collaboration `/root/e338_astra_r1_review_revalidate`(gpt-6-astra/max/fork none)가 사전 설계 SHA33ce7af67ab52a7437e7c425cc26e9202fadc4edfb83e651c1dbc2ffbdfd908d를 읽고 conditional implementable로 판정했다. ROOT는 M1–M5 전부와 선택 권고를 아래처럼 반영한다. 이 리뷰는 지정8개 source/JSON의 읽기·SHA 확인만 수행했고 원본 NPZ/header/value/full closure/F6·시험·편집0이다. 실제 호출값은 dispatch evidence이며 backend 내부 self-attestation은 주장하지 않는다.

M1. API 기본 complete와 descriptor 없는 observation의 기존 호출 호환성은 유지하되 각각 명시 scope로 표현한다. 성공 receipt truth table은 complete+descriptor=(historical_match null, inherited_roster_match true, unrecorded []), historical_selected+descriptor=(historical_match true, inherited_roster_match null, unrecorded actual-recorded), observation+no descriptor=(둘 다 null, unrecorded null)이다. historical_selected+None, observation+descriptor, unknown scope를 거부한다. Make-freeze binding은 control/native/reference에 descriptor 필수+historical_selected, live/startup에는 descriptor None+observation으로 고정하며 scope 누락/변조를 gate가 거부한다. Default complete+None을 유지할 경우 receipt scope를 observation으로 명시적으로 정규화하되 historical_selected로 fallback하지 않는다. 목록이 우연히 같아도 historical scope에서는 complete inherited match를 주장하지 않는다. Generated decoder는 scope 변경 인자가 없고 complete+full descriptor를 명시 호출한다.

M2. 비교 오류와 관측 검증을 분리한다. missing_recorded_members=expected-actual, unrecorded_members=actual-expected, 공통 필드 차이는 expected/observed로 고정한다. 안전한 전체 scan 뒤 mismatch를 먼저 기억하고 같은 deadline의 posthash를 수행한다. 이어 posthash/timeout이 실패하면 comparison 오류를 first로 유지하고 후속 오류를 secondary에 남기며 posthash_verified=false다. 안전성 도중 실패는 partial observation/full_header_scan_completed=false를 표시한다. Qualification 성공이 아닌 failure_observation에만 실패 metadata를 넣고, role schema 오류도 실제 역할과 이미 얻은 관측을 보존한다. Structured failure detail 직렬화 상한64KiB를 적용하며 초과 시 metadata_truncated=true, 생략 항목/사유를 명시한다. 필수 첫 오류·상태·비교 scope는 생략하지 않는다. 전체 output2MiB·절대시계·철회 규칙은 유지한다. ZIP CRC independent recomputation0이다.

M3. 새 freeze/result/receipt는 R1 v2로 구분한다. Current R1 설계/실제 review/ROOT reflection/postmortem source+contract+감독+PID/report/actual-failure final-decision을 byte pin으로 연결하고 reflection이 정확히 current R1 design과 review, all M1–M5 반영을 참조하는지 검증한다. FIRST 디자인/review/reflection은 역사 pins로 보존하며 이것만으로 R1을 승인하지 않는다. FIRST v1 freeze/review/generated proof/source2 snapshots도 역사 artifact로 보존한다. 새 generated proof는 v2/current owned2와 현재 R1 실행 증거에 귀속되며 최초 PASS를 재사용해 current qualification=true라고 만들지 않는다. Mutable ledger/checkpoint는 closure에서 제외한다.

M4. 새 값으로 생성한7개 control/기록6개 fixture, default complete 거부/historical 승인/full decoder partial 거부/activation value 미해석을 검증한다. Original identity constants(역할별5 path/SHA)를 바꾸지 않는다. Make-freeze는 mock하지 않고 생성 JSON fixture를 직접 읽어 descriptor 필수/역할 scope/R1 reflection exact 결합을 검증한다. 생성 JSON만의 작은 pin namespace는 실제 metadata pin 검사를 사용하며 원본 path/SHA는 symbolic JSON 문자열로만 기록한다. 이 시험은 actual full closure/hash 성공으로 부르지 않는다. Gate의 v1/scope 변조는 leaf/full closure 접근 전에 거부하는 것을 sentinel로 확인한다. Worker 생성 회귀에서는 별도 local generated gate fixture를 명시 구분하여 publication/readback mismatch, mismatch→posthash/timeout secondary, role-schema failure, unrecorded unsafe member, no original decode/import를 실제 실패로 검사한다. 원본 archive 및 실제 full source/input closure hash 접근은 sentinel로 차단한다.

M5. ROOT current4.91436899999826/90초와 Sol FIRST1.6528338999996777/90초에서 실패 포함 계속 계상한다. 최초 디자인의4.7982383는 역사 스냅샷이며.1161307 감사는 중복 charge하지 않는다. 실제 E338 metadata job1/retry0와 별도 control diagnostic job1은 서로 다른 기록이다. ONE R1/SECOND 원본 header/value/full closure/CLI0, F6 exposure0, 실제 engineeringfalse/H337 not_evaluated/adoptionfalse/F1–F8false를 유지한다. Future complete-descriptor observation 재사용은 별도 현재 source review/과학적 계약을 필요로 한다.
