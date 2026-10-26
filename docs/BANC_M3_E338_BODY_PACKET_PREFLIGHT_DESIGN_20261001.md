# E338 신체 대조 자료 사전 검사와 선택 로더

2026-10-01 ROOT 설계. 기존 finite goal의 자연 보행 F1–F8을 유지한다. 이 단계는 E337의 실제 로더 오류를 수정하는 필수 구현 checkpoint이며, 보행 가설 시험 또는 새로운 보행 후보가 아니다. E337의 technical invalid/원본1/retry0와 frozen source·실패 산출물을 보존한다. E335 기각, M2, 미사용 F6, UI는 그대로 둔다.

## 입력 오류와 단계 결과

E337은 `input_members[path]`를 member map으로 사용했지만 실제 E336 inventory는 `{sha256, bytes, members}` 컨테이너였다. Command NPZ의 `command`, `tendon`, `motor_rate32`, `control_tick`, `network_tick`, `physics_time_s`를 바깥 객체 키와 비교해 selected payload/constructor 전에 실패했다. 생성 fixture는 해당 실제 중첩 스키마를 포함하지 않았다. 원본51.2932739초를 한 번 합산한 현재 cumulative9565.21586319952/9660초, 예약0, 잔여94.78413680047925초다. 독립 saved artifact audit8 PASS는 invalid 보존만 확인했으며 보행 수치 검산은 하지 않았다.

이번 결과는 원본 값을 읽거나 신체를 실행하기 전에 **5개 입력 archive의 모든 헤더를 검증하고, 물리 비교가 사용하는 선택 schema를 확정하는 실행 가능한 사전 검사**다. 원본 E337 비교의 자동 재실행0. Phase/파형/블렌드/판정 기준/몸체/신경 회로·효능은 변경하지 않는다. Science outcome은 not evaluated로 유지한다.

## 구현 범위와 계약

GPT6.1 Sol이 `tools/qualify_c_banc_m3_body_packet.py`와 `tests/test_c_banc_body_packet.py`만 작성한다. 과거 tools/tests를 덮어쓰거나 monkeypatch하지 않는다. 새 module은 E337 frozen source를 읽기만 하는 `QualifiedReplayDependencies` adapter를 제공할 수 있다. Adapter의 실제 값을 읽는 API는 향후 별도 원본 예약/계약 아래 사용할 후보 API이며 이번 CLI에서 호출하지 않는다. 기존 런타임 연결·신체 생성·자료 전체 load·채택0.

검사 대상은 exact pinned E335 `control-raw.npz`와 `native-contact.npz`, E332 `reference.npz`, E336 `live-smooth.npz`, E103 `startup-inputs.npz`의5 archive다. E337 freeze의 해당 file SHA와 E336 nested descriptors를 그대로 상속한다. Live/startup은 inherited `input_members`에 없으므로 누락을 flat descriptor fallback으로 메우지 않는다. 정확한 frozen file SHA를 검증한 뒤 전체 typed header inventory를 기록하고, 아래 역할의 고정 selected schema와 비교한다. Full file byte identity가 해당 archive 내용의 provenance이며 새 header inventory는 실제 읽은 metadata로 표시한다.

역할별 선택 항목은 control=`command[301,42]/float64`, `control_tick[301]/int64`, `physics_time_s[301]/float64`; native=`time_s[15000]`, `normal_load_bw[15000,6]`, `slip_mm_s[15000,6]`, `upright[15000]`의float64와 `slip_valid[15000,6]`, `nonfoot_contact[15000]`의bool; reference=`curves[6,128,7]/float64`; live=`q[201,42]`, `z[201,6,2]`, `R[201,6]`, `phase_phase[201,6]`의float64와 `phase_valid[200,6]/bool`; startup=`target_command[99,42]`, `target_tendon[99,6]/float64`다. 모든 leg ordering은 lf,lm,lh,rf,rm,rh다. Live phase의 의미·값·R·역진·보행 quality를 헤더로 판정하지 않는다.

Nested descriptor는 container의 `sha256/bytes/members`를 검증한 뒤 `members` 전체 roster를 actual header와 대조한다. 전체 member의 dtype/byte order/shape/Fortran flag/NPY version/header length/ZIP compression/compressed and uncompressed length/CRC가 제공되면 모두 exact 비교한다. 기존 nominal dtype를 NumPy dtype metadata로 정규화하되 endian 또는 Fortran flag를 잃지 않는다. 임의 평탄화/누락 키 fallback/검증 제거0. File bytes/SHA는 source pin과 descriptor가 일치해야 한다.

Archive는 모든 member를 검사한다. Duplicate member, 경로 포함 이름, non-NPY, object/structured/subarray, 미지원 NPY version, 잘못된 차원/크기/형식,128개 초과 또는 declared decompressed128MiB 초과를 거부한다. 각 stream은 NPY header까지만 읽고 payload value는 decode하지 않는다. Headers 검사 후 파일 SHA를 재확인하여 검사 도중 변경을 거부한다. 선택하지 않는 neuron 배열도 header 안전성은 확인하지만 값은 읽지 않는다.

생성 자료에서만 사용할 선택 값 reader는 이 검증을 먼저 수행하고 allow_pickle=False로 역할의 selected keys만 decode한다. Requested key는 역할 allowlist의 subset이며 추가 neural keys를 허용하지 않는다. Selected dtype/shape/finite 검사를 반복한다. Slip의 valid NaN은 별도 validity mask로 확인하며 unloaded NaN 때문에 unsafe dtype를 허용하지 않는다. Header에 Fortran flag가 있으면서 singleton dimension에서 NumPy C/F flag가 둘 다 true일 수 있으므로 header ordering 검증을 array flag로 대체하지 않는다. 원본 CLI는 값 reader/old Dependency.load를 호출하지 않으며 원본 numeric/body/neural/CUDA0을 직접 증명한다.

## 자격, 고정 판정과 자원

ROOT 설계 → actual gpt-6-astra/max read-only 설계 리뷰 → ROOT 반영 → actual gpt-6.1-sol/high 구현 → ROOT FIRST → 필요시 같은 Sol ONE R1 → ROOT SECOND → 잔여 ROOT 수정이다. Effort high는 nested provenance, full archive header, selective decode, fail-closed publication과 물리 reader adapter의 조합에 적합하다. 설계 리뷰 단계에서 구현을 시작하지 않는다.

Sol/ROOT generated CPU각90초, host2GiB, retained64MiB, 실패 포함/owned PID 종료 후 결과 확인. 생성 NPZ에 실제 중첩 metadata 구조와 선택하지 않는 neuron 배열을 포함한다. E337 구 loader는 그 fixture에서 roster 오류로 거부하고 새 loader는 승인할 것을 직접 확인한다. 추가·누락·duplicate/unsafe member, dtype/endian/order/size/SHA 변경, selected-only decode, 원본 CLI의 payload 금지를 검증한다. 기존 오류를 회피하도록 fixture를 바꾸지 않는다. 과거 실패·source SHA는 보존한다.

ROOT 원본 metadata-only preflight는 감독30/internal25/preparation+headers20/final5초/CPU1/GPUhidden/thread1/host2GiB/output2MiB, job1, original numeric analysis0이다. Mutable checkpoint를 frozen 입력으로 넣지 않는다. E337 source207/input2951의 immutable closure와 현재 owned2·actual review/reflection/current generated proof를 byte pin으로 검사한 후 target5 header를 검사한다. Full inherited file validation은 byte hashing이고 원본 numerical analysis가 아니다. 시간은 gate 이전 s0에서 시작해 고정하며, finalization/readback/posthash에도 적용한다. 새 original cumulative allocation 상승·original reserve0.

Fresh literal output `verification/banc-natural-gait-m3-20260919/e338-body-packet-preflight`에 typed header/selected schemas/검사 receipt와 첫 오류·secondary 오류·고정 deadline·0 실행 카운터를 저장한다. 결과는 all5 schemas valid AND all source/input pins valid AND generated proof current AND publication/readback/deadline valid일 때만 engineering qualified다. 부분/timeout은 technical invalid며 자동 retry0. 성공이 신체 replay exact, rollout health, gait, H337, NN 학습 가능성 또는 F1–F8을 의미하지 않는다.

성공 후 새로운 과학적 방향을 설계할 때 이 후보 adapter를 사용하되 이미 끝난 E337 원본을 자동 재시도하지 않는다. 현재 목표의 미해결 핵심은 neural-only 운동 출력이 다리의 stance/swing 추진과 협응으로 전달되는지이며, 어떤 다음 개입도 그 근거와 새 비교 계약을 먼저 작성한다.

## Actual Astra/max M1–M6 반영: 이전 본문의 해당 조건보다 우선

실제 collaboration `/root/e338_astra_design_review`의 GPT6 Astra/max가 사전 설계 SHA22333cf522f5c8437f5cfb71809b0407d5c85c977db13c4c64819f403464b94e를 읽고 조건부 구현 가능으로 판정했다. 지정 source/JSON descriptor만 읽었으며 NPZ header/value 또는 F1–F8 전문은 재감사하지 않았다. ROOT는 아래6항목 모두 수용한다. 원본 archive/header/runtime 검증을 리뷰가 수행한 것으로 소급하지 않는다.

M1. Container 필수 키는 정확히 sha256/bytes/members이며 각 member 필수 키는 member/dtype/shape/fortran_order/npy_version/header_bytes/zip_compression/compressed_bytes/uncompressed_bytes/crc32다. 필수 키·타입·member=key+'.npy'·전체 roster를 검사하며 누락 fallback은 없다. E336 helper의 fortran/compression을 inventory의 fortran_order/zip_compression에 대응한다. header_bytes는 NPY 시작부터 payload 시작까지 stream.tell()다. Live/startup은 inherited_roster_match가 아니라 frozen file SHA에 귀속된 새 actual header observation이다. E338 immutable 입력에는 E337 final-decision SHA c476cb52f867747ef2318398a5f8a5c7e5a03af176836bf6c7f068fd2310fe82 및 그 decision의 evidence map과 실패 terminal pins를 추가한다. E337의 원본 allocation/authorization 필드는 실행 권한으로 상속하지 않는다.

M2. 이번 새 module은 **독립 header qualification API + 생성 자료 전용 selected decode API**만 구현한다. QualifiedReplayDependencies adapter 또는 기존 load()/run() 연결은 이번 범위에서 제외한다. 기존 static Dependencies.selected()는 생성 임시 root/NPZ/nested descriptor로 직접 호출하여 원래 정확한 roster 오류를 재현할 수 있지만, 기존 GeneratedDependencies.load() 우회 fixture를 자격 증거로 사용하지 않는다. Metadata CLI에는 decoder 활성화·generated original 우회 옵션이 없다. 생성 decode API는 원본 repository의5 archive 경로와 해당 원본 SHA를 거부하며, 실제 원본 값 읽기에 대한 authorization으로 해석하지 않는다. 향후 원본 연결은 별도 계약·예약·현재 소스 리뷰가 필요하다.

M3. 역할은 E338 freeze의 exact role/path/pin과 결합하고 shape/dtype/finite 정책을 caller가 교체하지 못한다. 생성 자료는 고정된 역할별 selected spec을 사용하며 원본 경로/SHA를 바꿔 허용하는 fixture는 금지한다. 허용 밖 key는 np.load 이전에 거부한다. Slip 요청에는 slip_valid와 normal_load_bw도 명시적으로 포함해야 하며 동반 항목을 몰래 읽지 않는다. 실제 생성 값은 isfinite(slip)==slip_valid, invalid 위치NaN/inf금지, load>=0, load>=.02BW이면 valid를 검사한다. 다른 subset 요청으로 이 검사를 수행할 수 없으면 거부한다.

M4. 역할 정책은 <f8/<i8/|b1으로 고정한다. Inherited nominal float64/int64는 **little-endian archive 계약**으로 명시적으로 정규화하고 host native endian에 맡기지 않는다. 원래 descriptor에 explicit endian 문자열이 기록됐다는 주장은 하지 않는다. 실제 새 헤더의 dtype/order는 observation이며 file SHA가 provenance다. 비선택 member도 같은 full header 정책으로 검사한다. 생성 Fortran 승인/order불일치/endian불일치/singleton C/F동시true 사례를 포함한다.

M5. E338 generated CPU allocation은 Sol90(ROOT FIRST와 동일 Sol ONE R1 포함), ROOT90(ROOT 생성 검산+실패+metadata preflight 포함)으로 고정한다. E337의 Sol72.4446824/90 및 ROOT47.0085676/90와 다른 단계별 비용은 별도 역사 기록을 보존하며0으로 초기화하지 않는다. E338 metadata job의 모든 실제 감독 wall은 ROOT E338 ledger에 실패 포함 한번 합산하고, original numerical-analysis cumulative9565.21586319952/9660은 변경하지 않는다. Header bytes/hash 작업0이라는 주장을 하지 않는다. Job1/자동 retry0이며 gate 실패와 output 미생성도 사용한 attempt다. s0는 worker gate 이전, 준비/검사<=s0+20, final deadline=min(s0+25, final_begin+5)이다. 마지막 posthash/게시/readback도 그 시계에 포함하고 지연 시 deadline을 늘리지 않는다. Scanner는 한 pass로 각 archive header를 읽고, 전체 byte hash와 required posthash를 구분해 기록한다. 전체 pin 검증에 반복 expensive process-tree sampling을 넣지 않고 cheap wall checks와 outer RSS supervisor를 사용하되 RSS 확인 cadence를 기록한다.

M6. Worker는 internal qualification/publication agreement와 **외부 ROOT engineering_qualified**를 구분한다. 외부 감독 성공/owned tree 종료/현재 원자료·terminal hash/readback/한도 충족까지 확인한 뒤만 ROOT가 engineering qualified를 결정한다. Metadata 접근과 numeric decode는 서로 다른 receipts다. Full-file hashing/ZIP 버퍼링은 payload bytes를 읽을 수 있으므로 수치 decode/분석0으로 제한하여 보고한다. CRC는 ZIP header 필드를 비교한 것으로 whole payload CRC 독립 재계산이 아니다. 생성 시험에서 selected decoder 및 금지 body/neural/CUDA 경로를 실제 실패 sentinel로 바꾸고 metadata path가 도달하지 않음을 확인한다. Publication 실패/last-readback 변조·timeout/posthash불일치/all sink failure를 확인하며 첫 오류를 유지한다. 마지막 읽기 뒤 deadline 실패는 success를 철회하고, 철회 sink까지 실패하면 외부 ROOT completion이 stale disk success를 채택하지 않는다.
